"""
Test-àncora: la penalità anti-overkill Numba deve coincidere con il reward shaping canonico.

Perché esiste questo test
-------------------------
La penalità "overkill briscola" ha due implementazioni:
- `training/reward_shaping.py` (Python puro su `PlayerObservation`) — la reference didattica;
- `numba/observation.py::_trump_overkill_penalty_numba` (kernel JIT su card id) — quella
  effettivamente usata da `train_a2c` nel rollout compilato.

Senza un test di parità, un drift tra le due farebbe allenare i modelli con uno shaping
diverso da quello documentato/testato. Qui giochiamo partite reali col dominio canonico e,
a ogni decisione da secondo di mano, confrontiamo le due penalità per ogni carta giocabile,
in entrambe le modalità (`flat` e `gap`) e con/senza soglia `low_lead_points_max`.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from briscola_ai.ai.numba.mlp import _overkill_penalty_mode_code
from briscola_ai.ai.numba.observation import _trump_overkill_penalty_numba
from briscola_ai.ai.training.reward_shaping import trump_overkill_penalty, trump_overkill_penalty_gap
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba

_BETA = 1.0
# `low_lead_points_max`: nel kernel Numba un valore negativo equivale a `None` (soglia disattivata).
_LOW_LEAD_CASES: tuple[tuple[int | None, int], ...] = ((2, 2), (None, -1))
_MODE_CASES: tuple[tuple[str, int], ...] = tuple((mode, _overkill_penalty_mode_code(mode)) for mode in ("flat", "gap"))


def _numba_arrays_from_state(state: GameState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converte mani e tavolo dello stato canonico negli array numerici del kernel JIT."""
    hands = np.full((2, 3), -1, dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)
    for player_index in range(2):
        hand = state.players[player_index].hand
        hand_sizes[player_index] = len(hand)
        for slot, card in enumerate(hand):
            hands[player_index, slot] = card_to_id(card)

    table_cards = np.full(1, -1, dtype=np.int64)
    table_players = np.full(1, -1, dtype=np.int64)
    if state.table_cards:
        table_cards[0] = card_to_id(state.table_cards[0][0])
        table_players[0] = int(state.table_cards[0][1])
    return hands, hand_sizes, table_cards, table_players


def _python_penalty(mode: str, observation, *, chosen: int, low_lead: int | None) -> float:
    """Reference canonica: `flat` e `gap` condividono la stessa firma a meno del nome."""
    if mode == "flat":
        return trump_overkill_penalty(observation, chosen_card_index=chosen, beta=_BETA, low_lead_points_max=low_lead)
    return trump_overkill_penalty_gap(observation, chosen_card_index=chosen, beta=_BETA, low_lead_points_max=low_lead)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8, 13, 21, 42, 99, 123, 2024])
def test_overkill_penalty_numba_matches_reward_shaping_on_real_games(seed: int) -> None:
    """Su partite reali, il kernel JIT e la reference Python devono dare la stessa penalità."""
    rng = random.Random(seed)
    state = new_game_state(num_players=2, seed=seed)
    nonzero_penalties = 0

    while not state.game_over:
        player_index = state.current_turn
        hand_size = len(state.players[player_index].hand)

        # Il caso interessante è "secondo di mano": confrontiamo ogni carta giocabile,
        # non solo quella effettivamente giocata, per massimizzare la copertura.
        if len(state.table_cards) == 1:
            observation = make_player_observation(state, player_index)
            hands, hand_sizes, table_cards, table_players = _numba_arrays_from_state(state)
            assert state.trump_card is not None
            trump_id = card_to_id(state.trump_card)

            for chosen in range(hand_size):
                for mode, mode_code in _MODE_CASES:
                    for low_lead_py, low_lead_nb in _LOW_LEAD_CASES:
                        expected = _python_penalty(mode, observation, chosen=chosen, low_lead=low_lead_py)
                        actual = _trump_overkill_penalty_numba(
                            hands,
                            hand_sizes,
                            table_cards,
                            table_players,
                            1,
                            trump_id,
                            player_index,
                            chosen,
                            _BETA,
                            low_lead_nb,
                            mode_code,
                        )
                        assert actual == pytest.approx(expected), (
                            f"seed={seed} player={player_index} chosen={chosen} "
                            f"mode={mode} low_lead={low_lead_py}: numba={actual} python={expected}"
                        )
                        if expected != 0.0:
                            nonzero_penalties += 1

        card_index = rng.randrange(hand_size)
        state, result = step(state, PlayCardAction(player_index=player_index, card_index=card_index))
        assert result.error is None

    # Nota didattica: non tutte le partite contengono un overkill; l'assenza qui non è un errore.
    # La non-vacuità complessiva è garantita dal test dedicato qui sotto.


def test_overkill_penalty_parity_is_not_vacuous() -> None:
    """Almeno una penalità non nulla deve esistere nel campione, altrimenti la parità non prova nulla."""
    found = 0
    for seed in range(60):
        rng = random.Random(seed)
        state = new_game_state(num_players=2, seed=seed)
        while not state.game_over and found == 0:
            player_index = state.current_turn
            hand_size = len(state.players[player_index].hand)
            if len(state.table_cards) == 1:
                observation = make_player_observation(state, player_index)
                for chosen in range(hand_size):
                    penalty = trump_overkill_penalty(
                        observation, chosen_card_index=chosen, beta=_BETA, low_lead_points_max=None
                    )
                    if penalty != 0.0:
                        found += 1
                        break
            card_index = rng.randrange(hand_size)
            state, result = step(state, PlayCardAction(player_index=player_index, card_index=card_index))
            assert result.error is None
        if found:
            break

    assert found > 0, "Nessun caso di overkill nel campione: aumentare i seed o rivedere il generatore"


def test_overkill_penalty_numba_disabled_with_zero_beta() -> None:
    """Come nella reference, `beta <= 0` disattiva la penalità nel kernel JIT."""
    hands = np.full((2, 3), -1, dtype=np.int64)
    hands[0, 0] = 6  # SETTE di bastoni
    hand_sizes = np.asarray([1, 0], dtype=np.int64)
    table_cards = np.asarray([16], dtype=np.int64)
    table_players = np.asarray([1], dtype=np.int64)
    penalty = _trump_overkill_penalty_numba(hands, hand_sizes, table_cards, table_players, 1, 6, 0, 0, 0.0, -1, 0)
    assert penalty == 0.0
