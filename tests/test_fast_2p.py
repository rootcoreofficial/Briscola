"""
Test di equivalenza per il motore sperimentale fast 2-player.

Il modulo fast è pensato per performance, quindi questi test lo vincolano al dominio
canonico: stesso seed e stesse azioni devono produrre gli stessi eventi essenziali.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from briscola_ai.ai.fast.state_2p import Fast2PState, new_fast_2p_state, play_random_fast_2p, step_fast_2p
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, StepResult, step
from briscola_ai.domain.models import Card
from briscola_ai.domain.state import GameState, new_game_state


def _card_ids(cards: Sequence[Card]) -> list[int]:
    """Converte una sequenza di `Card` canoniche in card id numerici."""
    return [card_to_id(card) for card in cards]


def _assert_state_equivalent(canonical: GameState, fast: Fast2PState) -> None:
    """
    Confronta lo stato osservabile del dominio canonico con lo stato fast.

    Il fast path non conserva le carte catturate, quindi confrontiamo punti e zone attive
    della partita: deck, mani, tavolo, turno e risultato finale.
    """
    assert canonical.num_players == 2
    assert canonical.trump_card is not None

    assert fast.deck == _card_ids(canonical.deck)
    assert fast.trump_card == card_to_id(canonical.trump_card)
    assert fast.hands == [_card_ids(canonical.players[0].hand), _card_ids(canonical.players[1].hand)]
    assert fast.points == [canonical.players[0].points, canonical.players[1].points]
    assert fast.table_cards == [card_to_id(card) for card, _ in canonical.table_cards]
    assert fast.table_players == [player_index for _, player_index in canonical.table_cards]
    assert fast.current_turn == canonical.current_turn
    assert fast.first_player == canonical.first_player
    assert fast.game_over == canonical.game_over
    assert fast.winner_index == canonical.winner_index


def _assert_step_result_equivalent(canonical: StepResult, fast_played_card: int, fast_player: int) -> None:
    """Confronta i campi dello step che hanno un equivalente nel motore fast."""
    assert canonical.error is None
    assert canonical.played_card is not None
    assert canonical.player is not None

    assert fast_played_card == card_to_id(canonical.played_card)
    assert fast_player == canonical.player


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_new_fast_2p_state_matches_canonical_initial_state(seed: int) -> None:
    """
    Il deal iniziale deve restare identico al dominio canonico.

    Questo protegge ordine mazzo, shuffle, briscola reinserita in testa e mani iniziali.
    """
    canonical = new_game_state(num_players=2, seed=seed)
    fast = new_fast_2p_state(seed=seed)

    _assert_state_equivalent(canonical, fast)


@pytest.mark.parametrize(("seed", "action_seed"), [(0, 0), (7, 19), (123, 999)])
def test_fast_2p_replays_same_full_game_as_canonical(seed: int, action_seed: int) -> None:
    """
    Una partita completa con azioni casuali deterministiche resta equivalente step-by-step.

    La scelta casuale usa l'indice carta nella mano canonica; se i due motori divergessero,
    il confronto successivo su mani/deck/turni intercetterebbe il problema.
    """
    canonical = new_game_state(num_players=2, seed=seed)
    fast = new_fast_2p_state(seed=seed)
    rng = random.Random(action_seed)

    while not canonical.game_over:
        player_index = canonical.current_turn
        card_index = rng.randrange(len(canonical.players[player_index].hand))

        canonical, canonical_result = step(
            canonical,
            PlayCardAction(player_index=player_index, card_index=card_index),
        )
        fast_result = step_fast_2p(fast, player_index=player_index, card_index=card_index)

        _assert_step_result_equivalent(canonical_result, fast_result.played_card, fast_result.player)
        assert fast_result.trick_completed == canonical_result.trick_completed
        assert fast_result.trick_winner == canonical_result.trick_winner
        assert fast_result.cards_dealt == canonical_result.cards_dealt
        _assert_state_equivalent(canonical, fast)

    assert sum(fast.points) == 120


def test_play_random_fast_2p_finishes_with_valid_terminal_state() -> None:
    """
    Lo smoke helper fast deve terminare con uno stato finale coerente.

    Non dimostra equivalenza da solo, ma protegge il loop caldo usato dal benchmark.
    """
    state = play_random_fast_2p(seed=5, action_seed=11)

    assert state.game_over is True
    assert state.deck == []
    assert state.hands == [[], []]
    assert sum(state.points) == 120
    if state.points[0] == state.points[1]:
        assert state.winner_index is None
    else:
        assert state.winner_index == int(state.points[1] > state.points[0])
