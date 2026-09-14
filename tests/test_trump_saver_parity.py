"""
Parità di `heuristic_trump_saver` attraverso i tre motori: dominio ↔ fast Python ↔ kernel Numba.

Perché questi test esistono (CLAUDE.md: il fast path DEVE restare coerente col dominio):
la sonda di exploitability nata dall'audit di produzione 2026-07-07 è stata tradotta nel
fast path e nei kernel JIT per poterla usare come avversario nei rollout di training
(`train_a2c.py --opponent-mix ... --fast-rollout numba`) e nelle valutazioni `--engine fast`.
La fonte di verità resta `HeuristicTrumpSaverAgent` (dominio): qui verifichiamo che le tre
implementazioni scelgano ESATTAMENTE lo stesso indice carta a ogni decisione di partite reali
(pattern "partite specchiate", come `test_numba_v4_parity`).

L'equivalenza aggregata fast ↔ dominio su interi match è coperta in `test_ai_evaluation.py`
(parametrizzazioni estese con `heuristic_trump_saver`).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from briscola_ai.ai.agents import HeuristicAgentV1, HeuristicAgentV2, HeuristicTrumpSaverAgent
from briscola_ai.ai.fast.evaluation import choose_fast_card_index
from briscola_ai.ai.fast.state_2p import Fast2PState, new_fast_2p_state, step_fast_2p
from briscola_ai.ai.numba.core import _choose_policy_card_index_numba, numba_agent_code
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba

# Agenti deterministici del dominio, indicizzati per nome fast-compatible.
_DOMAIN_AGENTS = {
    "heuristic_v1": HeuristicAgentV1(),
    "heuristic_v2": HeuristicAgentV2(),
    "heuristic_trump_saver": HeuristicTrumpSaverAgent(),
}


def _numba_choose_card_index(
    agent_name: str,
    *,
    hands: list[list[int]],
    player_index: int,
    table_cards: list[int],
    table_players: list[int],
    deck_size: int,
    trump_card: int,
    seen_cards_onehot: tuple[int, ...],
) -> int:
    """
    Invoca il dispatch JIT `_choose_policy_card_index_numba` su uno stato numerico esplicito.

    Le mani sono paddate a larghezza 3 (massimo in Briscola 2p) con -1: i kernel leggono
    solo i primi `hand_sizes[player]` slot, quindi il padding non è mai osservato.
    """
    hands_arr = np.full((2, 3), -1, dtype=np.int64)
    for p in range(2):
        for j, card_id in enumerate(hands[p]):
            hands_arr[p, j] = card_id
    hand_sizes = np.asarray([len(hands[0]), len(hands[1])], dtype=np.int64)
    table_cards_arr = np.zeros(2, dtype=np.int64)
    table_players_arr = np.zeros(2, dtype=np.int64)
    for j, card_id in enumerate(table_cards):
        table_cards_arr[j] = card_id
        table_players_arr[j] = table_players[j]
    return int(
        _choose_policy_card_index_numba(
            numba_agent_code(agent_name),
            hands_arr,
            hand_sizes,
            player_index,
            table_cards_arr,
            table_players_arr,
            len(table_cards),
            deck_size,
            trump_card,
            np.asarray(seen_cards_onehot, dtype=np.int64),
        )
    )


def _numba_choose_from_fast_state(agent_name: str, state: Fast2PState, seen: list[int]) -> int:
    """Adatta lo stato fast mutabile alla firma del dispatch JIT."""
    return _numba_choose_card_index(
        agent_name,
        hands=state.hands,
        player_index=state.current_turn,
        table_cards=state.table_cards,
        table_players=state.table_players,
        deck_size=len(state.deck),
        trump_card=state.trump_card,
        seen_cards_onehot=tuple(seen),
    )


@pytest.mark.parametrize("opponent_name", ["random", "heuristic_v1", "heuristic_v2", "heuristic_trump_saver"])
@pytest.mark.parametrize("seed", [0, 7, 42, 123, 2026])
@pytest.mark.parametrize("saver_seat", [0, 1])
def test_trump_saver_three_engine_parity_on_mirrored_games(opponent_name: str, seed: int, saver_seat: int) -> None:
    """
    Partite complete specchiate dominio/fast: a OGNI decisione deterministica i tre motori
    devono scegliere lo stesso indice carta.

    Le mosse del seat `random` (quando presente) sono estratte una sola volta e applicate a
    entrambi i motori: servono a diversificare gli stati visitati, non vengono confrontate.
    """
    domain_state = new_game_state(2, seed=seed)
    fast_state = new_fast_2p_state(seed=seed)
    seen = [0] * 40
    seen[fast_state.trump_card] = 1
    rng_random_moves = random.Random(seed ^ 0x5EED)

    agent_names = {saver_seat: "heuristic_trump_saver", 1 - saver_seat: opponent_name}
    compared_decisions = 0

    while not fast_state.game_over:
        current = fast_state.current_turn
        assert current == domain_state.current_turn
        name = agent_names[current]

        if name == "random":
            card_index = rng_random_moves.randrange(len(fast_state.hands[current]))
        else:
            observation = make_player_observation(domain_state, current)
            # Gli agenti confrontati sono deterministici: l'RNG passato non viene consumato.
            domain_idx = _DOMAIN_AGENTS[name].choose_card_index(observation, rng=random.Random(0))
            fast_idx = choose_fast_card_index(
                name, fast_state, current, rng=random.Random(0), seen_cards_onehot=tuple(seen)
            )
            numba_idx = _numba_choose_from_fast_state(name, fast_state, seen)
            assert fast_idx == domain_idx, f"fast!=domain al turno di {name} (seed={seed})"
            assert numba_idx == domain_idx, f"numba!=domain al turno di {name} (seed={seed})"
            card_index = domain_idx
            compared_decisions += 1

        result = step_fast_2p(fast_state, player_index=current, card_index=card_index)
        seen[result.played_card] = 1
        domain_state, domain_result = step(domain_state, PlayCardAction(player_index=current, card_index=card_index))
        assert domain_result.error is None

    assert domain_state.game_over
    assert fast_state.points == [domain_state.players[0].points, domain_state.players[1].points]
    assert sum(fast_state.points) == 120
    # Almeno tutte le decisioni del trump saver sono state confrontate (20 per partita).
    assert compared_decisions >= 20


def test_numba_kernel_cuts_carico_with_smallest_trump() -> None:
    """Kernel JIT, pattern n.3: il carico avversario si taglia con la briscola MINIMA."""
    trump_card = card_to_id(Card(Suit.CLUBS, Rank.ACE))
    hand = [card_to_id(Card(Suit.CLUBS, Rank.KING)), card_to_id(Card(Suit.CLUBS, Rank.TWO))]
    hands = [[card_to_id(Card(Suit.COINS, Rank.ACE))], hand]  # la mano di P0 è irrilevante

    idx = _numba_choose_card_index(
        "heuristic_trump_saver",
        hands=hands,
        player_index=1,
        table_cards=[card_to_id(Card(Suit.COINS, Rank.ACE))],
        table_players=[0],
        deck_size=20,
        trump_card=trump_card,
        seen_cards_onehot=(0,) * 40,
    )
    assert hand[idx] == card_to_id(Card(Suit.CLUBS, Rank.TWO))


def test_numba_kernel_never_wastes_trump_on_poor_trick() -> None:
    """Kernel JIT, pattern n.4: mai briscola su piatti poveri durante le pescate."""
    trump_card = card_to_id(Card(Suit.CLUBS, Rank.ACE))
    hand = [
        card_to_id(Card(Suit.CLUBS, Rank.TWO)),
        card_to_id(Card(Suit.CUPS, Rank.FOUR)),
        card_to_id(Card(Suit.SWORDS, Rank.KING)),
    ]
    hands = [[card_to_id(Card(Suit.COINS, Rank.TWO))], hand]

    idx = _numba_choose_card_index(
        "heuristic_trump_saver",
        hands=hands,
        player_index=1,
        table_cards=[card_to_id(Card(Suit.COINS, Rank.TWO))],
        table_players=[0],
        deck_size=20,
        trump_card=trump_card,
        seen_cards_onehot=(0,) * 40,
    )
    assert hand[idx] == card_to_id(Card(Suit.CUPS, Rank.FOUR))  # scarto economico, briscola in mano


def test_numba_kernel_endgame_lead_cashes_master() -> None:
    """Kernel JIT: a mazzo vuoto un carico divenuto imbattibile (master) si incassa da primi."""
    trump_card = card_to_id(Card(Suit.CLUBS, Rank.KING))
    hand = [card_to_id(Card(Suit.CUPS, Rank.ACE)), card_to_id(Card(Suit.SWORDS, Rank.FOUR))]
    seen = [0] * 40
    for rank in Rank:  # tutte le briscole sono uscite: nessun taglio possibile
        seen[card_to_id(Card(Suit.CLUBS, rank))] = 1

    idx = _numba_choose_card_index(
        "heuristic_trump_saver",
        hands=[hand, [card_to_id(Card(Suit.COINS, Rank.TWO))]],
        player_index=0,
        table_cards=[],
        table_players=[],
        deck_size=0,
        trump_card=trump_card,
        seen_cards_onehot=tuple(seen),
    )
    assert hand[idx] == card_to_id(Card(Suit.CUPS, Rank.ACE))


def test_numba_kernel_leads_liscio_never_carico() -> None:
    """Kernel JIT, pattern n.1: si apre liscio (0 punti, non briscola), mai un carico."""
    trump_card = card_to_id(Card(Suit.CLUBS, Rank.KING))
    hand = [
        card_to_id(Card(Suit.SWORDS, Rank.ACE)),
        card_to_id(Card(Suit.CUPS, Rank.THREE)),
        card_to_id(Card(Suit.COINS, Rank.FOUR)),
    ]

    idx = _numba_choose_card_index(
        "heuristic_trump_saver",
        hands=[hand, [card_to_id(Card(Suit.COINS, Rank.TWO))]],
        player_index=0,
        table_cards=[],
        table_players=[],
        deck_size=20,
        trump_card=trump_card,
        seen_cards_onehot=(0,) * 40,
    )
    assert hand[idx] == card_to_id(Card(Suit.COINS, Rank.FOUR))
