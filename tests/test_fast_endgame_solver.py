"""Parita' tra solver endgame canonico e fast solver numerico."""

from __future__ import annotations

import pytest

from briscola_ai.ai.endgame.fast_solver import solve_endgame_fast
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.rules import trick_points
from briscola_ai.domain.state import GameState, PlayerState, new_game_state

# Default condiviso a livello di modulo: `Card` e' frozen, quindi riusarlo e' sicuro
# (B008 vieta le chiamate nei default perche' non distingue i costruttori immutabili).
_DEFAULT_TRUMP = Card(Suit.COINS, Rank.SEVEN)


def _endgame_state(
    hand0: tuple[Card, ...],
    hand1: tuple[Card, ...],
    *,
    current_turn: int,
    table_cards: tuple[tuple[Card, int], ...] = (),
    trump_card: Card = _DEFAULT_TRUMP,
) -> GameState:
    """Costruisce uno stato endgame minimale con punti coerenti."""
    first_player = table_cards[0][1] if table_cards else current_turn
    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState("P0", hand0, tuple(), trick_points(tuple())),
            PlayerState("P1", hand1, tuple(), trick_points(tuple())),
        ),
        deck=tuple(),
        trump_card=trump_card,
        table_cards=table_cards,
        current_turn=current_turn,
        first_player=first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def _assert_same_solution(state: GameState) -> None:
    """Confronta mossa, delta e principal variation dei due solver."""
    canonical = solve_endgame(state)
    fast = solve_endgame_fast(state)

    assert fast.best_card_index == canonical.best_card_index
    assert fast.final_delta_p0_p1 == canonical.final_delta_p0_p1
    assert fast.principal_variation == canonical.principal_variation


@pytest.mark.parametrize(
    "state",
    [
        _endgame_state(
            hand0=(Card(Suit.CUPS, Rank.ACE),),
            hand1=(Card(Suit.CUPS, Rank.KING),),
            current_turn=0,
        ),
        _endgame_state(
            hand0=(Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)),
            hand1=(Card(Suit.COINS, Rank.KING), Card(Suit.CLUBS, Rank.TWO)),
            current_turn=0,
        ),
        _endgame_state(
            hand0=(Card(Suit.SWORDS, Rank.FOUR),),
            hand1=(Card(Suit.CLUBS, Rank.ACE), Card(Suit.SWORDS, Rank.FIVE)),
            current_turn=1,
            table_cards=((Card(Suit.CLUBS, Rank.TWO), 0),),
        ),
    ],
)
def test_fast_solver_matches_canonical_on_hand_built_cases(state: GameState) -> None:
    """Gli esempi didattici del solver canonico devono avere la stessa soluzione."""
    _assert_same_solution(state)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 12345])
def test_fast_solver_matches_canonical_on_real_endgames(seed: int) -> None:
    """Su finali raggiunti dal dominio reale, il fast solver resta bit-for-bit equivalente."""
    state = new_game_state(num_players=2, seed=seed)
    while len(state.deck) > 0 and not state.game_over:
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None

    assert not state.game_over
    assert len(state.deck) == 0
    _assert_same_solution(state)


@pytest.mark.parametrize("seed", [3, 11, 99])
def test_fast_solver_matches_canonical_when_second_player_must_respond(seed: int) -> None:
    """La parita' copre anche lo stato con una carta gia' sul tavolo."""
    state = new_game_state(num_players=2, seed=seed)
    while len(state.deck) > 0 and not state.game_over:
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None

    assert not state.game_over
    assert len(state.deck) == 0
    state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
    assert result.error is None
    assert len(state.table_cards) == 1

    _assert_same_solution(state)
