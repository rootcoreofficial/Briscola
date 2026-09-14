"""Parita' della scelta endgame Numba rispetto al solver canonico."""

from __future__ import annotations

import pytest

from briscola_ai.ai.endgame.numba_solver import (
    _arrays_from_state,
    choose_endgame_card_numba,
    choose_endgame_card_numba_arrays,
    warm_up_numba_endgame_solver,
)
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.rules import trick_points
from briscola_ai.domain.state import GameState, PlayerState, new_game_state

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba

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


def _assert_same_best_card(state: GameState) -> None:
    """
    Il kernel Numba choose-only deve mantenere lo stesso tie-break del solver completo.

    Verifichiamo anche `final_delta_p0_p1`: è il valore di foglia usato dal value-lookahead,
    quindi un errore sul delta che non cambia la carta ottima corromperebbe comunque il
    ranking delle foglie senza far fallire il confronto sulla sola mossa.
    """
    expected = solve_endgame(state)
    assert choose_endgame_card_numba(state) == expected.best_card_index
    best_card_index, final_delta = choose_endgame_card_numba_arrays(*_arrays_from_state(state))
    assert int(best_card_index) == expected.best_card_index
    assert int(final_delta) == expected.final_delta_p0_p1


def test_warm_up_numba_endgame_solver_compiles_kernel() -> None:
    """Il warm-up deve essere richiamabile dagli script lunghi prima del loop caldo."""
    warm_up_numba_endgame_solver()


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
            hand0=(Card(Suit.COINS, Rank.THREE), Card(Suit.CLUBS, Rank.TWO)),
            hand1=(Card(Suit.CUPS, Rank.ACE), Card(Suit.CLUBS, Rank.FOUR)),
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
def test_numba_solver_matches_canonical_on_hand_built_cases(state: GameState) -> None:
    """Copre casi didattici e lo stato secondo di mano."""
    _assert_same_best_card(state)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 12345])
def test_numba_solver_matches_canonical_on_real_endgames(seed: int) -> None:
    """Su finali reali raggiunti dal dominio, la mossa Numba coincide col solver completo."""
    state = new_game_state(num_players=2, seed=seed)
    while len(state.deck) > 0 and not state.game_over:
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None

    assert not state.game_over
    assert len(state.deck) == 0
    _assert_same_best_card(state)


@pytest.mark.parametrize("seed", [3, 11, 99])
def test_numba_solver_matches_canonical_when_second_player_must_respond(seed: int) -> None:
    """La parita' include finali con una carta gia' sul tavolo."""
    state = new_game_state(num_players=2, seed=seed)
    while len(state.deck) > 0 and not state.game_over:
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None

    state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
    assert result.error is None
    assert len(state.table_cards) == 1

    _assert_same_best_card(state)
