"""
Test di flusso per il dominio (`GameState + step`).

Obiettivo:
- validare la transizione di stato quando una mano si completa
- verificare la logica di pesca post-mano in modalità 2 giocatori
- assicurare che azioni invalide non corrompano lo stato
"""

from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.state import GameState, PlayerState


def test_trick_completion_updates_turn_and_points() -> None:
    """
    Esegue una mano completa in modalità 2 giocatori con mani impostate a mano.

    Verifica:
    - `table_cards` si svuota dopo la mano
    - `current_turn` e `first_player` diventano il vincitore della mano
    - le carte vengono aggiunte a `captured_cards` del vincitore e i punti aggiornati
    """
    trump = Card(Suit.CUPS, Rank.TWO)
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="A", hand=(Card(Suit.SWORDS, Rank.FOUR),), captured_cards=tuple(), points=0),
            PlayerState(name="B", hand=(Card(Suit.SWORDS, Rank.ACE),), captured_cards=tuple(), points=0),
        ),
        deck=tuple(),
        trump_card=trump,
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )

    # A gioca 4 di spade (seme di uscita), B gioca asso di spade: B deve vincere
    state, r1 = step(state, PlayCardAction(player_index=0, card_index=0))
    assert r1.error is None
    assert r1.trick_completed is False
    assert len(state.table_cards) == 1
    assert state.current_turn == 1

    state, r2 = step(state, PlayCardAction(player_index=1, card_index=0))
    assert r2.error is None
    assert r2.trick_completed is True
    assert r2.trick_winner == 1
    assert state.table_cards == tuple()
    assert state.current_turn == 1
    assert state.first_player == 1

    assert len(state.players[1].captured_cards) == 2
    assert state.players[1].points == Rank.ACE.points + Rank.FOUR.points


def test_after_trick_in_2p_cards_are_dealt_starting_from_trick_winner() -> None:
    """
    In 2 giocatori, dopo una mano completa si pescano due carte (una per giocatore)
    partendo dal vincitore della mano.
    """
    trump = Card(Suit.CUPS, Rank.TWO)

    # Deck: `pop()` pesca dalla fine. Prepariamo due carte note.
    drawn_by_winner = Card(Suit.COINS, Rank.KING)
    drawn_by_other = Card(Suit.SWORDS, Rank.TWO)
    deck = (trump, drawn_by_other, drawn_by_winner)

    # A perde: A gioca 4 spade, B gioca 7 coppe (briscola) => B vince.
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="A", hand=(Card(Suit.SWORDS, Rank.FOUR),), captured_cards=tuple(), points=0),
            PlayerState(name="B", hand=(Card(Suit.CUPS, Rank.SEVEN),), captured_cards=tuple(), points=0),
        ),
        deck=deck,
        trump_card=trump,
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )

    state, r1 = step(state, PlayCardAction(player_index=0, card_index=0))
    assert r1.error is None
    state, result = step(state, PlayCardAction(player_index=1, card_index=0))
    assert result.error is None

    assert result.trick_completed is True
    assert result.trick_winner == 1
    assert result.cards_dealt is True

    # Dopo la mano: B pesca per primo (winner), quindi prende `drawn_by_winner`.
    assert state.players[1].hand == (drawn_by_winner,)
    assert state.players[0].hand == (drawn_by_other,)
    assert state.deck == (trump,)


def test_invalid_action_does_not_modify_state() -> None:
    """Un'azione invalida (indice carta fuori range) deve restituire errore senza side-effect."""
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="A", hand=(Card(Suit.SWORDS, Rank.FOUR),), captured_cards=tuple(), points=0),
            PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=0),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.TWO),
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )

    before = state
    after, out = step(state, PlayCardAction(player_index=0, card_index=999))
    assert out.error is not None
    assert after == before
