"""
Test del campo `out_of_play_cards_onehot` (Fase 5G, step 3).

Obiettivo: separare "visto pubblicamente" (`seen_cards_onehot`) da "non più disponibile"
(`out_of_play_cards_onehot`). La differenza si concentra tutta sulla briscola scoperta, che resta
"vista" ma è "fuori gioco" solo quando finisce in una presa o sul tavolo.
"""

from __future__ import annotations

from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.rules import trick_points
from briscola_ai.domain.state import GameState, PlayerState, new_game_state


def _state(
    *,
    players: tuple[PlayerState, PlayerState],
    deck: tuple[Card, ...],
    trump_card: Card,
    table_cards: tuple[tuple[Card, int], ...] = (),
) -> GameState:
    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=players,
        deck=deck,
        trump_card=trump_card,
        table_cards=table_cards,
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def test_trump_still_in_deck_is_seen_but_not_out_of_play() -> None:
    """Briscola ancora nel mazzo: vista (pubblica) ma non fuori gioco; nessuna carta fuori gioco."""
    state = new_game_state(num_players=2, seed=1)
    assert len(state.deck) > 0  # la briscola scoperta è "sotto il mazzo"

    obs = make_player_observation(state, player_index=0)
    trump_id = card_to_id(state.trump_card)  # type: ignore[arg-type]

    assert obs.seen_cards_onehot[trump_id] == 1
    assert obs.out_of_play_cards_onehot[trump_id] == 0
    assert sum(obs.out_of_play_cards_onehot) == 0  # inizio partita: nulla è ancora fuori gioco


def test_trump_drawn_into_hand_is_seen_but_not_out_of_play() -> None:
    """Briscola pescata in mano (mazzo vuoto) ma non ancora giocata: vista, non fuori gioco."""
    trump = Card(Suit.COINS, Rank.ACE)
    players = (
        PlayerState("P0", (trump, Card(Suit.CUPS, Rank.TWO)), (), 0),
        PlayerState("P1", (Card(Suit.SWORDS, Rank.TWO), Card(Suit.CLUBS, Rank.FOUR)), (), 0),
    )
    state = _state(players=players, deck=(), trump_card=trump)

    obs = make_player_observation(state, player_index=0)
    trump_id = card_to_id(trump)

    assert obs.seen_cards_onehot[trump_id] == 1
    assert obs.out_of_play_cards_onehot[trump_id] == 0
    assert sum(obs.out_of_play_cards_onehot) == 0


def test_trump_captured_is_seen_and_out_of_play() -> None:
    """Briscola finita in una presa: sia vista sia fuori gioco."""
    trump = Card(Suit.COINS, Rank.ACE)
    players = (
        PlayerState("P0", (Card(Suit.CUPS, Rank.TWO),), (), 0),
        PlayerState("P1", (Card(Suit.SWORDS, Rank.TWO),), (trump,), trick_points((trump,))),
    )
    state = _state(players=players, deck=(), trump_card=trump)

    obs = make_player_observation(state, player_index=0)
    trump_id = card_to_id(trump)

    assert obs.seen_cards_onehot[trump_id] == 1
    assert obs.out_of_play_cards_onehot[trump_id] == 1


def test_table_and_captured_cards_are_out_of_play() -> None:
    """Carte sul tavolo e nelle prese sono fuori gioco (oltre che viste)."""
    trump = Card(Suit.COINS, Rank.SEVEN)
    table_card = Card(Suit.CUPS, Rank.FIVE)
    captured_card = Card(Suit.SWORDS, Rank.KING)
    players = (
        PlayerState("P0", (Card(Suit.CLUBS, Rank.TWO),), (captured_card,), trick_points((captured_card,))),
        PlayerState("P1", (Card(Suit.CLUBS, Rank.THREE),), (), 0),
    )
    state = _state(players=players, deck=(), trump_card=trump, table_cards=((table_card, 1),))

    obs = make_player_observation(state, player_index=0)

    for card in (table_card, captured_card):
        cid = card_to_id(card)
        assert obs.out_of_play_cards_onehot[cid] == 1
        assert obs.seen_cards_onehot[cid] == 1


def test_out_of_play_is_always_subset_of_seen() -> None:
    """Invariante: ogni carta fuori gioco è anche vista (out_of_play ⊆ seen)."""
    state = new_game_state(num_players=2, seed=7)
    obs = make_player_observation(state, player_index=0)

    assert len(obs.out_of_play_cards_onehot) == 40
    for cid in range(40):
        if obs.out_of_play_cards_onehot[cid] == 1:
            assert obs.seen_cards_onehot[cid] == 1
