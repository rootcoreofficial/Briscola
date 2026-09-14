"""
Test di casi limite del dominio (Phase 3).

Focus:
- pescata dell'ultima carta in 2-player (briscola in fondo al mazzo)
- gestione dei pareggi a fine partita (2-player e 4-player)

Questi test sono volutamente "piccoli" e molto espliciti, per essere letti come esempi didattici.
"""

from __future__ import annotations

from collections.abc import Iterable

from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.state import GameState, PlayerState


def _all_cards() -> list[Card]:
    """Crea il mazzo completo (40 carte) in maniera deterministica."""
    return [Card(suit, rank) for suit in Suit for rank in Rank]


def _points(cards: Iterable[Card]) -> int:
    """Somma punti Briscola di una sequenza di carte."""
    return sum(card.rank.points for card in cards)


def test_2p_last_card_draw_is_trump_card() -> None:
    """
    In 2-player la briscola scoperta viene reinserita in testa al mazzo (indice 0) e quindi
    deve essere pescata per ultima.

    Questo test costruisce uno stato "post mano" dove:
    - entrambi i player hanno 1 carta in mano
    - nel deck restano esattamente 2 carte: [TRUMP, OTHER]
    - si completa una mano -> il vincitore pesca `OTHER` e l'altro pesca `TRUMP`
    """
    trump = Card(Suit.CUPS, Rank.TWO)
    other = Card(Suit.SWORDS, Rank.FOUR)

    # Mano: A gioca un 7 qualsiasi, B gioca un 6 stesso seme -> A vince (7 > 6).
    a_card = Card(Suit.CLUBS, Rank.SEVEN)
    b_card = Card(Suit.CLUBS, Rank.SIX)

    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="A", hand=(a_card,), captured_cards=tuple(), points=0),
            PlayerState(name="B", hand=(b_card,), captured_cards=tuple(), points=0),
        ),
        deck=(trump, other),  # trump in testa: deve uscire per ultimo (pop() dalla fine)
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
    assert r1.trick_completed is False

    state, r2 = step(state, PlayCardAction(player_index=1, card_index=0))
    assert r2.error is None
    assert r2.trick_completed is True
    assert r2.trick_winner == 0
    assert r2.cards_dealt is True

    # Winner (A) pesca per primo: deve prendere `other` (l'ultima del deck).
    assert state.players[0].hand == (other,)
    # L'altro player pesca dopo e prende `trump`.
    assert state.players[1].hand == (trump,)
    assert state.deck == tuple()


def test_2p_tie_sets_winner_index_to_none() -> None:
    """
    Se a fine partita i punti sono pari, `winner_index` deve essere None.

    Costruiamo uno stato dove la prossima mano completa chiude la partita e produce un pareggio:
    - A ha già 60 punti, B ha già 60 punti
    - la mano finale contiene solo carte da 0 punti
    """
    # Strategia per costruire un pareggio "pulito":
    # - I punti totali del mazzo sono 120.
    # - Le carte che danno punti sono esattamente: Asso(11), Tre(10), Re(4), Cavallo(3), Fante(2).
    #   Per ogni seme, la somma punti è 30; su 4 semi fa 120.
    # - Per ottenere 60/60 basta assegnare a ciascun player le carte "a punti" di 2 semi.
    all_cards = _all_cards()
    trump = next(card for card in all_cards if card.suit == Suit.CUPS and card.rank == Rank.TWO)

    scoring_ranks = {Rank.ACE, Rank.THREE, Rank.KING, Rank.KNIGHT, Rank.JACK}
    zero_ranks = {Rank.TWO, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN}

    # A: semi CUPS+COINS => 60 punti. B: semi CLUBS+SWORDS => 60 punti.
    a_scoring = [c for c in all_cards if c.suit in (Suit.CUPS, Suit.COINS) and c.rank in scoring_ranks]
    b_scoring = [c for c in all_cards if c.suit in (Suit.CLUBS, Suit.SWORDS) and c.rank in scoring_ranks]
    assert _points(a_scoring) == 60
    assert _points(b_scoring) == 60

    zero_cards = [c for c in all_cards if c.rank in zero_ranks]
    # Riserviamo due carte (0 punti) per l'ultima mano, una per player.
    a_hand_card, b_hand_card = zero_cards[0], zero_cards[1]
    remaining_zero = zero_cards[2:]

    # Distribuiamo le altre 18 carte da 0 punti (9 a testa) nelle raccolte.
    a_zero = remaining_zero[:9]
    b_zero = remaining_zero[9:18]

    a_captured = tuple(a_scoring + a_zero)
    b_captured = tuple(b_scoring + b_zero)

    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="A", hand=(a_hand_card,), captured_cards=a_captured, points=_points(a_captured)),
            PlayerState(name="B", hand=(b_hand_card,), captured_cards=b_captured, points=_points(b_captured)),
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

    state, r1 = step(state, PlayCardAction(player_index=0, card_index=0))
    assert r1.error is None
    state, r2 = step(state, PlayCardAction(player_index=1, card_index=0))
    assert r2.error is None
    assert r2.trick_completed is True

    assert state.game_over is True
    assert state.winner_index is None


def test_4p_tie_sets_winning_team_to_none() -> None:
    """
    In 4-player, se i punti di squadra sono pari a fine partita, `winning_team` deve essere None.

    Nota:
    - In 4-player il deck è sempre vuoto e le mani si esauriscono insieme.
    - Qui impostiamo punti pregressi a 60 vs 60 e una mano finale a 0 punti.
    """
    all_cards = _all_cards()
    trump = next(card for card in all_cards if card.suit == Suit.CUPS and card.rank == Rank.TWO)

    scoring_ranks = {Rank.ACE, Rank.THREE, Rank.KING, Rank.KNIGHT, Rank.JACK}
    zero_ranks = {Rank.TWO, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN}

    # Team 0 (player 0 + 2): CUPS + COINS => 60. Team 1 (player 1 + 3): CLUBS + SWORDS => 60.
    p0_scoring = [c for c in all_cards if c.suit == Suit.CUPS and c.rank in scoring_ranks]
    p2_scoring = [c for c in all_cards if c.suit == Suit.COINS and c.rank in scoring_ranks]
    p1_scoring = [c for c in all_cards if c.suit == Suit.CLUBS and c.rank in scoring_ranks]
    p3_scoring = [c for c in all_cards if c.suit == Suit.SWORDS and c.rank in scoring_ranks]
    assert _points(p0_scoring) == 30
    assert _points(p1_scoring) == 30
    assert _points(p2_scoring) == 30
    assert _points(p3_scoring) == 30

    zero_cards = [c for c in all_cards if c.rank in zero_ranks]
    # Riserviamo 4 carte (0 punti) per l'ultima mano: una per giocatore.
    last_hand_cards = zero_cards[:4]
    remaining_zero = zero_cards[4:]

    # Distribuiamo le altre 16 carte da 0 punti (4 a testa) nelle raccolte.
    p0_zero = remaining_zero[0:4]
    p1_zero = remaining_zero[4:8]
    p2_zero = remaining_zero[8:12]
    p3_zero = remaining_zero[12:16]

    p0_captured = tuple(p0_scoring + p0_zero)
    p1_captured = tuple(p1_scoring + p1_zero)
    p2_captured = tuple(p2_scoring + p2_zero)
    p3_captured = tuple(p3_scoring + p3_zero)

    state = GameState(
        num_players=4,
        is_team_game=True,
        teams=((0, 2), (1, 3)),
        players=(
            PlayerState(
                name="A",
                hand=(last_hand_cards[0],),
                captured_cards=p0_captured,
                points=_points(p0_captured),
            ),
            PlayerState(
                name="B",
                hand=(last_hand_cards[1],),
                captured_cards=p1_captured,
                points=_points(p1_captured),
            ),
            PlayerState(
                name="C",
                hand=(last_hand_cards[2],),
                captured_cards=p2_captured,
                points=_points(p2_captured),
            ),
            PlayerState(
                name="D",
                hand=(last_hand_cards[3],),
                captured_cards=p3_captured,
                points=_points(p3_captured),
            ),
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

    # 4-player: quattro step per completare la mano.
    state, r1 = step(state, PlayCardAction(player_index=0, card_index=0))
    assert r1.error is None
    state, r2 = step(state, PlayCardAction(player_index=1, card_index=0))
    assert r2.error is None
    state, r3 = step(state, PlayCardAction(player_index=2, card_index=0))
    assert r3.error is None
    state, r4 = step(state, PlayCardAction(player_index=3, card_index=0))
    assert r4.error is None
    assert r4.trick_completed is True

    assert state.game_over is True
    assert state.winning_team is None
