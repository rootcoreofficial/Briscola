import random

from briscola_ai.ai.agents import HeuristicAgentV2
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _make_obs_second_hand(
    *,
    hand: tuple[Card, ...],
    lead_card: Card,
    trump_card: Card,
    deck_size: int,
    seen_cards_onehot: tuple[int, ...],
) -> PlayerObservation:
    """
    Crea una `PlayerObservation` minimale per testare la scelta “secondo di mano” (2-player).

    Nota:
    Qui NON costruiamo il `GameState` completo: per gli agenti è sufficiente l'osservazione lecita.
    """
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=1,
        player_name="IA",
        hand=hand,
        trump_card=trump_card,
        deck_size=deck_size,
        table_cards=((lead_card, 0),),
        current_turn=1,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(3, 3),
        seen_cards_onehot=seen_cards_onehot,
    )


def test_heuristic_v2_late_game_takes_control_with_cheap_trump() -> None:
    """
    In late game (o con poche briscole residue), `heuristic_v2` può decidere di prendere
    anche una mano a basso valore usando una briscola *economica*, per ottenere il controllo.

    Questo comportamento è intenzionale e deriva dall'uso di `seen_cards_onehot`:
    se quasi tutte le briscole sono già uscite, prendere con una briscola bassa è “più sicuro”.
    """
    trump_card = Card(Suit.CLUBS, Rank.ACE)
    lead_card = Card(Suit.COINS, Rank.TWO)  # 0 punti

    # Abbiamo una briscola molto economica + scarti.
    hand = (Card(Suit.CLUBS, Rank.TWO), Card(Suit.CUPS, Rank.FOUR), Card(Suit.SWORDS, Rank.FOUR))

    # Simuliamo che 8 briscole su 10 siano già "viste" pubblicamente.
    seen = [0] * 40
    for rank in (
        Rank.ACE,
        Rank.THREE,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.JACK,
        Rank.KNIGHT,
        Rank.KING,
    ):
        seen[card_to_id(Card(Suit.CLUBS, rank))] = 1
    obs = _make_obs_second_hand(
        hand=hand,
        lead_card=lead_card,
        trump_card=trump_card,
        deck_size=2,
        seen_cards_onehot=tuple(seen),
    )

    agent = HeuristicAgentV2()
    idx = agent.choose_card_index(obs, rng=random.Random(0))
    assert idx == 0  # gioca clubs TWO (briscola bassa) per prendere il controllo


def test_heuristic_v2_picks_minimum_winning_trump_when_trumping() -> None:
    """Se decide di prendere con briscola, deve scegliere la briscola vincente minima (anti-overkill)."""
    trump_card = Card(Suit.CLUBS, Rank.ACE)
    lead_card = Card(Suit.COINS, Rank.ACE)  # 11 punti: vale sempre la pena prendere

    hand = (Card(Suit.CLUBS, Rank.TWO), Card(Suit.CLUBS, Rank.KING), Card(Suit.CUPS, Rank.FOUR))
    obs = _make_obs_second_hand(
        hand=hand,
        lead_card=lead_card,
        trump_card=trump_card,
        deck_size=20,
        seen_cards_onehot=(0,) * 40,
    )

    agent = HeuristicAgentV2()
    idx = agent.choose_card_index(obs, rng=random.Random(0))
    assert idx == 0  # clubs TWO vince e costa meno di clubs KING
