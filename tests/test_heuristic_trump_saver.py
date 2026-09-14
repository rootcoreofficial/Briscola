"""
Test di `heuristic_trump_saver`: la sonda di exploitability nata dall'audit di produzione
del 2026-07-07 (7 vittorie umane contro `bc_model_pimc_belief_64x10` su base v10).

Ogni test codifica UNO dei comportamenti osservati nei vincitori umani; se un test
fallisce, l'euristica non sta più rappresentando lo stile che deve sondare.
"""

import random

from briscola_ai.ai.agents import HeuristicTrumpSaverAgent, build_agent
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _make_obs(
    *,
    hand: tuple[Card, ...],
    trump_card: Card,
    deck_size: int,
    table_cards: tuple[tuple[Card, int], ...] = (),
    seen_cards_onehot: tuple[int, ...] = (0,) * 40,
    player_index: int = 1,
) -> PlayerObservation:
    """Observation minimale 2-player (stesso pattern di `test_heuristic_v2`)."""
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=player_index,
        player_name="IA",
        hand=hand,
        trump_card=trump_card,
        deck_size=deck_size,
        table_cards=table_cards,
        current_turn=player_index,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(len(hand), len(hand)),
        seen_cards_onehot=seen_cards_onehot,
    )


def test_registry_builds_trump_saver() -> None:
    """L'agente è costruibile dal registry col nome canonico (per evaluate_agents & co.)."""
    agent = build_agent("heuristic_trump_saver")
    assert isinstance(agent, HeuristicTrumpSaverAgent)
    assert agent.name == "heuristic_trump_saver"


def test_cashes_carico_as_second_player() -> None:
    """
    Pattern di campo n.2: il carico si incassa DA SECONDI (in 2p nessuno gioca dopo).

    L'avversario apre 7 di coppe (0 punti); possiamo vincere sia col Re (4pt) sia con
    l'Asso (11pt) di coppe. v2 sceglierebbe il Re (vincente economica); il trump saver
    banca l'Asso: 11 punti al sicuro invece di lasciarlo esposto a un taglio futuro.
    """
    trump = Card(Suit.CLUBS, Rank.ACE)
    hand = (Card(Suit.CUPS, Rank.KING), Card(Suit.CUPS, Rank.ACE), Card(Suit.SWORDS, Rank.FOUR))
    obs = _make_obs(
        hand=hand,
        trump_card=trump,
        deck_size=20,
        table_cards=((Card(Suit.CUPS, Rank.SEVEN), 0),),
    )
    idx = HeuristicTrumpSaverAgent().choose_card_index(obs, rng=random.Random(0))
    assert hand[idx] == Card(Suit.CUPS, Rank.ACE)


def test_never_wastes_trump_on_poor_trick() -> None:
    """
    Pattern di campo n.4: mai briscola su piatti poveri durante le pescate.

    L'IA di produzione ha tagliato 8 volte piatti da <=2 punti: qui l'avversario apre
    un 2 (0 punti) e, pur avendo la briscolina che vincerebbe, il trump saver scarta
    il liscio e conserva la briscola per i carichi.
    """
    trump = Card(Suit.CLUBS, Rank.ACE)
    hand = (Card(Suit.CLUBS, Rank.TWO), Card(Suit.CUPS, Rank.FOUR), Card(Suit.SWORDS, Rank.KING))
    obs = _make_obs(
        hand=hand,
        trump_card=trump,
        deck_size=20,
        table_cards=((Card(Suit.COINS, Rank.TWO), 0),),
    )
    idx = HeuristicTrumpSaverAgent().choose_card_index(obs, rng=random.Random(0))
    assert hand[idx] == Card(Suit.CUPS, Rank.FOUR)  # scarto economico, briscola in mano


def test_cuts_opponent_carico_with_smallest_trump() -> None:
    """
    Pattern di campo n.3: il carico avversario si taglia, e con la briscola MINIMA.

    È l'exploit-chiave: v10 ha guidato 9 carichi perdendone 8 proprio contro questo
    comportamento. Qui l'avversario apre l'Asso di denari: tagliamo col 2 di briscola,
    non col Re (anti-overkill).
    """
    trump = Card(Suit.CLUBS, Rank.ACE)
    hand = (Card(Suit.CLUBS, Rank.KING), Card(Suit.CLUBS, Rank.TWO), Card(Suit.CUPS, Rank.FOUR))
    obs = _make_obs(
        hand=hand,
        trump_card=trump,
        deck_size=20,
        table_cards=((Card(Suit.COINS, Rank.ACE), 0),),
    )
    idx = HeuristicTrumpSaverAgent().choose_card_index(obs, rng=random.Random(0))
    assert hand[idx] == Card(Suit.CLUBS, Rank.TWO)


def test_holds_trump_ace_on_poor_trick() -> None:
    """
    Pattern di campo n.3 (rovescio): l'asso di briscola non si spende per piatti vuoti.

    I vincitori umani hanno tenuto l'asso di briscola oltre ogni raccomandazione del
    modello. Qui l'unica vincente è l'asso di briscola ma il piatto vale 0: si scarta.
    """
    trump = Card(Suit.CLUBS, Rank.KING)
    hand = (Card(Suit.CLUBS, Rank.ACE), Card(Suit.CUPS, Rank.FOUR), Card(Suit.SWORDS, Rank.FIVE))
    obs = _make_obs(
        hand=hand,
        trump_card=trump,
        deck_size=20,
        table_cards=((Card(Suit.COINS, Rank.TWO), 0),),
    )
    idx = HeuristicTrumpSaverAgent().choose_card_index(obs, rng=random.Random(0))
    assert hand[idx] == Card(Suit.CUPS, Rank.FOUR)


def test_leads_liscio_never_carico() -> None:
    """
    Pattern di campo n.1: si apre liscio, mai un carico (19/35 aperture umane erano
    lisci non briscola; v10 invece guidava i carichi ed è stata punita).
    """
    trump = Card(Suit.CLUBS, Rank.KING)
    hand = (Card(Suit.SWORDS, Rank.ACE), Card(Suit.CUPS, Rank.THREE), Card(Suit.COINS, Rank.FOUR))
    obs = _make_obs(hand=hand, trump_card=trump, deck_size=20, player_index=0)
    idx = HeuristicTrumpSaverAgent().choose_card_index(obs, rng=random.Random(0))
    assert hand[idx] == Card(Suit.COINS, Rank.FOUR)


def test_endgame_lead_cashes_master() -> None:
    """
    A mazzo vuoto, un carico divenuto imbattibile (tutte le briscole viste, nessuna
    coppa superiore in giro) si incassa subito: è il "tenere l'asso fino a t20" visto
    nelle partite 3a053071 e 895ceaa3, seguito dall'incasso a colpo sicuro.
    """
    trump = Card(Suit.CLUBS, Rank.KING)
    hand = (Card(Suit.CUPS, Rank.ACE), Card(Suit.SWORDS, Rank.FOUR))
    seen = [0] * 40
    for rank in Rank:  # tutte le briscole sono uscite
        seen[card_to_id(Card(Suit.CLUBS, rank))] = 1
    obs = _make_obs(
        hand=hand,
        trump_card=trump,
        deck_size=0,
        seen_cards_onehot=tuple(seen),
        player_index=0,
    )
    idx = HeuristicTrumpSaverAgent().choose_card_index(obs, rng=random.Random(0))
    assert hand[idx] == Card(Suit.CUPS, Rank.ACE)


def test_endgame_lead_avoids_beatable_carico() -> None:
    """
    A mazzo vuoto ma con briscole ancora ignote, il carico NON è master e non si guida
    (v1/v2 qui giocano la carta più forte: è il comportamento che regala punti).
    """
    trump = Card(Suit.CLUBS, Rank.KING)
    hand = (Card(Suit.CUPS, Rank.THREE), Card(Suit.SWORDS, Rank.TWO))
    obs = _make_obs(hand=hand, trump_card=trump, deck_size=0, player_index=0)
    idx = HeuristicTrumpSaverAgent().choose_card_index(obs, rng=random.Random(0))
    assert hand[idx] == Card(Suit.SWORDS, Rank.TWO)
