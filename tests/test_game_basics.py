"""
Test di base per il dominio (`GameState + step`).

Questi test verificano invarianti "semplici" ma fondamentali:
- il setup (distribuzione e briscola) è coerente per 2 e 4 giocatori
- una partita completa termina e totalizza 120 punti complessivi (somma di tutti i giocatori)
"""

import random

from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.state import new_game_state


def _all_cards_in_state(state) -> list:
    """
    Helper: restituisce tutte le carte "fisiche" presenti nello stato.

    Nota importante:
    - Non includiamo `state.trump_card` perché può essere un riferimento ridondante:
      - in 2-player la briscola scoperta è re-inserita nel deck (quindi è già in `state.deck`)
      - in 4-player la briscola è una carta che sta già in mano a un giocatore
    """
    cards = []
    cards.extend(state.deck)
    for p in state.players:
        cards.extend(p.hand)
        cards.extend(p.captured_cards)
    cards.extend([card for card, _ in state.table_cards])
    return cards


def test_start_game_deals_correctly_for_2_players() -> None:
    """In 2 giocatori: 3 carte in mano ciascuno, briscola impostata e deck coerente."""
    state = new_game_state(num_players=2, player_names=["A", "B"], seed=123)

    assert state.is_team_game is False
    assert state.trump_card is not None
    assert len(state.deck) == 34  # 40 - 6 dealt; trump reinserted on bottom (index 0)
    assert state.deck[0] == state.trump_card
    assert len(state.players) == 2
    assert [len(p.hand) for p in state.players] == [3, 3]

    # Invariante: lo stato contiene sempre 40 carte uniche.
    cards = _all_cards_in_state(state)
    assert len(cards) == 40
    assert len(set(cards)) == 40


def test_start_game_deals_correctly_for_4_players() -> None:
    """In 4 giocatori: partita a squadre e 10 carte in mano a ciascun giocatore."""
    state = new_game_state(num_players=4, seed=456)

    assert state.is_team_game is True
    assert state.trump_card is not None
    assert len(state.deck) == 0
    assert [len(p.hand) for p in state.players] == [10, 10, 10, 10]

    cards = _all_cards_in_state(state)
    assert len(cards) == 40
    assert len(set(cards)) == 40


def test_complete_random_2p_game_ends_and_points_sum_to_120() -> None:
    """Esegue azioni valide fino al game over e verifica il totale punti (120)."""
    state = new_game_state(num_players=2, player_names=["A", "B"], seed=999)
    chooser = random.Random(999)

    safety = 1000
    while not state.game_over and safety > 0:
        safety -= 1

        current = state.current_turn
        hand_size = len(state.players[current].hand)
        assert hand_size > 0, "Se la partita non è finita deve esserci almeno una carta in mano"
        card_index = chooser.randrange(hand_size)

        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None

    assert safety > 0, "Loop di sicurezza: la partita dovrebbe terminare"
    assert state.game_over is True
    assert sum(p.points for p in state.players) == 120


def test_complete_random_4p_game_ends_and_points_sum_to_120() -> None:
    """Come il 2-player, ma in modalità 4 giocatori (a squadre)."""
    state = new_game_state(num_players=4, player_names=["A", "B", "C", "D"], seed=2025)
    chooser = random.Random(2025)

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1

        current = state.current_turn
        hand_size = len(state.players[current].hand)
        assert hand_size > 0
        card_index = chooser.randrange(hand_size)

        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None

    assert safety > 0
    assert state.game_over is True
    assert sum(p.points for p in state.players) == 120
