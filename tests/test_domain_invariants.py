"""
Test di invarianti per il dominio (Phase 3).

Scopo didattico:
- avere una rete di sicurezza per refactor futuri del motore (`GameState + step`)
- verificare proprietà "sempre vere" del gioco, indipendenti dalla UI o dall'API

Questi test NON cercano di coprire ogni regola nel dettaglio (per quello ci sono test mirati),
ma di assicurare che lo stato rimanga coerente durante l'evoluzione di una partita.
"""

from __future__ import annotations

import random

import pytest

from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card
from briscola_ai.domain.state import GameState, new_game_state


def _all_physical_cards_in_state(state: GameState) -> list[Card]:
    """
    Raccoglie tutte le carte "fisiche" presenti nello stato.

    Nota importante:
    - Non includiamo `state.trump_card` perché può essere un riferimento ridondante:
      - in 2-player la briscola scoperta è reinserita nel deck (quindi è già in `state.deck`)
      - in 4-player la briscola è una carta che sta già in mano a un giocatore
    """
    cards: list[Card] = []
    cards.extend(state.deck)
    for p in state.players:
        cards.extend(p.hand)
        cards.extend(p.captured_cards)
    cards.extend([card for card, _ in state.table_cards])
    return cards


def _assert_state_invariants(state: GameState) -> None:
    """
    Verifica invarianti forti che devono valere sempre.

    Queste regole sono pensate per catturare bug "strutturali", ad esempio:
    - duplicazioni di carte (stessa carta in due posti diversi)
    - turni fuori range o incoerenti
    - carte sul tavolo attribuite a un player inesistente o duplicate nello stesso round
    """
    assert state.num_players in (2, 4)
    assert len(state.players) == state.num_players

    assert 0 <= state.current_turn < state.num_players
    assert 0 <= state.first_player < state.num_players

    # Il tavolo non può contenere più carte del numero di player.
    assert len(state.table_cards) <= state.num_players

    # Le carte sul tavolo devono essere attribuite a player validi e ogni player può giocare al massimo
    # una carta per round (quindi niente duplicati di player_index nella stessa mano).
    table_player_indices = [player_idx for _, player_idx in state.table_cards]
    assert all(0 <= idx < state.num_players for idx in table_player_indices)
    assert len(set(table_player_indices)) == len(table_player_indices)

    # In 4-player il mazzo viene distribuito interamente all'inizio.
    if state.is_team_game:
        assert state.num_players == 4
        assert state.teams is not None
        assert len(state.deck) == 0
    else:
        assert state.num_players == 2
        assert state.teams is None

    # Invariante fondamentale: il sistema contiene sempre 40 carte uniche.
    all_cards = _all_physical_cards_in_state(state)
    assert len(all_cards) == 40
    assert len(set(all_cards)) == 40

    # Invariante punti: lo score del player è sempre la somma dei punti delle carte raccolte.
    for p in state.players:
        assert p.points == sum(card.rank.points for card in p.captured_cards)

    # A game_over non deve esserci una mano "in sospeso" sul tavolo.
    if state.game_over:
        assert len(state.table_cards) == 0


@pytest.mark.parametrize(
    "num_players,seed",
    [
        (2, 123),
        (4, 456),
    ],
)
def test_new_game_state_satisfies_invariants(num_players: int, seed: int) -> None:
    """Lo stato iniziale deve rispettare tutte le invarianti strutturali."""
    state = new_game_state(num_players=num_players, seed=seed)
    _assert_state_invariants(state)


@pytest.mark.parametrize(
    "num_players,seed,chooser_seed",
    [
        (2, 42, 999),
        (4, 7, 123),
    ],
)
def test_random_game_preserves_invariants_and_terminates(num_players: int, seed: int, chooser_seed: int) -> None:
    """
    Simula una partita random deterministica e verifica invarianti ad ogni step.

    Nota:
    - scegliamo `card_index` in base alla dimensione della mano del player corrente.
    - in caso di bug (duplicazioni, turni fuori range, tavolo incoerente) vogliamo fallire
      il più vicino possibile al momento in cui lo stato si rompe.
    """
    state = new_game_state(num_players=num_players, seed=seed)
    chooser = random.Random(chooser_seed)

    _assert_state_invariants(state)

    # Ogni partita ha 40 giocate totali (una per carta). Tenere un safety più alto per robustezza.
    safety = 300
    while not state.game_over and safety > 0:
        safety -= 1

        current = state.current_turn
        hand_size = len(state.players[current].hand)
        assert hand_size > 0

        card_index = chooser.randrange(hand_size)
        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None

        _assert_state_invariants(state)

    assert safety > 0, "La partita dovrebbe terminare in un numero finito di mosse"
    assert state.game_over is True
    assert sum(p.points for p in state.players) == 120
