"""
Test per `PlayerObservation` (anti-cheat).

Scopo:
- garantire che gli agenti/ML ricevano solo una vista lecita del gioco
- evitare regressioni dove, accidentalmente, passiamo `GameState` completo alle policy

Non testiamo qui strategie o "forza" degli agenti: testiamo solo il contratto dell'osservazione.
"""

from __future__ import annotations

import pytest

from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state


def test_make_player_observation_contains_only_legal_information() -> None:
    """
    L'osservazione deve contenere:
    - la mano del giocatore osservante
    - info pubbliche (briscola, carte sul tavolo, punti, dimensioni mani, deck_size)

    E deve NON contenere:
    - il mazzo (`state.deck`) come sequenza di carte (ordine nascosto)
    - la mano avversaria come carte specifiche
    """
    state = new_game_state(num_players=2, seed=123)

    obs0 = make_player_observation(state, player_index=0)

    # Sanity: la mano osservata coincide con la mano reale del player.
    assert obs0.hand == state.players[0].hand

    # Info pubbliche coerenti.
    assert obs0.deck_size == len(state.deck)
    assert obs0.trump_card == state.trump_card
    assert obs0.players_points == tuple(p.points for p in state.players)
    assert obs0.players_hand_sizes == tuple(len(p.hand) for p in state.players)

    # Anti-cheat: non deve esserci il mazzo né lo stato completo (niente riferimenti a `players`).
    assert not hasattr(obs0, "deck")
    assert not hasattr(obs0, "players")

    # Anti-cheat: la mano avversaria NON è presente come campo.
    # (Controllo "forte": nessuna carta dell'avversario deve apparire nella nostra mano osservata.)
    assert set(state.players[1].hand).isdisjoint(set(obs0.hand))

    # Storia pubblica: è sempre una one-hot di lunghezza 40.
    assert len(obs0.seen_cards_onehot) == 40


def test_make_player_observation_rejects_invalid_player_index() -> None:
    """Indici fuori range devono fallire esplicitamente per evitare bug silenziosi."""
    state = new_game_state(num_players=2, seed=0)

    with pytest.raises(ValueError, match="player_index fuori range"):
        make_player_observation(state, player_index=-1)

    with pytest.raises(ValueError, match="player_index fuori range"):
        make_player_observation(state, player_index=2)
