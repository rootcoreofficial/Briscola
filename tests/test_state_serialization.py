"""
Test del round-trip di serializzazione `GameState <-> dict` (per game store condiviso).
"""

from __future__ import annotations

import json

from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.serialization import game_state_from_dict, game_state_to_dict
from briscola_ai.domain.state import new_game_state


def _play_some(state, moves: int):
    for _ in range(moves):
        if state.game_over:
            break
        state, _ = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
    return state


def test_roundtrip_initial_2p() -> None:
    """Serializzare e rideserializzare lo stato iniziale 2p deve restituire un GameState
    identico (== campo per campo)."""
    state = new_game_state(num_players=2, seed=42)
    restored = game_state_from_dict(game_state_to_dict(state))
    assert restored == state


def test_roundtrip_initial_4p() -> None:
    """Il round-trip di serializzazione deve preservare anche lo stato iniziale 4p (a squadre),
    restituendo un GameState identico."""
    state = new_game_state(num_players=4, seed=7)
    restored = game_state_from_dict(game_state_to_dict(state))
    assert restored == state


def test_roundtrip_midgame_and_endgame_2p() -> None:
    """Il round-trip deve restare fedele anche a stato mid-game (tavolo parziale, prese accumulate)
    e a stato endgame (game_over) 2p."""
    state = new_game_state(num_players=2, seed=123)
    # Mid-game (tavolo parziale o pieno, prese accumulate).
    state = _play_some(state, 7)
    assert game_state_from_dict(game_state_to_dict(state)) == state
    # Endgame: gioca fino alla fine.
    state = _play_some(state, 1000)
    assert state.game_over
    assert game_state_from_dict(game_state_to_dict(state)) == state


def test_serialized_is_json_safe() -> None:
    """Il dict non deve contenere enum/oggetti: deve passare per json.dumps/loads invariato."""
    state = new_game_state(num_players=2, seed=5)
    state = _play_some(state, 3)
    as_dict = game_state_to_dict(state)
    roundtrip_json = json.loads(json.dumps(as_dict))
    assert game_state_from_dict(roundtrip_json) == state
