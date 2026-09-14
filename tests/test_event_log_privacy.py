"""Test per gli helper privacy dei payload dataset."""

from __future__ import annotations

from briscola_ai.backend.event_log_privacy import sanitize_dataset_payload


def test_sanitize_dataset_payload_replaces_player_names_but_keeps_features() -> None:
    """I nomi liberi non devono finire nei dataset, ma indici/feature restano invariati."""
    payload = {
        "player_names": ["Alice", "Bot"],
        "observation": {
            "players": [
                {"index": 0, "name": "Alice", "points": 12, "hand_size": 2},
                {"index": 1, "name": "Bot", "points": 8, "hand_size": 3},
            ],
            "valid_actions": [0, 1],
            "seen_cards_onehot": [0, 1, 0],
        },
        "winner_index": 1,
        "winner_name": "Bot",
        "client_id": "client_pseudonym",
    }

    cleaned = sanitize_dataset_payload(payload)

    assert cleaned["player_names"] == ["player_0", "player_1"]
    assert cleaned["observation"]["players"][0]["name"] == "player_0"
    assert cleaned["observation"]["players"][1]["name"] == "player_1"
    assert cleaned["winner_name"] == "player_1"
    assert cleaned["client_id"] == "client_pseudonym"
    assert cleaned["observation"]["valid_actions"] == [0, 1]
    assert cleaned["observation"]["seen_cards_onehot"] == [0, 1, 0]
