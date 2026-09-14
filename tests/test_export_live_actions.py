"""
Test per `scripts/export_live_actions.py`.

L'export serve al prossimo audit di campo: deve fondere mosse umane e IA nello stesso
JSONL ordinato, senza richiedere il log debug completo e senza esporre `client_id` di
default.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from briscola_ai.backend.event_log_reader import EventLogEventRow, EventLogGameRow

_ROOT = Path(__file__).resolve().parents[1]
_EXPORTER_PATH = _ROOT / "scripts" / "export_live_actions.py"
_spec = importlib.util.spec_from_file_location("export_live_actions", _EXPORTER_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[misc]

ExportLiveActionsConfig = _mod.ExportLiveActionsConfig
export_live_actions_from_reader = _mod.export_live_actions_from_reader


def _read_jsonl(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


class _FakeLiveReader:
    """Reader fake read-only: simula il log Postgres senza connessioni esterne."""

    backend_name = "postgres"

    def close(self) -> None:
        pass

    def list_completed_game_ids(self) -> set[str]:
        return {"human_game", "bot_game"}

    def iter_games(self) -> list[EventLogGameRow]:
        return [
            EventLogGameRow(
                game_id="human_game",
                created_at=1.0,
                num_players=2,
                seed=123,
                code_version="0.32.0",
                rules_version="1",
                client_id="human_client",
                finished_at=20.0,
                aborted_at=None,
                aborted_reason=None,
            ),
            EventLogGameRow(
                game_id="bot_game",
                created_at=2.0,
                num_players=2,
                seed=124,
                code_version="0.32.0",
                rules_version="1",
                client_id="loadtest-bot",
                finished_at=21.0,
                aborted_at=None,
                aborted_reason=None,
            ),
        ]

    def iter_events(self) -> list[EventLogEventRow]:
        human_card = {"suit": "coins", "rank": "ACE", "number": 1, "points": 11}
        ai_card = {"suit": "cups", "rank": "TWO", "number": 2, "points": 0}
        return [
            EventLogEventRow(
                id=1,
                game_id="human_game",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.32.0",
                        "rules_version": "1",
                        "num_players": 2,
                        "seed": 123,
                        "ai_agent": "bc_model_pimc_belief_16x8",
                        "ai_model_id": "best_a2c_v11.npz",
                        "consent_to_data_collection": True,
                        "client_id": "must_not_leak",
                    }
                ),
            ),
            EventLogEventRow(
                id=2,
                game_id="human_game",
                server_version=1,
                player_index=0,
                event_type="human_action",
                payload_json=json.dumps(
                    {
                        "player_index": 0,
                        "card_index": 0,
                        "observation": {
                            "my_index": 0,
                            "my_hand": [human_card],
                            "table_cards": [],
                            "trick_history": [],
                            "cards_remaining_in_deck": 20,
                            "num_players": 2,
                            "players": [
                                {"index": 0, "name": "Nome Umano", "points": 0, "hand_size": 1},
                                {"index": 1, "name": "Nome IA", "points": 0, "hand_size": 1},
                            ],
                        },
                        "reward": 0,
                        "done": False,
                        "next_observation": {"my_index": 0, "my_hand": [], "table_cards": []},
                        "client_decision_time_ms": 321,
                        "client_id": "must_not_leak",
                    }
                ),
            ),
            EventLogEventRow(
                id=3,
                game_id="human_game",
                server_version=2,
                player_index=1,
                event_type="ai_action",
                payload_json=json.dumps(
                    {
                        "is_ai": True,
                        "player_index": 1,
                        "ai_agent": "bc_model_pimc_belief_16x8",
                        "ai_model_id": "best_a2c_v11.npz",
                        "card_index": 0,
                        "action_coerced": False,
                        "observation": {
                            "my_index": 1,
                            "my_hand": [ai_card],
                            "table_cards": [{"card": human_card, "player_index": 0}],
                            "trick_history": [],
                            "cards_remaining_in_deck": 20,
                            "num_players": 2,
                        },
                        "reward": -11,
                        "done": False,
                        "next_observation": {"my_index": 1, "my_hand": [], "table_cards": []},
                        "result": {
                            "played_card": ai_card,
                            "trick_completed": True,
                            "trick_winner": 0,
                            "trick_cards": [
                                {"card": human_card, "player_index": 0},
                                {"card": ai_card, "player_index": 1},
                            ],
                        },
                        "decision_trace": {"decision_type": "search", "determinizations": 16},
                    }
                ),
            ),
            EventLogEventRow(
                id=4,
                game_id="human_game",
                server_version=3,
                player_index=None,
                event_type="game_finished",
                payload_json=json.dumps(
                    {
                        "game_over": True,
                        "final_points_by_player_index": [61, 59],
                        "winning_player_index": 0,
                    }
                ),
            ),
            EventLogEventRow(
                id=5,
                game_id="bot_game",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.32.0",
                        "ai_agent": "bc_model_pimc_belief_16x8",
                        "ai_model_id": "best_a2c_v11.npz",
                    }
                ),
            ),
            EventLogEventRow(
                id=6,
                game_id="bot_game",
                server_version=1,
                player_index=0,
                event_type="human_action",
                payload_json=json.dumps(
                    {
                        "player_index": 0,
                        "card_index": 0,
                        "observation": {"my_hand": [human_card], "table_cards": [], "cards_remaining_in_deck": 20},
                    }
                ),
            ),
        ]


def test_export_live_actions_merges_human_and_ai_records_without_client_id(tmp_path: Path) -> None:
    out_path = tmp_path / "live_actions.jsonl"

    summary = export_live_actions_from_reader(
        _FakeLiveReader(),
        config=ExportLiveActionsConfig(
            db_path=None,
            out_path=out_path,
            ai_agent="bc_model_pimc_belief_16x8",
            exclude_client_ids=("loadtest-bot",),
        ),
    )

    assert summary["backend"] == "postgres"
    assert summary["counters"]["records_written"] == 2
    assert summary["counters"]["actions_skipped_filter"] == 1

    raw_output = out_path.read_text(encoding="utf-8")
    assert "client_id" not in raw_output
    assert "must_not_leak" not in raw_output

    records = _read_jsonl(out_path)
    assert [record["actor"] for record in records] == ["human", "ai"]

    human = records[0]
    assert human["action"]["card"]["rank"] == "ACE"  # inferito da observation.my_hand
    assert human["phase"]["is_lead"] is True
    assert human["metadata"]["final_points_by_player_index"] == [61, 59]
    assert human["client"]["decision_time_ms"] == 321
    assert human["observation"]["players"][0]["name"] == "player_0"

    ai = records[1]
    assert ai["phase"]["is_lead"] is False
    assert ai["ai"]["decision_type"] == "search"
    assert ai["trick"]["completed"] is True
    assert ai["trick"]["winner_index"] == 0
