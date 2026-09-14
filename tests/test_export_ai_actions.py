"""
Test per `scripts/export_ai_actions.py`.

Lo script esporta il dettaglio delle mosse IA/PIMC gia' salvate in event log come
`ai_action`. Il requisito principale non e' addestrativo ma diagnostico: poter
ispezionare fallback/solver/search senza stampare payload grezzi o identificativi
come `client_id`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from briscola_ai.backend.event_log_reader import EventLogEventRow, EventLogGameRow

_ROOT = Path(__file__).resolve().parents[1]
_EXPORTER_PATH = _ROOT / "scripts" / "export_ai_actions.py"
_spec = importlib.util.spec_from_file_location("export_ai_actions", _EXPORTER_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[misc]

ExportAIActionsConfig = _mod.ExportAIActionsConfig
export_ai_actions_from_reader = _mod.export_ai_actions_from_reader


def _read_jsonl(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


class _FakeAIActionsReader:
    """Reader fake read-only: simula Postgres senza aprire connessioni reali."""

    backend_name = "postgres"

    def close(self) -> None:
        pass

    def list_completed_game_ids(self) -> set[str]:
        return {"pimc_game", "other_game"}

    def iter_games(self) -> list[EventLogGameRow]:
        return [
            EventLogGameRow(
                game_id="pimc_game",
                created_at=1.0,
                num_players=2,
                seed=777,
                code_version="0.15.0",
                rules_version="1",
                client_id="client_hidden",
                finished_at=20.0,
                aborted_at=None,
                aborted_reason=None,
            ),
            EventLogGameRow(
                game_id="other_game",
                created_at=2.0,
                num_players=2,
                seed=778,
                code_version="0.15.0",
                rules_version="1",
                client_id="client_hidden_other",
                finished_at=21.0,
                aborted_at=None,
                aborted_reason=None,
            ),
        ]

    def iter_events(self) -> list[EventLogEventRow]:
        return [
            EventLogEventRow(
                id=1,
                game_id="pimc_game",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.15.0",
                        "rules_version": "1",
                        "ai_agent": "bc_model_pimc_16x8",
                        "ai_model_id": "best_a2c_v6.npz",
                        "consent_to_data_collection": True,
                        "client_id": "must_not_leak",
                    }
                ),
            ),
            EventLogEventRow(
                id=2,
                game_id="pimc_game",
                server_version=12,
                player_index=1,
                event_type="ai_action",
                payload_json=json.dumps(
                    {
                        "is_ai": True,
                        "player_index": 1,
                        "ai_agent": "bc_model_pimc_16x8",
                        "ai_model_id": "best_a2c_v6.npz",
                        "card_index": 0,
                        "action_coerced": False,
                        "decision_trace": {
                            "decision_type": "search",
                            "determinizations": 16,
                            "max_unknown_cards": 8,
                        },
                        "observation": {
                            "cards_remaining_in_deck": 8,
                            "my_hand": [{"suit": "coins", "rank": "three"}],
                            "table_cards": [],
                            "players": [
                                {"index": 0, "name": "Marco"},
                                {"index": 1, "name": "Bot"},
                            ],
                            "client_id": "nested_must_not_leak",
                        },
                        "next_observation": {
                            "cards_remaining_in_deck": 7,
                            "my_hand": [],
                            "table_cards": [{"suit": "coins", "rank": "three"}],
                            "players": [
                                {"index": 0, "name": "Marco"},
                                {"index": 1, "name": "Bot"},
                            ],
                        },
                        "result": {
                            "played_card": {"suit": "coins", "rank": "three"},
                            "winner_index": 1,
                            "winner_name": "Bot",
                            "client_id": "result_must_not_leak",
                        },
                        "reward": 3,
                        "done": False,
                        "client_id": "top_must_not_leak",
                    }
                ),
            ),
            EventLogEventRow(
                id=3,
                game_id="other_game",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.15.0",
                        "rules_version": "1",
                        "ai_agent": "bc_model_hybrid_endgame",
                        "ai_model_id": "best_a2c_v6.npz",
                        "consent_to_data_collection": True,
                    }
                ),
            ),
            EventLogEventRow(
                id=4,
                game_id="other_game",
                server_version=8,
                player_index=1,
                event_type="ai_action",
                payload_json=json.dumps(
                    {
                        "is_ai": True,
                        "player_index": 1,
                        "card_index": 1,
                        "decision_trace": {"decision_type": "solver"},
                        "observation": {"cards_remaining_in_deck": 0, "my_hand": [], "table_cards": []},
                    }
                ),
            ),
        ]


def test_export_ai_actions_writes_sanitized_detailed_records(tmp_path: Path) -> None:
    """L'export deve includere il dettaglio PIMC ma non campi sensibili o nomi liberi."""
    out_path = tmp_path / "ai_actions.jsonl"

    summary = export_ai_actions_from_reader(
        _FakeAIActionsReader(),
        config=ExportAIActionsConfig(
            db_path=None,
            out_path=out_path,
            game_id="pimc_game",
            include_observations=True,
        ),
    )

    assert summary["backend"] == "postgres"
    assert summary["counters"]["ai_actions_seen"] == 2
    assert summary["counters"]["ai_actions_skipped_filter"] == 1
    assert summary["counters"]["records_written"] == 1
    assert summary["decision_types"] == {"search": 1}

    raw_output = out_path.read_text(encoding="utf-8")
    assert "client_id" not in raw_output
    assert "must_not_leak" not in raw_output

    records = _read_jsonl(out_path)
    assert len(records) == 1
    record = records[0]
    assert record["game_id"] == "pimc_game"
    assert record["event_id"] == 2
    assert record["player_index"] == 1
    assert record["metadata"]["ai_agent"] == "bc_model_pimc_16x8"
    assert record["metadata"]["ai_model_id"] == "best_a2c_v6.npz"
    assert "client_id" not in record["metadata"]
    assert record["phase"] == {"deck_size": 8, "hand_size": 1, "table_size": 0}
    assert record["decision_type"] == "search"
    assert record["decision_trace"]["determinizations"] == 16
    assert record["action"] == {
        "card_index": 0,
        "card": {"suit": "coins", "rank": "three"},
        "coerced": False,
    }
    assert record["observation"]["players"][0]["name"] == "player_0"
    assert record["observation"]["players"][1]["name"] == "player_1"
    assert record["result"]["winner_name"] == "player_1"


def test_export_ai_actions_can_omit_observations_but_keep_phase(tmp_path: Path) -> None:
    """`--no-observations` produce un file piu' leggero senza perdere il sommario fase/decisione."""
    out_path = tmp_path / "ai_actions_summary.jsonl"

    summary = export_ai_actions_from_reader(
        _FakeAIActionsReader(),
        config=ExportAIActionsConfig(
            db_path=None,
            out_path=out_path,
            ai_agent="bc_model_pimc_16x8",
            include_observations=False,
        ),
    )

    assert summary["counters"]["records_written"] == 1
    assert summary["decision_types"] == {"search": 1}

    record = _read_jsonl(out_path)[0]
    assert record["phase"]["deck_size"] == 8
    assert record["decision_type"] == "search"
    assert record["observation"] is None
    assert record["next_observation"] is None
