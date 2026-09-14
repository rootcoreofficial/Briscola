"""
Test per `scripts/audit_event_log_games.py`.

Lo script serve a distinguere tre casi operativi importanti:
- partita PIMC in log dataset minimale: sappiamo l'avversario, ma non possiamo auditare le mosse IA;
- partita PIMC in log dataset con `ai_action`: auditabile senza usare il log debug completo;
- partita PIMC in log debug/full: eventi IA presenti, quindi auditabile;
- log legacy senza metadati agente: mosse umane presenti, ma avversario non ricostruibile.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from briscola_ai.backend.event_log_reader import EventLogEventRow, EventLogGameRow

_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_PATH = _ROOT / "scripts" / "audit_event_log_games.py"
_spec = importlib.util.spec_from_file_location("audit_event_log_games", _AUDIT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[misc]

AuditConfig = _mod.AuditConfig
build_event_log_games_audit_from_reader = _mod.build_event_log_games_audit_from_reader


class _FakeAuditReader:
    """Reader fake read-only: simula righe normalizzate senza aprire DB reali."""

    backend_name = "postgres"

    def close(self) -> None:
        pass

    def list_completed_game_ids(self) -> set[str]:
        return {"pimc_dataset"}

    def iter_games(self) -> list[EventLogGameRow]:
        return [
            EventLogGameRow(
                game_id="pimc_dataset",
                created_at=1.0,
                num_players=2,
                seed=10,
                code_version="0.14.1",
                rules_version="1",
                client_id="client_hidden",
                finished_at=20.0,
                aborted_at=None,
                aborted_reason=None,
            ),
            EventLogGameRow(
                game_id="pimc_debug",
                created_at=2.0,
                num_players=2,
                seed=11,
                code_version="0.14.1",
                rules_version="1",
                client_id="client_hidden",
                finished_at=None,
                aborted_at=None,
                aborted_reason=None,
            ),
            EventLogGameRow(
                game_id="pimc_dataset_ai",
                created_at=2.5,
                num_players=2,
                seed=13,
                code_version="0.14.1",
                rules_version="1",
                client_id="client_hidden",
                finished_at=None,
                aborted_at=None,
                aborted_reason=None,
            ),
            EventLogGameRow(
                game_id="legacy_human",
                created_at=3.0,
                num_players=2,
                seed=12,
                code_version="0.12.1",
                rules_version="1",
                client_id="client_hidden",
                finished_at=None,
                aborted_at=None,
                aborted_reason=None,
            ),
        ]

    def iter_events(self) -> list[EventLogEventRow]:
        return [
            EventLogEventRow(
                id=1,
                game_id="pimc_dataset",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.14.1",
                        "rules_version": "1",
                        "ai_agent": "bc_model_pimc_16x8",
                        "ai_model_id": "best_a2c_v6.npz",
                        "consent_to_data_collection": True,
                    }
                ),
            ),
            EventLogEventRow(
                id=2,
                game_id="pimc_dataset",
                server_version=1,
                player_index=0,
                event_type="human_action",
                payload_json=json.dumps(
                    {
                        "player_index": 0,
                        "card_index": 1,
                        "observation": {"my_turn": True},
                        "next_observation": {"my_turn": False},
                        "done": False,
                    }
                ),
            ),
            EventLogEventRow(
                id=3,
                game_id="pimc_dataset",
                server_version=40,
                player_index=None,
                event_type="game_finished",
                payload_json=json.dumps({"game_over": True}),
            ),
            EventLogEventRow(
                id=4,
                game_id="pimc_dataset_ai",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.14.1",
                        "rules_version": "1",
                        "ai_agent": "bc_model_pimc_16x8",
                        "ai_model_id": "best_a2c_v6.npz",
                        "consent_to_data_collection": True,
                    }
                ),
            ),
            EventLogEventRow(
                id=5,
                game_id="pimc_dataset_ai",
                server_version=1,
                player_index=0,
                event_type="human_action",
                payload_json=json.dumps(
                    {
                        "player_index": 0,
                        "card_index": 1,
                        "observation": {"my_turn": True},
                        "next_observation": {"my_turn": False},
                        "done": False,
                    }
                ),
            ),
            EventLogEventRow(
                id=6,
                game_id="pimc_dataset_ai",
                server_version=2,
                player_index=1,
                event_type="ai_action",
                payload_json=json.dumps(
                    {
                        "is_ai": True,
                        "player_index": 1,
                        "card_index": 0,
                        "observation": {"my_turn": True},
                        "next_observation": {"my_turn": False},
                        "done": False,
                        "decision_trace": {"decision_type": "search"},
                    }
                ),
            ),
            EventLogEventRow(
                id=7,
                game_id="value_lookahead_dataset_ai",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.16.0",
                        "rules_version": "1",
                        "ai_agent": "bc_model_value_lookahead_8x8",
                        "ai_model_id": "best_a2c_v6.npz",
                        "consent_to_data_collection": True,
                    }
                ),
            ),
            EventLogEventRow(
                id=8,
                game_id="value_lookahead_dataset_ai",
                server_version=1,
                player_index=0,
                event_type="human_action",
                payload_json=json.dumps(
                    {
                        "player_index": 0,
                        "card_index": 1,
                        "observation": {"my_turn": True},
                        "next_observation": {"my_turn": False},
                        "done": False,
                    }
                ),
            ),
            EventLogEventRow(
                id=9,
                game_id="value_lookahead_dataset_ai",
                server_version=2,
                player_index=1,
                event_type="ai_action",
                payload_json=json.dumps(
                    {
                        "is_ai": True,
                        "player_index": 1,
                        "card_index": 0,
                        "observation": {"my_turn": True},
                        "next_observation": {"my_turn": False},
                        "done": False,
                        "decision_trace": {"decision_type": "lookahead"},
                    }
                ),
            ),
            EventLogEventRow(
                id=10,
                game_id="pimc_debug",
                server_version=0,
                player_index=None,
                event_type="game_created",
                payload_json=json.dumps(
                    {
                        "code_version": "0.14.1",
                        "rules_version": "1",
                        "ai_agent": "bc_model_pimc_16x8",
                        "ai_model_id": "best_a2c_v6.npz",
                        "consent_to_data_collection": True,
                    }
                ),
            ),
            EventLogEventRow(
                id=11,
                game_id="pimc_debug",
                server_version=1,
                player_index=0,
                event_type="action_play_card",
                payload_json=json.dumps({"is_ai": False, "player_index": 0, "card_index": 2}),
            ),
            EventLogEventRow(
                id=12,
                game_id="pimc_debug",
                server_version=1,
                player_index=1,
                event_type="ai_card_reveal",
                payload_json=json.dumps({"card_index": 0, "card": {"suit": "cups", "number": 1}}),
            ),
            EventLogEventRow(
                id=13,
                game_id="pimc_debug",
                server_version=2,
                player_index=1,
                event_type="action_play_card",
                payload_json=json.dumps({"is_ai": True, "player_index": 1, "card_index": 0}),
            ),
            EventLogEventRow(
                id=14,
                game_id="pimc_debug",
                server_version=2,
                player_index=None,
                event_type="trick_result",
                payload_json=json.dumps({"winner_index": 1, "points": 11}),
            ),
            EventLogEventRow(
                id=15,
                game_id="legacy_human",
                server_version=1,
                player_index=0,
                event_type="human_action",
                payload_json=json.dumps({"player_index": 0, "card_index": 5, "observation": None}),
            ),
        ]


def test_audit_distinguishes_pimc_dataset_from_auditable_debug_log() -> None:
    """Una partita PIMC dataset-only non deve essere scambiata per auditabile."""
    report = build_event_log_games_audit_from_reader(
        _FakeAuditReader(),
        config=AuditConfig(db_path=None),
    )

    assert report["backend"] == "postgres"
    assert report["games"]["total"] == 5
    assert report["games"]["finished"] == 1
    assert report["games"]["pimc_games"] == 3
    assert report["games"]["pimc_with_ai_events"] == 2
    assert report["games"]["pimc_without_ai_events"] == 1
    assert report["gameplay_events"]["human_actions"] == 4
    assert report["gameplay_events"]["human_actions_missing_observation"] == 1
    assert report["gameplay_events"]["human_actions_missing_next_observation"] == 1
    assert report["gameplay_events"]["action_play_card_human"] == 1
    assert report["gameplay_events"]["action_play_card_ai"] == 1
    assert report["gameplay_events"]["ai_actions"] == 2
    assert report["gameplay_events"]["ai_actions_search"] == 1
    assert report["gameplay_events"]["ai_actions_lookahead"] == 1
    assert report["gameplay_events"]["ai_actions_unknown_decision"] == 0
    assert report["gameplay_events"]["ai_card_reveals"] == 1
    assert report["gameplay_events"]["trick_results"] == 1
    assert report["by_code_version"] == {"0.12.1": 1, "0.14.1": 3, "0.16.0": 1}
    assert report["by_ai_agent"]["bc_model_pimc_16x8"] == 3
    assert report["by_ai_agent"]["bc_model_value_lookahead_8x8"] == 1
    assert report["by_ai_agent"]["<none>"] == 1
    assert report["by_mode_guess"] == {
        "dataset_minimal": 4,
        "debug_or_full": 1,
    }
    assert report["by_audit_status"] == {
        "ai_metadata_only_no_ai_moves": 1,
        "ai_moves_auditable": 3,
        "human_dataset_no_ai_metadata": 1,
    }


def test_audit_filters_and_game_details_are_safe() -> None:
    """Il dettaglio per-partita deve rispettare i filtri senza stampare client_id/payload."""
    report = build_event_log_games_audit_from_reader(
        _FakeAuditReader(),
        config=AuditConfig(
            db_path=None,
            code_version="0.14.1",
            ai_agent="bc_model_pimc_16x8",
            include_games=True,
            game_limit=10,
        ),
    )

    assert report["games"]["total"] == 3
    assert len(report["games_detail"]) == 3
    assert {game["game_id"] for game in report["games_detail"]} == {
        "pimc_dataset",
        "pimc_dataset_ai",
        "pimc_debug",
    }
    assert {game["audit_status"] for game in report["games_detail"]} == {
        "ai_metadata_only_no_ai_moves",
        "ai_moves_auditable",
    }
    assert all("client_id" not in game for game in report["games_detail"])
    assert all("payload_json" not in game for game in report["games_detail"])
