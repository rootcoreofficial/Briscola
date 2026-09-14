"""
Test per `scripts/report_event_log.py`.

Il report deve restare aggregato e read-only: conta partite, consenso ed eventi
dataset senza esporre payload o identificatori client.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

from briscola_ai.backend.event_log import EventLog, EventLogConfig

_ROOT = Path(__file__).resolve().parents[1]
_REPORT_PATH = _ROOT / "scripts" / "report_event_log.py"
_spec = importlib.util.spec_from_file_location("report_event_log", _REPORT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[misc]

ReportConfig = _mod.ReportConfig
build_event_log_report = _mod.build_event_log_report


def test_report_counts_games_consent_and_dataset_quality(tmp_path: Path) -> None:
    """Il report deve contare complete, abortite, consenso e anomalie `human_action`."""
    now = time.time()
    db_path = tmp_path / "events.sqlite3"
    log = EventLog(EventLogConfig(path=str(db_path)))
    try:
        log.ensure_game("done", num_players=2, seed=1, code_version="0.10.0", rules_version="1")
        log.set_client_id("done", client_id="client_a")
        log.log_event("done", "game_created", {"consent_to_data_collection": True}, server_version=0)
        log.log_event(
            "done",
            "human_action",
            {
                "player_index": 0,
                "card_index": 1,
                "observation": {"my_turn": True},
                "next_observation": {"my_turn": False},
                "done": True,
            },
            server_version=1,
            player_index=0,
        )
        assert log.try_mark_game_finished("done", finished_at=now - 10.0) is True
        log.log_event("done", "game_finished", {"game_over": True}, server_version=2)

        log.ensure_game("aborted", num_players=2, seed=2, code_version="0.10.0", rules_version="1")
        log.log_event("aborted", "game_created", {"consent_to_data_collection": True}, server_version=0)
        assert log.try_mark_game_aborted("aborted", aborted_reason="inactive_timeout", aborted_at=now - 20.0) is True
        log.log_event("aborted", "game_aborted", {"reason": "inactive_timeout"}, server_version=1)

        log.ensure_game("open", num_players=2, seed=3, code_version="0.10.0", rules_version="1")
        log.log_event("open", "game_created", {"consent_to_data_collection": False}, server_version=0)
        log.log_event(
            "open",
            "human_action",
            {"player_index": 0, "card_index": 5, "observation": None, "next_observation": None},
            server_version=1,
            player_index=0,
        )
    finally:
        log.close()

    report = build_event_log_report(ReportConfig(db_path=db_path), now=now + 1.0)

    assert report["backend"] == "sqlite"
    assert report["games"]["total"] == 3
    assert report["games"]["finished"] == 1
    assert report["games"]["aborted"] == 1
    assert report["games"]["open"] == 1
    assert report["games"]["with_client_id"] == 1
    assert report["consent"]["games_with_consent"] == 2
    assert report["consent"]["games_without_consent"] == 1
    assert report["dataset_quality"]["human_actions"] == 2
    assert report["dataset_quality"]["human_actions_missing_observation"] == 1
    assert report["dataset_quality"]["human_actions_missing_next_observation"] == 1
    assert report["aborted_reasons"] == {"inactive_timeout": 1}
