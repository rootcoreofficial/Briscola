"""
Test per `scripts/export_dataset.py`.

Obiettivo didattico
-------------------
Vogliamo garantire che l'exporter:
- esporti *di default* solo partite complete (game_over=true);
- supporti il nuovo evento `human_action` (modalità dataset, DB più piccolo);
- resti compatibile con DB legacy (action_play_card + observation_sent).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from briscola_ai.backend.event_log import EventLog, EventLogConfig
from briscola_ai.backend.event_log_reader import EventLogEventRow, EventLogGameRow

# Nota:
# `scripts/` non è un package Python installato, quindi in test carichiamo il modulo
# via path (import dinamico). In questo modo testiamo la logica reale dello script
# senza spostarla nel package.
_ROOT = Path(__file__).resolve().parents[1]
_EXPORTER_PATH = _ROOT / "scripts" / "export_dataset.py"
_spec = importlib.util.spec_from_file_location("export_dataset", _EXPORTER_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[misc]

ExportConfig = _mod.ExportConfig
export_dataset = _mod.export_dataset


def _read_jsonl(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


class _FakePostgresReader:
    """Reader fake: simula Postgres senza aprire connessioni reali."""

    backend_name = "postgres"

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def list_completed_game_ids(self) -> set[str]:
        return {"pg_game"}

    def iter_games(self) -> list[EventLogGameRow]:
        return [
            EventLogGameRow(
                game_id="pg_game",
                created_at=1.0,
                num_players=2,
                seed=99,
                code_version="0.10.0",
                rules_version="1",
                client_id="client_anon",
                finished_at=10.0,
                aborted_at=None,
                aborted_reason=None,
            )
        ]

    def iter_events(self) -> list[EventLogEventRow]:
        return [
            EventLogEventRow(
                id=1,
                game_id="pg_game",
                server_version=1,
                player_index=0,
                event_type="human_action",
                payload_json=json.dumps(
                    {
                        "player_index": 0,
                        "card_index": 7,
                        "observation": {"my_turn": True, "valid_actions": [7], "game_over": False},
                        "reward": 0,
                        "done": False,
                        "next_observation": {"my_turn": False, "valid_actions": [], "game_over": False},
                        "client_decision_time_ms": 250,
                    }
                ),
            )
        ]


def test_export_only_completed_games_human_action(tmp_path: Path) -> None:
    """
    Caso principale: backend in `BRISCOLA_EVENT_LOG_MODE=dataset` logga `human_action`
    e un marker `game_finished`. L'exporter deve produrre record puliti e completi.
    """
    db_path = tmp_path / "events.sqlite3"
    out_path = tmp_path / "dataset.jsonl"

    log = EventLog(EventLogConfig(path=str(db_path)))
    try:
        game_id = "game_complete"
        log.ensure_game(game_id, num_players=2, seed=123, code_version="0.0.0", rules_version="1")
        log.log_event(game_id, "game_created", {"seed": 123, "num_players": 2}, server_version=0)
        log.log_event(
            game_id,
            "human_action",
            {
                "player_index": 0,
                "card_index": 1,
                "observation": {
                    "type": "observation",
                    "game_over": False,
                    "my_turn": True,
                    "valid_actions": [1],
                    "players": [
                        {"index": 0, "name": "Nome Libero", "points": 0, "hand_size": 2},
                        {"index": 1, "name": "Giocatore AI", "points": 0, "hand_size": 3},
                    ],
                },
                "reward": 0,
                "done": False,
                "next_observation": {
                    "type": "observation",
                    "game_over": False,
                    "my_turn": False,
                    "valid_actions": [],
                    "players": [
                        {"index": 0, "name": "Nome Libero", "points": 0, "hand_size": 2},
                        {"index": 1, "name": "Giocatore AI", "points": 0, "hand_size": 3},
                    ],
                },
                "result": {
                    "played_card": {"suit": "coins", "rank": "ACE", "number": 1, "points": 11},
                    "trick_completed": False,
                },
                "client_observed_server_version": 0,
                "client_decision_time_ms": 1234,
            },
            server_version=1,
            player_index=0,
        )
        log.log_event(game_id, "game_finished", {"game_over": True}, server_version=2)
    finally:
        log.close()

    cfg = ExportConfig(
        db_path=db_path,
        out_path=out_path,
        player_index=0,
        include_ai=False,
        include_next_state=True,
        only_completed_games=True,
    )
    counters = export_dataset(cfg)
    assert counters["records_written"] == 1

    records = _read_jsonl(out_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["game_id"] == "game_complete"
    assert rec["player_index"] == 0
    assert rec["is_ai"] is False
    assert rec["action"] == {"card_index": 1}
    assert rec["result"]["played_card"]["rank"] == "ACE"
    assert rec["observation"]["my_turn"] is True
    assert rec["observation"]["players"][0]["name"] == "player_0"
    assert rec["observation"]["players"][1]["name"] == "player_1"
    assert isinstance(rec["next_observation"], dict)
    assert rec["next_observation"]["players"][0]["name"] == "player_0"
    assert rec["client"]["observed_server_version"] == 0
    assert rec["client"]["decision_time_ms"] == 1234


def test_export_reads_postgres_reader_when_database_url_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con `database_url` l'exporter deve usare il reader Postgres, senza richiedere un DB SQLite."""
    out_path = tmp_path / "dataset.jsonl"
    fake_reader = _FakePostgresReader()

    def fake_open_reader(*, sqlite_path: Path | None, database_url: str | None) -> _FakePostgresReader:
        assert sqlite_path is None
        assert database_url == "postgresql://fake/db"
        return fake_reader

    monkeypatch.setattr(_mod, "open_event_log_reader", fake_open_reader)

    cfg = ExportConfig(
        db_path=None,
        database_url="postgresql://fake/db",
        out_path=out_path,
        player_index=0,
        include_ai=False,
        include_next_state=True,
        only_completed_games=True,
    )
    counters = export_dataset(cfg)

    assert fake_reader.closed is True
    assert counters["records_written"] == 1
    records = _read_jsonl(out_path)
    assert records[0]["game_id"] == "pg_game"
    assert records[0]["metadata"]["client_id"] == "client_anon"
    assert records[0]["action"] == {"card_index": 7}


def test_export_skips_incomplete_games_by_default(tmp_path: Path) -> None:
    """
    Se manca un marker di completezza (`game_finished` o snapshot `game_over=true`),
    l'exporter deve scartare la partita quando `only_completed_games=True`.
    """
    db_path = tmp_path / "events.sqlite3"
    out_path = tmp_path / "dataset.jsonl"

    log = EventLog(EventLogConfig(path=str(db_path)))
    try:
        game_id = "game_incomplete"
        log.ensure_game(game_id, num_players=2, seed=1, code_version="0.0.0", rules_version="1")
        log.log_event(
            game_id,
            "human_action",
            {
                "player_index": 0,
                "card_index": 0,
                "observation": {"type": "observation", "game_over": False, "my_turn": True, "valid_actions": [0]},
                "reward": 0,
                "done": False,
                "next_observation": None,
            },
            server_version=1,
            player_index=0,
        )
    finally:
        log.close()

    cfg = ExportConfig(
        db_path=db_path,
        out_path=out_path,
        player_index=0,
        include_ai=False,
        include_next_state=True,
        only_completed_games=True,
    )
    counters = export_dataset(cfg)
    assert counters["records_written"] == 0
    assert out_path.read_text(encoding="utf-8").strip() == ""


def test_export_legacy_action_play_card_and_observation_sent(tmp_path: Path) -> None:
    """
    Compatibilità: DB legacy senza `human_action`.

    L'exporter ricostruisce (s, a, s') usando:
    - observation_sent prima dell'azione (my_turn=true + valid_actions coerente)
    - action_play_card
    - observation_sent dopo (game_over=true) per marcare la partita come completa
    """
    db_path = tmp_path / "events.sqlite3"
    out_path = tmp_path / "dataset.jsonl"

    log = EventLog(EventLogConfig(path=str(db_path)))
    try:
        game_id = "game_legacy"
        log.ensure_game(game_id, num_players=2, seed=42, code_version="0.0.0", rules_version="1")

        # observation prima dell'azione (coerente)
        log.log_event(
            game_id,
            "observation_sent",
            {"type": "observation", "my_turn": True, "valid_actions": [2], "game_over": False, "my_index": 0},
            server_version=0,
            player_index=0,
        )
        # azione
        log.log_event(
            game_id,
            "action_play_card",
            {
                "is_ai": False,
                "player_index": 0,
                "card_index": 2,
                "result": {"trick_completed": False},
            },
            server_version=1,
            player_index=0,
        )
        # observation dopo: game_over=true per segnare partita completa + next_state
        log.log_event(
            game_id,
            "observation_sent",
            {"type": "observation", "my_turn": False, "valid_actions": [], "game_over": True, "my_index": 0},
            server_version=2,
            player_index=0,
        )
    finally:
        log.close()

    cfg = ExportConfig(
        db_path=db_path,
        out_path=out_path,
        player_index=0,
        include_ai=False,
        include_next_state=True,
        only_completed_games=True,
    )
    counters = export_dataset(cfg)
    assert counters["records_written"] == 1

    records = _read_jsonl(out_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["game_id"] == "game_legacy"
    assert rec["action"] == {"card_index": 2}
    assert rec["next_observation"]["game_over"] is True
