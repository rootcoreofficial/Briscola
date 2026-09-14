"""
Test dei reader read-only per event log.

Il reader Postgres viene esercitato con una connessione fake: verifichiamo mapping
e query senza contattare database reali.
"""

from __future__ import annotations

import json
from typing import Any

from briscola_ai.backend.event_log_reader import PostgresEventLogReader


class _FakeReadCursor:
    def __init__(self, conn: _FakeReadConn) -> None:
        self._conn = conn
        self._rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> _FakeReadCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        normalized = " ".join(sql.split())
        self._conn.executed.append((normalized, tuple(params)))
        self._rows = self._conn.rows_for(normalized, tuple(params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeReadConn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.closed = False

    def cursor(self) -> _FakeReadCursor:
        return _FakeReadCursor(self)

    def close(self) -> None:
        self.closed = True

    def rows_for(self, sql: str, params: tuple) -> list[tuple[Any, ...]]:
        if "information_schema.columns" in sql and params == ("games",):
            return [
                ("game_id",),
                ("created_at",),
                ("num_players",),
                ("seed",),
                ("code_version",),
                ("rules_version",),
                ("client_id",),
                ("finished_at",),
                ("aborted_at",),
                ("aborted_reason",),
            ]
        if sql.startswith("SELECT DISTINCT game_id FROM events WHERE event_type = 'game_finished'"):
            return [("pg_game",)]
        if sql.startswith("SELECT game_id, created_at, num_players"):
            return [("pg_game", 10.0, 2, 123, "0.10.0", "1", "client_x", 50.0, None, None)]
        if sql.startswith("SELECT id, game_id, server_version"):
            return [
                (
                    7,
                    "pg_game",
                    3,
                    0,
                    "human_action",
                    json.dumps({"player_index": 0, "card_index": 4}),
                )
            ]
        return []


def test_postgres_event_log_reader_maps_fake_rows() -> None:
    """Il reader Postgres deve normalizzare righe games/events senza possedere la connessione iniettata."""
    conn = _FakeReadConn()
    reader = PostgresEventLogReader(conn=conn)

    assert reader.list_completed_game_ids() == {"pg_game"}
    games = list(reader.iter_games())
    events = list(reader.iter_events())
    reader.close()

    assert games[0].game_id == "pg_game"
    assert games[0].client_id == "client_x"
    assert games[0].finished_at == 50.0
    assert events[0].id == 7
    assert events[0].event_type == "human_action"
    assert conn.closed is False  # le connessioni iniettate restano di proprietà del test/chiamante
    assert any("information_schema.columns" in sql for sql, _ in conn.executed)
