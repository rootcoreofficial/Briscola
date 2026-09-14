"""
Lettura read-only dell'event log (`games`/`events`) su SQLite o Postgres.

Il backend applicativo usa `event_log.py` per scrivere eventi. Gli script offline
di osservabilità/export hanno invece bisogno di leggere lo stesso schema da due
backend diversi:

- SQLite locale, comodo in sviluppo e nei test;
- Postgres/Neon in produzione, selezionato tramite `DATABASE_URL`.

Questo modulo isola il dialetto SQL minimo necessario agli script, senza
duplicare la logica di export/report e senza contattare servizi esterni nei test
(le classi accettano connessioni iniettate).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class EventLogGameRow:
    """Riga normalizzata della tabella `games`."""

    game_id: str
    created_at: float | None
    num_players: int | None
    seed: int | None
    code_version: str | None
    rules_version: str | None
    client_id: str | None
    finished_at: float | None
    aborted_at: float | None
    aborted_reason: str | None


@dataclass(frozen=True)
class EventLogEventRow:
    """Riga normalizzata della tabella `events`."""

    id: int
    game_id: str
    server_version: int | None
    player_index: int | None
    event_type: str
    payload_json: str


class EventLogReader(Protocol):
    """Interfaccia read-only comune per SQLite e Postgres."""

    @property
    def backend_name(self) -> str: ...

    def close(self) -> None: ...

    def list_completed_game_ids(self) -> set[str]: ...

    def iter_games(self) -> Iterable[EventLogGameRow]: ...

    def iter_events(self) -> Iterable[EventLogEventRow]: ...


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _game_from_mapping(data: dict[str, Any]) -> EventLogGameRow:
    """Converte una riga DB parziale in `EventLogGameRow`.

    Alcuni DB SQLite locali possono essere stati creati prima dell'aggiunta di
    colonne come `client_id` o `finished_at`: i campi mancanti diventano `None`.
    """

    return EventLogGameRow(
        game_id=str(data["game_id"]),
        created_at=_optional_float(data.get("created_at")),
        num_players=_optional_int(data.get("num_players")),
        seed=_optional_int(data.get("seed")),
        code_version=_optional_str(data.get("code_version")),
        rules_version=_optional_str(data.get("rules_version")),
        client_id=_optional_str(data.get("client_id")),
        finished_at=_optional_float(data.get("finished_at")),
        aborted_at=_optional_float(data.get("aborted_at")),
        aborted_reason=_optional_str(data.get("aborted_reason")),
    )


class SQLiteEventLogReader:
    """Reader read-only per DB SQLite dell'event log."""

    backend_name = "sqlite"

    def __init__(self, path: str | Path, *, conn: sqlite3.Connection | None = None) -> None:
        self._owns_connection = conn is None
        self._conn = conn if conn is not None else sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def _columns(self, table: str) -> set[str]:
        return {str(row[1]) for row in self._conn.execute(f"PRAGMA table_info({table});").fetchall()}

    def list_completed_game_ids(self) -> set[str]:
        completed: set[str] = set()
        for row in self._conn.execute("SELECT DISTINCT game_id FROM events WHERE event_type = 'game_finished';"):
            completed.add(str(row["game_id"]))
        for row in self._conn.execute(
            """
            SELECT DISTINCT game_id
            FROM events
            WHERE event_type = 'observation_sent'
              AND payload_json LIKE '%"game_over":true%';
            """
        ):
            completed.add(str(row["game_id"]))
        return completed

    def iter_games(self) -> Iterable[EventLogGameRow]:
        game_columns = self._columns("games")
        select_cols = [
            c
            for c in (
                "game_id",
                "created_at",
                "num_players",
                "seed",
                "code_version",
                "rules_version",
                "client_id",
                "finished_at",
                "aborted_at",
                "aborted_reason",
            )
            if c in game_columns
        ]
        order_col = "created_at" if "created_at" in game_columns else "game_id"
        for row in self._conn.execute(f"SELECT {', '.join(select_cols)} FROM games ORDER BY {order_col}, game_id;"):
            yield _game_from_mapping({col: row[col] for col in select_cols})

    def iter_events(self) -> Iterable[EventLogEventRow]:
        for row in self._conn.execute(
            """
            SELECT id, game_id, server_version, player_index, event_type, payload_json
            FROM events
            ORDER BY game_id, id;
            """
        ):
            yield EventLogEventRow(
                id=int(row["id"]),
                game_id=str(row["game_id"]),
                server_version=_optional_int(row["server_version"]),
                player_index=_optional_int(row["player_index"]),
                event_type=str(row["event_type"]),
                payload_json=str(row["payload_json"]),
            )


class PostgresEventLogReader:
    """Reader read-only per event log Postgres.

    `psycopg` viene importato solo quando serve. Nei test è possibile iniettare
    una connessione fake compatibile con il sottoinsieme `cursor()/execute()/fetchall()`.
    """

    backend_name = "postgres"

    def __init__(self, dsn: str | None = None, *, conn: Any = None) -> None:
        self._owns_connection = conn is None
        if conn is not None:
            self._conn = conn
        elif dsn is not None:
            import psycopg  # import lazy: gli script locali SQLite non caricano Postgres

            self._conn = psycopg.connect(dsn)
        else:
            raise ValueError("PostgresEventLogReader richiede `dsn` oppure `conn`.")

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def _columns(self, table: str) -> set[str]:
        rows = self._fetchall(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s;
            """,
            (table,),
        )
        return {str(row[0]) for row in rows}

    def list_completed_game_ids(self) -> set[str]:
        rows = self._fetchall(
            """
            SELECT DISTINCT game_id FROM events WHERE event_type = 'game_finished'
            UNION
            SELECT DISTINCT game_id
            FROM events
            WHERE event_type = 'observation_sent'
              AND payload_json LIKE %s;
            """,
            ('%"game_over":true%',),
        )
        return {str(row[0]) for row in rows}

    def iter_games(self) -> Iterable[EventLogGameRow]:
        game_columns = self._columns("games")
        select_cols = [
            c
            for c in (
                "game_id",
                "created_at",
                "num_players",
                "seed",
                "code_version",
                "rules_version",
                "client_id",
                "finished_at",
                "aborted_at",
                "aborted_reason",
            )
            if c in game_columns
        ]
        order_col = "created_at" if "created_at" in game_columns else "game_id"
        rows = self._fetchall(f"SELECT {', '.join(select_cols)} FROM games ORDER BY {order_col}, game_id;")
        for row in rows:
            yield _game_from_mapping(dict(zip(select_cols, row, strict=True)))

    def iter_events(self) -> Iterable[EventLogEventRow]:
        rows = self._fetchall(
            """
            SELECT id, game_id, server_version, player_index, event_type, payload_json
            FROM events
            ORDER BY game_id, id;
            """
        )
        for row in rows:
            yield EventLogEventRow(
                id=int(row[0]),
                game_id=str(row[1]),
                server_version=_optional_int(row[2]),
                player_index=_optional_int(row[3]),
                event_type=str(row[4]),
                payload_json=str(row[5]),
            )


def open_event_log_reader(*, sqlite_path: str | Path | None, database_url: str | None) -> EventLogReader:
    """Apre il reader corretto: Postgres se `database_url` è presente, altrimenti SQLite."""

    if database_url:
        return PostgresEventLogReader(database_url)
    if sqlite_path is not None:
        return SQLiteEventLogReader(sqlite_path)
    raise ValueError("Serve `database_url` oppure `sqlite_path` per leggere l'event log.")
