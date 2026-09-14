"""
Event log “da laboratorio” (SQLite) per Briscola AI.

Obiettivo didattico
-------------------
Quando iniziamo a fare ML (dataset, self-play, valutazione), diventa fondamentale poter:
- riprodurre una partita (seed + sequenza di azioni);
- capire *cosa* è successo e *quando* (ordering, `server_version`);
- esportare i dati in un formato adatto al training (es. JSONL).

Questo modulo implementa un event log append-only su SQLite.
È volutamente semplice:
- usa solo la stdlib (`sqlite3`, `json`);
- non impone uno schema “finale” dei payload: i dettagli vivono in `payload_json`.

Configurazione
-------------
Il percorso del DB è configurabile (env/CLI) dal livello applicativo.
Se non viene fornito alcun path, la feature può restare disabilitata.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class EventLogConfig:
    """
    Configurazione del logger.

    - `path`: percorso del file SQLite (es. `./data/briscola_events.sqlite3`).
      Se `:memory:` usa un database in memoria (utile nei test).
    """

    path: str


@runtime_checkable
class EventLogProtocol(Protocol):
    """
    Interfaccia comune dei backend event log (SQLite locale, Postgres in cloud).

    Permette al resto dell'app di non dipendere dall'implementazione concreta:
    `main.py` sceglie il backend (factory) e `server.py` usa solo questi metodi.
    """

    @property
    def path(self) -> str: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def database_name(self) -> str | None: ...

    @property
    def database_host(self) -> str | None: ...

    def health_check(self) -> bool: ...

    def count_games(self) -> int | None: ...

    def count_games_by_model(self) -> list[dict] | None: ...

    def close(self) -> None: ...

    def ensure_game(
        self,
        game_id: str,
        *,
        num_players: int,
        seed: int | None = None,
        code_version: str | None = None,
        rules_version: str | None = None,
    ) -> None: ...

    def set_client_id(self, game_id: str, *, client_id: str) -> None: ...

    def try_mark_game_finished(self, game_id: str, *, finished_at: float | None = None) -> bool: ...

    def try_mark_game_aborted(self, game_id: str, *, aborted_reason: str, aborted_at: float | None = None) -> bool: ...

    def log_event(
        self,
        game_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        server_version: int | None = None,
        player_index: int | None = None,
        created_at: float | None = None,
    ) -> None: ...


class EventLog:
    """
    Logger append-only su SQLite.

    Note implementative
    -------------------
    - Usare SQLite in un server async è OK per carichi bassi e per un progetto didattico.
      Qui rendiamo le scritture thread-safe con un lock.
    - Abilitiamo WAL per migliorare la concorrenza (letture mentre scriviamo).
    """

    def __init__(self, config: EventLogConfig):
        self._config = config
        self._lock = threading.Lock()
        self._conn = self._connect(config.path)
        self._init_schema()

    @property
    def path(self) -> str:
        """Percorso del DB (utile per debug)."""
        return self._config.path

    @property
    def backend_name(self) -> str:
        """Nome breve del backend, esposto solo come diagnostica runtime."""
        return "sqlite"

    @property
    def database_name(self) -> str | None:
        """Nome file SQLite, senza path completo."""
        if self._config.path == ":memory:":
            return ":memory:"
        return os.path.basename(self._config.path) or None

    @property
    def database_host(self) -> str | None:
        """SQLite locale non ha un host di rete."""
        return None

    def health_check(self) -> bool:
        """Verifica rapida che la connessione SQLite sia ancora utilizzabile."""
        try:
            with self._lock:
                self._conn.execute("SELECT 1;")
        except Exception:
            return False
        return True

    def count_games(self) -> int | None:
        """Numero di partite registrate (None se la query fallisce: diagnostica best-effort)."""
        try:
            with self._lock:
                row = self._conn.execute("SELECT count(*) FROM games;").fetchone()
            return int(row[0]) if row else None
        except Exception:
            return None

    def count_games_by_model(self) -> list[dict] | None:
        """
        Partite (totali e completate) raggruppate per modello/agente avversario.

        Il modello e' letto dal payload dell'evento `game_created` (json), quindi il
        conteggio copre anche lo storico. `finished_at` marca le completate. Best-effort:
        None su errore.
        """
        sql = """
            SELECT COALESCE(
                       json_extract(e.payload_json, '$.ai_model_id'),
                       json_extract(e.payload_json, '$.ai_agent'),
                       'sconosciuto'
                   ) AS model,
                   count(*) AS total,
                   count(g.finished_at) AS completed
            FROM games g
            JOIN events e ON e.game_id = g.game_id AND e.event_type = 'game_created'
            GROUP BY model
            ORDER BY total DESC;
        """
        try:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
            return [{"model": r[0], "total": int(r[1]), "completed": int(r[2])} for r in rows]
        except Exception:
            return None

    def close(self) -> None:
        """Chiude la connessione SQLite."""
        with self._lock:
            self._conn.close()

    def _connect(self, path: str) -> sqlite3.Connection:
        """
        Apre una connessione SQLite.

        Se il path è un file su disco, crea la directory padre se manca.
        """
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)

        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        """Crea tabelle e indici se non esistono."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    num_players INTEGER NOT NULL,
                    seed INTEGER,
                    code_version TEXT,
                    rules_version TEXT,
                    client_id TEXT,
                    finished_at REAL,
                    aborted_at REAL,
                    aborted_reason TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    server_version INTEGER,
                    player_index INTEGER,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                );
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_game_id ON events(game_id);")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
            # Compatibilità DB già esistenti (created prima di aggiungere colonne).
            self._ensure_column("games", "code_version", "TEXT")
            self._ensure_column("games", "rules_version", "TEXT")
            self._ensure_column("games", "client_id", "TEXT")
            self._ensure_column("games", "finished_at", "REAL")
            self._ensure_column("games", "aborted_at", "REAL")
            self._ensure_column("games", "aborted_reason", "TEXT")
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, col_type: str) -> None:
        """
        Migrazione minimale: aggiunge una colonna se manca.

        SQLite supporta `ALTER TABLE ... ADD COLUMN` per aggiunte semplici.
        Questo è sufficiente per un progetto didattico e mantiene compatibilità con DB già creati.
        """
        existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table});").fetchall()}
        if column in existing:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")

    def ensure_game(
        self,
        game_id: str,
        *,
        num_players: int,
        seed: int | None = None,
        code_version: str | None = None,
        rules_version: str | None = None,
    ) -> None:
        """
        Inserisce la riga della partita (idempotente).

        La tabella `games` serve principalmente come metadato e come “anchor” per le FK.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO games(game_id, created_at, num_players, seed, code_version, rules_version)
                VALUES(?, ?, ?, ?, ?, ?);
                """,
                (game_id, now, num_players, seed, code_version, rules_version),
            )
            self._conn.commit()

    def set_client_id(self, game_id: str, *, client_id: str) -> None:
        """
        Salva un identificatore pseudonimo del client (best-effort).

        Nota privacy:
        questo campo serve a poter fare split train/val "per giocatore" senza salvare PII.
        È responsabilità del frontend generare un UUID (localStorage) o un identificatore
        equivalente non riconducibile alla persona.
        """
        cleaned = str(client_id).strip()
        if not cleaned:
            return
        with self._lock:
            self._conn.execute(
                """
                UPDATE games
                SET client_id = COALESCE(client_id, ?)
                WHERE game_id = ?;
                """,
                (cleaned, game_id),
            )
            self._conn.commit()

    def try_mark_game_finished(self, game_id: str, *, finished_at: float | None = None) -> bool:
        """
        Marca una partita come conclusa (`game_over=true`) in modo idempotente.

        Ritorna True se lo stato è stato aggiornato (prima volta), False se era già marcata.
        """
        ts = time.time() if finished_at is None else float(finished_at)
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE games
                SET finished_at = ?
                WHERE game_id = ? AND finished_at IS NULL;
                """,
                (ts, game_id),
            )
            self._conn.commit()
            return bool(cur.rowcount == 1)

    def try_mark_game_aborted(
        self,
        game_id: str,
        *,
        aborted_reason: str,
        aborted_at: float | None = None,
    ) -> bool:
        """
        Marca una partita come abortita (timeout/inactivity) in modo idempotente.

        Nota:
        non abortiamo una partita già finita (`finished_at` non null).
        """
        ts = time.time() if aborted_at is None else float(aborted_at)
        reason = str(aborted_reason).strip()[:200]
        if not reason:
            reason = "unknown"

        with self._lock:
            row = self._conn.execute(
                "SELECT finished_at, aborted_at FROM games WHERE game_id = ?;",
                (game_id,),
            ).fetchone()
            if row is None:
                return False
            finished_at, existing_aborted_at = row[0], row[1]
            if finished_at is not None:
                return False
            if existing_aborted_at is not None:
                return False

            cur = self._conn.execute(
                """
                UPDATE games
                SET aborted_at = ?, aborted_reason = ?
                WHERE game_id = ? AND finished_at IS NULL AND aborted_at IS NULL;
                """,
                (ts, reason, game_id),
            )
            self._conn.commit()
            return bool(cur.rowcount == 1)

    def log_event(
        self,
        game_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        server_version: int | None = None,
        player_index: int | None = None,
        created_at: float | None = None,
    ) -> None:
        """
        Appende un evento alla tabella `events`.

        Parametri
        ---------
        - `event_type`: stringa breve e stabile (es. `game_created`, `action_play_card`, `observation_sent`).
        - `payload`: dict JSON-serializzabile (DTO o informazioni minimali).
        - `server_version`: versione monotona (se nota) per ordering/debug.
        - `player_index`: destinatario o autore (se applicabile).
        """
        ts = created_at if created_at is not None else time.time()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events(game_id, created_at, server_version, player_index, event_type, payload_json)
                VALUES(?, ?, ?, ?, ?, ?);
                """,
                (game_id, ts, server_version, player_index, event_type, payload_json),
            )
            self._conn.commit()


def _postgres_database_name_from_dsn(dsn: str | None) -> str | None:
    """Estrae solo il nome database da un DSN Postgres, senza esporre host/utente/segreti."""
    if not dsn:
        return None

    parsed = urlparse(dsn)
    if parsed.scheme in {"postgres", "postgresql"}:
        database_name = unquote(parsed.path.lstrip("/"))
        return database_name or None

    # Supporta anche DSN keyword-style: "dbname=neondb user=...".
    for token in shlex.split(dsn):
        key, separator, value = token.partition("=")
        if separator and key == "dbname":
            return value or None
    return None


def _postgres_database_host_from_dsn(dsn: str | None) -> str | None:
    """Estrae solo l'host Postgres da un DSN, senza esporre utente/password."""
    if not dsn:
        return None

    parsed = urlparse(dsn)
    if parsed.scheme in {"postgres", "postgresql"}:
        return parsed.hostname or None

    # Supporta anche DSN keyword-style: "host=ep-... dbname=neondb user=...".
    for token in shlex.split(dsn):
        key, separator, value = token.partition("=")
        if separator and key == "host":
            return value or None
    return None


class PostgresEventLog:
    """
    Event log append-only su **Postgres** (deploy multi-replica, es. Neon).

    Stessa interfaccia di `EventLog` (vedi `EventLogProtocol`), ma persistente e condiviso tra
    repliche (a differenza dell'SQLite locale, che in cloud è per-replica ed effimero).

    Note implementative
    -------------------
    - `psycopg` (v3) è importato lazy: la dipendenza è installata, ma il modulo si carica solo se
      si usa davvero Postgres (cioè se è impostata `DATABASE_URL`).
    - Connessione in `autocommit` + `threading.Lock`: scritture best-effort, serializzate (come
      l'SQLite locale). Per il traffico hobby di un event log append-only è adeguato.
    - Le operazioni "mark finished/aborted" sono UPDATE atomici con guardia in `WHERE` e usano
      `rowcount` per l'idempotenza (nessun SELECT-then-UPDATE).
    - Il client può essere iniettato (`conn=`) per i test senza un Postgres reale.
    """

    def __init__(self, dsn: str | None = None, *, conn: Any = None) -> None:
        self._lock = threading.Lock()
        self._dsn = dsn
        if conn is not None:
            self._conn = conn
        elif dsn is not None:
            self._conn = self._connect()
        else:
            raise ValueError("PostgresEventLog richiede `dsn` oppure `conn`.")
        self._init_schema()

    @property
    def path(self) -> str:
        """Identità del backend (per il confronto di ricreazione nel lifespan)."""
        return self._dsn or "postgres"

    @property
    def backend_name(self) -> str:
        """Nome breve del backend, esposto solo come diagnostica runtime."""
        return "postgres"

    @property
    def database_name(self) -> str | None:
        """Nome del database Postgres, estratto dal DSN senza rivelare credenziali."""
        return _postgres_database_name_from_dsn(self._dsn)

    @property
    def database_host(self) -> str | None:
        """Host Postgres, utile per confrontare project/branch Neon senza rivelare segreti."""
        return _postgres_database_host_from_dsn(self._dsn)

    def close(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._conn.close()

    def _connect(self) -> Any:
        """Apre una nuova connessione Postgres dal DSN configurato."""
        if not self._dsn:
            raise ValueError("Impossibile riconnettere Postgres senza DSN.")
        import psycopg  # import lazy: solo se si usa Postgres

        return psycopg.connect(self._dsn, autocommit=True)

    def _reconnect_locked(self) -> None:
        """Ricrea la connessione Postgres; chiamare solo con `_lock` gia' acquisito."""
        with contextlib.suppress(Exception):
            self._conn.close()
        self._conn = self._connect()

    def _execute_once_locked(self, sql: str, params: tuple = ()) -> int:
        """Esegue una statement usando la connessione corrente; richiede `_lock` acquisito."""
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return int(cur.rowcount)

    def _execute(self, sql: str, params: tuple = ()) -> int:
        """
        Esegue una statement e ritorna `rowcount` (per l'idempotenza).

        Neon e altre piattaforme serverless possono chiudere connessioni rimaste inattive.
        In quel caso il logger non deve restare "available ma muto": tentiamo una
        riconnessione e ritentiamo una volta la statement.
        """
        with self._lock:
            try:
                return self._execute_once_locked(sql, params)
            except Exception:
                if not self._dsn:
                    raise
                self._reconnect_locked()
                return self._execute_once_locked(sql, params)

    def health_check(self) -> bool:
        """Verifica che la connessione Postgres corrente sia utilizzabile, riconnettendo se serve."""
        with self._lock:
            try:
                self._execute_once_locked("SELECT 1;")
                return True
            except Exception:
                if not self._dsn:
                    return False
                try:
                    self._reconnect_locked()
                    self._execute_once_locked("SELECT 1;")
                except Exception:
                    return False
                return True

    def count_games(self) -> int | None:
        """Numero di partite registrate (None se la query fallisce: diagnostica best-effort)."""
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM games;")
                    row = cur.fetchone()
                return int(row[0]) if row else None
            except Exception:
                return None

    def count_games_by_model(self) -> list[dict] | None:
        """Come per SQLite: raggruppa per modello dal payload di `game_created` (best-effort)."""
        sql = """
            SELECT COALESCE(
                       (e.payload_json)::jsonb->>'ai_model_id',
                       (e.payload_json)::jsonb->>'ai_agent',
                       'sconosciuto'
                   ) AS model,
                   count(*) AS total,
                   count(g.finished_at) AS completed
            FROM games g
            JOIN events e ON e.game_id = g.game_id AND e.event_type = 'game_created'
            GROUP BY model
            ORDER BY total DESC;
        """
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                return [{"model": r[0], "total": int(r[1]), "completed": int(r[2])} for r in rows]
            except Exception:
                return None

    def _init_schema(self) -> None:
        """Crea tabelle e indici se non esistono (schema completo: niente migrazioni)."""
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    created_at DOUBLE PRECISION NOT NULL,
                    num_players INTEGER NOT NULL,
                    seed BIGINT,
                    code_version TEXT,
                    rules_version TEXT,
                    client_id TEXT,
                    finished_at DOUBLE PRECISION,
                    aborted_at DOUBLE PRECISION,
                    aborted_reason TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id BIGSERIAL PRIMARY KEY,
                    game_id TEXT NOT NULL REFERENCES games(game_id),
                    created_at DOUBLE PRECISION NOT NULL,
                    server_version INTEGER,
                    player_index INTEGER,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_game_id ON events(game_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")

    def ensure_game(
        self,
        game_id: str,
        *,
        num_players: int,
        seed: int | None = None,
        code_version: str | None = None,
        rules_version: str | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO games(game_id, created_at, num_players, seed, code_version, rules_version)
            VALUES(%s, %s, %s, %s, %s, %s)
            ON CONFLICT (game_id) DO NOTHING;
            """,
            (game_id, time.time(), num_players, seed, code_version, rules_version),
        )

    def set_client_id(self, game_id: str, *, client_id: str) -> None:
        cleaned = str(client_id).strip()
        if not cleaned:
            return
        self._execute(
            "UPDATE games SET client_id = COALESCE(client_id, %s) WHERE game_id = %s;",
            (cleaned, game_id),
        )

    def try_mark_game_finished(self, game_id: str, *, finished_at: float | None = None) -> bool:
        ts = time.time() if finished_at is None else float(finished_at)
        rc = self._execute(
            "UPDATE games SET finished_at = %s WHERE game_id = %s AND finished_at IS NULL;",
            (ts, game_id),
        )
        return rc == 1

    def try_mark_game_aborted(self, game_id: str, *, aborted_reason: str, aborted_at: float | None = None) -> bool:
        ts = time.time() if aborted_at is None else float(aborted_at)
        reason = (str(aborted_reason).strip() or "unknown")[:200]
        rc = self._execute(
            """
            UPDATE games SET aborted_at = %s, aborted_reason = %s
            WHERE game_id = %s AND finished_at IS NULL AND aborted_at IS NULL;
            """,
            (ts, reason, game_id),
        )
        return rc == 1

    def log_event(
        self,
        game_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        server_version: int | None = None,
        player_index: int | None = None,
        created_at: float | None = None,
    ) -> None:
        ts = created_at if created_at is not None else time.time()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._execute(
            """
            INSERT INTO events(game_id, created_at, server_version, player_index, event_type, payload_json)
            VALUES(%s, %s, %s, %s, %s, %s);
            """,
            (game_id, ts, server_version, player_index, event_type, payload_json),
        )


def resolve_database_url() -> str | None:
    """URL Postgres dalle env candidate (override esplicito prima), o None."""
    for name in ("BRISCOLA_DATABASE_URL", "DATABASE_URL"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def build_event_log(*, sqlite_path: str | None, database_url: str | None) -> EventLogProtocol | None:
    """
    Crea il backend event log: Postgres se `database_url` è presente, altrimenti SQLite se è dato un
    path, altrimenti `None` (feature disabilitata). In cloud multi-replica usare sempre Postgres:
    l'SQLite locale è per-replica ed effimero.
    """
    if database_url:
        return PostgresEventLog(database_url)
    if sqlite_path:
        return EventLog(EventLogConfig(path=sqlite_path))
    return None


def parse_event_db_path(raw: str | None) -> str | None:
    """
    Normalizza un path di configurazione (env/CLI).

    Ritorna `None` se la feature deve essere disabilitata.
    """
    if raw is None:
        return None
    cleaned = raw.strip()
    if cleaned == "":
        return None
    return cleaned
