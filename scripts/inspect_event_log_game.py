#!/usr/bin/env python3
"""
Ispezione read-only di una singola partita nell'event log.

Serve quando un export filtrato per `game_id` torna vuoto: prima di inseguire
bug nell'export, questo script verifica se quella partita esiste davvero nel DB
che stiamo interrogando e quali eventi sono presenti (`game_created`,
`human_action`, `ai_action`, `game_finished`, ...).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from briscola_ai.backend.event_log import resolve_database_url
from briscola_ai.backend.event_log_reader import EventLogEventRow, EventLogGameRow, open_event_log_reader

DEFAULT_SQLITE_DB = Path("./data/briscola_events.sqlite3")


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    """Parsa un payload evento, restituendo None se non e' un oggetto JSON valido."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _game_row_to_dict(row: EventLogGameRow | None) -> dict[str, Any] | None:
    """Converte la riga `games` in JSON minimale e stabile."""
    if row is None:
        return None
    return {
        "game_id": row.game_id,
        "created_at": row.created_at,
        "finished_at": row.finished_at,
        "aborted_at": row.aborted_at,
        "aborted_reason": row.aborted_reason,
        "num_players": row.num_players,
        "seed": row.seed,
        "code_version": row.code_version,
        "rules_version": row.rules_version,
        "client_id": row.client_id,
    }


def _event_summary(row: EventLogEventRow) -> dict[str, Any]:
    """Riassunto compatto di un evento, senza serializzare osservazioni intere."""
    payload = _safe_json_loads(row.payload_json)
    summary: dict[str, Any] = {
        "id": row.id,
        "server_version": row.server_version,
        "player_index": row.player_index,
        "event_type": row.event_type,
        "payload_valid_json": payload is not None,
    }
    if payload is None:
        return summary

    if row.event_type == "game_created":
        for key in (
            "code_version",
            "rules_version",
            "ai_agent",
            "ai_model_id",
            "consent_to_data_collection",
            "num_players",
            "player_names",
        ):
            if key in payload:
                summary[key] = payload[key]
    elif row.event_type == "ai_action":
        trace = payload.get("decision_trace")
        decision_type = trace.get("decision_type") if isinstance(trace, dict) else None
        summary["decision_type"] = decision_type
        summary["action_id"] = payload.get("action_id")
        summary["done"] = payload.get("done")
    elif row.event_type == "human_action":
        summary["action_id"] = payload.get("action_id")
        summary["done"] = payload.get("done")
    elif row.event_type == "game_finished":
        for key in ("final_scores", "winner_indices"):
            if key in payload:
                summary[key] = payload[key]
    elif row.event_type == "game_aborted":
        summary["reason"] = payload.get("reason")

    return summary


def inspect_game(*, game_id: str, db_path: Path | None, database_url: str | None) -> dict[str, Any]:
    """Legge il DB selezionato e restituisce il riepilogo della partita richiesta."""
    reader = open_event_log_reader(sqlite_path=db_path, database_url=database_url)
    try:
        game_row = next((row for row in reader.iter_games() if row.game_id == game_id), None)
        events = [row for row in reader.iter_events() if row.game_id == game_id]
    finally:
        reader.close()

    event_counts = Counter(row.event_type for row in events)
    decision_counts: Counter[str] = Counter()
    malformed_payloads = 0
    for row in events:
        payload = _safe_json_loads(row.payload_json)
        if payload is None:
            malformed_payloads += 1
            continue
        if row.event_type == "ai_action":
            trace = payload.get("decision_trace")
            decision_type = trace.get("decision_type") if isinstance(trace, dict) else None
            decision_counts[str(decision_type or "<missing>")] += 1

    return {
        "game_id": game_id,
        "found_in_games_table": game_row is not None,
        "found_events": bool(events),
        "game": _game_row_to_dict(game_row),
        "event_count": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "ai_decision_counts": dict(sorted(decision_counts.items())),
        "malformed_payloads": malformed_payloads,
        "events": [_event_summary(row) for row in events],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ispeziona una singola partita nell'event log SQLite/Postgres.")
    parser.add_argument("game_id", help="ID partita da cercare.")
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Path DB SQLite. Se omesso e DATABASE_URL/BRISCOLA_DATABASE_URL e' presente, "
            "legge da Postgres; altrimenti usa ./data/briscola_events.sqlite3."
        ),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="DSN Postgres esplicito. Default: BRISCOLA_DATABASE_URL o DATABASE_URL, se presenti.",
    )
    args = parser.parse_args()

    database_url = (
        str(args.database_url).strip() if args.database_url else (None if args.db else resolve_database_url())
    )
    db_path = Path(args.db) if args.db else None
    if db_path is None and not database_url:
        db_path = DEFAULT_SQLITE_DB
    if db_path is not None and not db_path.exists():
        print(f"DB non trovato: {db_path}")
        return 2

    result = inspect_game(game_id=args.game_id, db_path=db_path, database_url=database_url)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
