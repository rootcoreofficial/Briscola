#!/usr/bin/env python3
"""
Report leggero sull'event log di Briscola AI.

Scopo
-----
Prima di usare dati umani per ML, serve capire se la produzione sta raccogliendo
abbastanza partite complete e consenzienti, e se il log contiene anomalie
grossolane. Questo script legge lo stesso schema `games`/`events` da:

- SQLite locale (`--db`, default in sviluppo);
- Postgres/Neon (`DATABASE_URL`/`BRISCOLA_DATABASE_URL`, default in cloud).

Il report è volutamente aggregato: non stampa DSN, `client_id`, payload di mosse
o altri dati potenzialmente sensibili.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from briscola_ai.backend.event_log import resolve_database_url
from briscola_ai.backend.event_log_reader import EventLogReader, open_event_log_reader

DEFAULT_SQLITE_DB = Path("./data/briscola_events.sqlite3")
DAY_SECONDS = 24 * 60 * 60
WEEK_SECONDS = 7 * DAY_SECONDS


@dataclass(frozen=True)
class ReportConfig:
    """Configurazione del report read-only."""

    db_path: Path | None
    database_url: str | None = None


def _is_recent(ts: float | None, *, now: float, window_seconds: int) -> bool:
    return ts is not None and ts >= now - window_seconds


def _empty_report(*, backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "games": {
            "total": 0,
            "finished": 0,
            "aborted": 0,
            "open": 0,
            "created_last_24h": 0,
            "created_last_7d": 0,
            "finished_last_24h": 0,
            "finished_last_7d": 0,
            "aborted_last_24h": 0,
            "aborted_last_7d": 0,
            "with_client_id": 0,
        },
        "consent": {
            "game_created_events": 0,
            "games_with_consent": 0,
            "games_without_consent": 0,
            "games_unknown": 0,
        },
        "events": {
            "total": 0,
            "by_type": {},
            "malformed_payload_json": 0,
        },
        "dataset_quality": {
            "human_actions": 0,
            "human_actions_missing_observation": 0,
            "human_actions_missing_next_observation": 0,
            "human_actions_done": 0,
        },
        "aborted_reasons": {},
    }


def build_event_log_report(config: ReportConfig, *, now: float | None = None) -> dict[str, Any]:
    """
    Costruisce il report aggregato.

    `now` è iniettabile per test deterministici. In produzione usa `time.time()`.
    """

    reader = open_event_log_reader(sqlite_path=config.db_path, database_url=config.database_url)
    try:
        return build_event_log_report_from_reader(reader, now=time.time() if now is None else now)
    finally:
        reader.close()


def build_event_log_report_from_reader(reader: EventLogReader, *, now: float) -> dict[str, Any]:
    """Versione testabile che riceve un reader già aperto."""

    report = _empty_report(backend=reader.backend_name)
    aborted_reasons: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    game_ids: set[str] = set()
    game_created_seen: set[str] = set()
    consent_true: set[str] = set()
    consent_false: set[str] = set()

    for game in reader.iter_games():
        game_ids.add(game.game_id)
        report["games"]["total"] += 1
        if game.finished_at is not None:
            report["games"]["finished"] += 1
        if game.aborted_at is not None:
            report["games"]["aborted"] += 1
            aborted_reasons[game.aborted_reason or "unknown"] += 1
        if game.finished_at is None and game.aborted_at is None:
            report["games"]["open"] += 1
        if _is_recent(game.created_at, now=now, window_seconds=DAY_SECONDS):
            report["games"]["created_last_24h"] += 1
        if _is_recent(game.created_at, now=now, window_seconds=WEEK_SECONDS):
            report["games"]["created_last_7d"] += 1
        if _is_recent(game.finished_at, now=now, window_seconds=DAY_SECONDS):
            report["games"]["finished_last_24h"] += 1
        if _is_recent(game.finished_at, now=now, window_seconds=WEEK_SECONDS):
            report["games"]["finished_last_7d"] += 1
        if _is_recent(game.aborted_at, now=now, window_seconds=DAY_SECONDS):
            report["games"]["aborted_last_24h"] += 1
        if _is_recent(game.aborted_at, now=now, window_seconds=WEEK_SECONDS):
            report["games"]["aborted_last_7d"] += 1
        if game.client_id:
            report["games"]["with_client_id"] += 1

    for event in reader.iter_events():
        report["events"]["total"] += 1
        event_types[event.event_type] += 1

        try:
            payload = json.loads(event.payload_json)
        except json.JSONDecodeError:
            report["events"]["malformed_payload_json"] += 1
            continue

        if event.event_type == "game_created":
            game_created_seen.add(event.game_id)
            if payload.get("consent_to_data_collection") is True:
                consent_true.add(event.game_id)
            elif payload.get("consent_to_data_collection") is False:
                consent_false.add(event.game_id)

        if event.event_type == "human_action":
            report["dataset_quality"]["human_actions"] += 1
            if not isinstance(payload.get("observation"), dict):
                report["dataset_quality"]["human_actions_missing_observation"] += 1
            if not isinstance(payload.get("next_observation"), dict):
                report["dataset_quality"]["human_actions_missing_next_observation"] += 1
            if payload.get("done") is True:
                report["dataset_quality"]["human_actions_done"] += 1

    report["consent"]["game_created_events"] = len(game_created_seen)
    report["consent"]["games_with_consent"] = len(consent_true)
    report["consent"]["games_without_consent"] = len(consent_false - consent_true)
    report["consent"]["games_unknown"] = len(game_ids - game_created_seen)
    report["events"]["by_type"] = dict(sorted(event_types.items()))
    report["aborted_reasons"] = dict(sorted(aborted_reasons.items()))
    return report


def _print_text_report(report: dict[str, Any]) -> None:
    games = report["games"]
    consent = report["consent"]
    events = report["events"]
    quality = report["dataset_quality"]

    print("Report event log Briscola AI")
    print(f"- backend: {report['backend']}")
    print("\nPartite")
    print(f"- totali: {games['total']}")
    print(f"- complete: {games['finished']}")
    print(f"- abortite: {games['aborted']}")
    print(f"- aperte/incomplete: {games['open']}")
    print(f"- create ultime 24h / 7d: {games['created_last_24h']} / {games['created_last_7d']}")
    print(f"- complete ultime 24h / 7d: {games['finished_last_24h']} / {games['finished_last_7d']}")
    print(f"- abortite ultime 24h / 7d: {games['aborted_last_24h']} / {games['aborted_last_7d']}")
    print(f"- con client_id pseudonimo: {games['with_client_id']}")

    print("\nConsenso")
    print(f"- game_created osservati: {consent['game_created_events']}")
    print(f"- partite con consenso: {consent['games_with_consent']}")
    print(f"- partite senza consenso: {consent['games_without_consent']}")
    print(f"- consenso non ricostruibile: {consent['games_unknown']}")

    print("\nEventi")
    print(f"- totali: {events['total']}")
    print(f"- payload JSON malformati: {events['malformed_payload_json']}")
    for event_type, count in events["by_type"].items():
        print(f"- {event_type}: {count}")

    print("\nQualità dataset")
    print(f"- human_action: {quality['human_actions']}")
    print(f"- human_action senza observation: {quality['human_actions_missing_observation']}")
    print(f"- human_action senza next_observation: {quality['human_actions_missing_next_observation']}")
    print(f"- human_action terminali: {quality['human_actions_done']}")

    if report["aborted_reasons"]:
        print("\nAbbandoni")
        for reason, count in report["aborted_reasons"].items():
            print(f"- {reason}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report read-only su event log SQLite/Postgres")
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Path DB SQLite. Se omesso e DATABASE_URL/BRISCOLA_DATABASE_URL è presente, "
            "il report legge da Postgres; altrimenti usa ./data/briscola_events.sqlite3."
        ),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="DSN Postgres esplicito. Default: BRISCOLA_DATABASE_URL o DATABASE_URL, se presenti.",
    )
    parser.add_argument("--json", action="store_true", help="Stampa il report come JSON.")
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

    report = build_event_log_report(ReportConfig(db_path=db_path, database_url=database_url))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
