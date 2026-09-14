#!/usr/bin/env python3
"""
Audit aggregato delle partite nell'event log.

Perche' esiste
--------------
`scripts/export_dataset.py` produce un JSONL orientato al training umano: in modalita'
`BRISCOLA_EVENT_LOG_MODE=dataset` contiene soprattutto eventi `human_action`.
Questo e' corretto per privacy/minimalita', ma non basta a rispondere a domande
operative come:

- quante partite sono state giocate contro PIMC?
- quali versioni/agent/model_id compaiono davvero nel DB di produzione?
- il log contiene anche mosse IA, quindi e' auditabile, oppure e' un log dataset minimale?

Questo script legge direttamente le tabelle `games`/`events` (SQLite locale o
Postgres/Neon) e produce solo aggregati. Non stampa DSN, `client_id` o payload di
mosse.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from briscola_ai.backend.event_log import resolve_database_url
from briscola_ai.backend.event_log_reader import EventLogReader, open_event_log_reader

DEFAULT_SQLITE_DB = Path("./data/briscola_events.sqlite3")
UNKNOWN = "<unknown>"
NONE = "<none>"


@dataclass(frozen=True)
class AuditConfig:
    """Configurazione read-only dell'audit partite."""

    db_path: Path | None
    database_url: str | None = None
    code_version: str | None = None
    ai_agent: str | None = None
    include_games: bool = False
    game_limit: int = 40


@dataclass
class _GameAudit:
    """Accumulator interno per una singola partita."""

    game_id: str
    code_version: str | None = None
    rules_version: str | None = None
    finished: bool = False
    aborted: bool = False
    ai_agent: str | None = None
    ai_model_id: str | None = None
    consent: bool | None = None
    game_created_seen: bool = False
    malformed_payload_json: int = 0
    events: Counter[str] = field(default_factory=Counter)
    human_actions: int = 0
    human_actions_missing_observation: int = 0
    human_actions_missing_next_observation: int = 0
    human_actions_done: int = 0
    action_play_card_human: int = 0
    action_play_card_ai: int = 0
    ai_actions: int = 0
    ai_actions_missing_observation: int = 0
    ai_actions_missing_next_observation: int = 0
    ai_actions_done: int = 0
    ai_actions_search: int = 0
    ai_actions_lookahead: int = 0
    ai_actions_solver: int = 0
    ai_actions_fallback: int = 0
    ai_actions_unknown_decision: int = 0
    ai_card_reveals: int = 0
    trick_results: int = 0
    game_finished_events: int = 0
    game_aborted_events: int = 0

    @property
    def has_ai_metadata(self) -> bool:
        return bool(self.ai_agent)

    @property
    def is_pimc(self) -> bool:
        return self.ai_agent == "bc_model_pimc_16x8"

    @property
    def has_ai_events(self) -> bool:
        return self.ai_actions > 0 or self.action_play_card_ai > 0 or self.ai_card_reveals > 0

    @property
    def is_open(self) -> bool:
        return not self.finished and not self.aborted

    @property
    def mode_guess(self) -> str:
        """
        Classificazione pragmatica del tipo di log.

        - `dataset_minimal`: eventi umani self-contained, ma niente eventi IA.
        - `debug_or_full`: eventi `action_play_card`/`ai_card_reveal` presenti.
        - `mixed`: entrambi i formati nella stessa partita.
        - `metadata_only`: solo lifecycle/metadati, niente mosse.
        """
        has_dataset = self.human_actions > 0
        has_full = self.action_play_card_human > 0 or self.action_play_card_ai > 0 or self.ai_card_reveals > 0
        if has_dataset and has_full:
            return "mixed"
        if has_dataset:
            return "dataset_minimal"
        if has_full:
            return "debug_or_full"
        return "metadata_only"

    @property
    def audit_status(self) -> str:
        """
        Stato di auditabilita' delle mosse IA.

        La presenza di `ai_agent` ci dice contro chi si e' giocato; la presenza di eventi IA
        determina se possiamo auditare le scelte dell'avversario dal log corrente.
        """
        if self.has_ai_events:
            return "ai_moves_auditable"
        if self.has_ai_metadata:
            return "ai_metadata_only_no_ai_moves"
        if self.human_actions > 0:
            return "human_dataset_no_ai_metadata"
        return "no_gameplay_events"


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    """Ordina un `Counter` in un dict stabile e JSON-friendly."""
    return {key: int(counter[key]) for key in sorted(counter)}


def _label(value: str | None) -> str:
    """Normalizza label aggregate, evitando `None` in output tabellari."""
    if value is None:
        return NONE
    cleaned = str(value).strip()
    return cleaned if cleaned else NONE


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    """Parsa un payload evento; ritorna None su JSON malformato o non-oggetto."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _game_matches_filters(game: _GameAudit, config: AuditConfig) -> bool:
    if config.code_version is not None and _label(game.code_version) != config.code_version:
        return False
    # SIM103 soppresso di proposito: l'ultima guardia resta simmetrica alle precedenti.
    if config.ai_agent is not None and _label(game.ai_agent) != config.ai_agent:  # noqa: SIM103
        return False
    return True


def build_event_log_games_audit(config: AuditConfig) -> dict[str, Any]:
    """Apre il reader corretto e costruisce l'audit aggregato."""
    reader = open_event_log_reader(sqlite_path=config.db_path, database_url=config.database_url)
    try:
        return build_event_log_games_audit_from_reader(reader, config=config)
    finally:
        reader.close()


def build_event_log_games_audit_from_reader(reader: EventLogReader, *, config: AuditConfig) -> dict[str, Any]:
    """
    Costruisce l'audit partite da un reader gia' aperto.

    Questa funzione contiene la logica testabile e non apre connessioni esterne.
    """
    games: dict[str, _GameAudit] = {}
    event_types: Counter[str] = Counter()
    malformed_total = 0

    for game_row in reader.iter_games():
        games[game_row.game_id] = _GameAudit(
            game_id=game_row.game_id,
            code_version=game_row.code_version,
            rules_version=game_row.rules_version,
            finished=game_row.finished_at is not None,
            aborted=game_row.aborted_at is not None,
        )

    for event_row in reader.iter_events():
        game = games.setdefault(event_row.game_id, _GameAudit(game_id=event_row.game_id))
        event_type = str(event_row.event_type)
        game.events[event_type] += 1
        event_types[event_type] += 1

        payload = _safe_json_loads(event_row.payload_json)
        if payload is None:
            game.malformed_payload_json += 1
            malformed_total += 1
            continue

        if event_type == "game_created":
            game.game_created_seen = True
            if isinstance(payload.get("code_version"), str):
                game.code_version = payload["code_version"]
            if isinstance(payload.get("rules_version"), str):
                game.rules_version = payload["rules_version"]
            if isinstance(payload.get("ai_agent"), str):
                game.ai_agent = payload["ai_agent"]
            if isinstance(payload.get("ai_model_id"), str):
                game.ai_model_id = payload["ai_model_id"]
            if isinstance(payload.get("consent_to_data_collection"), bool):
                game.consent = bool(payload["consent_to_data_collection"])
            continue

        if event_type == "human_action":
            game.human_actions += 1
            if not isinstance(payload.get("observation"), dict):
                game.human_actions_missing_observation += 1
            if not isinstance(payload.get("next_observation"), dict):
                game.human_actions_missing_next_observation += 1
            if payload.get("done") is True:
                game.human_actions_done += 1
            continue

        if event_type == "action_play_card":
            if payload.get("is_ai") is True:
                game.action_play_card_ai += 1
            else:
                game.action_play_card_human += 1
            continue

        if event_type == "ai_action":
            game.ai_actions += 1
            if not isinstance(payload.get("observation"), dict):
                game.ai_actions_missing_observation += 1
            if not isinstance(payload.get("next_observation"), dict):
                game.ai_actions_missing_next_observation += 1
            if payload.get("done") is True:
                game.ai_actions_done += 1

            trace = payload.get("decision_trace")
            decision_type = trace.get("decision_type") if isinstance(trace, dict) else None
            if decision_type == "search":
                game.ai_actions_search += 1
            elif decision_type == "lookahead":
                game.ai_actions_lookahead += 1
            elif decision_type == "solver":
                game.ai_actions_solver += 1
            elif decision_type == "fallback":
                game.ai_actions_fallback += 1
            else:
                game.ai_actions_unknown_decision += 1
            continue

        if event_type == "ai_card_reveal":
            game.ai_card_reveals += 1
            continue

        if event_type == "trick_result":
            game.trick_results += 1
            continue

        if event_type == "game_finished":
            game.finished = True
            game.game_finished_events += 1
            continue

        if event_type == "game_aborted":
            game.aborted = True
            game.game_aborted_events += 1
            continue

    selected_games = [game for game in games.values() if _game_matches_filters(game, config)]
    selected_games.sort(key=lambda game: (game.code_version or "", game.game_id))

    by_version: Counter[str] = Counter()
    by_agent: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_agent_model: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    by_audit_status: Counter[str] = Counter()
    consent_counter: Counter[str] = Counter()

    summary = {
        "total": 0,
        "finished": 0,
        "aborted": 0,
        "open": 0,
        "with_game_created": 0,
        "with_ai_agent": 0,
        "pimc_games": 0,
        "pimc_finished": 0,
        "pimc_with_ai_events": 0,
        "pimc_without_ai_events": 0,
    }
    gameplay = {
        "human_actions": 0,
        "human_actions_missing_observation": 0,
        "human_actions_missing_next_observation": 0,
        "human_actions_done": 0,
        "action_play_card_human": 0,
        "action_play_card_ai": 0,
        "ai_actions": 0,
        "ai_actions_missing_observation": 0,
        "ai_actions_missing_next_observation": 0,
        "ai_actions_done": 0,
        "ai_actions_search": 0,
        "ai_actions_lookahead": 0,
        "ai_actions_solver": 0,
        "ai_actions_fallback": 0,
        "ai_actions_unknown_decision": 0,
        "ai_card_reveals": 0,
        "trick_results": 0,
    }

    details: list[dict[str, Any]] = []
    for game in selected_games:
        summary["total"] += 1
        summary["finished"] += int(game.finished)
        summary["aborted"] += int(game.aborted)
        summary["open"] += int(game.is_open)
        summary["with_game_created"] += int(game.game_created_seen)
        summary["with_ai_agent"] += int(game.has_ai_metadata)
        summary["pimc_games"] += int(game.is_pimc)
        summary["pimc_finished"] += int(game.is_pimc and game.finished)
        summary["pimc_with_ai_events"] += int(game.is_pimc and game.has_ai_events)
        summary["pimc_without_ai_events"] += int(game.is_pimc and not game.has_ai_events)

        gameplay["human_actions"] += game.human_actions
        gameplay["human_actions_missing_observation"] += game.human_actions_missing_observation
        gameplay["human_actions_missing_next_observation"] += game.human_actions_missing_next_observation
        gameplay["human_actions_done"] += game.human_actions_done
        gameplay["action_play_card_human"] += game.action_play_card_human
        gameplay["action_play_card_ai"] += game.action_play_card_ai
        gameplay["ai_actions"] += game.ai_actions
        gameplay["ai_actions_missing_observation"] += game.ai_actions_missing_observation
        gameplay["ai_actions_missing_next_observation"] += game.ai_actions_missing_next_observation
        gameplay["ai_actions_done"] += game.ai_actions_done
        gameplay["ai_actions_search"] += game.ai_actions_search
        gameplay["ai_actions_lookahead"] += game.ai_actions_lookahead
        gameplay["ai_actions_solver"] += game.ai_actions_solver
        gameplay["ai_actions_fallback"] += game.ai_actions_fallback
        gameplay["ai_actions_unknown_decision"] += game.ai_actions_unknown_decision
        gameplay["ai_card_reveals"] += game.ai_card_reveals
        gameplay["trick_results"] += game.trick_results

        by_version[_label(game.code_version if game.code_version is not None else UNKNOWN)] += 1
        by_agent[_label(game.ai_agent)] += 1
        by_model[_label(game.ai_model_id)] += 1
        by_agent_model[f"{_label(game.ai_agent)} | {_label(game.ai_model_id)}"] += 1
        by_mode[game.mode_guess] += 1
        by_audit_status[game.audit_status] += 1
        if game.consent is True:
            consent_counter["true"] += 1
        elif game.consent is False:
            consent_counter["false"] += 1
        else:
            consent_counter["unknown"] += 1

        if config.include_games and len(details) < max(0, config.game_limit):
            details.append(
                {
                    "game_id": game.game_id,
                    "game_id_short": game.game_id[:8],
                    "code_version": game.code_version,
                    "rules_version": game.rules_version,
                    "finished": game.finished,
                    "aborted": game.aborted,
                    "open": game.is_open,
                    "ai_agent": game.ai_agent,
                    "ai_model_id": game.ai_model_id,
                    "consent_to_data_collection": game.consent,
                    "mode_guess": game.mode_guess,
                    "audit_status": game.audit_status,
                    "human_actions": game.human_actions,
                    "ai_actions": game.ai_actions,
                    "ai_actions_search": game.ai_actions_search,
                    "ai_actions_lookahead": game.ai_actions_lookahead,
                    "ai_actions_solver": game.ai_actions_solver,
                    "ai_actions_fallback": game.ai_actions_fallback,
                    "action_play_card_ai": game.action_play_card_ai,
                    "ai_card_reveals": game.ai_card_reveals,
                    "trick_results": game.trick_results,
                    "events": _counter_to_dict(game.events),
                }
            )

    return {
        "backend": reader.backend_name,
        "filters": {
            "code_version": config.code_version,
            "ai_agent": config.ai_agent,
        },
        "games": summary,
        "gameplay_events": gameplay,
        "consent": _counter_to_dict(consent_counter),
        "by_code_version": _counter_to_dict(by_version),
        "by_ai_agent": _counter_to_dict(by_agent),
        "by_ai_model_id": _counter_to_dict(by_model),
        "by_ai_agent_model": _counter_to_dict(by_agent_model),
        "by_mode_guess": _counter_to_dict(by_mode),
        "by_audit_status": _counter_to_dict(by_audit_status),
        "events": {
            "total": int(sum(event_types.values())),
            "by_type": _counter_to_dict(event_types),
            "malformed_payload_json": int(malformed_total),
        },
        "games_detail": details,
    }


def _print_counter(title: str, data: dict[str, int]) -> None:
    print(f"\n{title}")
    if not data:
        print("- nessun dato")
        return
    for key, count in sorted(data.items(), key=lambda item: (-item[1], item[0])):
        print(f"- {key}: {count}")


def _print_text_report(report: dict[str, Any]) -> None:
    games = report["games"]
    gameplay = report["gameplay_events"]

    print("Audit event log Briscola AI")
    print(f"- backend: {report['backend']}")
    filters = {k: v for k, v in report["filters"].items() if v is not None}
    if filters:
        print(f"- filtri: {filters}")

    print("\nPartite")
    print(f"- totali: {games['total']}")
    print(f"- complete: {games['finished']}")
    print(f"- abortite: {games['aborted']}")
    print(f"- aperte/incomplete: {games['open']}")
    print(f"- con game_created: {games['with_game_created']}")
    print(f"- con ai_agent: {games['with_ai_agent']}")
    print(f"- PIMC 16x8: {games['pimc_games']}")
    print(f"- PIMC complete: {games['pimc_finished']}")

    print("\nAudit mosse IA")
    print(f"- partite PIMC con eventi IA auditabili: {games['pimc_with_ai_events']}")
    print(f"- partite PIMC senza eventi IA nel log: {games['pimc_without_ai_events']}")
    print(f"- ai_action dataset: {gameplay['ai_actions']}")
    print(
        "- ai_action search/lookahead/solver/fallback: "
        f"{gameplay['ai_actions_search']} / "
        f"{gameplay['ai_actions_lookahead']} / "
        f"{gameplay['ai_actions_solver']} / "
        f"{gameplay['ai_actions_fallback']}"
    )
    print(f"- action_play_card IA: {gameplay['action_play_card_ai']}")
    print(f"- ai_card_reveal: {gameplay['ai_card_reveals']}")
    print(
        "- nota: se una partita PIMC risulta senza eventi IA, il log identifica l'avversario "
        "ma non permette di giudicare le singole mosse dell'IA."
    )

    print("\nMosse umane / qualita' dataset")
    print(f"- human_action: {gameplay['human_actions']}")
    print(f"- human_action senza observation: {gameplay['human_actions_missing_observation']}")
    print(f"- human_action senza next_observation: {gameplay['human_actions_missing_next_observation']}")
    print(f"- human_action terminali: {gameplay['human_actions_done']}")

    _print_counter("Versioni", report["by_code_version"])
    _print_counter("Agenti IA", report["by_ai_agent"])
    _print_counter("Modelli IA", report["by_ai_model_id"])
    _print_counter("Modo log stimato", report["by_mode_guess"])
    _print_counter("Audit status", report["by_audit_status"])

    if report["games_detail"]:
        print("\nDettaglio partite")
        for game in report["games_detail"]:
            print(
                "- "
                f"{game['game_id_short']} "
                f"version={game['code_version'] or NONE} "
                f"agent={game['ai_agent'] or NONE} "
                f"model={game['ai_model_id'] or NONE} "
                f"finished={game['finished']} "
                f"mode={game['mode_guess']} "
                f"audit={game['audit_status']} "
                f"human={game['human_actions']} "
                f"ai_dataset={game['ai_actions']} "
                f"ai_actions={game['action_play_card_ai']} "
                f"ai_reveals={game['ai_card_reveals']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit aggregato delle partite in event log SQLite/Postgres")
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
    parser.add_argument("--code-version", default=None, help="Filtra una sola code_version, es. 0.14.1.")
    parser.add_argument("--ai-agent", default=None, help="Filtra un solo ai_agent, es. bc_model_pimc_16x8.")
    parser.add_argument("--show-games", action="store_true", help="Mostra anche un dettaglio per-partita.")
    parser.add_argument("--limit", type=int, default=40, help="Numero massimo di partite in dettaglio. Default: 40.")
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

    report = build_event_log_games_audit(
        AuditConfig(
            db_path=db_path,
            database_url=database_url,
            code_version=args.code_version,
            ai_agent=args.ai_agent,
            include_games=bool(args.show_games),
            game_limit=max(0, int(args.limit)),
        )
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
