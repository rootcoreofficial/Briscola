#!/usr/bin/env python3
"""
Export dettagliato delle mosse IA salvate come `ai_action`.

Uso tipico:

  DATABASE_URL=... python scripts/export_ai_actions.py \
    --game-id <game_id> \
    --out data/ai_actions.jsonl

Perche' non basta `export_dataset.py`?
--------------------------------------
`export_dataset.py` produce un dataset orientato alle mosse umane (`human_action`).
Per debuggare PIMC serve invece guardare le mosse del bot: quale carta ha giocato,
se la decisione era fallback/solver/search, e quale observation lecita vedeva l'IA.

Questo script e' read-only e non stampa DSN, `client_id` o payload non filtrati.
L'output JSONL contiene solo eventi `ai_action`, gia' sanitizzati dal backend e
sanificati nuovamente in modo difensivo.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from briscola_ai.backend.event_log import resolve_database_url
from briscola_ai.backend.event_log_privacy import sanitize_dataset_payload
from briscola_ai.backend.event_log_reader import EventLogReader, open_event_log_reader

DEFAULT_SQLITE_DB = Path("./data/briscola_events.sqlite3")
SENSITIVE_EXPORT_KEYS = frozenset({"client_id", "payload_json"})


@dataclass(frozen=True)
class ExportAIActionsConfig:
    """Configurazione read-only dell'export dettagliato `ai_action`."""

    db_path: Path | None
    out_path: Path
    database_url: str | None = None
    game_id: str | None = None
    code_version: str | None = None
    ai_agent: str | None = None
    include_observations: bool = True
    schema_version: int = 1


@dataclass
class _GameMeta:
    """Metadati per partita, senza campi identificativi sensibili come `client_id`."""

    game_id: str
    num_players: int | None = None
    seed: int | None = None
    code_version: str | None = None
    rules_version: str | None = None
    finished_at: float | None = None
    aborted_at: float | None = None
    aborted_reason: str | None = None
    ai_agent: str | None = None
    ai_model_id: str | None = None
    consent_to_data_collection: bool | None = None

    def as_record(self) -> dict[str, Any]:
        """Serializza i metadati safe da includere in ogni record JSONL."""
        return {
            "num_players": self.num_players,
            "seed": self.seed,
            "code_version": self.code_version,
            "rules_version": self.rules_version,
            "finished_at": self.finished_at,
            "aborted_at": self.aborted_at,
            "aborted_reason": self.aborted_reason,
            "ai_agent": self.ai_agent,
            "ai_model_id": self.ai_model_id,
            "consent_to_data_collection": self.consent_to_data_collection,
        }


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _strip_sensitive_export_fields(value: Any) -> Any:
    """Rimuove ricorsivamente campi che non devono finire nell'export dettagliato."""
    if isinstance(value, list):
        return [_strip_sensitive_export_fields(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        str(key): _strip_sensitive_export_fields(item)
        for key, item in value.items()
        if str(key) not in SENSITIVE_EXPORT_KEYS
    }


def _payload_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if isinstance(value, str) and value.strip() else None


def _payload_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    return bool(value) if isinstance(value, bool) else None


def _decision_type(payload: dict[str, Any]) -> str:
    trace = payload.get("decision_trace")
    if isinstance(trace, dict):
        raw = trace.get("decision_type")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return "unknown"


def _phase_from_observation(observation: Any) -> dict[str, Any]:
    """Estrae indicatori rapidi della fase partita dall'observation IA, se presente."""
    if not isinstance(observation, dict):
        return {"deck_size": None, "hand_size": None, "table_size": None}
    hand = observation.get("my_hand")
    table = observation.get("table_cards")
    deck_size = observation.get("cards_remaining_in_deck")
    return {
        "deck_size": int(deck_size) if isinstance(deck_size, int) else None,
        "hand_size": len(hand) if isinstance(hand, list) else None,
        "table_size": len(table) if isinstance(table, list) else None,
    }


def _game_matches(meta: _GameMeta, config: ExportAIActionsConfig) -> bool:
    if config.game_id is not None and meta.game_id != config.game_id:
        return False
    if config.code_version is not None and meta.code_version != config.code_version:
        return False
    # SIM103 soppresso di proposito: l'ultima guardia resta simmetrica alle precedenti.
    if config.ai_agent is not None and meta.ai_agent != config.ai_agent:  # noqa: SIM103
        return False
    return True


def export_ai_actions(config: ExportAIActionsConfig) -> dict[str, Any]:
    """Apre il reader corretto ed esporta gli eventi `ai_action` filtrati."""
    reader = open_event_log_reader(sqlite_path=config.db_path, database_url=config.database_url)
    try:
        return export_ai_actions_from_reader(reader, config=config)
    finally:
        reader.close()


def export_ai_actions_from_reader(reader: EventLogReader, *, config: ExportAIActionsConfig) -> dict[str, Any]:
    """
    Esporta `ai_action` da un reader gia' aperto.

    Ritorna contatori utili per log/test; l'output dettagliato e' scritto in JSONL.
    """
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text("", encoding="utf-8")

    games: dict[str, _GameMeta] = {}
    for game in reader.iter_games():
        games[game.game_id] = _GameMeta(
            game_id=game.game_id,
            num_players=game.num_players,
            seed=game.seed,
            code_version=game.code_version,
            rules_version=game.rules_version,
            finished_at=game.finished_at,
            aborted_at=game.aborted_at,
            aborted_reason=game.aborted_reason,
        )

    counters = Counter[str]()
    decision_types = Counter[str]()
    rows: list[dict[str, Any]] = []

    # Primo passaggio sugli eventi: arricchisce i metadati da `game_created` e conserva gli ai_action.
    for event in reader.iter_events():
        counters["events_read"] += 1
        payload = _safe_json_loads(event.payload_json)
        if payload is None:
            counters["malformed_payload_json"] += 1
            continue

        meta = games.setdefault(event.game_id, _GameMeta(game_id=event.game_id))
        if event.event_type == "game_created":
            meta.ai_agent = _payload_str(payload, "ai_agent") or meta.ai_agent
            meta.ai_model_id = _payload_str(payload, "ai_model_id") or meta.ai_model_id
            meta.code_version = _payload_str(payload, "code_version") or meta.code_version
            meta.rules_version = _payload_str(payload, "rules_version") or meta.rules_version
            meta.consent_to_data_collection = _payload_bool(payload, "consent_to_data_collection")
            counters["game_created_seen"] += 1
            continue

        if event.event_type != "ai_action":
            continue
        counters["ai_actions_seen"] += 1
        rows.append(
            {
                "_event_id": event.id,
                "_game_id": event.game_id,
                "_server_version": event.server_version,
                "_player_index": event.player_index,
                "_payload": payload,
            }
        )

    with config.out_path.open("a", encoding="utf-8") as out:
        for row in rows:
            meta = games.setdefault(str(row["_game_id"]), _GameMeta(game_id=str(row["_game_id"])))
            if not _game_matches(meta, config):
                counters["ai_actions_skipped_filter"] += 1
                continue

            payload = _strip_sensitive_export_fields(sanitize_dataset_payload(row["_payload"]))
            assert isinstance(payload, dict)
            decision_type = _decision_type(payload)
            decision_types[decision_type] += 1

            observation = payload.get("observation")
            next_observation = payload.get("next_observation")
            record: dict[str, Any] = {
                "schema_version": config.schema_version,
                "game_id": row["_game_id"],
                "event_id": row["_event_id"],
                "server_version": row["_server_version"],
                "player_index": row["_player_index"],
                "metadata": meta.as_record(),
                "ai": {
                    "agent": payload.get("ai_agent") or meta.ai_agent,
                    "model_id": payload.get("ai_model_id") or meta.ai_model_id,
                },
                "phase": _phase_from_observation(observation),
                "decision_type": decision_type,
                "decision_trace": payload.get("decision_trace"),
                "action": {
                    "card_index": payload.get("card_index"),
                    "card": (
                        (payload.get("result") or {}).get("played_card")
                        if isinstance(payload.get("result"), dict)
                        else None
                    ),
                    "coerced": payload.get("action_coerced"),
                },
                "reward": payload.get("reward"),
                "done": payload.get("done"),
                "result": payload.get("result"),
                "observation": observation if config.include_observations else None,
                "next_observation": next_observation if config.include_observations else None,
            }
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counters["records_written"] += 1

            if not isinstance(observation, dict):
                counters["records_missing_observation"] += 1
            if not isinstance(next_observation, dict):
                counters["records_missing_next_observation"] += 1

    return {
        "backend": reader.backend_name,
        "out_path": str(config.out_path),
        "filters": {
            "game_id": config.game_id,
            "code_version": config.code_version,
            "ai_agent": config.ai_agent,
            "include_observations": config.include_observations,
        },
        "counters": dict(sorted(counters.items())),
        "decision_types": dict(sorted(decision_types.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export dettagliato eventi ai_action da event log SQLite/Postgres")
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
    parser.add_argument("--out", required=True, help="Path output JSONL dettagliato.")
    parser.add_argument("--game-id", default=None, help="Filtra una singola partita.")
    parser.add_argument("--code-version", default=None, help="Filtra una sola code_version.")
    parser.add_argument("--ai-agent", default=None, help="Filtra un solo ai_agent.")
    parser.add_argument(
        "--no-observations",
        action="store_true",
        help="Non includere observation/next_observation nel JSONL, lasciando solo summary decisionale.",
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

    summary = export_ai_actions(
        ExportAIActionsConfig(
            db_path=db_path,
            database_url=database_url,
            out_path=Path(args.out),
            game_id=args.game_id,
            code_version=args.code_version,
            ai_agent=args.ai_agent,
            include_observations=not bool(args.no_observations),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
