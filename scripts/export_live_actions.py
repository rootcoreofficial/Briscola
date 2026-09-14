#!/usr/bin/env python3
"""
Export unico delle azioni giocate nel live event log.

Per il prossimo audit di campo non basta guardare solo il dataset umano
(`export_dataset.py`) o solo le mosse IA (`export_ai_actions.py`): serve una sequenza
ordinata di azioni umane e IA, con le observation lecite, i metadati partita e il risultato
finale. Questo script produce proprio quel JSONL, leggendo SQLite locale o Postgres live.

Uso tipico:

  DATABASE_URL=... python scripts/export_live_actions.py \
    --ai-agent bc_model_pimc_belief_16x8 \
    --ai-model-id best_a2c_v14.npz \
    --exclude-client-id loadtest-bot \
    --out data/live_actions_v14.jsonl

L'export non stampa DSN e non include `client_id` nel JSONL salvo `--include-client-id`.
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
SENSITIVE_KEYS = frozenset({"payload_json"})


@dataclass(frozen=True)
class ExportLiveActionsConfig:
    """Configurazione read-only dell'export azioni live."""

    db_path: Path | None
    out_path: Path
    database_url: str | None = None
    code_version: str | None = None
    ai_agent: str | None = None
    ai_model_id: str | None = None
    only_completed_games: bool = True
    include_observations: bool = True
    include_client_id: bool = False
    exclude_client_ids: tuple[str, ...] = ()
    schema_version: int = 1


@dataclass
class _GameMeta:
    """Metadati partita necessari per filtri e record export."""

    game_id: str
    num_players: int | None = None
    seed: int | None = None
    code_version: str | None = None
    rules_version: str | None = None
    client_id: str | None = None
    finished_at: float | None = None
    aborted_at: float | None = None
    aborted_reason: str | None = None
    ai_agent: str | None = None
    ai_model_id: str | None = None
    consent_to_data_collection: bool | None = None
    final_points_by_player_index: list[int] | None = None
    winning_player_index: int | None = None

    def as_record(self, *, include_client_id: bool) -> dict[str, Any]:
        record: dict[str, Any] = {
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
            "final_points_by_player_index": self.final_points_by_player_index,
            "winning_player_index": self.winning_player_index,
        }
        if include_client_id:
            record["client_id"] = self.client_id
        return record


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _payload_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if isinstance(value, str) and value.strip() else None


def _payload_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    return bool(value) if isinstance(value, bool) else None


def _strip_sensitive_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_sensitive_fields(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        str(key): _strip_sensitive_fields(item)
        for key, item in value.items()
        if str(key) not in SENSITIVE_KEYS and str(key) != "client_id"
    }


def _card_from_observation(observation: Any, card_index: Any) -> dict[str, Any] | None:
    """Ricava la carta giocata da `observation.my_hand[card_index]`, utile per log vecchi."""
    if not isinstance(observation, dict) or not isinstance(card_index, int):
        return None
    hand = observation.get("my_hand")
    if not isinstance(hand, list) or card_index < 0 or card_index >= len(hand):
        return None
    card = hand[card_index]
    return card if isinstance(card, dict) else None


def _played_card(payload: dict[str, Any], observation: Any) -> dict[str, Any] | None:
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("played_card"), dict):
        return result["played_card"]
    return _card_from_observation(observation, payload.get("card_index"))


def _decision_type(payload: dict[str, Any]) -> str | None:
    trace = payload.get("decision_trace")
    if not isinstance(trace, dict):
        return None
    raw = trace.get("decision_type")
    return str(raw).strip() if isinstance(raw, str) and raw.strip() else None


def _phase_from_observation(observation: Any, *, meta: _GameMeta) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {
            "deck_size": None,
            "hand_size": None,
            "table_size": None,
            "trick_index": None,
            "is_lead": None,
            "in_draw_phase": None,
        }
    hand = observation.get("my_hand")
    table = observation.get("table_cards")
    history = observation.get("trick_history")
    deck_size = observation.get("cards_remaining_in_deck")
    table_size = len(table) if isinstance(table, list) else None
    return {
        "deck_size": int(deck_size) if isinstance(deck_size, int) else None,
        "hand_size": len(hand) if isinstance(hand, list) else None,
        "table_size": table_size,
        "trick_index": (len(history) + 1) if isinstance(history, list) else None,
        "is_lead": table_size == 0 if table_size is not None else None,
        "in_draw_phase": bool(isinstance(deck_size, int) and deck_size > 0),
        "num_players": meta.num_players,
    }


def _trick_completed(payload: dict[str, Any], observation: Any, *, meta: _GameMeta) -> bool | None:
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("trick_completed"), bool):
        return bool(result["trick_completed"])
    if not isinstance(observation, dict):
        return None
    table = observation.get("table_cards")
    num_players = (
        observation.get("num_players") if isinstance(observation.get("num_players"), int) else meta.num_players
    )
    if isinstance(table, list) and isinstance(num_players, int):
        return len(table) + 1 >= num_players
    return None


def _trick_cards_if_completed(
    payload: dict[str, Any], observation: Any, played_card: Any
) -> list[dict[str, Any]] | None:
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("trick_cards"), list):
        return result["trick_cards"]
    if not isinstance(observation, dict) or not isinstance(played_card, dict):
        return None
    table = observation.get("table_cards")
    player_index = payload.get("player_index")
    if not isinstance(table, list) or not isinstance(player_index, int):
        return None
    return [*table, {"card": played_card, "player_index": player_index}]


def _game_matches(meta: _GameMeta, config: ExportLiveActionsConfig, completed_ids: set[str]) -> bool:
    if config.only_completed_games and meta.game_id not in completed_ids and meta.finished_at is None:
        return False
    if config.code_version is not None and meta.code_version != config.code_version:
        return False
    if config.ai_agent is not None and meta.ai_agent != config.ai_agent:
        return False
    if config.ai_model_id is not None and meta.ai_model_id != config.ai_model_id:
        return False
    return not (meta.client_id is not None and meta.client_id in set(config.exclude_client_ids))


def export_live_actions(config: ExportLiveActionsConfig) -> dict[str, Any]:
    """Apre il reader corretto e scrive un JSONL action-by-action."""
    reader = open_event_log_reader(sqlite_path=config.db_path, database_url=config.database_url)
    try:
        return export_live_actions_from_reader(reader, config=config)
    finally:
        reader.close()


def export_live_actions_from_reader(reader: EventLogReader, *, config: ExportLiveActionsConfig) -> dict[str, Any]:
    """Esporta azioni umane e IA da un reader già aperto."""
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text("", encoding="utf-8")

    completed_ids = set(reader.list_completed_game_ids())
    games: dict[str, _GameMeta] = {}
    for row in reader.iter_games():
        games[row.game_id] = _GameMeta(
            game_id=row.game_id,
            num_players=row.num_players,
            seed=row.seed,
            code_version=row.code_version,
            rules_version=row.rules_version,
            client_id=row.client_id,
            finished_at=row.finished_at,
            aborted_at=row.aborted_at,
            aborted_reason=row.aborted_reason,
        )
        if row.finished_at is not None:
            completed_ids.add(row.game_id)

    counters = Counter[str]()
    rows: list[dict[str, Any]] = []

    for event in reader.iter_events():
        counters["events_read"] += 1
        payload = _safe_json_loads(event.payload_json)
        if payload is None:
            counters["malformed_payload_json"] += 1
            continue

        meta = games.setdefault(event.game_id, _GameMeta(game_id=event.game_id))

        if event.event_type == "game_created":
            meta.code_version = _payload_str(payload, "code_version") or meta.code_version
            meta.rules_version = _payload_str(payload, "rules_version") or meta.rules_version
            meta.num_players = (
                payload.get("num_players") if isinstance(payload.get("num_players"), int) else meta.num_players
            )
            meta.seed = payload.get("seed") if isinstance(payload.get("seed"), int) else meta.seed
            meta.ai_agent = _payload_str(payload, "ai_agent") or meta.ai_agent
            meta.ai_model_id = _payload_str(payload, "ai_model_id") or meta.ai_model_id
            meta.consent_to_data_collection = _payload_bool(payload, "consent_to_data_collection")
            counters["game_created_seen"] += 1
            continue

        if event.event_type == "game_finished":
            if isinstance(payload.get("final_points_by_player_index"), list):
                meta.final_points_by_player_index = [
                    int(value) for value in payload["final_points_by_player_index"] if isinstance(value, int)
                ]
            if isinstance(payload.get("winning_player_index"), int):
                meta.winning_player_index = int(payload["winning_player_index"])
            completed_ids.add(event.game_id)
            counters["game_finished_seen"] += 1
            continue

        if event.event_type not in {"human_action", "ai_action"}:
            continue

        counters[f"{event.event_type}_seen"] += 1
        rows.append(
            {
                "event_id": event.id,
                "game_id": event.game_id,
                "server_version": event.server_version,
                "player_index": event.player_index,
                "event_type": event.event_type,
                "payload": _strip_sensitive_fields(sanitize_dataset_payload(payload)),
            }
        )

    with config.out_path.open("a", encoding="utf-8") as out:
        for row in rows:
            meta = games.setdefault(row["game_id"], _GameMeta(game_id=row["game_id"]))
            if not _game_matches(meta, config, completed_ids):
                counters["actions_skipped_filter"] += 1
                continue

            payload = row["payload"]
            assert isinstance(payload, dict)
            observation = payload.get("observation")
            next_observation = payload.get("next_observation")
            played_card = _played_card(payload, observation)
            result = payload.get("result") if isinstance(payload.get("result"), dict) else None
            is_ai = row["event_type"] == "ai_action"
            action_record: dict[str, Any] = {
                "schema_version": config.schema_version,
                "game_id": row["game_id"],
                "event_id": row["event_id"],
                "server_version": row["server_version"],
                "event_type": row["event_type"],
                "actor": "ai" if is_ai else "human",
                "player_index": payload.get("player_index")
                if isinstance(payload.get("player_index"), int)
                else row["player_index"],
                "metadata": meta.as_record(include_client_id=config.include_client_id),
                "phase": _phase_from_observation(observation, meta=meta),
                "action": {
                    "card_index": payload.get("card_index"),
                    "card": played_card,
                    "coerced": payload.get("action_coerced") if is_ai else None,
                },
                "trick": {
                    "completed": _trick_completed(payload, observation, meta=meta),
                    "winner_index": result.get("trick_winner") if isinstance(result, dict) else None,
                    "cards": _trick_cards_if_completed(payload, observation, played_card),
                },
                "reward": payload.get("reward"),
                "done": payload.get("done"),
                "ai": {
                    "agent": payload.get("ai_agent") or meta.ai_agent,
                    "model_id": payload.get("ai_model_id") or meta.ai_model_id,
                    "decision_type": _decision_type(payload),
                    "decision_trace": payload.get("decision_trace") if is_ai else None,
                }
                if is_ai
                else None,
                "client": {
                    "decision_time_ms": payload.get("client_decision_time_ms"),
                    "observed_server_version": payload.get("client_observed_server_version"),
                }
                if not is_ai
                else None,
                "result": result,
                "observation": observation if config.include_observations else None,
                "next_observation": next_observation if config.include_observations else None,
            }
            out.write(json.dumps(action_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counters["records_written"] += 1

            if not isinstance(observation, dict):
                counters["records_missing_observation"] += 1
            if not isinstance(next_observation, dict):
                counters["records_missing_next_observation"] += 1
            if played_card is None:
                counters["records_missing_played_card"] += 1

    return {
        "backend": reader.backend_name,
        "out_path": str(config.out_path),
        "filters": {
            "code_version": config.code_version,
            "ai_agent": config.ai_agent,
            "ai_model_id": config.ai_model_id,
            "only_completed_games": config.only_completed_games,
            "include_observations": config.include_observations,
            "include_client_id": config.include_client_id,
            "exclude_client_ids": list(config.exclude_client_ids),
        },
        "counters": dict(sorted(counters.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export action-by-action da event log live SQLite/Postgres")
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
    parser.add_argument("--out", required=True, help="Path output JSONL.")
    parser.add_argument("--code-version", default=None, help="Filtra una sola code_version.")
    parser.add_argument("--ai-agent", default=None, help="Filtra un solo ai_agent.")
    parser.add_argument("--ai-model-id", default=None, help="Filtra un solo ai_model_id.")
    parser.add_argument("--include-incomplete", action="store_true", help="Include partite incomplete.")
    parser.add_argument("--no-observations", action="store_true", help="Omette observation/next_observation dal JSONL.")
    parser.add_argument(
        "--include-client-id",
        action="store_true",
        help="Include il client_id pseudonimo nei metadati del JSONL. Default: no.",
    )
    parser.add_argument(
        "--exclude-client-id",
        action="append",
        default=[],
        help="Esclude partite di un client_id pseudonimo. Ripetibile, es. loadtest-bot.",
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

    summary = export_live_actions(
        ExportLiveActionsConfig(
            db_path=db_path,
            database_url=database_url,
            out_path=Path(args.out),
            code_version=args.code_version,
            ai_agent=args.ai_agent,
            ai_model_id=args.ai_model_id,
            only_completed_games=not bool(args.include_incomplete),
            include_observations=not bool(args.no_observations),
            include_client_id=bool(args.include_client_id),
            exclude_client_ids=tuple(str(item) for item in args.exclude_client_id),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
