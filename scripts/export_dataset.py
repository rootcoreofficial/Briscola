#!/usr/bin/env python3
"""
Export dataset da event log (SQLite locale o Postgres cloud) a JSONL.

Perché JSONL?
------------
JSON Lines (JSONL) è un formato “streaming friendly”: ogni riga è un JSON valido.
È comodo per:
- append e processing riga-per-riga;
- import in tool di data processing (anche senza caricare tutto in memoria);
- training pipelines dove ogni record è una transizione o un esempio (observation → action).

Schema (versione 1)
-------------------
Ogni record è una singola azione “gioca carta”.

Fonti supportate (in ordine di preferenza):
- `human_action` (raccolta umana in modalità `BRISCOLA_EVENT_LOG_MODE=dataset`): evento già self-contained,
  con observation prima dell'azione e (opzionale) next_observation.
- legacy: `action_play_card` + `observation_sent` (il record viene ricostruito cercando la migliore observation
  disponibile *prima* dell'azione e la observation *subito dopo*).

Campi principali:
- `schema_version`: versione dello schema record (int)
- `game_id`: id della partita
- `event_id`: id dell'evento `action_play_card` nel DB
- `server_version`: versione monotona server-side (ordering/debug)
- `player_index`: indice del player che ha giocato
- `is_ai`: se l'azione è stata fatta dall'IA
- `observation`: snapshot (DTO observation) prima dell'azione (può essere `null` se mancante)
- `action`: `{ "card_index": int }`
- `reward`: reward “sparse” (punti della mano, signed)
- `next_observation`: observation dopo l'azione (se presente)
- `done`: True se `next_observation.game_over` è True, altrimenti False (o `null` se mancante)

Nota didattica
--------------
Questo export è una base “minima ma utile” per:
- imitation learning (supervised): basta `observation` + `action`;
- RL: si può usare anche `reward` + `next_observation` + `done`.

Focus attuale:
- il progetto (UI e percorso didattico) è focalizzato sul **2-player** (umano vs IA);
- in 4-player (a squadre) il formato observation resta valido, ma il reward “signed” (pro/contro il singolo player)
  è una semplificazione: per un training “team-play” conviene usare reward per squadra e osservazioni parziali
  progettate esplicitamente (vedi `PLAN.md`).

Uso SQLite locale
-----------------
  python scripts/export_dataset.py --db ./data/briscola_events.sqlite3 --out ./data/dataset.jsonl

Uso Postgres/Neon
-----------------
  DATABASE_URL=... python scripts/export_dataset.py --out ./data/dataset.jsonl

  # Oppure esplicito, utile in automazioni:
  python scripts/export_dataset.py --database-url "$DATABASE_URL" --out ./data/dataset.jsonl

Per esportare solo azioni “umane” del player 0 (default UI):
  python scripts/export_dataset.py --player-index 0 --exclude-ai

Per esportare *solo* partite complete (default):
- una partita è “completa” se ha `game_over=true`
- nel DB questo è tracciato preferibilmente con un evento `game_finished`
- fallback legacy: l'exporter riconosce anche snapshot `observation_sent` con `"game_over": true`

Per includere anche partite incomplete (sconsigliato per dataset principale):
  python scripts/export_dataset.py --include-incomplete
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from briscola_ai.backend.event_log import resolve_database_url
from briscola_ai.backend.event_log_privacy import sanitize_dataset_payload
from briscola_ai.backend.event_log_reader import open_event_log_reader

DEFAULT_SQLITE_DB = Path("./data/briscola_events.sqlite3")


@dataclass(frozen=True)
class ExportConfig:
    """Configurazione dell'export."""

    db_path: Path | None
    out_path: Path
    player_index: int | None
    include_ai: bool
    include_next_state: bool
    only_completed_games: bool
    database_url: str | None = None
    schema_version: int = 1


def _compute_trick_points(trick_cards: Any) -> int:
    """
    Calcola i punti della mano a partire da `trick_cards` (DTO).

    `trick_cards` è atteso come lista di oggetti:
      { "card": { ..., "points": int }, "player_index": int }

    Se il formato non è quello atteso, ritorniamo 0 (best-effort).
    """
    if not isinstance(trick_cards, list):
        return 0

    total = 0
    for item in trick_cards:
        if not isinstance(item, dict):
            continue
        card = item.get("card")
        if not isinstance(card, dict):
            continue
        points = card.get("points")
        if isinstance(points, int):
            total += points
    return total


def _find_best_observation(observations: list[dict[str, Any]], *, card_index: int) -> dict[str, Any] | None:
    """
    Trova la migliore observation “prima dell'azione”.

    Strategia:
    - scorriamo all'indietro (più recente → più vecchia)
    - scegliamo la prima observation che:
      - `my_turn == true` (è effettivamente il turno del player)
      - `valid_actions` contiene `card_index` (l'azione è coerente con lo snapshot)

    Se non troviamo una observation coerente, ritorniamo l'ultima disponibile (fallback),
    oppure `None` se non c'è nulla.
    """
    for obs in reversed(observations):
        if not isinstance(obs, dict):
            continue
        if (
            obs.get("my_turn") is True
            and isinstance(obs.get("valid_actions"), list)
            and card_index in obs["valid_actions"]
        ):
            return obs

    return observations[-1] if observations else None


def export_dataset(config: ExportConfig) -> dict[str, int]:
    """
    Esegue l'export event log → JSONL.

    Ritorna un riepilogo (contatori) utile per debug:
    - `rows_total`: eventi letti
    - `records_written`: righe JSONL scritte
    - `records_skipped_ai`: azioni IA scartate per filtro
    - `records_skipped_player`: azioni scartate per filtro player_index
    - `records_missing_observation`: record scritti senza observation coerente
    - `records_missing_next_observation`: record flushati senza next_observation
    """
    config.out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = open_event_log_reader(sqlite_path=config.db_path, database_url=config.database_url)

    completed_game_ids: set[str] = reader.list_completed_game_ids() if config.only_completed_games else set()
    games: dict[str, dict[str, Any]] = {}
    for row in reader.iter_games():
        meta: dict[str, Any] = {}
        if row.num_players is not None:
            meta["num_players"] = row.num_players
        if row.seed is not None:
            meta["seed"] = row.seed
        if row.code_version is not None:
            meta["code_version"] = row.code_version
        if row.rules_version is not None:
            meta["rules_version"] = row.rules_version
        if row.client_id is not None:
            meta["client_id"] = row.client_id
        if row.finished_at is not None:
            meta["finished_at"] = row.finished_at
        if row.aborted_at is not None:
            meta["aborted_at"] = row.aborted_at
        if row.aborted_reason is not None:
            meta["aborted_reason"] = row.aborted_reason
        games[row.game_id] = meta

    counters = {
        "rows_total": 0,
        "records_written": 0,
        "records_skipped_ai": 0,
        "records_skipped_player": 0,
        "records_skipped_incomplete_game": 0,
        "records_missing_observation": 0,
        "records_missing_next_observation": 0,
    }

    current_game_id: str | None = None
    observations_by_player: dict[int, list[dict[str, Any]]] = {}
    pending_by_player: dict[int, dict[str, Any]] = {}
    skip_current_game = False

    def flush_game_pending() -> None:
        """
        Flush best-effort delle transizioni rimaste senza `next_observation`.

        In un log “completo” non dovrebbe succedere spesso (dopo ogni azione inviamo observation),
        ma questo rende l'export robusto a disconnessioni o log incompleti.
        """
        nonlocal pending_by_player
        if not pending_by_player:
            return

        with config.out_path.open("a", encoding="utf-8") as f:
            for _, record in list(pending_by_player.items()):
                record["next_observation"] = None
                record["done"] = None
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters["records_written"] += 1
                counters["records_missing_next_observation"] += 1
            pending_by_player = {}

    # Puliamo il file di output se esiste: export deterministico e ripetibile.
    config.out_path.write_text("", encoding="utf-8")

    try:
        for row in reader.iter_events():
            counters["rows_total"] += 1
            game_id = row.game_id

            if current_game_id is None:
                current_game_id = game_id
                skip_current_game = bool(config.only_completed_games and str(game_id) not in completed_game_ids)
            elif current_game_id != game_id:
                flush_game_pending()
                current_game_id = game_id
                observations_by_player = {}
                pending_by_player = {}
                skip_current_game = bool(config.only_completed_games and str(game_id) not in completed_game_ids)

            if skip_current_game:
                counters["records_skipped_incomplete_game"] += 1
                continue

            event_type = row.event_type
            server_version = row.server_version
            db_player_index = row.player_index

            try:
                payload = json.loads(row.payload_json)
            except json.JSONDecodeError:
                continue
            payload = sanitize_dataset_payload(payload)

            if event_type == "human_action":
                # Nuovo formato (raccolta umana "dataset mode"): evento self-contained.
                player_idx = payload.get("player_index")
                card_index = payload.get("card_index")
                observation = payload.get("observation")
                reward = payload.get("reward", 0)
                done = payload.get("done")
                next_obs = payload.get("next_observation")
                result = payload.get("result")
                decision_time_ms = payload.get("client_decision_time_ms")
                observed_server_version = payload.get("client_observed_server_version")

                if not isinstance(player_idx, int) or not isinstance(card_index, int):
                    continue

                if config.player_index is not None and player_idx != config.player_index:
                    counters["records_skipped_player"] += 1
                    continue

                record: dict[str, Any] = {
                    "schema_version": config.schema_version,
                    "game_id": game_id,
                    "event_id": row.id,
                    "server_version": server_version,
                    "player_index": player_idx,
                    "is_ai": False,
                    "metadata": games.get(game_id, {}),
                    "observation": observation if isinstance(observation, dict) else None,
                    "action": {"card_index": card_index},
                    "reward": int(reward) if isinstance(reward, int) else 0,
                    "result": result if isinstance(result, dict) else None,
                    "next_observation": next_obs
                    if (config.include_next_state and isinstance(next_obs, dict))
                    else None,
                    "done": bool(done is True) if config.include_next_state else None,
                    "client": {
                        "decision_time_ms": int(decision_time_ms) if isinstance(decision_time_ms, int) else None,
                        "observed_server_version": int(observed_server_version)
                        if isinstance(observed_server_version, int)
                        else None,
                    },
                }

                if record["observation"] is None:
                    counters["records_missing_observation"] += 1

                if config.include_next_state and record["next_observation"] is None:
                    counters["records_missing_next_observation"] += 1

                if not config.include_next_state:
                    record["done"] = bool((record["observation"] or {}).get("game_over") is True)

                with config.out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters["records_written"] += 1
                continue

            if event_type == "observation_sent":
                # Observation inviata al player: ci serve per costruire (s, a, r, s', done).
                if isinstance(db_player_index, int):
                    player_idx = db_player_index
                else:
                    # Fallback: alcuni payload includono `my_index`.
                    player_idx_raw = payload.get("my_index")
                    player_idx = int(player_idx_raw) if isinstance(player_idx_raw, int) else -1

                if player_idx < 0:
                    continue

                observations_by_player.setdefault(player_idx, []).append(payload)

                if config.include_next_state and player_idx in pending_by_player:
                    record = pending_by_player.pop(player_idx)
                    record["next_observation"] = payload
                    record["done"] = bool(payload.get("game_over") is True)
                    with config.out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counters["records_written"] += 1

                continue

            if event_type != "action_play_card":
                continue

            is_ai = bool(payload.get("is_ai") is True)
            player_idx_raw = payload.get("player_index")
            card_index_raw = payload.get("card_index")
            result = payload.get("result")

            if (
                not isinstance(player_idx_raw, int)
                or not isinstance(card_index_raw, int)
                or not isinstance(result, dict)
            ):
                continue

            player_idx = player_idx_raw
            card_index = card_index_raw

            if not config.include_ai and is_ai:
                counters["records_skipped_ai"] += 1
                continue
            if config.player_index is not None and player_idx != config.player_index:
                counters["records_skipped_player"] += 1
                continue

            obs_history = observations_by_player.get(player_idx, [])
            observation = _find_best_observation(obs_history, card_index=card_index)

            if observation is None:
                counters["records_missing_observation"] += 1

            # Reward “sparse”: quando una mano si completa, assegniamo i punti della mano.
            # Usiamo un reward signed:
            # - +points se vince il player che ha agito
            # - -points se perde (stesso valore assoluto)
            # - 0 se la mano non si completa in questa azione
            reward = 0
            if result.get("trick_completed") is True:
                trick_points = _compute_trick_points(result.get("trick_cards"))
                winner = result.get("trick_winner")
                if isinstance(winner, int):
                    reward = trick_points if winner == player_idx else -trick_points

            record: dict[str, Any] = {
                "schema_version": config.schema_version,
                "game_id": game_id,
                "event_id": row.id,
                "server_version": server_version,
                "player_index": player_idx,
                "is_ai": is_ai,
                "metadata": games.get(game_id, {}),
                "observation": observation,
                "action": {"card_index": card_index},
                "reward": reward,
                "result": result,
                "next_observation": None,
                "done": None,
            }

            if config.include_next_state:
                # Aspettiamo la prossima observation per completare la transizione.
                pending_by_player[player_idx] = record
            else:
                # Export "supervised-only": scriviamo subito (observation → action).
                record["done"] = bool((observation or {}).get("game_over") is True)
                with config.out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters["records_written"] += 1

        flush_game_pending()
        return counters
    finally:
        reader.close()


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Export dataset da event log SQLite/Postgres a JSONL")
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Path DB SQLite (event log). Se omesso e DATABASE_URL/BRISCOLA_DATABASE_URL è presente, "
            "l'export legge da Postgres; altrimenti usa ./data/briscola_events.sqlite3."
        ),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="DSN Postgres esplicito. Default: BRISCOLA_DATABASE_URL o DATABASE_URL, se presenti.",
    )
    parser.add_argument("--out", default="./data/dataset.jsonl", help="Path output JSONL")
    parser.add_argument(
        "--player-index",
        type=int,
        default=0,
        help="Player da esportare (default: 0; in UI 2-player è l'umano).",
    )
    parser.add_argument("--all-players", action="store_true", help="Esporta azioni di tutti i player")
    parser.add_argument("--include-ai", action="store_true", help="Includi anche le azioni dell'IA")
    parser.add_argument(
        "--exclude-ai",
        action="store_true",
        help="Escludi le azioni dell'IA (override di --include-ai). Default: True per dataset umano.",
    )
    parser.add_argument(
        "--no-next-state",
        action="store_true",
        help="Esporta senza `next_observation` (solo observation → action).",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help=(
            "Include anche partite non complete. Default: esporta solo partite complete "
            "(definite come `game_over=true`, tracciate via evento `game_finished` o snapshot finali)."
        ),
    )
    args = parser.parse_args()

    player_index = None if args.all_players else args.player_index
    include_ai = bool(args.include_ai)
    if args.exclude_ai:
        include_ai = False

    database_url = (
        str(args.database_url).strip() if args.database_url else (None if args.db else resolve_database_url())
    )
    db_path = Path(args.db) if args.db else None
    if db_path is None and not database_url:
        db_path = DEFAULT_SQLITE_DB

    config = ExportConfig(
        db_path=db_path,
        out_path=Path(args.out),
        player_index=player_index,
        include_ai=include_ai,
        include_next_state=not args.no_next_state,
        only_completed_games=bool(not args.include_incomplete),
        database_url=database_url,
    )

    if config.db_path is not None and not config.db_path.exists():
        print(f"DB non trovato: {config.db_path}")
        return 2

    counters = export_dataset(config)
    print("Export completato.")
    for k, v in counters.items():
        print(f"- {k}: {v}")
    print(f"- output: {config.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
