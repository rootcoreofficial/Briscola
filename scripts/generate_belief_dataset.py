#!/usr/bin/env python3
"""
Genera il dataset osservazione -> mano avversaria per la belief network.

Il dataset storico usava una sola policy mirror. Belief v1 puo' invece ricevere un
roster JSON di stili: ogni partita usa lo stesso stile su entrambi i posti e salva
``opponent_id`` come **solo metadato di split**, mai come feature. Questa scelta rende
compatibili due invarianti che altrimenti entrerebbero in conflitto:

- tutti i record della stessa partita restano nello stesso split;
- un fold leave-one-opponent-out esclude davvero uno stile completo.

Per ogni decisione nella finestra configurata salviamo:

- ``x``: encoder v4 della sola ``PlayerObservation`` lecita;
- ``y``: mano avversaria vera, usata esclusivamente come label offline;
- ``unknown``: maschera delle carte ignote su cui calcolare la loss;
- ``opp_hand_size`` e ``game_index``;
- ``opponent_id``: indice dello stile mirror, solo per split e metriche.

Senza ``--roster`` lo script mantiene il comportamento legacy: una sola policy passata
con ``--policy-model`` gioca entrambi i lati.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.agents.pimc import unknown_live_card_count
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4, encode_player_observation_2p
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state
from briscola_ai.versioning import get_code_version

ROSTER_SCHEMA = "briscola.belief_roster.v1"


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """Uno stile mirror del roster belief, con peso intero di scheduling."""

    opponent_id: str
    agent_name: str
    model_path: Path | None
    weight: int


def _sha256(path: Path) -> str:
    """SHA-256 streaming di un file di configurazione o modello."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_roster_entry(raw: Any, *, index: int) -> RosterEntry:
    """Valida una voce del roster senza affidarsi a parsing testuale ad hoc."""
    if not isinstance(raw, dict):
        raise ValueError(f"roster.items[{index}] deve essere un oggetto")
    opponent_id = str(raw.get("id", "")).strip()
    agent_name = str(raw.get("agent", "")).strip()
    if not opponent_id or not agent_name:
        raise ValueError(f"roster.items[{index}] richiede id e agent non vuoti")
    weight = int(raw.get("weight", 1))
    if weight <= 0:
        raise ValueError(f"roster.items[{index}].weight deve essere > 0")
    model_raw = raw.get("model_path")
    model_path = Path(str(model_raw)) if isinstance(model_raw, str) and model_raw.strip() else None
    if agent_name == "bc_model" and model_path is None:
        raise ValueError(f"roster.items[{index}] bc_model richiede model_path")
    if agent_name != "bc_model" and model_path is not None:
        raise ValueError(f"roster.items[{index}] model_path e' valido solo per bc_model")
    if model_path is not None and not model_path.is_file():
        raise ValueError(f"roster.items[{index}] modello non trovato: {model_path}")
    return RosterEntry(opponent_id=opponent_id, agent_name=agent_name, model_path=model_path, weight=weight)


def load_roster(
    roster_path: Path | None, *, legacy_policy_model: Path
) -> tuple[tuple[RosterEntry, ...], dict[str, Any]]:
    """Carica il roster versionato oppure costruisce la singola voce legacy."""
    if roster_path is None:
        if not legacy_policy_model.is_file():
            raise ValueError(f"Policy legacy non trovata: {legacy_policy_model}")
        entries = (RosterEntry("mirror", "bc_model", legacy_policy_model, 1),)
        return entries, {
            "schema": ROSTER_SCHEMA,
            "source": "legacy_policy_model",
            "items": [
                {
                    "id": "mirror",
                    "agent": "bc_model",
                    "model_path": str(legacy_policy_model),
                    "model_sha256": _sha256(legacy_policy_model),
                    "weight": 1,
                }
            ],
        }

    payload = json.loads(roster_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != ROSTER_SCHEMA:
        raise ValueError(f"Roster senza schema {ROSTER_SCHEMA!r}: {roster_path}")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("roster.items deve essere una lista non vuota")
    entries = tuple(_parse_roster_entry(raw, index=index) for index, raw in enumerate(items))
    ids = [entry.opponent_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Roster con id duplicati: {ids}")

    normalized_items: list[dict[str, Any]] = []
    for entry in entries:
        item: dict[str, Any] = {
            "id": entry.opponent_id,
            "agent": entry.agent_name,
            "weight": entry.weight,
        }
        if entry.model_path is not None:
            item["model_path"] = str(entry.model_path)
            item["model_sha256"] = _sha256(entry.model_path)
        normalized_items.append(item)
    return entries, {
        "schema": ROSTER_SCHEMA,
        "source": str(roster_path),
        "source_sha256": _sha256(roster_path),
        "description": payload.get("description"),
        "items": normalized_items,
    }


def build_weighted_schedule(entries: tuple[RosterEntry, ...], *, num_games: int, seed: int) -> list[int]:
    """
    Costruisce uno schedule bilanciato per blocchi e riproducibile.

    Ogni blocco contiene ``weight`` copie di ciascuno stile e viene rimescolato. In
    questo modo il prefisso di una run lunga rispetta quasi esattamente i pesi, senza
    affidarsi a estrazioni indipendenti che potrebbero sbilanciare i fold piccoli.
    """
    if num_games < 0:
        raise ValueError("num_games deve essere >= 0")
    block = [index for index, entry in enumerate(entries) for _ in range(entry.weight)]
    if not block:
        raise ValueError("Roster senza peso totale")
    rng = random.Random(int(seed) ^ 0xB3113F)
    schedule: list[int] = []
    while len(schedule) < num_games:
        shuffled = list(block)
        rng.shuffle(shuffled)
        schedule.extend(shuffled)
    return schedule[:num_games]


def _build_roster_agents(entries: tuple[RosterEntry, ...]) -> tuple[Agent, ...]:
    """Costruisce una sola istanza stateless per stile mirror."""
    agents: list[Agent] = []
    for entry in entries:
        if entry.model_path is not None:
            agents.append(build_agent(entry.agent_name, model_path=entry.model_path))
        else:
            agents.append(build_agent(entry.agent_name))
    return tuple(agents)


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera dataset belief (osservazione -> mano avversaria)")
    parser.add_argument("--out", required=True, help="Path .npz di output")
    parser.add_argument("--num-games", type=int, default=20000, help="Partite mirror (default 20000)")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilita')")
    parser.add_argument(
        "--policy-model",
        default="data/models/best_a2c_v14.npz",
        help="Modello mirror legacy usato quando --roster e' assente.",
    )
    parser.add_argument(
        "--roster",
        default="",
        help="Roster JSON versionato; ogni partita usa uno stile mirror e salva opponent_id.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Probabilita' di mossa casuale per diversificare gli stati (default 0.05).",
    )
    parser.add_argument(
        "--max-unknown-cards",
        type=int,
        default=10,
        help="Registra stati con carte vive ignote <= soglia (0 = nessun filtro; default 10).",
    )
    parser.add_argument("--log-every", type=int, default=5000, help="Log ogni N partite")
    args = parser.parse_args()

    if int(args.num_games) <= 0:
        raise ValueError("--num-games deve essere > 0")
    if not 0.0 <= float(args.epsilon) <= 1.0:
        raise ValueError("--epsilon deve essere in [0,1]")

    roster_path = Path(args.roster) if str(args.roster).strip() else None
    entries, roster_metadata = load_roster(roster_path, legacy_policy_model=Path(args.policy_model))
    agents = _build_roster_agents(entries)
    schedule = build_weighted_schedule(entries, num_games=int(args.num_games), seed=int(args.seed))

    rng = random.Random(args.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    unknowns: list[np.ndarray] = []
    opp_hand_sizes: list[int] = []
    game_indexes: list[int] = []
    opponent_ids: list[int] = []
    game_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()

    started = time.perf_counter()
    for game_index, roster_index in enumerate(schedule):
        entry = entries[roster_index]
        mirror_agent = agents[roster_index]
        game_counts[entry.opponent_id] += 1
        state = new_game_state(2, seed=rng.randrange(0, 2**32))
        action_rng = random.Random(rng.randrange(0, 2**32))

        while not state.game_over:
            current = state.current_turn
            observation = make_player_observation(state, current)

            in_window = bool(state.deck) and (
                args.max_unknown_cards <= 0 or unknown_live_card_count(observation) <= args.max_unknown_cards
            )
            if in_window:
                encoded = encode_player_observation_2p(observation, version="v4")
                x = np.asarray(encoded.features, dtype=np.float16)

                opponent_hand = state.players[1 - current].hand
                y = np.zeros(40, dtype=np.int8)
                for card in opponent_hand:
                    y[card_to_id(card)] = 1

                unknown = np.ones(40, dtype=np.int8)
                for card in observation.hand:
                    unknown[card_to_id(card)] = 0
                for card_id, flag in enumerate(observation.out_of_play_cards_onehot):
                    if flag:
                        unknown[card_id] = 0

                if int((y & (1 - unknown)).sum()) != 0:
                    raise RuntimeError("Label incoerente: carta avversaria non ignota all'osservatore")
                if int(y.sum()) != len(opponent_hand):
                    raise RuntimeError("Label incoerente: cardinalita' mano avversaria")

                xs.append(x)
                ys.append(y)
                unknowns.append(unknown)
                opp_hand_sizes.append(len(opponent_hand))
                game_indexes.append(game_index)
                opponent_ids.append(roster_index)
                record_counts[entry.opponent_id] += 1

            hand_size = len(state.players[current].hand)
            if action_rng.random() < float(args.epsilon):
                card_index = action_rng.randrange(hand_size)
            else:
                card_index = mirror_agent.choose_card_index(observation, rng=action_rng)
            state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio in self-play: {result.error}")

        if args.log_every > 0 and (game_index + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"partite {game_index + 1}/{args.num_games} | record {len(xs)} | {elapsed:.1f}s")

    metadata = {
        "format": "belief_dataset_v2",
        "encoder_version": "v4",
        "feature_dim": int(FEATURE_DIM_2P_V4),
        "epsilon": float(args.epsilon),
        "max_unknown_cards": int(args.max_unknown_cards),
        "num_games": int(args.num_games),
        "num_records": len(xs),
        "seed": int(args.seed),
        "code_version": get_code_version(),
        "roster": roster_metadata,
        "opponent_index": [entry.opponent_id for entry in entries],
        "schedule": {
            "kind": "weighted_mirror_blocks_v1",
            "weight_total": sum(entry.weight for entry in entries),
            "games_by_opponent": dict(sorted(game_counts.items())),
            "records_by_opponent": dict(sorted(record_counts.items())),
        },
        "anti_cheat": "opponent_id e full-state label non entrano mai in x; inference usa solo PlayerObservation",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=np.stack(xs) if xs else np.zeros((0, FEATURE_DIM_2P_V4), dtype=np.float16),
        y=np.stack(ys) if ys else np.zeros((0, 40), dtype=np.int8),
        unknown=np.stack(unknowns) if unknowns else np.zeros((0, 40), dtype=np.int8),
        opp_hand_size=np.asarray(opp_hand_sizes, dtype=np.int8),
        game_index=np.asarray(game_indexes, dtype=np.int32),
        opponent_id=np.asarray(opponent_ids, dtype=np.int16),
        metadata_json=json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
    )
    elapsed = time.perf_counter() - started
    print(f"Salvato {out_path} | record={len(xs)} | partite={args.num_games} | stili={len(entries)} | {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
