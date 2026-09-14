#!/usr/bin/env python3
"""
Genera un dataset `.npz` per value training usando il fast path Numba.

È la versione ad alto throughput di `generate_value_dataset.py` per run lunghe: salva
direttamente feature già encodate e target numerici, evitando oggetti dominio e JSONL.
Il target è sempre una continuazione deterministica `policy .npz + solver endgame`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np

from briscola_ai.ai.models import BCModelAgent, MLPBCModel
from briscola_ai.ai.numba.value_dataset import (
    COLLECT_ALL,
    COLLECT_WINDOW,
    PHASE_EARLY,
    PHASE_ENDGAME,
    PHASE_MID,
    PHASE_PIMC_WINDOW,
    collect_value_dataset_batch_numba,
)

CollectModeName = Literal["all", "window"]


def _collect_mode_code(name: str) -> int:
    normalized = str(name).strip().lower()
    if normalized == "all":
        return COLLECT_ALL
    if normalized == "window":
        return COLLECT_WINDOW
    raise ValueError(f"--collect-mode non supportato: {name!r}")


def _load_policy(path: Path) -> tuple[MLPBCModel, bool]:
    """Carica una policy MLP e il relativo flag guard runtime."""
    agent = BCModelAgent.from_npz(path)
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("Il generatore Numba supporta solo policy `.npz` MLP con w1/b1/w2/b2.")
    return agent.model, bool(agent.overkill_guard_enabled)


def _phase_counts(phase: np.ndarray) -> dict[str, int]:
    """Conta le fasi salvate nel dataset."""
    return {
        "phase_early": int(np.sum(phase == PHASE_EARLY)),
        "phase_mid": int(np.sum(phase == PHASE_MID)),
        "phase_pimc_window": int(np.sum(phase == PHASE_PIMC_WINDOW)),
        "phase_endgame": int(np.sum(phase == PHASE_ENDGAME)),
    }


def _initial_capacity(
    *,
    num_games: int,
    collect_mode: CollectModeName,
    include_endgame: bool,
    max_records: int | None,
) -> int:
    """
    Stima una capacità iniziale ragionevole.

    In modalità `window` non salviamo tutte le 40 plie: con `max_unknown_cards=8` osserviamo
    circa 5 stati semifinale + 6 stati endgame per partita. Partire dal worst-case 40x su
    run da milioni di partite sprecherebbe molta RAM, quindi cresciamo dinamicamente se serve.
    """
    hard_cap = int(max_records) if max_records is not None else int(num_games) * 40
    if max_records is not None:
        return hard_cap
    if collect_mode == "all":
        estimate_per_game = 40
    elif include_endgame:
        estimate_per_game = 12
    else:
        estimate_per_game = 6
    return min(hard_cap, max(1024, int(num_games) * estimate_per_game))


def _grow_capacity(
    *,
    required: int,
    hard_cap: int,
    x: np.ndarray,
    current_delta: np.ndarray,
    final_delta: np.ndarray,
    phase: np.ndarray,
    unknown_live_cards: np.ndarray,
    deck_size: np.ndarray,
    exploratory_action: np.ndarray,
    game_ids: np.ndarray,
    game_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Espande gli array output quando la stima iniziale non basta."""
    old_capacity = int(x.shape[0])
    if required <= old_capacity:
        return (
            x,
            current_delta,
            final_delta,
            phase,
            unknown_live_cards,
            deck_size,
            exploratory_action,
            game_ids,
            game_seeds,
        )
    new_capacity = min(int(hard_cap), max(required, max(old_capacity + 1, int(old_capacity * 1.5))))
    if new_capacity <= old_capacity:
        return (
            x,
            current_delta,
            final_delta,
            phase,
            unknown_live_cards,
            deck_size,
            exploratory_action,
            game_ids,
            game_seeds,
        )

    def grow(arr: np.ndarray) -> np.ndarray:
        new_arr = np.empty((new_capacity, *arr.shape[1:]), dtype=arr.dtype)
        new_arr[:old_capacity] = arr
        return new_arr

    return (
        grow(x),
        grow(current_delta),
        grow(final_delta),
        grow(phase),
        grow(unknown_live_cards),
        grow(deck_size),
        grow(exploratory_action),
        grow(game_ids),
        grow(game_seeds),
    )


def generate_value_dataset_numba(
    *,
    model_path: Path,
    out_path: Path,
    num_games: int,
    seed: int,
    epsilon: float,
    batch_games: int,
    collect_mode: CollectModeName,
    max_unknown_cards: int,
    include_endgame: bool,
    max_records: int | None,
    feature_dtype: str,
) -> dict[str, int | float | str | bool]:
    """Genera un dataset compatto `.npz` e ritorna un summary serializzabile."""
    if num_games <= 0:
        raise ValueError("--num-games deve essere > 0")
    if batch_games <= 0:
        raise ValueError("--batch-games deve essere > 0")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("--epsilon deve essere in [0,1]")
    if max_unknown_cards < 0:
        raise ValueError("--max-unknown-cards deve essere >= 0")
    if feature_dtype not in {"float16", "float32"}:
        raise ValueError("--feature-dtype deve essere float16 o float32")

    policy, overkill_guard = _load_policy(model_path)
    feature_dim = int(policy.feature_dim)
    collect_code = _collect_mode_code(collect_mode)
    # Nel value dataset possiamo salvare fino a 40 plie/partita, ma in modalità window la densità reale è più bassa.
    hard_cap = int(max_records) if max_records is not None else int(num_games) * 40
    capacity = _initial_capacity(
        num_games=int(num_games),
        collect_mode=collect_mode,
        include_endgame=bool(include_endgame),
        max_records=max_records,
    )
    if capacity <= 0:
        raise ValueError("--max-records deve essere > 0 quando specificato")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if feature_dtype == "float16" else np.float32
    x = np.empty((capacity, feature_dim), dtype=dtype)
    current_delta = np.empty(capacity, dtype=np.float32)
    final_delta = np.empty(capacity, dtype=np.float32)
    phase = np.empty(capacity, dtype=np.int8)
    unknown_live_cards = np.empty(capacity, dtype=np.int8)
    deck_size = np.empty(capacity, dtype=np.int8)
    exploratory_action = np.empty(capacity, dtype=np.bool_)
    # ``game_ids`` e' l'ordinale univoco nella run; ``game_seeds`` rende la
    # provenienza ricostruibile senza usare il seed come identificativo (potrebbe ripetersi).
    game_ids = np.empty(capacity, dtype=np.int64)
    record_game_seeds = np.empty(capacity, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    records = 0
    games_completed = 0
    exploratory_records = 0
    started = time.perf_counter()

    while games_completed < int(num_games) and records < capacity:
        n_games = min(int(batch_games), int(num_games) - games_completed)
        game_seeds = rng.integers(0, np.iinfo(np.int64).max, size=n_games, dtype=np.int64)
        batch = collect_value_dataset_batch_numba(
            w1=policy.w1,
            b1=policy.b1,
            w2=policy.w2,
            b2=policy.b2,
            overkill_guard_enabled=overkill_guard,
            epsilon=float(epsilon),
            collect_mode=collect_code,
            max_unknown_cards=int(max_unknown_cards),
            include_endgame=bool(include_endgame),
            game_seeds=game_seeds,
        )
        mask = batch.valid.reshape(-1)
        batch_records = int(np.sum(mask))
        if batch_records > 0:
            required = records + batch_records
            if max_records is None and required > capacity:
                (
                    x,
                    current_delta,
                    final_delta,
                    phase,
                    unknown_live_cards,
                    deck_size,
                    exploratory_action,
                    game_ids,
                    record_game_seeds,
                ) = _grow_capacity(
                    required=required,
                    hard_cap=hard_cap,
                    x=x,
                    current_delta=current_delta,
                    final_delta=final_delta,
                    phase=phase,
                    unknown_live_cards=unknown_live_cards,
                    deck_size=deck_size,
                    exploratory_action=exploratory_action,
                    game_ids=game_ids,
                    game_seeds=record_game_seeds,
                )
                capacity = int(x.shape[0])
            take = min(batch_records, capacity - records)
            flat_x = batch.xs.reshape(-1, feature_dim)[mask]
            flat_current = batch.current_delta.reshape(-1)[mask]
            flat_final = batch.final_delta.reshape(-1)[mask]
            flat_phase = batch.phase.reshape(-1)[mask]
            flat_unknown = batch.unknown_live_cards.reshape(-1)[mask]
            flat_deck = batch.deck_size.reshape(-1)[mask]
            flat_exploratory = batch.exploratory_action.reshape(-1)[mask]
            game_ordinals = np.arange(games_completed, games_completed + n_games, dtype=np.int64)
            flat_game_ids = np.broadcast_to(game_ordinals[:, None], batch.valid.shape).reshape(-1)[mask]
            flat_game_seeds = np.broadcast_to(game_seeds[:, None], batch.valid.shape).reshape(-1)[mask]

            end = records + take
            x[records:end] = flat_x[:take].astype(dtype, copy=False)
            current_delta[records:end] = flat_current[:take]
            final_delta[records:end] = flat_final[:take]
            phase[records:end] = flat_phase[:take].astype(np.int8, copy=False)
            unknown_live_cards[records:end] = flat_unknown[:take].astype(np.int8, copy=False)
            deck_size[records:end] = flat_deck[:take].astype(np.int8, copy=False)
            exploratory_action[records:end] = flat_exploratory[:take]
            game_ids[records:end] = flat_game_ids[:take]
            record_game_seeds[records:end] = flat_game_seeds[:take]
            exploratory_records += int(np.sum(flat_exploratory[:take]))
            records = end
        games_completed += int(batch.games_completed)

    elapsed = time.perf_counter() - started
    final_x = x[:records]
    final_current = current_delta[:records]
    final_final = final_delta[:records]
    final_phase = phase[:records]
    final_unknown = unknown_live_cards[:records]
    final_deck = deck_size[:records]
    final_exploratory = exploratory_action[:records]
    final_game_ids = game_ids[:records]
    final_game_seeds = record_game_seeds[:records]
    residual = final_final - final_current

    metadata = {
        "format": "value_dataset_npz_v2",
        "schema_version": 2,
        "dataset_kind": "value_observation_numba",
        "model_path": str(model_path),
        "policy_label": str(policy.metadata.get("label", "")),
        "policy_feature_dim": int(policy.feature_dim),
        "encoder_version": str(policy.metadata.get("encoder_version") or policy.metadata.get("encoder") or "v3"),
        "overkill_guard_enabled": bool(overkill_guard),
        "num_games_requested": int(num_games),
        "games_completed": int(games_completed),
        "records_written": int(records),
        "games_with_records": int(np.unique(final_game_ids).size),
        "seed": int(seed),
        "epsilon": float(epsilon),
        "label_mode": "policy_solver_continuation",
        "collect_mode": str(collect_mode),
        "max_unknown_cards": int(max_unknown_cards),
        "include_endgame": bool(include_endgame),
        "feature_dtype": str(feature_dtype),
        "split_unit": "game",
        "split_group_key": "game_ids",
    }
    np.savez(
        out_path,
        x=final_x,
        current_delta=final_current,
        final_delta=final_final,
        residual_delta=residual.astype(np.float32, copy=False),
        phase=final_phase,
        unknown_live_cards=final_unknown,
        deck_size=final_deck,
        exploratory_action=final_exploratory,
        game_ids=final_game_ids,
        game_seeds=final_game_seeds,
        metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2),
    )

    summary: dict[str, int | float | str | bool] = {
        **metadata,
        **_phase_counts(final_phase),
        "exploratory_records": int(exploratory_records),
        "exploration_rate_observed": float(exploratory_records / records) if records else 0.0,
        "elapsed_seconds": float(elapsed),
        "games_per_second": float(games_completed / elapsed) if elapsed > 0 else 0.0,
        "records_per_second": float(records / elapsed) if elapsed > 0 else 0.0,
        "out_path": str(out_path),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera dataset value_observation compatto con Numba")
    parser.add_argument("--model", required=True, help="Policy .npz usata per self-play e continuazione")
    parser.add_argument("--out", required=True, help="Path .npz output")
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--batch-games", type=int, default=4096)
    parser.add_argument("--collect-mode", choices=["all", "window"], default="window")
    parser.add_argument("--max-unknown-cards", type=int, default=8)
    parser.add_argument("--include-endgame", action="store_true")
    parser.add_argument("--max-records", type=int, default=0, help="Stop dopo N record; 0 = fino a fine run")
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    summary = generate_value_dataset_numba(
        model_path=Path(args.model),
        out_path=Path(args.out),
        num_games=int(args.num_games),
        seed=int(args.seed),
        epsilon=float(args.epsilon),
        batch_games=int(args.batch_games),
        collect_mode=args.collect_mode,
        max_unknown_cards=int(args.max_unknown_cards),
        include_endgame=bool(args.include_endgame),
        max_records=int(args.max_records) if int(args.max_records) > 0 else None,
        feature_dtype=str(args.feature_dtype),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
