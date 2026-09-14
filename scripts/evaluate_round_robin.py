#!/usr/bin/env python3
"""
Valuta una popolazione di modelli/baseline con round-robin seat-fair.

Esempio:
  python scripts/evaluate_round_robin.py --benchmark medium --suite standard --out-json /tmp/round_robin.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from briscola_ai.ai.evaluation.round_robin import (
    RoundRobinPlayer,
    default_round_robin_players,
    evaluate_round_robin,
    format_round_robin_table,
)


def _parse_player_spec(raw: str) -> list[RoundRobinPlayer]:
    """
    Converte una CSV di player in spec.

    Formati supportati:
    - `nome=path/modello.npz` -> modello
    - `path/modello.npz` -> modello con nome `stem`
    - `heuristic_v1` -> baseline fast-compatible
    """
    players: list[RoundRobinPlayer] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            players.append(RoundRobinPlayer(name.strip(), "model", path.strip()))
            continue
        if item.endswith(".npz"):
            path = Path(item)
            players.append(RoundRobinPlayer(path.stem, "model", str(path)))
            continue
        players.append(RoundRobinPlayer(item, "fast"))
    return players


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta modelli/baseline con round-robin offline seat-fair.")
    parser.add_argument(
        "--engine",
        choices=["domain", "numba"],
        default="numba",
        help="Engine evaluation. Default: numba.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["small", "medium", "big"],
        default="medium",
        help="Taglia benchmark per matchup. Default: medium.",
    )
    parser.add_argument(
        "--suite",
        choices=["standard", "holdout", "both"],
        default="standard",
        help="Seed suite da usare. Default: standard.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilita').")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Livello CI per rating, matchup e gate cicli. Default: 0.95.",
    )
    parser.add_argument(
        "--models-dir",
        default="data/models",
        help="Directory usata dalla popolazione di default. Default: data/models.",
    )
    parser.add_argument(
        "--players",
        default="",
        help=(
            "Player CSV opzionale. Esempio: "
            "v7=data/models/best_a2c_v7.npz,v8=data/models/best_a2c_v8.npz,heuristic_v1. "
            "Se omesso usa legacy v2 + storici v3-v7 + heuristic_v1."
        ),
    )
    parser.add_argument(
        "--holdout-start",
        type=int,
        default=1_000_000,
        help="Range start per suite holdout. Default: 1_000_000.",
    )
    parser.add_argument(
        "--standard-start",
        type=int,
        default=0,
        help="Range start per suite standard. Default: 0.",
    )
    parser.add_argument(
        "--range-step",
        type=int,
        default=1,
        help="Step per le suite generate via range. Default: 1.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Se valorizzato, salva il round-robin in JSON oltre alla stampa a schermo.",
    )

    args = parser.parse_args()
    players = _parse_player_spec(args.players) if args.players else default_round_robin_players(args.models_dir)

    start = time.perf_counter()
    try:
        result = evaluate_round_robin(
            players=players,
            benchmark=args.benchmark,
            seed=args.seed,
            suite=args.suite,
            engine=args.engine,
            confidence=args.confidence,
            standard_start=args.standard_start,
            holdout_start=args.holdout_start,
            range_step=args.range_step,
        )
    except Exception as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 2

    elapsed = time.perf_counter() - start
    print(format_round_robin_table(result), end="")
    print(f"\nElapsed: {elapsed:.1f}s")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_json_text(), encoding="utf-8")
        print(f"JSON salvato in: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
