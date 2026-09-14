#!/usr/bin/env python3
"""
Valuta un modello `.npz` su una evaluation matrix (avversari × suite).

Esempio:
  python scripts/evaluate_matrix.py --model ./data/a2c_shaped.npz --benchmark medium --out-json /tmp/matrix.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from briscola_ai.ai.evaluation.matrix import default_opponents, evaluate_model_matrix, format_matrix_table


def _print_rich_table_if_available(*, matrix) -> bool:
    """
    Stampa una tabella “pretty” usando Rich.

    Nota:
    teniamo Rich opzionale per robustezza (fallback a CSV-like) anche se è una dipendenza runtime.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
    except Exception:
        return False

    console = Console()

    title = (
        "Evaluation matrix | "
        f"model={Path(matrix.model_path).name} | "
        f"engine={matrix.engine} | "
        f"benchmark={matrix.benchmark} games={matrix.num_games}"
    )
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Opponent", style="bold")
    table.add_column("Suite")
    table.add_column("Avg diff", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Losses", justify="right")
    table.add_column("Draws", justify="right")

    for row in matrix.rows:
        stats = row.stats
        wins = int(stats.wins_agent_a)
        losses = int(stats.wins_agent_b)
        draws = int(stats.draws)
        total = max(1, wins + losses + draws)
        win_pct = 100.0 * wins / total
        avg = float(stats.avg_point_diff_agent_a_minus_agent_b)

        avg_style = "green" if avg > 0 else ("red" if avg < 0 else "yellow")
        suite_style = "cyan" if row.suite.name == "holdout" else "white"

        table.add_row(
            row.opponent,
            f"[{suite_style}]{row.suite.name}[/{suite_style}]",
            f"[{avg_style}]{avg:+.2f}[/{avg_style}]",
            f"{win_pct:5.1f}",
            str(wins),
            str(losses),
            str(draws),
        )

    console.print(table)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta un modello `.npz` su una evaluation matrix offline.")
    parser.add_argument("--model", required=True, help="Path al modello `.npz` (BC/RL) da valutare.")
    parser.add_argument(
        "--engine",
        choices=["domain", "numba"],
        default="domain",
        help=(
            "Engine evaluation. `domain` usa il motore canonico; `numba` usa rollout MLP full-JIT "
            "per modelli MLP, opponent fast-compatible e opponent `bc_model` MLP (default: domain)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=["small", "medium", "big"],
        default="medium",
        help="Taglia benchmark (tutte seat-fair). Default: medium.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità).")
    parser.add_argument(
        "--opponents",
        default="",
        help="Lista avversari CSV (default: heuristic_v1,random,greedy_points).",
    )
    parser.add_argument(
        "--opponent-model",
        default="",
        help="Path modello `.npz` da usare quando `--opponents` contiene `bc_model`.",
    )
    parser.add_argument(
        "--holdout-start",
        type=int,
        default=1_000_000,
        help="Range start per suite holdout (default: 1_000_000).",
    )
    parser.add_argument(
        "--standard-start",
        type=int,
        default=0,
        help="Range start per suite standard (default: 0).",
    )
    parser.add_argument(
        "--range-step",
        type=int,
        default=1,
        help="Step per le suite generate via range (default: 1).",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Se valorizzato, salva la matrice in JSON oltre alla stampa a schermo.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Numero processi per parallelizzare le righe della matrix con engine=domain. "
            "Con engine=numba viene ignorato: il parallelismo usa thread Numba interni."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "rich"],
        default="auto",
        help=(
            "Formato output a schermo. "
            "`auto` usa una tabella Rich se disponibile e se stdout è un TTY, altrimenti CSV-like. "
            "Default: auto."
        ),
    )
    args = parser.parse_args()
    if int(args.workers) <= 0:
        raise ValueError("--workers deve essere > 0")

    opponents = default_opponents()
    if args.opponents.strip():
        opponents = [p.strip() for p in args.opponents.split(",") if p.strip()]
        if not opponents:
            raise ValueError("`--opponents` non contiene elementi validi.")

    effective_workers = 1 if args.engine == "numba" else int(args.workers)
    t0 = time.perf_counter()
    matrix = evaluate_model_matrix(
        model_path=Path(args.model),
        opponents=opponents,
        benchmark=args.benchmark,
        seed=args.seed,
        standard_start=args.standard_start,
        holdout_start=args.holdout_start,
        range_step=args.range_step,
        workers=effective_workers,
        engine=args.engine,
        opponent_model_path=Path(args.opponent_model) if args.opponent_model.strip() else None,
    )
    elapsed = time.perf_counter() - t0

    should_try_rich = args.format in {"auto", "rich"}
    printed_rich = False
    if should_try_rich and (args.format == "rich" or sys.stdout.isatty()):
        printed_rich = _print_rich_table_if_available(matrix=matrix)
        if args.format == "rich" and not printed_rich:
            raise RuntimeError("Formato `--format rich` richiesto, ma Rich non è disponibile.")

    if not printed_rich:
        print(format_matrix_table(matrix), end="")
    parallelism = "numba_threads" if args.engine == "numba" else f"workers={effective_workers}"
    print(f"elapsed_seconds={elapsed:.3f} {parallelism}", file=sys.stderr)

    if args.out_json.strip():
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(matrix.to_json_text(), encoding="utf-8")
        if printed_rich:
            print(f"Wrote JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
