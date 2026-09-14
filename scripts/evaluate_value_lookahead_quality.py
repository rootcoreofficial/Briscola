#!/usr/bin/env python3
"""
Valuta decision-quality di V-lookahead contro heuristic_v1.

Lo script confronta, sullo stesso benchmark/seed:
- candidato: `v6 + solver + V-lookahead depth-1`
- baseline: `v6 + solver`

La qualità è misurata sull'agente A, quindi ogni run è A vs `heuristic_v1` seat-fair.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from briscola_ai.ai.agents import HeuristicAgentV1, HybridEndgameAgent, ValueLookaheadAgent
from briscola_ai.ai.evaluation.decision_quality import evaluate_seat_fair_match_2p_with_quality_parallel
from briscola_ai.ai.models import BCModelAgent, load_value_model_npz


def _build_control(policy_model_path: Path, *, name: str) -> HybridEndgameAgent:
    """Costruisce `v6 + solver` con un nuovo BCModelAgent."""
    return HybridEndgameAgent(fallback=BCModelAgent.from_npz(policy_model_path), name=name)


def _run_quality(agent, *, num_games: int, seed: int, workers: int):
    """Esegue A vs heuristic_v1 con quality tracking su A."""
    return evaluate_seat_fair_match_2p_with_quality_parallel(
        agent,
        HeuristicAgentV1(),
        num_games=num_games,
        seed=seed,
        workers=workers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta decision-quality di ValueLookaheadAgent vs heuristic_v1.")
    parser.add_argument("--policy-model", default="data/models/best_a2c_v14.npz")
    parser.add_argument("--value-model", required=True)
    parser.add_argument("--benchmark", choices=["small", "medium", "big"], default="small")
    parser.add_argument("--num-games", type=int, default=0, help="Override esplicito per smoke/test; 0 usa benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument("--max-unknown-cards", type=int, default=8)
    parser.add_argument(
        "--disable-overkill-guard",
        action="store_true",
        help="Disabilita il guard anti-overkill sulle decisioni V-lookahead.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    num_games = (
        int(args.num_games)
        if int(args.num_games) > 0
        else {"small": 2000, "medium": 10000, "big": 100000}[args.benchmark]
    )
    if num_games % 2 != 0:
        raise ValueError("--num-games deve essere pari")
    policy_model_path = Path(args.policy_model)
    value_model_path = Path(args.value_model)
    value_model = load_value_model_npz(value_model_path)
    workers = int(args.workers)
    if workers <= 0:
        raise ValueError("--workers deve essere > 0")

    candidate_control = _build_control(policy_model_path, name=f"control_solver({policy_model_path.name})")
    candidate = ValueLookaheadAgent(
        value_model=value_model,
        fallback=candidate_control,
        continuation_agent=candidate_control,
        num_determinizations=int(args.determinizations),
        max_unknown_cards=int(args.max_unknown_cards),
        overkill_guard_enabled=not bool(args.disable_overkill_guard),
        name=f"value_lookahead({value_model_path.name})",
    )
    baseline = _build_control(policy_model_path, name=f"control_solver({policy_model_path.name})")

    started = time.perf_counter()
    candidate_out = _run_quality(candidate, num_games=num_games, seed=int(args.seed), workers=workers)
    candidate_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    baseline_out = _run_quality(baseline, num_games=num_games, seed=int(args.seed), workers=workers)
    baseline_elapsed = time.perf_counter() - started

    def print_block(label: str, out, elapsed: float) -> None:
        print(f"{label}: {out.match.agent_a_name} vs {out.match.agent_b_name}")
        print(f"- games: {out.match.num_games} seed={int(args.seed)} workers={workers}")
        print(f"- elapsed: {elapsed:.2f}s ({out.match.num_games / elapsed:.1f} games/sec)")
        print(f"- wins A: {out.match.wins_agent_a} | wins B: {out.match.wins_agent_b} | draws: {out.match.draws}")
        print(f"- avg diff punti (A-B): {out.match.avg_point_diff_agent_a_minus_agent_b:+.2f}")
        print(f"- trump_waste: {out.quality.num_trump_waste} (rate={out.quality.trump_waste_rate * 100.0:.2f}%)")
        print(
            f"- trump_overkill: {out.quality.num_trump_overkill} (rate={out.quality.trump_overkill_rate * 100.0:.2f}%)"
        )
        print(
            f"- trump_overkill_low_lead_points: {out.quality.num_trump_overkill_low_lead_points} "
            f"(rate={out.quality.trump_overkill_rate_low_lead_points * 100.0:.2f}%)"
        )

    print_block("candidate", candidate_out, candidate_elapsed)
    print(
        "candidate_runtime | "
        f"lookahead={candidate.metrics.lookahead_decisions} fallback={candidate.metrics.fallback_decisions} "
        f"solver={candidate.metrics.endgame_solver_decisions} "
        f"sec/lookahead={candidate.metrics.seconds_per_lookahead_decision:.4f} "
        f"guard_adjustments={candidate.metrics.overkill_guard_adjustments} "
        f"failed_det={candidate.metrics.failed_determinizations} "
        f"failed_leaf={candidate.metrics.failed_leaf_evaluations}"
    )
    print_block("baseline", baseline_out, baseline_elapsed)

    if args.out_json.strip():
        payload = {
            "benchmark": args.benchmark,
            "num_games": num_games,
            "seed": int(args.seed),
            "workers": workers,
            "policy_model": str(policy_model_path),
            "value_model": str(value_model_path),
            "determinizations": int(args.determinizations),
            "max_unknown_cards": int(args.max_unknown_cards),
            "overkill_guard_enabled": not bool(args.disable_overkill_guard),
            "candidate_elapsed_seconds": candidate_elapsed,
            "baseline_elapsed_seconds": baseline_elapsed,
            "candidate": {
                "match": asdict(candidate_out.match),
                "quality": {
                    **asdict(candidate_out.quality),
                    "trump_waste_rate": candidate_out.quality.trump_waste_rate,
                    "trump_overkill_rate": candidate_out.quality.trump_overkill_rate,
                    "trump_overkill_rate_low_lead_points": candidate_out.quality.trump_overkill_rate_low_lead_points,
                },
                "runtime": {
                    **asdict(candidate.metrics),
                    "seconds_per_lookahead_decision": candidate.metrics.seconds_per_lookahead_decision,
                },
            },
            "baseline": {
                "match": asdict(baseline_out.match),
                "quality": {
                    **asdict(baseline_out.quality),
                    "trump_waste_rate": baseline_out.quality.trump_waste_rate,
                    "trump_overkill_rate": baseline_out.quality.trump_overkill_rate,
                    "trump_overkill_rate_low_lead_points": baseline_out.quality.trump_overkill_rate_low_lead_points,
                },
            },
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON salvato in: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
