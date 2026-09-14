#!/usr/bin/env python3
"""
Confronta direttamente due value model dentro lo stesso agente V-lookahead.

Questo script serve per gli esperimenti incrementali su `V`: quando una nuova rete di valore
sembra migliore offline, il confronto corretto non è contro `v6/v7 + solver`, ma contro il
value-lookahead corrente, a parità di policy base, determinizzazioni e finestra.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from briscola_ai.ai.agents import HybridEndgameAgent, ValueLookaheadAgent
from briscola_ai.ai.evaluation import evaluate_seat_fair_match_2p
from briscola_ai.ai.evaluation.round_robin import seat_fair_avg_point_diff_ci, seat_fair_score_rate_ci
from briscola_ai.ai.models import BCModelAgent, load_value_model_npz


def _score_rate(stats) -> float:
    total = max(1, stats.wins_agent_a + stats.wins_agent_b + stats.draws)
    return (stats.wins_agent_a + 0.5 * stats.draws) / total


def _build_value_agent(
    *,
    label: str,
    policy_model_path: Path,
    value_model_path: Path,
    determinizations: int,
    max_unknown_cards: int,
    overkill_guard_enabled: bool,
) -> ValueLookaheadAgent:
    """Costruisce un agente value-lookahead isolato, con fallback/continuation propri."""
    policy = BCModelAgent.from_npz(policy_model_path)
    control = HybridEndgameAgent(
        fallback=policy,
        name=f"{label}_control_solver({policy_model_path.name})",
    )
    return ValueLookaheadAgent(
        value_model=load_value_model_npz(value_model_path),
        fallback=control,
        continuation_agent=control,
        num_determinizations=int(determinizations),
        max_unknown_cards=int(max_unknown_cards),
        overkill_guard_enabled=bool(overkill_guard_enabled),
        name=label,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Confronta due value model nello stesso V-lookahead harness.")
    parser.add_argument("--policy-model", default="data/models/best_a2c_v14.npz")
    parser.add_argument("--value-model-a", required=True)
    parser.add_argument("--value-model-b", required=True)
    parser.add_argument("--label-a", default="value_lookahead_a")
    parser.add_argument("--label-b", default="value_lookahead_b")
    parser.add_argument("--num-games", type=int, default=2000, help="Partite seat-fair, deve essere pari.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument("--max-unknown-cards", type=int, default=8)
    parser.add_argument("--disable-overkill-guard", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    policy_model_path = Path(args.policy_model)
    value_model_a = Path(args.value_model_a)
    value_model_b = Path(args.value_model_b)
    guard_enabled = not bool(args.disable_overkill_guard)
    agent_a = _build_value_agent(
        label=str(args.label_a),
        policy_model_path=policy_model_path,
        value_model_path=value_model_a,
        determinizations=int(args.determinizations),
        max_unknown_cards=int(args.max_unknown_cards),
        overkill_guard_enabled=guard_enabled,
    )
    agent_b = _build_value_agent(
        label=str(args.label_b),
        policy_model_path=policy_model_path,
        value_model_path=value_model_b,
        determinizations=int(args.determinizations),
        max_unknown_cards=int(args.max_unknown_cards),
        overkill_guard_enabled=guard_enabled,
    )

    started = time.perf_counter()
    stats = evaluate_seat_fair_match_2p(
        agent_a,
        agent_b,
        num_games=int(args.num_games),
        seed=int(args.seed),
    )
    elapsed = time.perf_counter() - started
    score_rate = _score_rate(stats)
    # CI calcolate sull'unità COPPIA seat-fair (stesso mazzo, seat scambiati): le due
    # partite di una coppia sono correlate, trattarle come indipendenti darebbe CI troppo strette.
    score_ci = seat_fair_score_rate_ci(stats, confidence=0.95)
    avg_diff_ci = seat_fair_avg_point_diff_ci(stats, confidence=0.95)
    avg_diff_ci_text = "n/a" if avg_diff_ci is None else f"{avg_diff_ci.low:+.2f}..{avg_diff_ci.high:+.2f}"

    print(
        "Value-lookahead pair | "
        f"A={value_model_a.name} | B={value_model_b.name} | policy={policy_model_path.name} | "
        f"games={args.num_games} | seed={args.seed}"
    )
    print(
        f"score_rate_A={score_rate:.4f} ci95={score_ci.low:.4f}..{score_ci.high:.4f} | "
        f"avg_diff_A_minus_B={stats.avg_point_diff_agent_a_minus_agent_b:+.2f} ci95={avg_diff_ci_text} | "
        f"wins_A={stats.wins_agent_a} wins_B={stats.wins_agent_b} draws={stats.draws}"
    )
    print(f"elapsed={elapsed:.2f}s | sec/game={elapsed / max(1, int(args.num_games)):.4f}")
    print(
        "metrics_A | "
        f"lookahead={agent_a.metrics.lookahead_decisions} fallback={agent_a.metrics.fallback_decisions} "
        f"solver={agent_a.metrics.endgame_solver_decisions} "
        f"sec/lookahead={agent_a.metrics.seconds_per_lookahead_decision:.4f} "
        f"guard={agent_a.metrics.overkill_guard_adjustments} failed_leaf={agent_a.metrics.failed_leaf_evaluations}"
    )
    print(
        "metrics_B | "
        f"lookahead={agent_b.metrics.lookahead_decisions} fallback={agent_b.metrics.fallback_decisions} "
        f"solver={agent_b.metrics.endgame_solver_decisions} "
        f"sec/lookahead={agent_b.metrics.seconds_per_lookahead_decision:.4f} "
        f"guard={agent_b.metrics.overkill_guard_adjustments} failed_leaf={agent_b.metrics.failed_leaf_evaluations}"
    )

    if args.out_json:
        payload = {
            "comparison": "value_lookahead_pair",
            "policy_model": str(policy_model_path),
            "value_model_a": str(value_model_a),
            "value_model_b": str(value_model_b),
            "label_a": str(args.label_a),
            "label_b": str(args.label_b),
            "num_games": int(args.num_games),
            "seed": int(args.seed),
            "determinizations": int(args.determinizations),
            "max_unknown_cards": int(args.max_unknown_cards),
            "overkill_guard_enabled": guard_enabled,
            "elapsed_seconds": elapsed,
            "score_rate_a": score_rate,
            "score_rate_ci95": asdict(score_ci),
            "avg_diff_a_minus_b": stats.avg_point_diff_agent_a_minus_agent_b,
            "avg_diff_ci95": asdict(avg_diff_ci) if avg_diff_ci is not None else None,
            "metrics_a": {
                **asdict(agent_a.metrics),
                "seconds_per_lookahead_decision": agent_a.metrics.seconds_per_lookahead_decision,
            },
            "metrics_b": {
                **asdict(agent_b.metrics),
                "seconds_per_lookahead_decision": agent_b.metrics.seconds_per_lookahead_decision,
            },
            "stats": asdict(stats),
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON salvato in: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
