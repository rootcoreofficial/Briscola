#!/usr/bin/env python3
"""
Valuta V-lookahead depth-1 contro la stessa policy con il solo solver finale.

Questo harness storico dell'ipotesi V-lookahead misura, in modo seat-fair, quanto la
lookahead aggiunge rispetto alla policy scelta con il solo solver. Il ramo di ricerca
value full-game e' chiuso; lo script resta utile per regressioni e confronti tra asset `V`.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta ValueLookaheadAgent con engine dominio offline.")
    parser.add_argument(
        "--policy-model",
        default="data/models/best_a2c_v14.npz",
        help="Path modello `.npz` usato da fallback, continuazione e baseline.",
    )
    parser.add_argument("--value-model", required=True, help="Path value model `.npz`.")
    parser.add_argument("--num-games", type=int, default=2000, help="Partite seat-fair, deve essere pari.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument("--max-unknown-cards", type=int, default=8)
    parser.add_argument(
        "--disable-overkill-guard",
        action="store_true",
        help="Disabilita il guard anti-overkill sulle decisioni V-lookahead.",
    )
    parser.add_argument("--out-json", default="", help="Path JSON opzionale per salvare il risultato.")
    args = parser.parse_args()

    policy_model_path = Path(args.policy_model)
    value_model_path = Path(args.value_model)
    value_model = load_value_model_npz(value_model_path)

    candidate_model_agent = BCModelAgent.from_npz(policy_model_path)
    candidate_control = HybridEndgameAgent(
        fallback=candidate_model_agent,
        name=f"control_solver({policy_model_path.name})",
    )
    candidate = ValueLookaheadAgent(
        value_model=value_model,
        fallback=candidate_control,
        continuation_agent=candidate_control,
        num_determinizations=int(args.determinizations),
        max_unknown_cards=int(args.max_unknown_cards),
        overkill_guard_enabled=not bool(args.disable_overkill_guard),
        name=f"value_lookahead({value_model_path.name})",
    )

    opponent_model_agent = BCModelAgent.from_npz(policy_model_path)
    opponent = HybridEndgameAgent(
        fallback=opponent_model_agent,
        name=f"control_solver({policy_model_path.name})",
    )

    started = time.perf_counter()
    stats = evaluate_seat_fair_match_2p(
        candidate,
        opponent,
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
    metrics = candidate.metrics

    print(
        "Value-lookahead evaluation | "
        f"policy={policy_model_path.name} | value={value_model_path.name} | games={args.num_games} | "
        f"determinizations={args.determinizations} | max_unknown={args.max_unknown_cards}"
    )
    print(
        f"score_rate={score_rate:.4f} ci95={score_ci.low:.4f}..{score_ci.high:.4f} | "
        f"avg_diff={stats.avg_point_diff_agent_a_minus_agent_b:+.2f} ci95={avg_diff_ci_text} | "
        f"wins={stats.wins_agent_a} losses={stats.wins_agent_b} draws={stats.draws}"
    )
    print(f"elapsed={elapsed:.2f}s | sec/game={elapsed / max(1, int(args.num_games)):.4f}")
    print(
        "value_lookahead_moves | "
        f"total={metrics.total_decisions} lookahead={metrics.lookahead_decisions} "
        f"fallback={metrics.fallback_decisions} endgame_solver={metrics.endgame_solver_decisions}"
    )
    print(
        "value_lookahead_latency | "
        f"search_elapsed={metrics.search_elapsed_seconds:.4f}s | "
        f"sec/lookahead_move={metrics.seconds_per_lookahead_decision:.4f} | "
        f"determinizations_ok={metrics.successful_determinizations} "
        f"determinizations_failed={metrics.failed_determinizations} | "
        f"leaf_evals={metrics.completed_leaf_evaluations} failed_leaf_evals={metrics.failed_leaf_evaluations}"
    )
    print(f"value_lookahead_guard | adjustments={metrics.overkill_guard_adjustments}")

    if args.out_json:
        payload = {
            "policy_model": str(policy_model_path),
            "value_model": str(value_model_path),
            "num_games": int(args.num_games),
            "seed": int(args.seed),
            "determinizations": int(args.determinizations),
            "max_unknown_cards": int(args.max_unknown_cards),
            "overkill_guard_enabled": not bool(args.disable_overkill_guard),
            "elapsed_seconds": elapsed,
            "seconds_per_game": elapsed / max(1, int(args.num_games)),
            "score_rate": score_rate,
            "score_rate_ci95": asdict(score_ci),
            "avg_point_diff_ci95": asdict(avg_diff_ci) if avg_diff_ci is not None else None,
            "value_lookahead_metrics": {
                **asdict(metrics),
                "seconds_per_lookahead_decision": metrics.seconds_per_lookahead_decision,
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
