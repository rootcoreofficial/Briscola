#!/usr/bin/env python3
"""
Valuta un match seat-fair e stampa metriche di qualità decisionale (didattico).

Obiettivo
---------
Integrare una metrica semplice per catturare comportamenti “miopi” tipici:
- spreco briscola: usare una briscola da secondi di mano quando una non-briscola avrebbe vinto.

Esempio:
  python scripts/evaluate_decision_quality.py \
    --agent-a bc_model --agent-a-model ./data/models/best_a2c.npz \
    --agent-b heuristic_v1 \
    --benchmark medium
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from briscola_ai.ai.agents import SuitSymmetrizedBCModelAgent, build_agent, list_agent_specs
from briscola_ai.ai.evaluation.decision_quality import (
    evaluate_bc_model_seat_fair_match_2p_with_quality_numba,
    evaluate_seat_fair_match_2p_with_quality_parallel,
)
from briscola_ai.ai.models import BCModelAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta match seat-fair + metriche qualità decisionale (2-player).")
    parser.add_argument("--benchmark", choices=["small", "medium", "big"], default="medium")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità).")
    parser.add_argument(
        "--engine",
        choices=["domain", "numba"],
        default="domain",
        help=(
            "Engine evaluation. `domain` supporta tutti gli agenti; `numba` supporta A=bc_model MLP "
            "contro baseline fast-compatible (default: domain)."
        ),
    )
    parser.add_argument(
        "--force-overkill-guard",
        action="store_true",
        help=(
            "Forza l'abilitazione del post-processing anti-overkill per `bc_model` impostando "
            "`BRISCOLA_BC_OVERKILL_GUARD=1` nel processo corrente. Utile per A/B test senza "
            "dover modificare i metadati del modello."
        ),
    )
    agent_names = [spec.name for spec in list_agent_specs()] + ["bc_model", "bc_model_suit_symmetrized"]
    parser.add_argument("--agent-a", default="bc_model", choices=agent_names, help="Agente A (misuriamo qualità su A).")
    parser.add_argument("--agent-b", default="heuristic_v1", choices=agent_names, help="Agente B (avversario).")
    parser.add_argument("--agent-a-model", default="", help="Path modello `.npz` se A=bc_model.")
    parser.add_argument("--agent-b-model", default="", help="Path modello `.npz` se B=bc_model.")
    parser.add_argument("--out-json", default="", help="Se valorizzato, salva risultato JSON.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Numero processi per parallelizzare le coppie seat-fair. "
            "Cambia solo la velocita', non i risultati a parita' di seed. "
            "Usato con engine=domain; engine=numba usa thread Numba interni."
        ),
    )
    args = parser.parse_args()

    num_games = {"small": 2000, "medium": 10000, "big": 100000}[args.benchmark]
    if num_games % 2 != 0:
        raise ValueError("Benchmark interno invalido (num_games deve essere pari).")
    if int(args.workers) <= 0:
        raise ValueError("--workers deve essere > 0")

    if bool(args.force_overkill_guard):
        os.environ["BRISCOLA_BC_OVERKILL_GUARD"] = "1"
    effective_workers = 1 if args.engine == "numba" else int(args.workers)

    def _build(*, agent_name: str, model_path: str, flag: str):
        if agent_name == "bc_model_suit_symmetrized":
            if not model_path.strip():
                raise ValueError(f"`--{flag}-model` obbligatorio quando `--{flag} bc_model_suit_symmetrized`.")
            return SuitSymmetrizedBCModelAgent.from_npz(Path(model_path.strip()))
        if agent_name == "bc_model":
            if not model_path.strip():
                raise ValueError(f"`--{flag}-model` obbligatorio quando `--{flag} bc_model`.")
            return BCModelAgent.from_npz(Path(model_path.strip()))
        if model_path.strip():
            raise ValueError(f"`--{flag}-model` è valido solo quando `--{flag} bc_model`.")
        return build_agent(agent_name)

    t0 = time.perf_counter()
    if args.engine == "numba":
        if args.agent_a != "bc_model":
            raise ValueError("`--engine numba` supporta per ora solo `--agent-a bc_model`.")
        if not args.agent_a_model.strip():
            raise ValueError("`--agent-a-model` obbligatorio quando `--engine numba --agent-a bc_model`.")
        agent_a = BCModelAgent.from_npz(Path(args.agent_a_model.strip()))
        agent_b_model = None
        if args.agent_b == "bc_model":
            if not args.agent_b_model.strip():
                raise ValueError("`--agent-b-model` obbligatorio quando `--engine numba --agent-b bc_model`.")
            agent_b_model = BCModelAgent.from_npz(Path(args.agent_b_model.strip()))
            agent_b_name = agent_b_model.name
        else:
            if args.agent_b_model.strip():
                raise ValueError("`--agent-b-model` è valido solo quando `--agent-b bc_model`.")
            agent_b_name = str(args.agent_b)
        out = evaluate_bc_model_seat_fair_match_2p_with_quality_numba(
            agent_a,
            agent_b_name,
            num_games=num_games,
            seed=int(args.seed),
            opponent_agent=agent_b_model,
        )
        agent_b_label = agent_b_name
    else:
        agent_a = _build(agent_name=args.agent_a, model_path=args.agent_a_model, flag="agent-a")
        agent_b = _build(agent_name=args.agent_b, model_path=args.agent_b_model, flag="agent-b")
        out = evaluate_seat_fair_match_2p_with_quality_parallel(
            agent_a,
            agent_b,
            num_games=num_games,
            seed=int(args.seed),
            workers=effective_workers,
        )
        agent_b_label = agent_b.name
    elapsed = time.perf_counter() - t0
    parallelism = "numba_threads" if args.engine == "numba" else f"workers={effective_workers}"

    print(f"Match 2-player ({args.engine}, seat-fair): {out.match.agent_a_name} (A) vs {out.match.agent_b_name} (B)")
    print(f"- games: {out.match.num_games} (seed={int(args.seed)}, {parallelism})")
    print(f"- elapsed: {elapsed:.3f}s ({out.match.num_games / elapsed:.1f} games/sec)")
    print(f"- wins A: {out.match.wins_agent_a} | wins B: {out.match.wins_agent_b} | draws: {out.match.draws}")
    print(f"- avg diff punti (A-B): {out.match.avg_point_diff_agent_a_minus_agent_b:+.2f}")
    print("Decision quality (A, secondo di mano):")
    print(f"- second_hand_decisions: {out.quality.num_second_hand_decisions}")
    print(f"- second_hand_with_winning_reply: {out.quality.num_second_hand_with_winning_reply}")
    print(f"- trump_waste: {out.quality.num_trump_waste} (rate={out.quality.trump_waste_rate * 100.0:.1f}%)")
    print(f"- trump_wins: {out.quality.num_second_hand_trump_wins}")
    print(f"- trump_overkill: {out.quality.num_trump_overkill} (rate={out.quality.trump_overkill_rate * 100.0:.1f}%)")
    print(
        f"- trump_overkill_low_lead_points: {out.quality.num_trump_overkill_low_lead_points} "
        f"(rate={out.quality.trump_overkill_rate_low_lead_points * 100.0:.1f}%)"
    )

    if args.out_json.strip():
        payload = {
            "benchmark": args.benchmark,
            "engine": args.engine,
            "num_games": num_games,
            "seed": int(args.seed),
            "workers": effective_workers,
            "parallelism": parallelism,
            "elapsed_seconds": elapsed,
            "agents": {"a": agent_a.name, "b": agent_b_label},
            "match": asdict(out.match),
            "quality": {
                **asdict(out.quality),
                "trump_waste_rate": out.quality.trump_waste_rate,
                "trump_overkill_rate": out.quality.trump_overkill_rate,
                "trump_overkill_rate_low_lead_points": out.quality.trump_overkill_rate_low_lead_points,
            },
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
