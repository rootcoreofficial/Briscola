#!/usr/bin/env python3
"""
Valuta agenti offline.

Esempi:
  python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 random --agent1 random
  python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 greedy_points --agent1 random
  python scripts/evaluate_agents.py --engine fast --num-games 1000 --seed 42 --agent0 greedy_points --agent1 random
  python scripts/evaluate_agents.py --seat-fair --num-games 2000 --seed-suite small \\
    --agent0 heuristic_v1 --agent1 random
  python scripts/evaluate_agents.py --seat-fair --num-games 2000 --seed-suite small \\
    --agent0 bc_model --agent0-model ./data/bc_model.npz --agent1 heuristic_v1
  python scripts/evaluate_agents.py --engine numba --benchmark medium \\
    --agent0 bc_model --agent0-model ./data/a2c_model.npz --agent1 heuristic_v1
  python scripts/evaluate_agents.py --seat-fair --num-games 100000 --seed-suite-range-start 0 \\
    --agent0 heuristic_v1 --agent1 random
  python scripts/evaluate_agents.py --benchmark medium --agent0 heuristic_v1 --agent1 random --out-json /tmp/medium.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from briscola_ai.ai.agents import (
    Agent,
    SuitSymmetrizedBCModelAgent,
    agent_uses_selected_model,
    build_agent,
    list_agent_specs,
)
from briscola_ai.ai.evaluation import evaluate_match_2p, evaluate_seat_fair_match_2p
from briscola_ai.ai.evaluation.round_robin import seat_fair_avg_point_diff_ci, seat_fair_score_rate_ci
from briscola_ai.ai.fast.evaluation import (
    FAST_EVALUATION_AGENT_NAMES,
    evaluate_fast_match_2p,
    evaluate_fast_seat_fair_match_2p,
)
from briscola_ai.ai.models import BCModelAgent, MLPBCModel
from briscola_ai.ai.numba.mlp import evaluate_mlp_policy_numba_2p


def _repo_root() -> Path:
    """Ritorna la root del repo (assumendo `scripts/` direttamente sotto root)."""
    return Path(__file__).resolve().parents[1]


def _load_seed_suite(path: Path) -> list[int]:
    """
    Carica una suite di seed (uno per riga).

    Formato:
    - righe vuote ignorate
    - righe che iniziano con `#` ignorate (commenti)
    - ogni riga valida deve essere un intero (base 10)
    """
    seeds: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        seeds.append(int(line))
    return seeds


def _load_bundled_seed_suite(name: str) -> list[int]:
    """
    Carica una seed suite “versionata” nel repo.

    Nota:
    le suite vivono in `seed_suites/` e sono pensate per benchmark/regressioni.
    """
    if name == "small":
        path = _repo_root() / "seed_suites" / "small_1000.txt"
        return _load_seed_suite(path)
    if name == "medium":
        path = _repo_root() / "seed_suites" / "medium_5000.txt"
        return _load_seed_suite(path)
    raise ValueError(f"Seed suite non supportata: {name!r}")


def _make_range_seed_suite(*, start: int, step: int, count: int) -> list[int]:
    """
    Genera una seed suite tramite un range aritmetico.

    È utile per “big” (es. 50k seed) per evitare file molto grandi versionati.

    Nota:
    normalizziamo i seed su 32 bit per allinearci all'uso tipico di `random.Random(seed)`.
    """
    if count < 0:
        raise ValueError(f"count deve essere >= 0, ottenuto {count}")
    if step <= 0:
        raise ValueError(f"step deve essere > 0, ottenuto {step}")
    return [((start + i * step) & 0xFFFFFFFF) for i in range(count)]


def _print_seat_fair_cis(stats) -> None:
    """
    Stampa le CI95 seat-fair calcolate sull'unità COPPIA (mazzo condiviso, seat scambiati).

    Se il produttore non ha tracciato le somme per coppia, gli helper ripiegano sulle CI
    per-partita (anti-conservative) e lo segnaliamo nell'etichetta.
    """
    paired_available = stats.sum_sq_pair_point_diff_agent_a_minus_agent_b is not None
    label = "CI95 (coppie)" if paired_available else "CI95 (per-partita, anti-conservativa)"
    diff_ci = seat_fair_avg_point_diff_ci(stats, confidence=0.95)
    score_ci = seat_fair_score_rate_ci(stats, confidence=0.95)
    score_rate = (stats.wins_agent_a + 0.5 * stats.draws) / max(1, stats.num_games)
    if diff_ci is not None:
        print(f"- avg point diff {label}: {diff_ci.low:+.2f}..{diff_ci.high:+.2f}")
    print(f"- score rate A: {score_rate:.4f} | {label}: {score_ci.low:.4f}..{score_ci.high:.4f}")


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Valuta agenti Briscola offline")
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--benchmark",
        choices=["small", "medium", "big"],
        default=None,
        help=(
            "Preset benchmark (tutti seat-fair): small=2000, medium=10000, big=100000. "
            "Imposta anche una seed suite coerente (small/medium versionate, big via range)."
        ),
    )
    benchmark_group.add_argument("--num-games", type=int, default=1000, help="Numero partite da simulare")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità)")
    parser.add_argument(
        "--engine",
        choices=["domain", "fast", "numba"],
        default="domain",
        help=(
            "Motore di valutazione. `domain` supporta tutti gli agenti; `fast` supporta baseline "
            "fast-compatible; `numba` supporta `agent0=bc_model` MLP contro baseline o `agent1=bc_model` MLP."
        ),
    )
    # Alcune sonde sono costruibili dal registry ma non compaiono in `list_agent_specs()`
    # perché non sono avversari offerti dalla UI: le abilitiamo solo per valutazioni CLI.
    agent_names = [spec.name for spec in list_agent_specs()] + [
        "bc_model",
        "bc_model_suit_symmetrized",
        "heuristic_trump_saver",
        "bc_model_pimc_belief_16x10",
    ]
    parser.add_argument(
        "--agent0",
        default="random",
        choices=agent_names,
        help="Agente per player 0",
    )
    parser.add_argument(
        "--agent1",
        default="random",
        choices=agent_names,
        help="Agente per player 1",
    )
    parser.add_argument(
        "--agent0-model",
        default="",
        help="Path al modello `.npz` se agent0 usa una policy neurale (output di scripts/train_bc.py).",
    )
    parser.add_argument(
        "--agent1-model",
        default="",
        help="Path al modello `.npz` se agent1 usa una policy neurale (output di scripts/train_bc.py).",
    )
    parser.add_argument(
        "--seat-fair",
        action="store_true",
        help=(
            "Valutazione seat-fair: per ogni seed si giocano due partite con agenti scambiati "
            "(riduce il bias dovuto a chi inizia = player 0). Richiede num-games pari."
        ),
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Se valorizzato, salva risultati e configurazione in JSON (oltre alla stampa a schermo).",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed-suite",
        choices=["small", "medium"],
        default=None,
        help=(
            "Usa una seed suite versionata nel repo (benchmark/regressioni). "
            "In seat-fair serve una seed per coppia (num-games/2)."
        ),
    )
    seed_group.add_argument(
        "--seed-suite-file",
        default="",
        help=(
            "Path a una suite di seed per lo shuffle (uno per riga). "
            "Se presente, rende la valutazione confrontabile nel tempo. "
            "In seat-fair serve una seed per coppia (num-games/2)."
        ),
    )
    seed_group.add_argument(
        "--seed-suite-range-start",
        type=int,
        default=None,
        help=(
            "Genera seed con `range()` a partire da START (utile per benchmark big senza file). "
            "In seat-fair genera num-games/2 seed, altrimenti num-games."
        ),
    )
    parser.add_argument(
        "--seed-suite-range-step",
        type=int,
        default=1,
        help="Step per `--seed-suite-range-start` (default: 1).",
    )
    args = parser.parse_args()

    if args.benchmark is not None:
        args.seat_fair = True
        if args.benchmark == "small":
            args.num_games = 2000
            if not (args.seed_suite or args.seed_suite_file.strip() or args.seed_suite_range_start is not None):
                args.seed_suite = "small"
        elif args.benchmark == "medium":
            args.num_games = 10000
            if not (args.seed_suite or args.seed_suite_file.strip() or args.seed_suite_range_start is not None):
                args.seed_suite = "medium"
        elif args.benchmark == "big":
            args.num_games = 100000
            if not (args.seed_suite or args.seed_suite_file.strip() or args.seed_suite_range_start is not None):
                args.seed_suite_range_start = 0
        else:
            raise ValueError(f"Benchmark non supportato: {args.benchmark!r}")

    def _build(*, agent_name: str, model_path: str, agent_flag: str) -> Agent:
        if agent_name == "bc_model_suit_symmetrized":
            if not model_path.strip():
                raise ValueError(
                    f"`--{agent_flag}-model` obbligatorio quando `--{agent_flag} bc_model_suit_symmetrized`."
                )
            return SuitSymmetrizedBCModelAgent.from_npz(model_path.strip())
        if agent_uses_selected_model(agent_name):
            # Vale per `bc_model` e per tutti gli agenti del catalogo che avvolgono un `.npz`
            # (hybrid endgame, value lookahead, PIMC): il registry sa come costruirli e
            # risolve da solo gli asset ausiliari (value/belief) dalla directory modelli.
            if not model_path.strip():
                raise ValueError(f"`--{agent_flag}-model` obbligatorio quando `--{agent_flag} {agent_name}`.")
            return build_agent(agent_name, model_path=Path(model_path.strip()))
        if model_path.strip():
            raise ValueError(f"`--{agent_flag}-model` è valido solo per gli agenti basati su modello `.npz`.")
        return build_agent(agent_name)

    if args.seed_suite_range_start is None and args.seed_suite_range_step != 1:
        raise ValueError("`--seed-suite-range-step` richiede anche `--seed-suite-range-start`.")

    needed = args.num_games // 2 if args.seat_fair else args.num_games
    game_seeds = None
    if args.seed_suite is not None:
        game_seeds = _load_bundled_seed_suite(args.seed_suite)[:needed]
    elif args.seed_suite_file.strip():
        game_seeds = _load_seed_suite(Path(args.seed_suite_file))[:needed]
    elif args.seed_suite_range_start is not None:
        game_seeds = _make_range_seed_suite(
            start=args.seed_suite_range_start,
            step=args.seed_suite_range_step,
            count=needed,
        )

    if args.engine == "numba":
        if args.agent0 != "bc_model":
            raise ValueError("`--engine numba` supporta per ora solo `--agent0 bc_model`.")
        if not args.agent0_model.strip():
            raise ValueError("`--agent0-model` obbligatorio quando `--engine numba --agent0 bc_model`.")

        model_agent = BCModelAgent.from_npz(args.agent0_model.strip())
        model = model_agent.model
        if not isinstance(model, MLPBCModel):
            raise ValueError("`--engine numba` supporta solo modelli `.npz` MLP con chiavi w1/b1/w2/b2.")

        opponent_agent: BCModelAgent | None = None
        opponent_model: MLPBCModel | None = None
        opponent_label = str(args.agent1)
        if args.agent1 == "bc_model":
            if not args.agent1_model.strip():
                raise ValueError("`--agent1-model` obbligatorio quando `--engine numba --agent1 bc_model`.")
            opponent_agent = BCModelAgent.from_npz(args.agent1_model.strip())
            if not isinstance(opponent_agent.model, MLPBCModel):
                raise ValueError("`--engine numba` supporta solo opponent `.npz` MLP con chiavi w1/b1/w2/b2.")
            opponent_model = opponent_agent.model
            opponent_label = opponent_agent.name
        else:
            if args.agent1_model.strip():
                raise ValueError("`--agent1-model` è valido solo quando `--agent1 bc_model`.")
            if args.agent1 not in FAST_EVALUATION_AGENT_NAMES:
                supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
                raise ValueError(
                    f"`--engine numba` supporta agent1: {supported} oppure bc_model. Ottenuto: {args.agent1!r}"
                )

        summary = evaluate_mlp_policy_numba_2p(
            w1=model.w1,
            b1=model.b1,
            w2=model.w2,
            b2=model.b2,
            opponent_name=opponent_label,
            num_games=args.num_games,
            seed=args.seed,
            seat_fair=bool(args.seat_fair),
            game_seeds=game_seeds,
            deterministic=True,
            policy_overkill_guard=bool(model_agent.overkill_guard_enabled),
            parallel=True,
            opponent_w1=opponent_model.w1 if opponent_model is not None else None,
            opponent_b1=opponent_model.b1 if opponent_model is not None else None,
            opponent_w2=opponent_model.w2 if opponent_model is not None else None,
            opponent_b2=opponent_model.b2 if opponent_model is not None else None,
            opponent_overkill_guard=bool(opponent_agent.overkill_guard_enabled)
            if opponent_agent is not None
            else False,
            policy_name=model_agent.name,
        )

        if args.seat_fair:
            stats = summary.to_seat_fair_stats()
            print(f"Match 2-player (numba, seat-fair): {stats.agent_a_name} (A) vs {stats.agent_b_name} (B)")
            print(f"- games: {stats.num_games} (seed={args.seed})")
            print(f"- wins A: {stats.wins_agent_a} | wins B: {stats.wins_agent_b} | draws: {stats.draws}")
            print(f"- avg points A: {stats.avg_points_agent_a:.2f} | avg points B: {stats.avg_points_agent_b:.2f}")
            print(f"- avg point diff (A-B): {stats.avg_point_diff_agent_a_minus_agent_b:.2f}")
            _print_seat_fair_cis(stats)
            if args.out_json.strip():
                payload = {
                    "mode": "seat_fair",
                    "engine": "numba",
                    "benchmark": args.benchmark,
                    "num_games": args.num_games,
                    "seed": args.seed,
                    "seed_suite": {
                        "name": args.seed_suite,
                        "file": args.seed_suite_file.strip() or None,
                        "range_start": args.seed_suite_range_start,
                        "range_step": args.seed_suite_range_step if args.seed_suite_range_start is not None else None,
                        "num_seeds_used": len(game_seeds) if game_seeds is not None else None,
                    },
                    "agents": {"agent0": model_agent.name, "agent1": opponent_label},
                    "stats": asdict(stats),
                }
                Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0

        stats = summary.to_match_stats()
        print(f"Match 2-player (numba): {stats.agent0_name} (P0) vs {stats.agent1_name} (P1)")
        print(f"- games: {stats.num_games} (seed={args.seed})")
        print(f"- wins P0: {stats.wins_agent0} | wins P1: {stats.wins_agent1} | draws: {stats.draws}")
        print(f"- avg points P0: {stats.avg_points_agent0:.2f} | avg points P1: {stats.avg_points_agent1:.2f}")
        print(f"- avg point diff (P0-P1): {stats.avg_point_diff_agent0_minus_agent1:.2f}")
        if args.out_json.strip():
            payload = {
                "mode": "plain",
                "engine": "numba",
                "benchmark": args.benchmark,
                "num_games": args.num_games,
                "seed": args.seed,
                "seed_suite": {
                    "name": args.seed_suite,
                    "file": args.seed_suite_file.strip() or None,
                    "range_start": args.seed_suite_range_start,
                    "range_step": args.seed_suite_range_step if args.seed_suite_range_start is not None else None,
                    "num_seeds_used": len(game_seeds) if game_seeds is not None else None,
                },
                "agents": {"agent0": model_agent.name, "agent1": opponent_label},
                "stats": asdict(stats),
            }
            Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    if args.engine == "fast":
        if args.agent0_model.strip() or args.agent1_model.strip():
            raise ValueError(
                "`--engine fast` non supporta modelli `.npz`; usa agenti fast-compatible senza path modello."
            )
        unsupported = [name for name in (args.agent0, args.agent1) if name not in FAST_EVALUATION_AGENT_NAMES]
        if unsupported:
            supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
            raise ValueError(f"`--engine fast` supporta solo: {supported}. Non supportati: {unsupported}")

        if args.seat_fair:
            stats = evaluate_fast_seat_fair_match_2p(
                args.agent0,
                args.agent1,
                num_games=args.num_games,
                seed=args.seed,
                game_seeds=game_seeds,
            )
            print(f"Match 2-player (fast, seat-fair): {stats.agent_a_name} (A) vs {stats.agent_b_name} (B)")
            print(f"- games: {stats.num_games} (seed={args.seed})")
            print(f"- wins A: {stats.wins_agent_a} | wins B: {stats.wins_agent_b} | draws: {stats.draws}")
            print(f"- avg points A: {stats.avg_points_agent_a:.2f} | avg points B: {stats.avg_points_agent_b:.2f}")
            print(f"- avg point diff (A-B): {stats.avg_point_diff_agent_a_minus_agent_b:.2f}")
            _print_seat_fair_cis(stats)
            if args.out_json.strip():
                payload = {
                    "mode": "seat_fair",
                    "engine": "fast",
                    "benchmark": args.benchmark,
                    "num_games": args.num_games,
                    "seed": args.seed,
                    "seed_suite": {
                        "name": args.seed_suite,
                        "file": args.seed_suite_file.strip() or None,
                        "range_start": args.seed_suite_range_start,
                        "range_step": args.seed_suite_range_step if args.seed_suite_range_start is not None else None,
                        "num_seeds_used": len(game_seeds) if game_seeds is not None else None,
                    },
                    "agents": {"agent0": args.agent0, "agent1": args.agent1},
                    "stats": asdict(stats),
                }
                Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0

        stats = evaluate_fast_match_2p(
            args.agent0, args.agent1, num_games=args.num_games, seed=args.seed, game_seeds=game_seeds
        )
        print(f"Match 2-player (fast): {stats.agent0_name} (P0) vs {stats.agent1_name} (P1)")
        print(f"- games: {stats.num_games} (seed={args.seed})")
        print(f"- wins P0: {stats.wins_agent0} | wins P1: {stats.wins_agent1} | draws: {stats.draws}")
        print(f"- avg points P0: {stats.avg_points_agent0:.2f} | avg points P1: {stats.avg_points_agent1:.2f}")
        print(f"- avg point diff (P0-P1): {stats.avg_point_diff_agent0_minus_agent1:.2f}")
        if args.out_json.strip():
            payload = {
                "mode": "plain",
                "engine": "fast",
                "benchmark": args.benchmark,
                "num_games": args.num_games,
                "seed": args.seed,
                "seed_suite": {
                    "name": args.seed_suite,
                    "file": args.seed_suite_file.strip() or None,
                    "range_start": args.seed_suite_range_start,
                    "range_step": args.seed_suite_range_step if args.seed_suite_range_start is not None else None,
                    "num_seeds_used": len(game_seeds) if game_seeds is not None else None,
                },
                "agents": {"agent0": args.agent0, "agent1": args.agent1},
                "stats": asdict(stats),
            }
            Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    agent0 = _build(agent_name=args.agent0, model_path=args.agent0_model, agent_flag="agent0")
    agent1 = _build(agent_name=args.agent1, model_path=args.agent1_model, agent_flag="agent1")

    if args.seat_fair:
        stats = evaluate_seat_fair_match_2p(
            agent0,
            agent1,
            num_games=args.num_games,
            seed=args.seed,
            game_seeds=game_seeds,
        )
        print(f"Match 2-player (seat-fair): {stats.agent_a_name} (A) vs {stats.agent_b_name} (B)")
        print(f"- games: {stats.num_games} (seed={args.seed})")
        print(f"- wins A: {stats.wins_agent_a} | wins B: {stats.wins_agent_b} | draws: {stats.draws}")
        print(f"- avg points A: {stats.avg_points_agent_a:.2f} | avg points B: {stats.avg_points_agent_b:.2f}")
        print(f"- avg point diff (A-B): {stats.avg_point_diff_agent_a_minus_agent_b:.2f}")
        _print_seat_fair_cis(stats)
        if args.out_json.strip():
            payload = {
                "mode": "seat_fair",
                "engine": "domain",
                "benchmark": args.benchmark,
                "num_games": args.num_games,
                "seed": args.seed,
                "seed_suite": {
                    "name": args.seed_suite,
                    "file": args.seed_suite_file.strip() or None,
                    "range_start": args.seed_suite_range_start,
                    "range_step": args.seed_suite_range_step if args.seed_suite_range_start is not None else None,
                    "num_seeds_used": len(game_seeds) if game_seeds is not None else None,
                },
                "agents": {"agent0": agent0.name, "agent1": agent1.name},
                "stats": asdict(stats),
            }
            Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    stats = evaluate_match_2p(agent0, agent1, num_games=args.num_games, seed=args.seed, game_seeds=game_seeds)
    print(f"Match 2-player: {stats.agent0_name} (P0) vs {stats.agent1_name} (P1)")
    print(f"- games: {stats.num_games} (seed={args.seed})")
    print(f"- wins P0: {stats.wins_agent0} | wins P1: {stats.wins_agent1} | draws: {stats.draws}")
    print(f"- avg points P0: {stats.avg_points_agent0:.2f} | avg points P1: {stats.avg_points_agent1:.2f}")
    print(f"- avg point diff (P0-P1): {stats.avg_point_diff_agent0_minus_agent1:.2f}")
    if args.out_json.strip():
        payload = {
            "mode": "plain",
            "engine": "domain",
            "benchmark": args.benchmark,
            "num_games": args.num_games,
            "seed": args.seed,
            "seed_suite": {
                "name": args.seed_suite,
                "file": args.seed_suite_file.strip() or None,
                "range_start": args.seed_suite_range_start,
                "range_step": args.seed_suite_range_step if args.seed_suite_range_start is not None else None,
                "num_seeds_used": len(game_seeds) if game_seeds is not None else None,
            },
            "agents": {"agent0": agent0.name, "agent1": agent1.name},
            "stats": asdict(stats),
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
