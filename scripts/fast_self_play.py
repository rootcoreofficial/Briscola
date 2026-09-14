#!/usr/bin/env python3
"""
Self-play veloce 2-player senza DB/event log completo.

Questo comando usa il fast path 2-player (`ai.fast.state_2p`) e salva al massimo
un JSONL minimale per partita. È utile per benchmark e per preparare un futuro
path di training più veloce, dove non vogliamo pagare il costo di DTO/SQLite.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from briscola_ai.ai.fast.evaluation import FAST_EVALUATION_AGENT_NAMES
from briscola_ai.ai.fast.self_play import FastSelfPlayAccumulator, iter_fast_self_play_2p


def _parse_agents(raw: str) -> tuple[str, str]:
    """Parsa `--agents agent0,agent1` e valida cardinalità minima."""
    names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(names) != 2:
        raise ValueError(f"`--agents` deve contenere esattamente 2 nomi. Ottenuti: {names}")
    return names[0], names[1]


def main() -> int:
    """Entry point CLI."""
    supported = ",".join(sorted(FAST_EVALUATION_AGENT_NAMES))
    parser = argparse.ArgumentParser(description="Fast self-play Briscola 2-player (summary-only)")
    parser.add_argument("--num-games", type=int, default=10000, help="Numero partite da simulare.")
    parser.add_argument("--seed", type=int, default=0, help="Seed base per game/action RNG.")
    parser.add_argument(
        "--agents",
        default="random,random",
        help=f"Agenti CSV per player 0/1. Supportati: {supported}.",
    )
    parser.add_argument(
        "--out-jsonl",
        default="",
        help="Path opzionale per salvare un JSONL minimale, una riga per partita.",
    )
    args = parser.parse_args()

    if int(args.num_games) < 0:
        raise ValueError("--num-games deve essere >= 0")

    agent0_name, agent1_name = _parse_agents(str(args.agents))
    unsupported = [name for name in (agent0_name, agent1_name) if name not in FAST_EVALUATION_AGENT_NAMES]
    if unsupported:
        raise ValueError(f"Agenti non supportati nel fast self-play: {unsupported}. Supportati: {supported}")

    out_path = Path(args.out_jsonl) if str(args.out_jsonl).strip() else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    accumulator = FastSelfPlayAccumulator(agent0_name=agent0_name, agent1_name=agent1_name)
    start = time.perf_counter()
    if out_path is None:
        for summary in iter_fast_self_play_2p(
            agent0_name=agent0_name,
            agent1_name=agent1_name,
            num_games=int(args.num_games),
            seed=int(args.seed),
        ):
            accumulator.add(summary)
    else:
        with out_path.open("w", encoding="utf-8") as fh:
            for summary in iter_fast_self_play_2p(
                agent0_name=agent0_name,
                agent1_name=agent1_name,
                num_games=int(args.num_games),
                seed=int(args.seed),
            ):
                accumulator.add(summary)
                fh.write(summary.to_json_line())

    elapsed = time.perf_counter() - start
    stats = accumulator.to_match_stats()
    games_per_sec = stats.num_games / elapsed if elapsed > 0 else 0.0

    print(f"Fast self-play completato in {elapsed:.3f}s ({games_per_sec:.1f} games/sec).")
    print(f"- agents: {stats.agent0_name},{stats.agent1_name}")
    print(f"- games: {stats.num_games} (seed={int(args.seed)})")
    print(f"- wins P0: {stats.wins_agent0} | wins P1: {stats.wins_agent1} | draws: {stats.draws}")
    print(f"- avg points P0: {stats.avg_points_agent0:.2f} | avg points P1: {stats.avg_points_agent1:.2f}")
    print(f"- avg point diff (P0-P1): {stats.avg_point_diff_agent0_minus_agent1:.2f}")
    if out_path is not None:
        print(f"- jsonl: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
