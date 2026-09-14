"""
Test per self-play fast summary-only.

Il fast self-play non scrive il DB e non costruisce DTO/osservazioni complete, quindi questi test
proteggono la cosa più importante: a parità di seed e agenti semplici, il risultato finale coincide
con il dominio canonico.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from briscola_ai.ai.agents import Agent, GreedyPointsAgent, HeuristicAgentV1, HeuristicAgentV2, RandomAgent
from briscola_ai.ai.evaluation import play_one_game_2p
from briscola_ai.ai.fast.self_play import (
    FastSelfPlayAccumulator,
    iter_fast_self_play_2p,
    run_fast_self_play_2p,
)
from briscola_ai.domain.state import GameState

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.slow


def _winner_index_domain_2p(state: GameState) -> int | None:
    """Calcola il vincitore 2-player dal dominio canonico."""
    p0 = state.players[0].points
    p1 = state.players[1].points
    if p0 > p1:
        return 0
    if p1 > p0:
        return 1
    return None


@pytest.mark.parametrize(
    ("agent0_name", "domain_agent0"),
    [
        ("greedy_points", GreedyPointsAgent()),
        ("heuristic_v1", HeuristicAgentV1()),
        ("heuristic_v2", HeuristicAgentV2()),
    ],
)
def test_fast_self_play_summaries_match_domain_for_supported_agents(agent0_name: str, domain_agent0: Agent) -> None:
    """
    Ogni summary fast deve coincidere con una partita dominio giocata con gli stessi seed.

    Usiamo `random` come secondo agente perché consuma RNG e protegge anche la riproducibilità
    di `action_seed`.
    """
    summaries = list(iter_fast_self_play_2p(agent0_name=agent0_name, agent1_name="random", num_games=10, seed=123))

    for summary in summaries:
        domain_state = play_one_game_2p(
            domain_agent0,
            RandomAgent(),
            rng=random.Random(summary.action_seed),
            game_seed=summary.game_seed,
        )
        assert summary.points0 == domain_state.players[0].points
        assert summary.points1 == domain_state.players[1].points
        assert summary.winner_index == _winner_index_domain_2p(domain_state)


def test_run_fast_self_play_matches_streaming_accumulator() -> None:
    """La funzione aggregata deve coincidere con l'accumulatore manuale sui summary."""
    accumulator = FastSelfPlayAccumulator(agent0_name="greedy_points", agent1_name="random")
    for summary in iter_fast_self_play_2p(agent0_name="greedy_points", agent1_name="random", num_games=25, seed=7):
        accumulator.add(summary)

    manual = accumulator.to_match_stats()
    direct = run_fast_self_play_2p(agent0_name="greedy_points", agent1_name="random", num_games=25, seed=7)

    assert direct == manual


def test_fast_self_play_cli_writes_minimal_jsonl(tmp_path: Path) -> None:
    """Smoke test del comando CLI e del JSONL summary-only."""
    out_jsonl = tmp_path / "fast_self_play.jsonl"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "fast_self_play.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--num-games",
            "3",
            "--seed",
            "11",
            "--agents",
            "greedy_points,random",
            "--out-jsonl",
            str(out_jsonl),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Fast self-play completato" in proc.stdout
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    first = json.loads(lines[0])
    assert first["schema"] == "fast_self_play_summary_v1"
    assert first["agent0_name"] == "greedy_points"
    assert first["agent1_name"] == "random"
    assert first["points0"] + first["points1"] == 120
