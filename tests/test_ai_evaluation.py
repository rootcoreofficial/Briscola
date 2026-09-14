"""
Test per la valutazione offline degli agenti.

Scopo:
- garantire riproducibilità: stessa seed → stessi risultati aggregati
- garantire coerenza: numero partite = wins + draws, punti medi in range plausibile

Non testiamo che un agente sia "forte": testiamo l'infrastruttura.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.agents import (
    GreedyPointsAgent,
    HeuristicAgentV1,
    HeuristicAgentV2,
    HeuristicTrumpSaverAgent,
    RandomAgent,
)
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V1
from briscola_ai.ai.evaluation import evaluate_match_2p, evaluate_seat_fair_match_2p
from briscola_ai.ai.fast.evaluation import evaluate_fast_match_2p, evaluate_fast_seat_fair_match_2p
from briscola_ai.ai.numba.core import evaluate_numba_seat_fair_match_2p, play_policy_game_numba

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = [pytest.mark.slow, pytest.mark.numba]


def test_evaluate_match_is_deterministic_for_fixed_seed() -> None:
    """
    Con la stessa seed, l'aggregato deve essere identico.

    Questo è importante perché useremo queste valutazioni come regressioni:
    se cambiamo un agente, vogliamo attribuire le differenze all'agente, non al rumore.
    """
    a0 = RandomAgent()
    a1 = RandomAgent()

    stats1 = evaluate_match_2p(a0, a1, num_games=50, seed=123)
    stats2 = evaluate_match_2p(a0, a1, num_games=50, seed=123)

    assert stats1 == stats2


def test_evaluate_match_counts_are_consistent() -> None:
    """Verifica che i contatori base tornino."""
    a0 = RandomAgent()
    a1 = RandomAgent()

    stats = evaluate_match_2p(a0, a1, num_games=25, seed=0)
    assert stats.wins_agent0 + stats.wins_agent1 + stats.draws == stats.num_games

    # In 2-player il totale punti per partita è 120, quindi la media per giocatore deve stare in [0, 120].
    assert 0.0 <= stats.avg_points_agent0 <= 120.0
    assert 0.0 <= stats.avg_points_agent1 <= 120.0


def test_seat_fair_evaluate_is_deterministic_for_fixed_seed() -> None:
    """Stesso input → stesso aggregato (anche in modalità seat-fair)."""
    a0 = RandomAgent()
    a1 = RandomAgent()

    stats1 = evaluate_seat_fair_match_2p(a0, a1, num_games=100, seed=123)
    stats2 = evaluate_seat_fair_match_2p(a0, a1, num_games=100, seed=123)
    assert stats1 == stats2


def test_evaluate_match_with_explicit_game_seeds_is_stable_for_deterministic_agents() -> None:
    """
    Se forniamo una suite di shuffle esplicita, la parte “game RNG” è fissata.

    Usiamo agenti deterministici (HeuristicAgentV1) così che cambiare `seed`
    (che controlla l'RNG delle scelte agente) non cambi l'esito aggregato.
    """
    a0 = HeuristicAgentV1()
    a1 = HeuristicAgentV1()

    game_seeds = list(range(100))
    stats1 = evaluate_match_2p(a0, a1, num_games=50, seed=1, game_seeds=game_seeds)
    stats2 = evaluate_match_2p(a0, a1, num_games=50, seed=999, game_seeds=game_seeds)

    assert stats1 == stats2


def test_evaluate_raises_if_game_seeds_is_insufficient() -> None:
    """
    Se la suite di seed è più corta del necessario, deve fallire esplicitamente.

    Questo evita regressioni “silenziose” dove un benchmark usa meno partite del previsto.
    """
    a0 = RandomAgent()
    a1 = RandomAgent()

    with pytest.raises(ValueError, match="game_seeds insufficiente"):
        evaluate_match_2p(a0, a1, num_games=10, seed=0, game_seeds=[1, 2, 3])

    # In seat-fair serve una seed per coppia: num_pairs = num_games // 2.
    with pytest.raises(ValueError, match="game_seeds insufficiente"):
        evaluate_seat_fair_match_2p(a0, a1, num_games=10, seed=0, game_seeds=[1, 2])


@pytest.mark.parametrize(
    ("agent0_name", "domain_agent0"),
    [
        ("greedy_points", GreedyPointsAgent()),
        ("heuristic_v1", HeuristicAgentV1()),
        ("heuristic_v2", HeuristicAgentV2()),
        ("heuristic_trump_saver", HeuristicTrumpSaverAgent()),
    ],
)
def test_fast_evaluate_match_matches_domain_for_supported_agents(agent0_name, domain_agent0) -> None:
    """
    Il path fast deve essere semanticamente equivalente al dominio per gli agenti supportati.
    """
    domain_stats = evaluate_match_2p(domain_agent0, RandomAgent(), num_games=100, seed=123)
    fast_stats = evaluate_fast_match_2p(agent0_name, "random", num_games=100, seed=123)

    assert fast_stats == domain_stats


@pytest.mark.parametrize(
    ("agent0_name", "domain_agent0"),
    [
        ("greedy_points", GreedyPointsAgent()),
        ("heuristic_v1", HeuristicAgentV1()),
        ("heuristic_v2", HeuristicAgentV2()),
        ("heuristic_trump_saver", HeuristicTrumpSaverAgent()),
    ],
)
def test_fast_seat_fair_matches_domain_for_supported_agents(agent0_name, domain_agent0) -> None:
    """Anche la modalità seat-fair fast deve coincidere col path canonico."""
    domain_stats = evaluate_seat_fair_match_2p(domain_agent0, RandomAgent(), num_games=100, seed=456)
    fast_stats = evaluate_fast_seat_fair_match_2p(agent0_name, "random", num_games=100, seed=456)

    assert fast_stats == domain_stats


def test_seat_fair_sum_sq_diff_matches_domain_and_fast() -> None:
    """Il nuovo campo varianza deve restare equivalente tra domain e fast."""
    num_games = 100
    seed = 456
    game_seeds = list(range(seed, seed + num_games // 2))

    domain_stats = evaluate_seat_fair_match_2p(
        HeuristicAgentV2(),
        HeuristicAgentV1(),
        num_games=num_games,
        seed=seed,
        game_seeds=game_seeds,
    )
    fast_stats = evaluate_fast_seat_fair_match_2p(
        "heuristic_v2",
        "heuristic_v1",
        num_games=num_games,
        seed=seed,
        game_seeds=game_seeds,
    )

    assert fast_stats == domain_stats
    assert domain_stats.sum_sq_point_diff_agent_a_minus_agent_b is not None
    assert domain_stats.sample_std_point_diff_agent_a_minus_agent_b is not None


def test_numba_seat_fair_sum_sq_diff_matches_manual_core_aggregation() -> None:
    """Il campo varianza Numba deve coincidere con l'aggregazione manuale dello stesso core."""
    num_games = 100
    seed = 456
    wins_a = wins_b = draws = 0
    sum_a = sum_b = sum_diff = sum_sq_diff = 0

    for pair_index in range(num_games // 2):
        game_seed = seed + pair_index

        p0, p1, winner = play_policy_game_numba("heuristic_v2", "heuristic_v1", seed=game_seed)
        sum_a += p0
        sum_b += p1
        diff = p0 - p1
        sum_diff += diff
        sum_sq_diff += diff * diff
        if winner == 0:
            wins_a += 1
        elif winner == 1:
            wins_b += 1
        else:
            draws += 1

        p0, p1, winner = play_policy_game_numba("heuristic_v1", "heuristic_v2", seed=game_seed)
        sum_a += p1
        sum_b += p0
        diff = p1 - p0
        sum_diff += diff
        sum_sq_diff += diff * diff
        if winner == 0:
            wins_b += 1
        elif winner == 1:
            wins_a += 1
        else:
            draws += 1

    stats = evaluate_numba_seat_fair_match_2p("heuristic_v2", "heuristic_v1", num_games=num_games, seed=seed)

    assert stats.wins_agent_a == wins_a
    assert stats.wins_agent_b == wins_b
    assert stats.draws == draws
    assert stats.avg_points_agent_a == pytest.approx(sum_a / num_games)
    assert stats.avg_points_agent_b == pytest.approx(sum_b / num_games)
    assert stats.avg_point_diff_agent_a_minus_agent_b == pytest.approx(sum_diff / num_games)
    assert stats.sum_sq_point_diff_agent_a_minus_agent_b == float(sum_sq_diff)
    assert stats.sample_std_point_diff_agent_a_minus_agent_b is not None


def test_fast_evaluation_rejects_unsupported_agents() -> None:
    """Il path fast deve fallire presto per agenti non ancora tradotti su card id."""
    with pytest.raises(ValueError, match="supporta solo"):
        evaluate_fast_match_2p("bc_model", "random", num_games=1, seed=0)


def test_evaluate_agents_cli_numba_engine_supports_mlp_model(tmp_path: Path) -> None:
    """Lo script diretto deve poter valutare un modello MLP con il core Numba."""
    d = int(FEATURE_DIM_2P_V1)
    h = 4
    model_path = tmp_path / "dummy_numba_eval.npz"
    out_json = tmp_path / "result.json"
    head_to_head_json = tmp_path / "head_to_head.json"
    np.savez(
        model_path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": d}, ensure_ascii=False),
    )

    script = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_agents.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--engine",
            "numba",
            "--num-games",
            "4",
            "--seed",
            "0",
            "--agent0",
            "bc_model",
            "--agent0-model",
            str(model_path),
            "--agent1",
            "heuristic_v1",
            "--out-json",
            str(out_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    stats = payload["stats"]
    assert payload["engine"] == "numba"
    assert stats["num_games"] == 4
    assert stats["wins_agent0"] + stats["wins_agent1"] + stats["draws"] == 4

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--engine",
            "numba",
            "--num-games",
            "4",
            "--seed",
            "0",
            "--agent0",
            "bc_model",
            "--agent0-model",
            str(model_path),
            "--agent1",
            "bc_model",
            "--agent1-model",
            str(model_path),
            "--out-json",
            str(head_to_head_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    h2h = json.loads(head_to_head_json.read_text(encoding="utf-8"))
    h2h_stats = h2h["stats"]
    assert h2h["engine"] == "numba"
    assert h2h_stats["num_games"] == 4
    assert h2h_stats["wins_agent0"] + h2h_stats["wins_agent1"] + h2h_stats["draws"] == 4
