"""
Test per il primo core Numba.

Questi test non confrontano la sequenza RNG con il dominio canonico: il core JIT usa un RNG
interno separato. Verificano però determinismo, invarianti di Briscola e aggregazione.
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.numba.core import (
    evaluate_numba_match_2p,
    evaluate_numba_seat_fair_match_2p,
    evaluate_random_numba_2p,
    play_policy_game_numba,
    play_random_game_numba,
    warm_up_numba,
    warm_up_numba_evaluation,
)

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba


def test_numba_random_game_is_deterministic_and_valid() -> None:
    """Stesso seed -> stesso risultato; i punti totali devono essere 120."""
    warm_up_numba()

    first = play_random_game_numba(123)
    second = play_random_game_numba(123)

    assert first == second
    points0, points1, winner = first
    assert points0 + points1 == 120
    assert winner in (-1, 0, 1)
    if points0 > points1:
        assert winner == 0
    elif points1 > points0:
        assert winner == 1
    else:
        assert winner == -1


def test_numba_random_evaluation_counts_are_consistent() -> None:
    """L'aggregazione Numba deve produrre contatori e medie coerenti."""
    summary = evaluate_random_numba_2p(num_games=50, seed=0)
    stats = summary.to_match_stats()

    assert stats.num_games == 50
    assert stats.wins_agent0 + stats.wins_agent1 + stats.draws == 50
    assert summary.sum0 + summary.sum1 == 50 * 120
    assert stats.avg_points_agent0 + stats.avg_points_agent1 == pytest.approx(120.0)


@pytest.mark.parametrize(
    ("agent0_name", "agent1_name"),
    [
        ("greedy_points", "random"),
        ("heuristic_v1", "random"),
        ("heuristic_v2", "random"),
        ("heuristic_trump_saver", "random"),
        ("heuristic_trump_saver", "heuristic_trump_saver"),
    ],
)
def test_numba_policy_game_is_deterministic_and_valid(agent0_name: str, agent1_name: str) -> None:
    """Le policy JIT devono produrre partite determinate per seed e terminali validi."""
    warm_up_numba_evaluation()

    first = play_policy_game_numba(agent0_name, agent1_name, seed=321)
    second = play_policy_game_numba(agent0_name, agent1_name, seed=321)

    assert first == second
    points0, points1, winner = first
    assert points0 + points1 == 120
    assert winner in (-1, 0, 1)


def test_numba_policy_match_counts_are_consistent() -> None:
    """La evaluation Numba non seat-fair deve aggregare contatori e punti corretti."""
    stats = evaluate_numba_match_2p("heuristic_v2", "random", num_games=50, seed=5)

    assert stats.num_games == 50
    assert stats.wins_agent0 + stats.wins_agent1 + stats.draws == 50
    assert stats.avg_points_agent0 + stats.avg_points_agent1 == pytest.approx(120.0)


def test_numba_policy_seat_fair_counts_are_consistent() -> None:
    """La evaluation Numba seat-fair deve aggregare contatori e punti corretti."""
    stats = evaluate_numba_seat_fair_match_2p("heuristic_v2", "heuristic_v1", num_games=50, seed=5)

    assert stats.num_games == 50
    assert stats.wins_agent_a + stats.wins_agent_b + stats.draws == 50
    assert stats.avg_points_agent_a + stats.avg_points_agent_b == pytest.approx(120.0)


def test_numba_policy_rejects_unsupported_agents() -> None:
    """Il wrapper Python deve fallire presto per agenti non tradotti nel core JIT."""
    with pytest.raises(ValueError, match="fast-compatible"):
        evaluate_numba_match_2p("bc_model", "random", num_games=1, seed=0)
