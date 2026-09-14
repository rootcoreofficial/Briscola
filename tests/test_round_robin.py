"""
Test per round-robin Elo offline.

Manteniamo questi test leggeri: non serve simulare migliaia di partite per verificare la parte
critica, cioe' normalizzazione prospettiva, aggregazione e rating.
"""

from __future__ import annotations

import json

import pytest

from briscola_ai.ai.evaluation import round_robin
from briscola_ai.ai.evaluation.match import SeatFairStats
from briscola_ai.ai.evaluation.matrix import SuiteSpec
from briscola_ai.ai.evaluation.round_robin import (
    RoundRobinMatchup,
    RoundRobinPlayer,
    compute_elo_ratings,
    evaluate_round_robin,
    find_non_transitive_cycles,
    invert_seat_fair_stats,
    score_rate,
    wilson_score_interval,
)


def _stats(
    *,
    a: str,
    b: str,
    wins_a: int,
    wins_b: int,
    draws: int = 0,
    avg_diff: float = 0.0,
) -> SeatFairStats:
    """Factory compatta per risultati seat-fair coerenti."""
    num_games = wins_a + wins_b + draws
    return SeatFairStats(
        num_games=num_games,
        agent_a_name=a,
        agent_b_name=b,
        wins_agent_a=wins_a,
        wins_agent_b=wins_b,
        draws=draws,
        avg_points_agent_a=60.0 + avg_diff / 2.0,
        avg_points_agent_b=60.0 - avg_diff / 2.0,
        avg_point_diff_agent_a_minus_agent_b=avg_diff,
    )


def test_score_rate_counts_draw_as_half_point() -> None:
    """Lo score rate usato dal rating deve trattare i pareggi come mezzo punto."""
    assert score_rate(wins=7, losses=1, draws=2) == pytest.approx(0.8)
    assert score_rate(wins=0, losses=0, draws=0) == 0.0


def test_wilson_score_interval_keeps_tight_matchup_uncertain() -> None:
    """Un 50.3% su 10k partite deve avere CI che attraversa 0.5."""
    interval = wilson_score_interval(wins=4883, losses=4818, draws=299)

    assert interval.low < 0.5 < interval.high


def test_invert_seat_fair_stats_preserves_numbers_from_new_perspective() -> None:
    """Invertire B-vs-A deve scambiare win/loss, punti medi e segno del margine."""
    original = _stats(a="model", b="heuristic_v1", wins_a=8, wins_b=2, avg_diff=12.0)

    inverted = invert_seat_fair_stats(original, agent_a_name="heuristic_v1", agent_b_name="model")

    assert inverted.agent_a_name == "heuristic_v1"
    assert inverted.agent_b_name == "model"
    assert inverted.wins_agent_a == 2
    assert inverted.wins_agent_b == 8
    assert inverted.avg_point_diff_agent_a_minus_agent_b == -12.0


def test_compute_elo_ratings_orders_stronger_player_higher() -> None:
    """Un player che batte entrambi gli altri deve finire sopra nel rating Elo-like."""
    suite = SuiteSpec(name="standard", range_start=0, range_step=1, num_seeds=5)
    players = [
        RoundRobinPlayer("strong", "model", "strong.npz"),
        RoundRobinPlayer("middle", "model", "middle.npz"),
        RoundRobinPlayer("weak", "model", "weak.npz"),
    ]
    matchups = [
        RoundRobinMatchup(suite, "strong", "middle", _stats(a="strong", b="middle", wins_a=7, wins_b=3)),
        RoundRobinMatchup(suite, "strong", "weak", _stats(a="strong", b="weak", wins_a=9, wins_b=1)),
        RoundRobinMatchup(suite, "middle", "weak", _stats(a="middle", b="weak", wins_a=7, wins_b=3)),
    ]

    ratings = compute_elo_ratings(players=players, matchups=matchups)

    assert ratings["strong"] > ratings["middle"] > ratings["weak"]
    assert sum(ratings.values()) / len(ratings) == pytest.approx(1500.0)


def test_find_non_transitive_cycles_ignores_uncertain_edges_by_default() -> None:
    """Un ciclo composto da vantaggi piccoli non deve passare il gate CI."""
    suite = SuiteSpec(name="standard", range_start=0, range_step=1, num_seeds=5)
    players = [
        RoundRobinPlayer("a", "model", "a.npz"),
        RoundRobinPlayer("b", "model", "b.npz"),
        RoundRobinPlayer("c", "model", "c.npz"),
    ]
    matchups = [
        RoundRobinMatchup(suite, "a", "b", _stats(a="a", b="b", wins_a=6, wins_b=4)),
        RoundRobinMatchup(suite, "b", "c", _stats(a="b", b="c", wins_a=6, wins_b=4)),
        RoundRobinMatchup(suite, "a", "c", _stats(a="a", b="c", wins_a=4, wins_b=6)),
    ]

    cycles = find_non_transitive_cycles(players=players, matchups=matchups)

    assert cycles == []


def test_find_non_transitive_cycles_reports_confident_cycle() -> None:
    """Un ciclo A>B, B>C, C>A con archi robusti deve essere segnalato."""
    suite = SuiteSpec(name="standard", range_start=0, range_step=1, num_seeds=50)
    players = [
        RoundRobinPlayer("a", "model", "a.npz"),
        RoundRobinPlayer("b", "model", "b.npz"),
        RoundRobinPlayer("c", "model", "c.npz"),
    ]
    matchups = [
        RoundRobinMatchup(suite, "a", "b", _stats(a="a", b="b", wins_a=80, wins_b=20)),
        RoundRobinMatchup(suite, "b", "c", _stats(a="b", b="c", wins_a=80, wins_b=20)),
        RoundRobinMatchup(suite, "a", "c", _stats(a="a", b="c", wins_a=20, wins_b=80)),
    ]

    cycles = find_non_transitive_cycles(players=players, matchups=matchups)

    assert cycles == [["a", "b", "c", "a"]]


def test_evaluate_round_robin_builds_all_pairs_with_fake_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La funzione pubblica deve costruire tutte le coppie e serializzare un JSON stabile."""
    monkeypatch.setattr(round_robin, "benchmark_num_games", lambda benchmark: 20)

    def fake_evaluate_pair(**kwargs) -> SeatFairStats:
        player_a = kwargs["player_a"]
        player_b = kwargs["player_b"]
        return _stats(a=player_a.name, b=player_b.name, wins_a=12, wins_b=6, draws=2, avg_diff=4.0)

    monkeypatch.setattr(round_robin, "_evaluate_pair", fake_evaluate_pair)
    players = [
        RoundRobinPlayer("random", "fast"),
        RoundRobinPlayer("greedy_points", "fast"),
        RoundRobinPlayer("heuristic_v1", "fast"),
    ]

    result = evaluate_round_robin(players=players, benchmark="small", seed=7, suite="standard", engine="numba")

    assert len(result.matchups) == 3
    assert [m.suite.name for m in result.matchups] == ["standard", "standard", "standard"]
    parsed = json.loads(result.to_json_text())
    assert parsed["benchmark"] == "small"
    assert parsed["engine"] == "numba"
    assert parsed["confidence"] == 0.95
    assert len(parsed["ratings"]) == 3
