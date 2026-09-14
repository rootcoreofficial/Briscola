"""
Test delle CI seat-fair calcolate sull'unità COPPIA.

Perché esistono questi test
---------------------------
Nelle valutazioni seat-fair le due partite di una coppia condividono il mazzo (seat
scambiati): non sono campioni indipendenti. Le CI storiche trattavano le partite come
indipendenti, risultando troppo strette (anti-conservative) — e le decisioni di
promozione dei modelli si basano su questi intervalli. Qui verifichiamo:
- la matematica delle nuove CI per coppia (esempi calcolati a mano);
- che con dati correlati la CI per coppia sia più larga di quella per-partita;
- che i produttori (dominio/fast/numba) accumulino le somme per coppia;
- che relabel/invert propaghino correttamente i nuovi campi.
"""

from __future__ import annotations

import math
import random

import pytest

from briscola_ai.ai.evaluation.match import SeatFairStats, evaluate_seat_fair_match_2p
from briscola_ai.ai.evaluation.round_robin import (
    invert_seat_fair_stats,
    mean_point_diff_interval,
    mean_point_diff_interval_paired,
    relabel_seat_fair_stats,
    score_rate_interval_paired,
    seat_fair_avg_point_diff_ci,
    seat_fair_score_rate_ci,
)
from briscola_ai.ai.fast.evaluation import evaluate_fast_seat_fair_match_2p
from briscola_ai.ai.numba.core import evaluate_numba_seat_fair_match_2p

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba


class _FirstCardAgent:
    """Agente deterministico minimale: gioca sempre la prima carta in mano."""

    name = "first_card"

    def choose_card_index(self, observation, *, rng: random.Random) -> int:
        return 0


def _stats_from_pair_diffs(pair_diffs: list[int]) -> SeatFairStats:
    """
    Costruisce uno `SeatFairStats` sintetico dove ogni coppia ha due partite identiche.

    Con partite perfettamente correlate dentro la coppia (d, d), il diff di coppia è 2d:
    è il caso in cui la CI per-partita sbaglia di più (sottostima la varianza effettiva).
    """
    game_diffs = [d for pair in pair_diffs for d in (pair, pair)]
    num_games = len(game_diffs)
    sum_diff = sum(game_diffs)
    return SeatFairStats(
        num_games=num_games,
        agent_a_name="A",
        agent_b_name="B",
        wins_agent_a=sum(1 for d in game_diffs if d > 0),
        wins_agent_b=sum(1 for d in game_diffs if d < 0),
        draws=sum(1 for d in game_diffs if d == 0),
        avg_points_agent_a=60.0,
        avg_points_agent_b=60.0,
        avg_point_diff_agent_a_minus_agent_b=sum_diff / num_games,
        sum_sq_point_diff_agent_a_minus_agent_b=float(sum(d * d for d in game_diffs)),
        sum_sq_pair_point_diff_agent_a_minus_agent_b=float(sum((2 * d) ** 2 for d in pair_diffs)),
        sum_sq_pair_score_agent_a=float(sum((1.0 if d > 0 else 0.0 if d < 0 else 0.5) ** 2 for d in pair_diffs)),
    )


def test_paired_mean_ci_matches_hand_computed_example() -> None:
    """La CI per coppia deve coincidere con il calcolo manuale su un esempio piccolo."""
    # 3 coppie con diff per coppia [+4, -2, +6] (somme dei due game della coppia).
    pair_diffs = [4, -2, 6]
    num_pairs = len(pair_diffs)
    num_games = num_pairs * 2
    sum_diff = sum(pair_diffs)
    mean_per_game = sum_diff / num_games
    sum_sq_pair = float(sum(d * d for d in pair_diffs))

    ci = mean_point_diff_interval_paired(
        mean=mean_per_game, num_games=num_games, sum_sq_pair=sum_sq_pair, confidence=0.95
    )
    assert ci is not None

    # Ricalcolo manuale: varianza campionaria dei pair diff, CI su mean(pair)/2.
    mean_pair = sum_diff / num_pairs
    variance_pair = sum((d - mean_pair) ** 2 for d in pair_diffs) / (num_pairs - 1)
    z = 1.959963984540054  # quantile normale per il 95%
    half_width = z * math.sqrt(variance_pair / num_pairs) / 2.0

    assert ci.low == pytest.approx(mean_per_game - half_width)
    assert ci.high == pytest.approx(mean_per_game + half_width)


def test_paired_ci_is_wider_than_per_game_ci_on_correlated_data() -> None:
    """
    Con partite perfettamente correlate dentro la coppia, la CI per coppia deve essere
    più larga di quella per-partita (che sovrastima il campione effettivo di un fattore 2).
    """
    stats = _stats_from_pair_diffs([10, -6, 8, -2, 12, 4, -8, 6])

    per_game = mean_point_diff_interval(
        mean=stats.avg_point_diff_agent_a_minus_agent_b,
        num_games=stats.num_games,
        sum_sq=stats.sum_sq_point_diff_agent_a_minus_agent_b,
    )
    paired = mean_point_diff_interval_paired(
        mean=stats.avg_point_diff_agent_a_minus_agent_b,
        num_games=stats.num_games,
        sum_sq_pair=stats.sum_sq_pair_point_diff_agent_a_minus_agent_b,
    )
    assert per_game is not None and paired is not None
    width_per_game = per_game.high - per_game.low
    width_paired = paired.high - paired.low
    assert width_paired > width_per_game
    # Correlazione perfetta => varianza di coppia = 4x quella per partita, e con n dimezzato
    # la larghezza cresce di circa sqrt(2) (a meno del fattore n-1 vs n).
    assert width_paired / width_per_game == pytest.approx(math.sqrt(2.0), rel=0.15)


def test_paired_score_rate_ci_center_is_score_rate() -> None:
    """
    Il centro della CI per coppia sullo score rate coincide con lo score rate per partita.

    Usiamo abbastanza coppie da evitare il clamp a [0,1] (che troncherebbe l'intervallo
    e sposterebbe il centro: comportamento corretto, ma non quello sotto test qui).
    """
    stats = _stats_from_pair_diffs([10, -6, 8, 4, -2, 12] * 4)
    ci = score_rate_interval_paired(
        wins=stats.wins_agent_a,
        losses=stats.wins_agent_b,
        draws=stats.draws,
        sum_sq_pair_score=stats.sum_sq_pair_score_agent_a,
    )
    assert ci is not None
    score_rate = (stats.wins_agent_a + 0.5 * stats.draws) / stats.num_games
    assert (ci.low + ci.high) / 2.0 == pytest.approx(score_rate, abs=1e-9)


def test_seat_fair_helpers_prefer_paired_and_fall_back() -> None:
    """Gli helper devono usare l'unità coppia se disponibile e ripiegare sul per-partita."""
    stats = _stats_from_pair_diffs([10, -6, 8, -2])
    paired = seat_fair_avg_point_diff_ci(stats)
    assert paired is not None
    expected = mean_point_diff_interval_paired(
        mean=stats.avg_point_diff_agent_a_minus_agent_b,
        num_games=stats.num_games,
        sum_sq_pair=stats.sum_sq_pair_point_diff_agent_a_minus_agent_b,
    )
    assert expected is not None
    assert paired.low == pytest.approx(expected.low)
    assert paired.high == pytest.approx(expected.high)

    # Senza i campi per coppia si torna al comportamento storico (documentato come fallback).
    legacy = SeatFairStats(
        num_games=stats.num_games,
        agent_a_name="A",
        agent_b_name="B",
        wins_agent_a=stats.wins_agent_a,
        wins_agent_b=stats.wins_agent_b,
        draws=stats.draws,
        avg_points_agent_a=stats.avg_points_agent_a,
        avg_points_agent_b=stats.avg_points_agent_b,
        avg_point_diff_agent_a_minus_agent_b=stats.avg_point_diff_agent_a_minus_agent_b,
        sum_sq_point_diff_agent_a_minus_agent_b=stats.sum_sq_point_diff_agent_a_minus_agent_b,
    )
    fallback = seat_fair_avg_point_diff_ci(legacy)
    per_game = mean_point_diff_interval(
        mean=legacy.avg_point_diff_agent_a_minus_agent_b,
        num_games=legacy.num_games,
        sum_sq=legacy.sum_sq_point_diff_agent_a_minus_agent_b,
    )
    assert fallback is not None and per_game is not None
    assert fallback.low == pytest.approx(per_game.low)
    # Il fallback score rate resta Wilson (nessuna somma per coppia disponibile).
    assert seat_fair_score_rate_ci(legacy) is not None


def test_domain_producer_tracks_pair_sums() -> None:
    """`evaluate_seat_fair_match_2p` deve popolare le somme per coppia con vincoli coerenti."""
    stats = evaluate_seat_fair_match_2p(_FirstCardAgent(), _FirstCardAgent(), num_games=20, seed=7)
    assert stats.sum_sq_pair_point_diff_agent_a_minus_agent_b is not None
    assert stats.sum_sq_pair_score_agent_a is not None
    assert stats.num_pairs == 10
    # Cauchy-Schwarz: sum(d_pair^2) >= (sum d_pair)^2 / n.
    sum_diff = stats.avg_point_diff_agent_a_minus_agent_b * stats.num_games
    assert stats.sum_sq_pair_point_diff_agent_a_minus_agent_b >= (sum_diff * sum_diff) / stats.num_pairs - 1e-9
    # Gli score per coppia sono in [0,1], quindi la somma dei quadrati è al massimo num_pairs.
    assert 0.0 <= stats.sum_sq_pair_score_agent_a <= stats.num_pairs


def test_fast_and_numba_producers_track_pair_sums() -> None:
    """Anche i path fast e numba devono popolare le somme per coppia."""
    fast_stats = evaluate_fast_seat_fair_match_2p("heuristic_v1", "random", num_games=8, seed=3)
    assert fast_stats.sum_sq_pair_point_diff_agent_a_minus_agent_b is not None
    assert fast_stats.sum_sq_pair_score_agent_a is not None

    numba_stats = evaluate_numba_seat_fair_match_2p("heuristic_v1", "random", num_games=8, seed=3)
    assert numba_stats.sum_sq_pair_point_diff_agent_a_minus_agent_b is not None
    assert numba_stats.sum_sq_pair_score_agent_a is not None


def test_relabel_and_invert_propagate_pair_fields() -> None:
    """relabel preserva i campi; invert due volte deve tornare ai valori originali."""
    stats = evaluate_seat_fair_match_2p(_FirstCardAgent(), _FirstCardAgent(), num_games=12, seed=11)

    relabeled = relabel_seat_fair_stats(stats, agent_a_name="X", agent_b_name="Y")
    assert relabeled.sum_sq_pair_point_diff_agent_a_minus_agent_b == stats.sum_sq_pair_point_diff_agent_a_minus_agent_b
    assert relabeled.sum_sq_pair_score_agent_a == stats.sum_sq_pair_score_agent_a

    inverted = invert_seat_fair_stats(stats, agent_a_name="B", agent_b_name="A")
    # I quadrati dei diff per coppia sono invarianti al segno.
    assert inverted.sum_sq_pair_point_diff_agent_a_minus_agent_b == stats.sum_sq_pair_point_diff_agent_a_minus_agent_b
    # Doppia inversione: si deve tornare esattamente alla somma dei quadrati originale.
    round_trip = invert_seat_fair_stats(inverted, agent_a_name="A", agent_b_name="B")
    assert round_trip.sum_sq_pair_score_agent_a == pytest.approx(stats.sum_sq_pair_score_agent_a)
    assert round_trip.wins_agent_a == stats.wins_agent_a
    assert round_trip.avg_point_diff_agent_a_minus_agent_b == pytest.approx(stats.avg_point_diff_agent_a_minus_agent_b)
