"""
Round-robin offline per confrontare una popolazione di modelli/baseline.

Perche' esiste accanto alla `evaluation matrix`
----------------------------------------------
La matrix risponde alla domanda: "questo candidato quanto va contro una lista di avversari?".
Il round-robin risponde a una domanda diversa: "in una popolazione di agenti, chi e' robusto e
quali matchup deboli emergono?". Questo e' utile prima di definire una nuova ipotesi di training:
se il modello migliore ha un singolo avversario sfavorevole, conviene investigare quel matchup
prima di lanciare altro self-play.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Literal

import numpy as np

from ..agents import build_agent
from ..fast.evaluation import FAST_EVALUATION_AGENT_NAMES, evaluate_fast_seat_fair_match_2p
from ..models.bc_model import BCModelAgent
from .match import SeatFairStats, evaluate_seat_fair_match_2p
from .matrix import (
    BenchmarkName,
    EvaluationEngine,
    SuiteSpec,
    _evaluate_numba_model_vs_opponent,
    benchmark_num_games,
    build_suites_for_benchmark,
    make_range_seed_suite,
)

RoundRobinPlayerKind = Literal["model", "fast"]
RoundRobinSuiteSelection = Literal["standard", "holdout", "both"]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Intervallo di confidenza semplice e JSON-serializzabile."""

    low: float
    high: float
    confidence: float

    def to_json_dict(self) -> dict:
        """Ritorna una rappresentazione JSON-serializzabile."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoundRobinPlayer:
    """Un partecipante del round-robin: modello `.npz` oppure baseline fast-compatible."""

    name: str
    kind: RoundRobinPlayerKind
    model_path: str | None = None

    def __post_init__(self) -> None:
        """Valida la coerenza minima della spec."""
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("Il nome del player round-robin non puo' essere vuoto.")
        if self.kind == "model" and not self.model_path:
            raise ValueError(f"Il player modello {self.name!r} richiede `model_path`.")
        if self.kind == "fast" and self.model_path is not None:
            raise ValueError(f"Il player fast {self.name!r} non deve avere `model_path`.")
        if self.kind == "fast" and self.name not in FAST_EVALUATION_AGENT_NAMES:
            supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
            raise ValueError(f"Baseline fast non supportata: {self.name!r}. Supportate: {supported}.")

    def to_json_dict(self) -> dict:
        """Ritorna una rappresentazione JSON-serializzabile."""
        data = {"name": self.name, "kind": self.kind}
        if self.model_path is not None:
            data["model_path"] = self.model_path
        return data


@dataclass(frozen=True, slots=True)
class RoundRobinMatchup:
    """Risultato di una coppia normalizzato dal punto di vista `player_a` vs `player_b`."""

    suite: SuiteSpec
    player_a: str
    player_b: str
    stats: SeatFairStats

    @property
    def score_rate_a(self) -> float:
        """Score rate Elo-like di A: win=1, draw=0.5, loss=0."""
        return score_rate(
            wins=self.stats.wins_agent_a,
            losses=self.stats.wins_agent_b,
            draws=self.stats.draws,
        )

    def score_rate_a_ci(self, *, confidence: float) -> ConfidenceInterval:
        """CI sullo score rate di A (unità coppia quando disponibile, altrimenti Wilson per-partita)."""
        return seat_fair_score_rate_ci(self.stats, confidence=confidence)

    def avg_point_diff_ci(self, *, confidence: float) -> ConfidenceInterval | None:
        """CI sul margine medio A-B (unità coppia quando disponibile, altrimenti per-partita)."""
        return seat_fair_avg_point_diff_ci(self.stats, confidence=confidence)

    def to_json_dict(self, *, confidence: float = 0.95) -> dict:
        """Ritorna una rappresentazione JSON-serializzabile."""
        score_ci = self.score_rate_a_ci(confidence=confidence)
        avg_diff_ci = self.avg_point_diff_ci(confidence=confidence)
        return {
            "suite": asdict(self.suite),
            "player_a": self.player_a,
            "player_b": self.player_b,
            "score_rate_a": self.score_rate_a,
            "score_rate_a_ci": score_ci.to_json_dict(),
            "avg_point_diff_ci": avg_diff_ci.to_json_dict() if avg_diff_ci is not None else None,
            "stats": asdict(self.stats),
        }


@dataclass(frozen=True, slots=True)
class RoundRobinRating:
    """Sintesi per player: rating, score aggregato e peggior matchup osservato."""

    player: str
    elo: float
    score_rate: float
    score_rate_ci_low: float
    score_rate_ci_high: float
    avg_point_diff: float
    avg_point_diff_ci_low: float | None
    avg_point_diff_ci_high: float | None
    worst_opponent: str | None
    worst_avg_point_diff: float | None

    def to_json_dict(self) -> dict:
        """Ritorna una rappresentazione JSON-serializzabile."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoundRobinResult:
    """Risultato completo del round-robin."""

    players: list[RoundRobinPlayer]
    benchmark: BenchmarkName
    num_games: int
    seed: int
    engine: EvaluationEngine
    confidence: float
    suites: list[SuiteSpec]
    matchups: list[RoundRobinMatchup]
    ratings: list[RoundRobinRating]
    non_transitive_cycles: list[list[str]]

    def to_json_dict(self) -> dict:
        """Ritorna una struttura JSON stabile per persistere il risultato."""
        return {
            "players": [p.to_json_dict() for p in self.players],
            "benchmark": self.benchmark,
            "num_games": self.num_games,
            "seed": self.seed,
            "engine": self.engine,
            "confidence": self.confidence,
            "suites": [asdict(s) for s in self.suites],
            "ratings": [r.to_json_dict() for r in self.ratings],
            "non_transitive_cycles": self.non_transitive_cycles,
            "matchups": [m.to_json_dict(confidence=self.confidence) for m in self.matchups],
        }

    def to_json_text(self) -> str:
        """JSON pretty-printed."""
        return json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def default_round_robin_players(models_dir: str | Path = "data/models") -> list[RoundRobinPlayer]:
    """
    Popolazione minima consigliata per la pre-validazione di nuovi candidati.

    `best_a2c.npz` e' trattato come ancora legacy v2 in valutazione; i modelli storici encoder v3
    restano invece v3/v4/v5/v6/v7. Il training multi-opponent non deve usare il legacy v2.
    """
    root = Path(models_dir)
    return [
        RoundRobinPlayer("best_a2c_v2", "model", str(root / "best_a2c.npz")),
        RoundRobinPlayer("best_a2c_v3", "model", str(root / "best_a2c_v3.npz")),
        RoundRobinPlayer("best_a2c_v4", "model", str(root / "best_a2c_v4.npz")),
        RoundRobinPlayer("best_a2c_v5", "model", str(root / "best_a2c_v5.npz")),
        RoundRobinPlayer("best_a2c_v6", "model", str(root / "best_a2c_v6.npz")),
        RoundRobinPlayer("best_a2c_v7", "model", str(root / "best_a2c_v7.npz")),
        RoundRobinPlayer("best_a2c_v8", "model", str(root / "best_a2c_v8.npz")),
        RoundRobinPlayer("best_a2c_v9", "model", str(root / "best_a2c_v9.npz")),
        RoundRobinPlayer("best_a2c_v10", "model", str(root / "best_a2c_v10.npz")),
        RoundRobinPlayer("heuristic_v1", "fast"),
    ]


def build_round_robin_suites(
    *,
    benchmark: BenchmarkName,
    suite: RoundRobinSuiteSelection,
    standard_start: int = 0,
    holdout_start: int = 1_000_000,
    range_step: int = 1,
) -> list[SuiteSpec]:
    """Costruisce e filtra le suite seed per il round-robin."""
    suites = build_suites_for_benchmark(
        benchmark=benchmark,
        standard_start=standard_start,
        holdout_start=holdout_start,
        step=range_step,
    )
    if suite == "both":
        return suites
    return [s for s in suites if s.name == suite]


def score_rate(*, wins: int, losses: int, draws: int) -> float:
    """Calcola score rate con pareggi a mezzo punto."""
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return 0.0
    return (int(wins) + 0.5 * int(draws)) / total


def _z_for_confidence(confidence: float) -> float:
    """Valore z two-sided per una confidence in (0, 1)."""
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence deve essere in (0, 1), ottenuto {confidence}")
    return float(NormalDist().inv_cdf(0.5 + confidence / 2.0))


def wilson_score_interval(*, wins: int, losses: int, draws: int, confidence: float = 0.95) -> ConfidenceInterval:
    """
    CI Wilson approssimata sullo score rate win/draw/loss.

    I pareggi valgono mezzo punto, quindi il numero di "successi" puo' essere frazionario. La formula resta una
    buona approssimazione operativa per gateare matchup stretti senza raccogliere risultati per-partita.
    """
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return ConfidenceInterval(low=0.0, high=0.0, confidence=float(confidence))

    z = _z_for_confidence(confidence)
    z2 = z * z
    p_hat = score_rate(wins=wins, losses=losses, draws=draws)
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half_width = z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total))) / denom
    return ConfidenceInterval(
        low=max(0.0, center - half_width),
        high=min(1.0, center + half_width),
        confidence=float(confidence),
    )


def mean_point_diff_interval(
    *,
    mean: float,
    num_games: int,
    sum_sq: float | None,
    confidence: float = 0.95,
) -> ConfidenceInterval | None:
    """
    CI normale sul margine medio punti A-B trattando le PARTITE come indipendenti.

    Attenzione: nelle valutazioni seat-fair le due partite di una coppia condividono il
    mazzo e sono correlate, quindi questa CI è anti-conservativa (troppo stretta).
    Usare `mean_point_diff_interval_paired` quando i dati per coppia sono disponibili;
    questa resta come fallback per i path che producono solo aggregati per partita.
    """
    if sum_sq is None or num_games <= 1:
        return None
    z = _z_for_confidence(confidence)
    n = int(num_games)
    sum_diff = float(mean) * n
    numerator = float(sum_sq) - (sum_diff * sum_diff / n)
    variance = max(0.0, numerator / (n - 1))
    half_width = z * math.sqrt(variance / n)
    return ConfidenceInterval(low=float(mean) - half_width, high=float(mean) + half_width, confidence=float(confidence))


def mean_point_diff_interval_paired(
    *,
    mean: float,
    num_games: int,
    sum_sq_pair: float | None,
    confidence: float = 0.95,
) -> ConfidenceInterval | None:
    """
    CI sul margine medio punti A-B PER PARTITA, con la coppia seat-fair come unità indipendente.

    Derivazione:
    - `d_pair = d_game1 + d_game2` (stesso mazzo, seat scambiati) è l'osservazione indipendente;
    - la media dei `d_pair` è `2 * mean` (perché `num_games = 2 * num_pairs`);
    - la CI viene calcolata su `mean(d_pair)` con `n = num_pairs` e poi divisa per 2 per
      riportarla alla scala per-partita usata in tutti i report.

    Rispetto alla versione per-partita, questa CI è più larga quando le due partite della
    coppia sono correlate positivamente (il caso tipico con agenti deterministici): è la
    larghezza corretta, l'altra sovrastima il campione effettivo.
    """
    if sum_sq_pair is None or num_games < 4 or num_games % 2 != 0:
        return None
    num_pairs = int(num_games) // 2
    z = _z_for_confidence(confidence)
    sum_pair_diff = float(mean) * num_games  # la somma dei diff per partita coincide con quella per coppia
    numerator = float(sum_sq_pair) - (sum_pair_diff * sum_pair_diff / num_pairs)
    variance_pair = max(0.0, numerator / (num_pairs - 1))
    half_width = z * math.sqrt(variance_pair / num_pairs) / 2.0
    return ConfidenceInterval(low=float(mean) - half_width, high=float(mean) + half_width, confidence=float(confidence))


def score_rate_interval_paired(
    *,
    wins: int,
    losses: int,
    draws: int,
    sum_sq_pair_score: float | None,
    confidence: float = 0.95,
) -> ConfidenceInterval | None:
    """
    CI normale sullo score rate con la coppia seat-fair come unità indipendente.

    Lo score per coppia è `s = (score_g1 + score_g2) / 2` con score 1/0.5/0: la sua media
    coincide con lo score rate per partita, quindi il centro non cambia; cambia (correttamente)
    la larghezza, calcolata su `n = num_pairs` osservazioni indipendenti.
    """
    total_games = int(wins) + int(losses) + int(draws)
    if sum_sq_pair_score is None or total_games < 4 or total_games % 2 != 0:
        return None
    num_pairs = total_games // 2
    z = _z_for_confidence(confidence)
    sum_pair_scores = (float(wins) + 0.5 * float(draws)) / 2.0
    mean = sum_pair_scores / num_pairs
    numerator = float(sum_sq_pair_score) - (sum_pair_scores * sum_pair_scores / num_pairs)
    variance_pair = max(0.0, numerator / (num_pairs - 1))
    half_width = z * math.sqrt(variance_pair / num_pairs)
    return ConfidenceInterval(
        low=max(0.0, mean - half_width),
        high=min(1.0, mean + half_width),
        confidence=float(confidence),
    )


def seat_fair_avg_point_diff_ci(stats: SeatFairStats, *, confidence: float = 0.95) -> ConfidenceInterval | None:
    """
    CI sul margine medio A-B a partire da uno `SeatFairStats`, preferendo l'unità coppia.

    È l'entry point consigliato per script e report: usa la CI corretta (per coppia) quando
    il produttore ha tracciato `sum_sq_pair_*`, altrimenti ripiega sulla CI per-partita
    (anti-conservativa, documentata come tale).
    """
    paired = mean_point_diff_interval_paired(
        mean=stats.avg_point_diff_agent_a_minus_agent_b,
        num_games=stats.num_games,
        sum_sq_pair=stats.sum_sq_pair_point_diff_agent_a_minus_agent_b,
        confidence=confidence,
    )
    if paired is not None:
        return paired
    return mean_point_diff_interval(
        mean=stats.avg_point_diff_agent_a_minus_agent_b,
        num_games=stats.num_games,
        sum_sq=stats.sum_sq_point_diff_agent_a_minus_agent_b,
        confidence=confidence,
    )


def seat_fair_score_rate_ci(stats: SeatFairStats, *, confidence: float = 0.95) -> ConfidenceInterval:
    """
    CI sullo score rate di A a partire da uno `SeatFairStats`, preferendo l'unità coppia.

    Fallback: Wilson per-partita (approssimazione anti-conservativa nelle valutazioni seat-fair).
    """
    paired = score_rate_interval_paired(
        wins=stats.wins_agent_a,
        losses=stats.wins_agent_b,
        draws=stats.draws,
        sum_sq_pair_score=stats.sum_sq_pair_score_agent_a,
        confidence=confidence,
    )
    if paired is not None:
        return paired
    return wilson_score_interval(
        wins=stats.wins_agent_a,
        losses=stats.wins_agent_b,
        draws=stats.draws,
        confidence=confidence,
    )


def relabel_seat_fair_stats(stats: SeatFairStats, *, agent_a_name: str, agent_b_name: str) -> SeatFairStats:
    """Mantiene i numeri ma assegna label coerenti con la spec round-robin."""
    return SeatFairStats(
        num_games=stats.num_games,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        wins_agent_a=stats.wins_agent_a,
        wins_agent_b=stats.wins_agent_b,
        draws=stats.draws,
        avg_points_agent_a=stats.avg_points_agent_a,
        avg_points_agent_b=stats.avg_points_agent_b,
        avg_point_diff_agent_a_minus_agent_b=stats.avg_point_diff_agent_a_minus_agent_b,
        sum_sq_point_diff_agent_a_minus_agent_b=stats.sum_sq_point_diff_agent_a_minus_agent_b,
        sum_sq_pair_point_diff_agent_a_minus_agent_b=stats.sum_sq_pair_point_diff_agent_a_minus_agent_b,
        sum_sq_pair_score_agent_a=stats.sum_sq_pair_score_agent_a,
    )


def _inverted_sum_sq_pair_score(stats: SeatFairStats) -> float | None:
    """
    Somma dei quadrati degli score per coppia dal punto di vista dell'agente B.

    Lo score per coppia di B è `1 - s` (dove `s` è quello di A), quindi:
    `sum((1-s)^2) = num_pairs - 2*sum(s) + sum(s^2)`, con `sum(s)` derivabile dagli aggregati.
    """
    if stats.sum_sq_pair_score_agent_a is None or stats.num_games % 2 != 0:
        return None
    num_pairs = stats.num_games // 2
    sum_scores_a = (float(stats.wins_agent_a) + 0.5 * float(stats.draws)) / 2.0
    return float(num_pairs) - 2.0 * sum_scores_a + float(stats.sum_sq_pair_score_agent_a)


def invert_seat_fair_stats(stats: SeatFairStats, *, agent_a_name: str, agent_b_name: str) -> SeatFairStats:
    """
    Inverte una valutazione B-vs-A per riportarla nella prospettiva richiesta A-vs-B.

    Serve quando l'engine Numba richiede che almeno uno dei due lati sia un modello MLP e quindi
    conviene valutare `model vs baseline` anche se, nella matrice round-robin, la coppia era ordinata
    come `baseline vs model`.
    """
    return SeatFairStats(
        num_games=stats.num_games,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        wins_agent_a=stats.wins_agent_b,
        wins_agent_b=stats.wins_agent_a,
        draws=stats.draws,
        avg_points_agent_a=stats.avg_points_agent_b,
        avg_points_agent_b=stats.avg_points_agent_a,
        avg_point_diff_agent_a_minus_agent_b=-stats.avg_point_diff_agent_a_minus_agent_b,
        # I quadrati dei diff (per partita e per coppia) sono invarianti al cambio di segno.
        sum_sq_point_diff_agent_a_minus_agent_b=stats.sum_sq_point_diff_agent_a_minus_agent_b,
        sum_sq_pair_point_diff_agent_a_minus_agent_b=stats.sum_sq_pair_point_diff_agent_a_minus_agent_b,
        sum_sq_pair_score_agent_a=_inverted_sum_sq_pair_score(stats),
    )


def compute_elo_ratings(
    *,
    players: list[RoundRobinPlayer],
    matchups: list[RoundRobinMatchup],
    base_rating: float = 1500.0,
) -> dict[str, float]:
    """
    Stima rating Elo-like dai risultati aggregati usando differenze Bradley-Terry.

    Per ogni coppia convertiamo lo score rate empirico in differenza Elo:
    `diff = 400 * log10(p / (1-p))`, con clipping per evitare infinito su sweep 100%.
    Poi risolviamo un least-squares con vincolo di media `base_rating`.
    """
    names = [p.name for p in players]
    if not names:
        return {}
    if len(names) == 1 or not matchups:
        return {name: float(base_rating) for name in names}

    index = {name: i for i, name in enumerate(names)}
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for matchup in matchups:
        if matchup.player_a not in index or matchup.player_b not in index:
            continue
        p = min(0.99, max(0.01, matchup.score_rate_a))
        empirical_diff = 400.0 * math.log10(p / (1.0 - p))
        # Il peso cresce con i game ma resta contenuto: tutte le suite standard pesano uguale,
        # eventuali run piu' piccole non dominano artificiosamente il fit.
        weight = math.sqrt(max(1, matchup.stats.num_games))
        row = np.zeros(len(names), dtype=np.float64)
        row[index[matchup.player_a]] = weight
        row[index[matchup.player_b]] = -weight
        rows.append(row)
        targets.append(empirical_diff * weight)

    if not rows:
        return {name: float(base_rating) for name in names}

    mean_constraint = np.ones(len(names), dtype=np.float64)
    rows.append(mean_constraint)
    targets.append(float(base_rating) * len(names))

    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(targets, dtype=np.float64), rcond=None)
    return {name: float(solution[index[name]]) for name in names}


def summarize_player_ratings(
    *,
    players: list[RoundRobinPlayer],
    matchups: list[RoundRobinMatchup],
    elo_by_player: dict[str, float],
    confidence: float = 0.95,
) -> list[RoundRobinRating]:
    """Aggrega score, margine medio e peggior matchup per ogni player."""
    ratings: list[RoundRobinRating] = []
    for player in players:
        wins = losses = draws = games = 0
        point_diff_sum = 0.0
        point_diff_sum_sq: float | None = 0.0
        worst_opponent: str | None = None
        worst_avg_diff: float | None = None

        for matchup in matchups:
            if player.name == matchup.player_a:
                wins += matchup.stats.wins_agent_a
                losses += matchup.stats.wins_agent_b
                draws += matchup.stats.draws
                games += matchup.stats.num_games
                diff = float(matchup.stats.avg_point_diff_agent_a_minus_agent_b)
                opponent = matchup.player_b
                sum_sq = matchup.stats.sum_sq_point_diff_agent_a_minus_agent_b
            elif player.name == matchup.player_b:
                wins += matchup.stats.wins_agent_b
                losses += matchup.stats.wins_agent_a
                draws += matchup.stats.draws
                games += matchup.stats.num_games
                diff = -float(matchup.stats.avg_point_diff_agent_a_minus_agent_b)
                opponent = matchup.player_a
                sum_sq = matchup.stats.sum_sq_point_diff_agent_a_minus_agent_b
            else:
                continue

            point_diff_sum += diff * matchup.stats.num_games
            if point_diff_sum_sq is not None:
                if sum_sq is None:
                    point_diff_sum_sq = None
                else:
                    point_diff_sum_sq += float(sum_sq)
            if worst_avg_diff is None or diff < worst_avg_diff:
                worst_avg_diff = diff
                worst_opponent = opponent

        player_score_rate = score_rate(wins=wins, losses=losses, draws=draws)
        score_ci = wilson_score_interval(wins=wins, losses=losses, draws=draws, confidence=confidence)
        avg_diff = point_diff_sum / games if games else 0.0
        avg_diff_ci = mean_point_diff_interval(
            mean=avg_diff,
            num_games=games,
            sum_sq=point_diff_sum_sq,
            confidence=confidence,
        )
        ratings.append(
            RoundRobinRating(
                player=player.name,
                elo=float(elo_by_player.get(player.name, 1500.0)),
                score_rate=player_score_rate,
                score_rate_ci_low=score_ci.low,
                score_rate_ci_high=score_ci.high,
                avg_point_diff=avg_diff,
                avg_point_diff_ci_low=avg_diff_ci.low if avg_diff_ci is not None else None,
                avg_point_diff_ci_high=avg_diff_ci.high if avg_diff_ci is not None else None,
                worst_opponent=worst_opponent,
                worst_avg_point_diff=worst_avg_diff,
            )
        )

    return sorted(ratings, key=lambda r: (r.elo, r.avg_point_diff, r.score_rate), reverse=True)


def find_non_transitive_cycles(
    *,
    players: list[RoundRobinPlayer],
    matchups: list[RoundRobinMatchup],
    score_threshold: float = 0.5,
    confidence: float = 0.95,
    require_confident_edge: bool = True,
) -> list[list[str]]:
    """
    Cerca cicli A>B, B>C, C>A nello score rate aggregato.

    Per default un arco entra nel grafo solo se la CI Wilson dello score rate sta interamente oltre la soglia.
    Questo evita che matchup al fotofinish (es. 0.503) vengano trattati come vittorie nette nei cicli.
    """
    names = [p.name for p in players]
    beats: set[tuple[str, str]] = set()
    for matchup in matchups:
        score_ci = matchup.score_rate_a_ci(confidence=confidence)
        a_confident = (not require_confident_edge) or score_ci.low > score_threshold
        b_confident = (not require_confident_edge) or score_ci.high < score_threshold
        if matchup.score_rate_a > score_threshold and a_confident:
            beats.add((matchup.player_a, matchup.player_b))
        elif (1.0 - matchup.score_rate_a) > score_threshold and b_confident:
            beats.add((matchup.player_b, matchup.player_a))

    cycles: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()
    for a in names:
        for b in names:
            if a == b or (a, b) not in beats:
                continue
            for c in names:
                if c in (a, b):
                    continue
                if (b, c) in beats and (c, a) in beats:
                    canonical = min(
                        (a, b, c),
                        (b, c, a),
                        (c, a, b),
                    )
                    if canonical not in seen:
                        seen.add(canonical)
                        cycles.append([a, b, c, a])
    return cycles


def validate_round_robin_players(players: list[RoundRobinPlayer]) -> None:
    """Fallisce presto se nomi/path non sono utilizzabili."""
    if len(players) < 2:
        raise ValueError("Il round-robin richiede almeno due player.")
    names = [p.name for p in players]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Nomi player duplicati: {', '.join(duplicates)}")
    for player in players:
        if player.kind == "model":
            assert player.model_path is not None
            path = Path(player.model_path)
            if not path.is_file():
                raise FileNotFoundError(f"Modello non trovato per {player.name!r}: {path}")


def _load_model(player: RoundRobinPlayer, model_cache: dict[str, BCModelAgent]) -> BCModelAgent:
    """Carica un modello una sola volta per path."""
    if player.kind != "model" or player.model_path is None:
        raise ValueError(f"Il player {player.name!r} non e' un modello.")
    key = str(Path(player.model_path))
    if key not in model_cache:
        model_cache[key] = BCModelAgent.from_npz(key)
    return model_cache[key]


def _build_domain_agent(player: RoundRobinPlayer, model_cache: dict[str, BCModelAgent]):
    """Costruisce l'agente per il path dominio."""
    if player.kind == "model":
        return _load_model(player, model_cache)
    return build_agent(player.name)


def _evaluate_pair(
    *,
    player_a: RoundRobinPlayer,
    player_b: RoundRobinPlayer,
    engine: EvaluationEngine,
    num_games: int,
    seed: int,
    game_seeds: list[int],
    model_cache: dict[str, BCModelAgent],
) -> SeatFairStats:
    """Valuta una coppia e normalizza sempre il risultato come A-vs-B."""
    if engine == "domain":
        stats = evaluate_seat_fair_match_2p(
            _build_domain_agent(player_a, model_cache),
            _build_domain_agent(player_b, model_cache),
            num_games=num_games,
            seed=seed,
            game_seeds=game_seeds,
        )
        return relabel_seat_fair_stats(stats, agent_a_name=player_a.name, agent_b_name=player_b.name)

    if engine != "numba":
        raise ValueError(f"engine non supportato: {engine!r}")

    if player_a.kind == "fast" and player_b.kind == "fast":
        stats = evaluate_fast_seat_fair_match_2p(
            player_a.name,
            player_b.name,
            num_games=num_games,
            seed=seed,
            game_seeds=game_seeds,
        )
        return relabel_seat_fair_stats(stats, agent_a_name=player_a.name, agent_b_name=player_b.name)

    if player_a.kind == "model":
        model = _load_model(player_a, model_cache)
        opponent_agent = _load_model(player_b, model_cache) if player_b.kind == "model" else None
        opponent_name = "bc_model" if opponent_agent is not None else player_b.name
        stats = _evaluate_numba_model_vs_opponent(
            model_agent=model,
            opponent_name=opponent_name,
            opponent_agent=opponent_agent,
            num_games=num_games,
            seed=seed,
            game_seeds=game_seeds,
        )
        return relabel_seat_fair_stats(stats, agent_a_name=player_a.name, agent_b_name=player_b.name)

    # A e' baseline, B e' modello: valutiamo B-vs-A per rispettare il wrapper Numba e poi invertiamo.
    model_b = _load_model(player_b, model_cache)
    stats = _evaluate_numba_model_vs_opponent(
        model_agent=model_b,
        opponent_name=player_a.name,
        opponent_agent=None,
        num_games=num_games,
        seed=seed,
        game_seeds=game_seeds,
    )
    return invert_seat_fair_stats(stats, agent_a_name=player_a.name, agent_b_name=player_b.name)


def evaluate_round_robin(
    *,
    players: list[RoundRobinPlayer],
    benchmark: BenchmarkName,
    seed: int,
    suite: RoundRobinSuiteSelection = "standard",
    engine: EvaluationEngine = "numba",
    confidence: float = 0.95,
    standard_start: int = 0,
    holdout_start: int = 1_000_000,
    range_step: int = 1,
) -> RoundRobinResult:
    """
    Valuta tutte le coppie di una popolazione.

    Ogni matchup e' seat-fair e usa la stessa seed suite, cosi' le differenze tra coppie non dipendono
    da shuffle diversi del mazzo.
    """
    validate_round_robin_players(players)
    num_games = benchmark_num_games(benchmark)
    suites = build_round_robin_suites(
        benchmark=benchmark,
        suite=suite,
        standard_start=standard_start,
        holdout_start=holdout_start,
        range_step=range_step,
    )
    if not suites:
        raise ValueError(f"Nessuna suite selezionata: {suite!r}")

    model_cache: dict[str, BCModelAgent] = {}
    matchups: list[RoundRobinMatchup] = []
    for suite_spec in suites:
        game_seeds = make_range_seed_suite(
            start=suite_spec.range_start,
            step=suite_spec.range_step,
            count=suite_spec.num_seeds,
        )
        for i, player_a in enumerate(players):
            for player_b in players[i + 1 :]:
                stats = _evaluate_pair(
                    player_a=player_a,
                    player_b=player_b,
                    engine=engine,
                    num_games=num_games,
                    seed=seed,
                    game_seeds=game_seeds,
                    model_cache=model_cache,
                )
                matchups.append(
                    RoundRobinMatchup(
                        suite=suite_spec,
                        player_a=player_a.name,
                        player_b=player_b.name,
                        stats=stats,
                    )
                )

    elo_by_player = compute_elo_ratings(players=players, matchups=matchups)
    ratings = summarize_player_ratings(
        players=players,
        matchups=matchups,
        elo_by_player=elo_by_player,
        confidence=confidence,
    )
    cycles = find_non_transitive_cycles(players=players, matchups=matchups, confidence=confidence)

    return RoundRobinResult(
        players=players,
        benchmark=benchmark,
        num_games=num_games,
        seed=int(seed),
        engine=engine,
        confidence=float(confidence),
        suites=suites,
        matchups=matchups,
        ratings=ratings,
        non_transitive_cycles=cycles,
    )


def format_round_robin_table(result: RoundRobinResult) -> str:
    """Tabella testuale compatta per terminale e log."""
    suite_names = ",".join(s.name for s in result.suites)

    def fmt_ci(low: float, high: float, *, signed: bool = False) -> str:
        """Formato compatto senza virgole interne, così la tabella resta CSV-like."""
        fmt = "{:+.2f}" if signed else "{:.4f}"
        return f"{fmt.format(low)}..{fmt.format(high)}"

    lines = [
        "Round-robin | "
        f"engine={result.engine} | benchmark={result.benchmark} | games={result.num_games} | "
        f"suites={suite_names} | ci={result.confidence:.2f}",
        "RATINGS",
        "player,elo,score_rate,score_ci,avg_diff,avg_diff_ci,worst_opponent,worst_avg_diff",
    ]
    for rating in result.ratings:
        worst = "" if rating.worst_opponent is None else rating.worst_opponent
        worst_diff = "" if rating.worst_avg_point_diff is None else f"{rating.worst_avg_point_diff:+.2f}"
        avg_diff_ci = (
            ""
            if rating.avg_point_diff_ci_low is None or rating.avg_point_diff_ci_high is None
            else fmt_ci(rating.avg_point_diff_ci_low, rating.avg_point_diff_ci_high, signed=True)
        )
        lines.append(
            f"{rating.player},{rating.elo:.1f},{rating.score_rate:.4f},"
            f"{fmt_ci(rating.score_rate_ci_low, rating.score_rate_ci_high)},"
            f"{rating.avg_point_diff:+.2f},{avg_diff_ci},{worst},{worst_diff}"
        )

    lines.extend(
        [
            "",
            "MATCHUPS",
            "suite,player_a,player_b,score_a,score_ci_a,avg_diff,avg_diff_ci,wins_a,wins_b,draws",
        ]
    )
    for matchup in result.matchups:
        stats = matchup.stats
        score_ci = matchup.score_rate_a_ci(confidence=result.confidence)
        matchup_avg_diff_ci = matchup.avg_point_diff_ci(confidence=result.confidence)
        avg_diff_ci_text = (
            ""
            if matchup_avg_diff_ci is None
            else fmt_ci(matchup_avg_diff_ci.low, matchup_avg_diff_ci.high, signed=True)
        )
        lines.append(
            f"{matchup.suite.name},{matchup.player_a},{matchup.player_b},{matchup.score_rate_a:.4f},"
            f"{fmt_ci(score_ci.low, score_ci.high)},"
            f"{stats.avg_point_diff_agent_a_minus_agent_b:+.2f},"
            f"{avg_diff_ci_text},"
            f"{stats.wins_agent_a},{stats.wins_agent_b},{stats.draws}"
        )

    if result.non_transitive_cycles:
        lines.extend(["", "NON_TRANSITIVE_CYCLES"])
        lines.extend(" > ".join(cycle) for cycle in result.non_transitive_cycles)

    return "\n".join(lines) + "\n"
