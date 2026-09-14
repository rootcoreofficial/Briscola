"""
DTO pubblici prodotti dai wrapper Numba.

Sono tenuti separati dai kernel JIT per chiarire il confine: i kernel lavorano su
array NumPy, i wrapper Python restituiscono oggetti leggibili per training/eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..evaluation import MatchStats, SeatFairStats


@dataclass(frozen=True, slots=True)
class NumbaMLPRolloutSummary:
    """Risultato aggregato del rollout full-JIT di una policy MLP."""

    num_games: int
    policy_name: str
    opponent_name: str
    wins_policy: int
    wins_opponent: int
    draws: int
    sum_policy: int
    sum_opponent: int
    sum_sq_point_diff_policy_minus_opponent: float | None = field(default=None, compare=False)
    # Somme per COPPIA seat-fair (unità statistica indipendente): disponibili solo quando il
    # rollout produce risultati per-partita accoppiabili (path parallelo seat-fair).
    sum_sq_pair_point_diff_policy_minus_opponent: float | None = field(default=None, compare=False)
    sum_sq_pair_score_policy: float | None = field(default=None, compare=False)

    def to_match_stats(self) -> MatchStats:
        """Converte il summary nel DTO statistico standard."""
        return MatchStats(
            num_games=self.num_games,
            agent0_name=self.policy_name,
            agent1_name=self.opponent_name,
            wins_agent0=self.wins_policy,
            wins_agent1=self.wins_opponent,
            draws=self.draws,
            avg_points_agent0=self.sum_policy / self.num_games if self.num_games else 0.0,
            avg_points_agent1=self.sum_opponent / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent0_minus_agent1=(
                (self.sum_policy - self.sum_opponent) / self.num_games if self.num_games else 0.0
            ),
        )

    def to_seat_fair_stats(self) -> SeatFairStats:
        """
        Converte il summary nel DTO seat-fair standard.

        Il core Numba aggrega sempre i risultati dal punto di vista della policy, indipendentemente dal seat
        con cui gioca. Per questo `policy` corrisponde ad Agent A e `opponent` ad Agent B.
        """
        return SeatFairStats(
            num_games=self.num_games,
            agent_a_name=self.policy_name,
            agent_b_name=self.opponent_name,
            wins_agent_a=self.wins_policy,
            wins_agent_b=self.wins_opponent,
            draws=self.draws,
            avg_points_agent_a=self.sum_policy / self.num_games if self.num_games else 0.0,
            avg_points_agent_b=self.sum_opponent / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent_a_minus_agent_b=(
                (self.sum_policy - self.sum_opponent) / self.num_games if self.num_games else 0.0
            ),
            sum_sq_point_diff_agent_a_minus_agent_b=self.sum_sq_point_diff_policy_minus_opponent,
            sum_sq_pair_point_diff_agent_a_minus_agent_b=self.sum_sq_pair_point_diff_policy_minus_opponent,
            sum_sq_pair_score_agent_a=self.sum_sq_pair_score_policy,
        )


@dataclass(frozen=True, slots=True)
class NumbaDecisionQualitySummary:
    """Risultato aggregato Numba per match seat-fair + metriche decision-quality della policy."""

    num_games: int
    policy_name: str
    opponent_name: str
    wins_policy: int
    wins_opponent: int
    draws: int
    sum_policy: int
    sum_opponent: int
    num_second_hand_decisions: int
    num_second_hand_with_winning_reply: int
    num_trump_waste: int
    num_second_hand_trump_wins: int
    num_trump_overkill: int
    num_second_hand_trump_wins_low_lead_points: int
    num_trump_overkill_low_lead_points: int
    sum_sq_point_diff_policy_minus_opponent: float | None = field(default=None, compare=False)
    sum_sq_pair_point_diff_policy_minus_opponent: float | None = field(default=None, compare=False)
    sum_sq_pair_score_policy: float | None = field(default=None, compare=False)

    def to_seat_fair_stats(self) -> SeatFairStats:
        """Converte match aggregato nel DTO seat-fair standard."""
        return SeatFairStats(
            num_games=self.num_games,
            agent_a_name=self.policy_name,
            agent_b_name=self.opponent_name,
            wins_agent_a=self.wins_policy,
            wins_agent_b=self.wins_opponent,
            draws=self.draws,
            avg_points_agent_a=self.sum_policy / self.num_games if self.num_games else 0.0,
            avg_points_agent_b=self.sum_opponent / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent_a_minus_agent_b=(
                (self.sum_policy - self.sum_opponent) / self.num_games if self.num_games else 0.0
            ),
            sum_sq_point_diff_agent_a_minus_agent_b=self.sum_sq_point_diff_policy_minus_opponent,
            sum_sq_pair_point_diff_agent_a_minus_agent_b=self.sum_sq_pair_point_diff_policy_minus_opponent,
            sum_sq_pair_score_agent_a=self.sum_sq_pair_score_policy,
        )


@dataclass(frozen=True, slots=True)
class NumbaA2CTrajectory:
    """Traiettoria A2C raccolta da una singola partita full-JIT."""

    policy_points: int
    opponent_points: int
    winner: int
    avg_entropy: float
    xs: np.ndarray
    z1s: np.ndarray
    hs: np.ndarray
    action_masks: np.ndarray
    probs: np.ndarray
    action_ids: np.ndarray
    value_preds: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True, slots=True)
class NumbaA2CBatch:
    """Batch di traiettorie A2C raccolte da Numba senza wrapper Python per partita."""

    policy_points: np.ndarray
    opponent_points: np.ndarray
    winners: np.ndarray
    step_counts: np.ndarray
    avg_entropies: np.ndarray
    xs: np.ndarray
    z1s: np.ndarray
    hs: np.ndarray
    action_masks: np.ndarray
    probs: np.ndarray
    action_ids: np.ndarray
    value_preds: np.ndarray
    rewards: np.ndarray
