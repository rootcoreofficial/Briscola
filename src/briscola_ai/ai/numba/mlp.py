"""
Wrapper Python per rollout/evaluation MLP su kernel Numba.

Il modulo `observation` contiene encoder e kernel JIT. Qui restano validazione,
normalizzazione tensori e conversione dei risultati in DTO: codice Python di bordo,
piu' leggibile e piu' vicino a training/evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4
from .core import ACTION_DIM, NUMBA_AGENT_CODE_MAX, numba_agent_code
from .observation import (
    _collect_mlp_policy_batch_numba,
    _collect_mlp_policy_game_numba,
    _evaluate_mlp_policy_numba,
    _evaluate_mlp_policy_numba_parallel_plain,
    _evaluate_mlp_policy_numba_parallel_seat_fair,
    _evaluate_mlp_policy_quality_numba,
    _evaluate_mlp_policy_quality_numba_parallel,
)
from .types import NumbaA2CBatch, NumbaA2CTrajectory, NumbaDecisionQualitySummary, NumbaMLPRolloutSummary
from .value_lookahead import _prepare_policy_belief_arrays


def _as_float32_matrix(name: str, value: np.ndarray) -> np.ndarray:
    """Normalizza un peso 2D a float32 e fallisce con errore leggibile se la shape è sbagliata."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} deve essere una matrice 2D, ottenuto shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _as_float32_vector(name: str, value: np.ndarray) -> np.ndarray:
    """Normalizza un bias 1D a float32 e fallisce con errore leggibile se la shape è sbagliata."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} deve essere un vettore 1D, ottenuto shape={arr.shape}")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True, slots=True)
class _PreparedA2CNumbaInputs:
    """Tensori validati per i wrapper A2C Numba."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    wv: np.ndarray
    opponent_code: int
    opponent_model_enabled: bool
    opponent_w1: np.ndarray
    opponent_b1: np.ndarray
    opponent_w2: np.ndarray
    opponent_b2: np.ndarray


def _prepare_a2c_numba_inputs(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    opponent_name: str,
    opponent_w1: np.ndarray | None,
    opponent_b1: np.ndarray | None,
    opponent_w2: np.ndarray | None,
    opponent_b2: np.ndarray | None,
) -> _PreparedA2CNumbaInputs:
    """Valida e normalizza i tensori condivisi dai wrapper A2C Numba."""
    w1_arr = _as_float32_matrix("w1", w1)
    b1_arr = _as_float32_vector("b1", b1)
    w2_arr = _as_float32_matrix("w2", w2)
    b2_arr = _as_float32_vector("b2", b2)
    wv_arr = _as_float32_vector("wv", wv)

    feature_dim = int(w1_arr.shape[0])
    hidden_dim = int(w1_arr.shape[1])
    if feature_dim not in (
        int(FEATURE_DIM_2P_V1),
        int(FEATURE_DIM_2P_V2),
        int(FEATURE_DIM_2P_V3),
        int(FEATURE_DIM_2P_V4),
        int(FEATURE_DIM_2P_V4) + 40,  # iterazione 1b: input v4 + 40 probabilita' belief
    ):
        raise ValueError(
            f"w1 feature_dim={feature_dim}; "
            f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
            f"{int(FEATURE_DIM_2P_V3)}, {int(FEATURE_DIM_2P_V4)} o {int(FEATURE_DIM_2P_V4) + 40}"
        )
    if b1_arr.shape != (hidden_dim,):
        raise ValueError(f"b1 shape={b1_arr.shape}; atteso {(hidden_dim,)}")
    if w2_arr.shape != (hidden_dim, ACTION_DIM):
        raise ValueError(f"w2 shape={w2_arr.shape}; atteso {(hidden_dim, ACTION_DIM)}")
    if b2_arr.shape != (ACTION_DIM,):
        raise ValueError(f"b2 shape={b2_arr.shape}; atteso {(ACTION_DIM,)}")
    if wv_arr.shape != (hidden_dim,):
        raise ValueError(f"wv shape={wv_arr.shape}; atteso {(hidden_dim,)}")

    opponent_model_enabled = (
        opponent_w1 is not None or opponent_b1 is not None or opponent_w2 is not None or opponent_b2 is not None
    )
    if opponent_model_enabled:
        if opponent_w1 is None or opponent_b1 is None or opponent_w2 is None or opponent_b2 is None:
            raise ValueError("Opponent model incompleto: servono opponent_w1/b1/w2/b2.")
        opponent_w1_arr = _as_float32_matrix("opponent_w1", opponent_w1)
        opponent_b1_arr = _as_float32_vector("opponent_b1", opponent_b1)
        opponent_w2_arr = _as_float32_matrix("opponent_w2", opponent_w2)
        opponent_b2_arr = _as_float32_vector("opponent_b2", opponent_b2)
        opp_feature_dim = int(opponent_w1_arr.shape[0])
        opp_hidden_dim = int(opponent_w1_arr.shape[1])
        if opp_feature_dim not in (
            int(FEATURE_DIM_2P_V1),
            int(FEATURE_DIM_2P_V2),
            int(FEATURE_DIM_2P_V3),
            int(FEATURE_DIM_2P_V4),
        ):
            raise ValueError(
                f"opponent_w1 feature_dim={opp_feature_dim}; "
                f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
                f"{int(FEATURE_DIM_2P_V3)} o {int(FEATURE_DIM_2P_V4)}"
            )
        if opponent_b1_arr.shape != (opp_hidden_dim,):
            raise ValueError(f"opponent_b1 shape={opponent_b1_arr.shape}; atteso {(opp_hidden_dim,)}")
        if opponent_w2_arr.shape != (opp_hidden_dim, ACTION_DIM):
            raise ValueError(f"opponent_w2 shape={opponent_w2_arr.shape}; atteso {(opp_hidden_dim, ACTION_DIM)}")
        if opponent_b2_arr.shape != (ACTION_DIM,):
            raise ValueError(f"opponent_b2 shape={opponent_b2_arr.shape}; atteso {(ACTION_DIM,)}")
        opponent_code = 0
    else:
        opponent_w1_arr = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
        opponent_b1_arr = np.zeros((1,), dtype=np.float32)
        opponent_w2_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)
        opponent_b2_arr = np.zeros((ACTION_DIM,), dtype=np.float32)
        opponent_code = numba_agent_code(opponent_name)

    return _PreparedA2CNumbaInputs(
        w1=w1_arr,
        b1=b1_arr,
        w2=w2_arr,
        b2=b2_arr,
        wv=wv_arr,
        opponent_code=int(opponent_code),
        opponent_model_enabled=bool(opponent_model_enabled),
        opponent_w1=opponent_w1_arr,
        opponent_b1=opponent_b1_arr,
        opponent_w2=opponent_w2_arr,
        opponent_b2=opponent_b2_arr,
    )


def _overkill_penalty_mode_code(mode: str) -> int:
    """Codifica la modalità reward shaping anti-overkill per il core JIT."""
    normalized = str(mode).strip().lower()
    if normalized == "flat":
        return 0
    if normalized == "gap":
        return 1
    raise ValueError(f"overkill_penalty_mode non supportato: {mode!r}")


def evaluate_mlp_policy_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_name: str,
    num_games: int,
    seed: int,
    seat_fair: bool = False,
    game_seeds: Sequence[int] | None = None,
    deterministic: bool = False,
    policy_overkill_guard: bool = False,
    parallel: bool = False,
    opponent_w1: np.ndarray | None = None,
    opponent_b1: np.ndarray | None = None,
    opponent_w2: np.ndarray | None = None,
    opponent_b2: np.ndarray | None = None,
    opponent_overkill_guard: bool = False,
    policy_name: str = "mlp_numba",
) -> NumbaMLPRolloutSummary:
    """
    Valuta una policy MLP con rollout full-JIT contro un opponent fast-compatible.

    Questo è ancora inference/evaluation: non raccoglie `StepRecord` e non aggiorna i pesi.
    Serve a validare il percorso completo stato numerico -> encoder -> MLP -> azione -> step.
    """
    num_games = int(num_games)
    if num_games < 0:
        raise ValueError("num_games deve essere >= 0")
    if bool(seat_fair) and num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari.")

    needed_seeds = num_games // 2 if bool(seat_fair) else num_games
    if game_seeds is None:
        seeds_arr = np.asarray([((int(seed) + i) & 0xFFFFFFFF) for i in range(needed_seeds)], dtype=np.int64)
    else:
        raw_seeds = [int(s) & 0xFFFFFFFF for s in game_seeds]
        if len(raw_seeds) < needed_seeds:
            raise ValueError(f"game_seeds insufficiente: attesi >= {needed_seeds}, ottenuti {len(raw_seeds)}")
        seeds_arr = np.asarray(raw_seeds[:needed_seeds], dtype=np.int64)

    w1_arr = _as_float32_matrix("w1", w1)
    b1_arr = _as_float32_vector("b1", b1)
    w2_arr = _as_float32_matrix("w2", w2)
    b2_arr = _as_float32_vector("b2", b2)
    feature_dim = int(w1_arr.shape[0])
    hidden_dim = int(w1_arr.shape[1])
    if feature_dim not in (
        int(FEATURE_DIM_2P_V1),
        int(FEATURE_DIM_2P_V2),
        int(FEATURE_DIM_2P_V3),
        int(FEATURE_DIM_2P_V4),
    ):
        raise ValueError(
            f"w1 feature_dim={feature_dim}; "
            f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
            f"{int(FEATURE_DIM_2P_V3)} o {int(FEATURE_DIM_2P_V4)}"
        )
    if b1_arr.shape != (hidden_dim,):
        raise ValueError(f"b1 shape={b1_arr.shape}; atteso {(hidden_dim,)}")
    if w2_arr.shape != (hidden_dim, ACTION_DIM):
        raise ValueError(f"w2 shape={w2_arr.shape}; atteso {(hidden_dim, ACTION_DIM)}")
    if b2_arr.shape != (ACTION_DIM,):
        raise ValueError(f"b2 shape={b2_arr.shape}; atteso {(ACTION_DIM,)}")

    opponent_model_enabled = (
        opponent_w1 is not None or opponent_b1 is not None or opponent_w2 is not None or opponent_b2 is not None
    )
    if opponent_model_enabled:
        if opponent_w1 is None or opponent_b1 is None or opponent_w2 is None or opponent_b2 is None:
            raise ValueError("Opponent model incompleto: servono opponent_w1/b1/w2/b2.")
        opponent_w1_arr = _as_float32_matrix("opponent_w1", opponent_w1)
        opponent_b1_arr = _as_float32_vector("opponent_b1", opponent_b1)
        opponent_w2_arr = _as_float32_matrix("opponent_w2", opponent_w2)
        opponent_b2_arr = _as_float32_vector("opponent_b2", opponent_b2)
        opp_feature_dim = int(opponent_w1_arr.shape[0])
        opp_hidden_dim = int(opponent_w1_arr.shape[1])
        if opp_feature_dim not in (
            int(FEATURE_DIM_2P_V1),
            int(FEATURE_DIM_2P_V2),
            int(FEATURE_DIM_2P_V3),
            int(FEATURE_DIM_2P_V4),
        ):
            raise ValueError(
                f"opponent_w1 feature_dim={opp_feature_dim}; "
                f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
                f"{int(FEATURE_DIM_2P_V3)} o {int(FEATURE_DIM_2P_V4)}"
            )
        if opponent_b1_arr.shape != (opp_hidden_dim,):
            raise ValueError(f"opponent_b1 shape={opponent_b1_arr.shape}; atteso {(opp_hidden_dim,)}")
        if opponent_w2_arr.shape != (opp_hidden_dim, ACTION_DIM):
            raise ValueError(f"opponent_w2 shape={opponent_w2_arr.shape}; atteso {(opp_hidden_dim, ACTION_DIM)}")
        if opponent_b2_arr.shape != (ACTION_DIM,):
            raise ValueError(f"opponent_b2 shape={opponent_b2_arr.shape}; atteso {(ACTION_DIM,)}")
        opponent_code = 0
    else:
        opponent_w1_arr = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
        opponent_b1_arr = np.zeros((1,), dtype=np.float32)
        opponent_w2_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)
        opponent_b2_arr = np.zeros((ACTION_DIM,), dtype=np.float32)
        opponent_code = numba_agent_code(opponent_name)

    if parallel:
        if bool(seat_fair):
            policy_points, opponent_points, winners = _evaluate_mlp_policy_numba_parallel_seat_fair(
                w1_arr,
                b1_arr,
                w2_arr,
                b2_arr,
                opponent_code,
                bool(opponent_model_enabled),
                opponent_w1_arr,
                opponent_b1_arr,
                opponent_w2_arr,
                opponent_b2_arr,
                bool(opponent_overkill_guard),
                seeds_arr,
                bool(deterministic),
                bool(policy_overkill_guard),
            )
        else:
            policy_points, opponent_points, winners = _evaluate_mlp_policy_numba_parallel_plain(
                w1_arr,
                b1_arr,
                w2_arr,
                b2_arr,
                opponent_code,
                bool(opponent_model_enabled),
                opponent_w1_arr,
                opponent_b1_arr,
                opponent_w2_arr,
                opponent_b2_arr,
                bool(opponent_overkill_guard),
                seeds_arr,
                bool(deterministic),
                bool(policy_overkill_guard),
            )
        wins_policy = int(np.count_nonzero(winners == 0))
        wins_opponent = int(np.count_nonzero(winners == 1))
        draws = int(np.count_nonzero(winners < 0))
        sum_policy = int(np.sum(policy_points))
        sum_opponent = int(np.sum(opponent_points))
        point_diffs = policy_points.astype(np.int64) - opponent_points.astype(np.int64)
        sum_sq_diff: float | None = float(np.sum(point_diffs * point_diffs))
        if bool(seat_fair):
            # Il kernel seat-fair produce le partite in coppie adiacenti (2i, 2i+1) con lo
            # stesso seed e seat scambiati, già dal punto di vista della policy: possiamo
            # quindi aggregare per COPPIA (l'unità statistica indipendente per le CI).
            pair_diffs = point_diffs.reshape(-1, 2).sum(axis=1)
            sum_sq_pair_diff: float | None = float(np.sum(pair_diffs * pair_diffs))
            game_scores = np.where(winners == 0, 1.0, np.where(winners == 1, 0.0, 0.5))
            pair_scores = game_scores.reshape(-1, 2).mean(axis=1)
            sum_sq_pair_score: float | None = float(np.sum(pair_scores * pair_scores))
        else:
            sum_sq_pair_diff = None
            sum_sq_pair_score = None
    else:
        wins_policy, wins_opponent, draws, sum_policy, sum_opponent = _evaluate_mlp_policy_numba(
            w1_arr,
            b1_arr,
            w2_arr,
            b2_arr,
            opponent_code,
            bool(opponent_model_enabled),
            opponent_w1_arr,
            opponent_b1_arr,
            opponent_w2_arr,
            opponent_b2_arr,
            bool(opponent_overkill_guard),
            seeds_arr,
            bool(seat_fair),
            bool(deterministic),
            bool(policy_overkill_guard),
        )
        sum_sq_diff = None
        sum_sq_pair_diff = None
        sum_sq_pair_score = None
    return NumbaMLPRolloutSummary(
        num_games=num_games,
        policy_name=policy_name,
        opponent_name=opponent_name,
        wins_policy=int(wins_policy),
        wins_opponent=int(wins_opponent),
        draws=int(draws),
        sum_policy=int(sum_policy),
        sum_opponent=int(sum_opponent),
        sum_sq_point_diff_policy_minus_opponent=sum_sq_diff,
        sum_sq_pair_point_diff_policy_minus_opponent=sum_sq_pair_diff,
        sum_sq_pair_score_policy=sum_sq_pair_score,
    )


def evaluate_mlp_policy_quality_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_name: str,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
    policy_overkill_guard: bool = False,
    parallel: bool = False,
    opponent_w1: np.ndarray | None = None,
    opponent_b1: np.ndarray | None = None,
    opponent_w2: np.ndarray | None = None,
    opponent_b2: np.ndarray | None = None,
    opponent_overkill_guard: bool = False,
    policy_name: str = "mlp_numba",
) -> NumbaDecisionQualitySummary:
    """
    Valuta seat-fair una policy MLP con Numba e raccoglie metriche decision-quality.

    Questa funzione replica la semantica inference di `BCModelAgent`: argmax deterministico,
    action mask e post-processing anti-overkill opzionale.
    """
    num_games = int(num_games)
    if num_games < 0:
        raise ValueError("num_games deve essere >= 0")
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari.")

    needed_seeds = num_games // 2
    if game_seeds is None:
        seeds_arr = np.asarray([((int(seed) + i) & 0xFFFFFFFF) for i in range(needed_seeds)], dtype=np.int64)
    else:
        raw_seeds = [int(s) & 0xFFFFFFFF for s in game_seeds]
        if len(raw_seeds) < needed_seeds:
            raise ValueError(f"game_seeds insufficiente: attesi >= {needed_seeds}, ottenuti {len(raw_seeds)}")
        seeds_arr = np.asarray(raw_seeds[:needed_seeds], dtype=np.int64)

    w1_arr = _as_float32_matrix("w1", w1)
    b1_arr = _as_float32_vector("b1", b1)
    w2_arr = _as_float32_matrix("w2", w2)
    b2_arr = _as_float32_vector("b2", b2)
    feature_dim = int(w1_arr.shape[0])
    hidden_dim = int(w1_arr.shape[1])
    if feature_dim not in (
        int(FEATURE_DIM_2P_V1),
        int(FEATURE_DIM_2P_V2),
        int(FEATURE_DIM_2P_V3),
        int(FEATURE_DIM_2P_V4),
    ):
        raise ValueError(
            f"w1 feature_dim={feature_dim}; "
            f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
            f"{int(FEATURE_DIM_2P_V3)} o {int(FEATURE_DIM_2P_V4)}"
        )
    if b1_arr.shape != (hidden_dim,):
        raise ValueError(f"b1 shape={b1_arr.shape}; atteso {(hidden_dim,)}")
    if w2_arr.shape != (hidden_dim, ACTION_DIM):
        raise ValueError(f"w2 shape={w2_arr.shape}; atteso {(hidden_dim, ACTION_DIM)}")
    if b2_arr.shape != (ACTION_DIM,):
        raise ValueError(f"b2 shape={b2_arr.shape}; atteso {(ACTION_DIM,)}")

    opponent_model_enabled = (
        opponent_w1 is not None or opponent_b1 is not None or opponent_w2 is not None or opponent_b2 is not None
    )
    if opponent_model_enabled:
        if opponent_w1 is None or opponent_b1 is None or opponent_w2 is None or opponent_b2 is None:
            raise ValueError("Opponent model incompleto: servono opponent_w1/b1/w2/b2.")
        opponent_w1_arr = _as_float32_matrix("opponent_w1", opponent_w1)
        opponent_b1_arr = _as_float32_vector("opponent_b1", opponent_b1)
        opponent_w2_arr = _as_float32_matrix("opponent_w2", opponent_w2)
        opponent_b2_arr = _as_float32_vector("opponent_b2", opponent_b2)
        opp_feature_dim = int(opponent_w1_arr.shape[0])
        opp_hidden_dim = int(opponent_w1_arr.shape[1])
        if opp_feature_dim not in (
            int(FEATURE_DIM_2P_V1),
            int(FEATURE_DIM_2P_V2),
            int(FEATURE_DIM_2P_V3),
            int(FEATURE_DIM_2P_V4),
        ):
            raise ValueError(
                f"opponent_w1 feature_dim={opp_feature_dim}; "
                f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
                f"{int(FEATURE_DIM_2P_V3)} o {int(FEATURE_DIM_2P_V4)}"
            )
        if opponent_b1_arr.shape != (opp_hidden_dim,):
            raise ValueError(f"opponent_b1 shape={opponent_b1_arr.shape}; atteso {(opp_hidden_dim,)}")
        if opponent_w2_arr.shape != (opp_hidden_dim, ACTION_DIM):
            raise ValueError(f"opponent_w2 shape={opponent_w2_arr.shape}; atteso {(opp_hidden_dim, ACTION_DIM)}")
        if opponent_b2_arr.shape != (ACTION_DIM,):
            raise ValueError(f"opponent_b2 shape={opponent_b2_arr.shape}; atteso {(ACTION_DIM,)}")
        opponent_code = 0
    else:
        opponent_w1_arr = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
        opponent_b1_arr = np.zeros((1,), dtype=np.float32)
        opponent_w2_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)
        opponent_b2_arr = np.zeros((ACTION_DIM,), dtype=np.float32)
        opponent_code = numba_agent_code(opponent_name)

    if parallel:
        (
            policy_points,
            opponent_points,
            winners,
            q_second_arr,
            q_second_with_win_arr,
            q_waste_arr,
            q_trump_wins_arr,
            q_trump_overkill_arr,
            q_trump_wins_low_arr,
            q_trump_overkill_low_arr,
        ) = _evaluate_mlp_policy_quality_numba_parallel(
            w1_arr,
            b1_arr,
            w2_arr,
            b2_arr,
            opponent_code,
            bool(opponent_model_enabled),
            opponent_w1_arr,
            opponent_b1_arr,
            opponent_w2_arr,
            opponent_b2_arr,
            bool(opponent_overkill_guard),
            seeds_arr,
            bool(policy_overkill_guard),
        )
        wins_policy = int(np.count_nonzero(winners == 0))
        wins_opponent = int(np.count_nonzero(winners == 1))
        draws = int(np.count_nonzero(winners < 0))
        sum_policy = int(np.sum(policy_points))
        sum_opponent = int(np.sum(opponent_points))
        q_second = int(np.sum(q_second_arr))
        q_second_with_win = int(np.sum(q_second_with_win_arr))
        q_waste = int(np.sum(q_waste_arr))
        q_trump_wins = int(np.sum(q_trump_wins_arr))
        q_trump_overkill = int(np.sum(q_trump_overkill_arr))
        q_trump_wins_low = int(np.sum(q_trump_wins_low_arr))
        q_trump_overkill_low = int(np.sum(q_trump_overkill_low_arr))
    else:
        (
            wins_policy,
            wins_opponent,
            draws,
            sum_policy,
            sum_opponent,
            q_second,
            q_second_with_win,
            q_waste,
            q_trump_wins,
            q_trump_overkill,
            q_trump_wins_low,
            q_trump_overkill_low,
        ) = _evaluate_mlp_policy_quality_numba(
            w1_arr,
            b1_arr,
            w2_arr,
            b2_arr,
            opponent_code,
            bool(opponent_model_enabled),
            opponent_w1_arr,
            opponent_b1_arr,
            opponent_w2_arr,
            opponent_b2_arr,
            bool(opponent_overkill_guard),
            seeds_arr,
            bool(policy_overkill_guard),
        )
    return NumbaDecisionQualitySummary(
        num_games=num_games,
        policy_name=policy_name,
        opponent_name=opponent_name,
        wins_policy=int(wins_policy),
        wins_opponent=int(wins_opponent),
        draws=int(draws),
        sum_policy=int(sum_policy),
        sum_opponent=int(sum_opponent),
        num_second_hand_decisions=int(q_second),
        num_second_hand_with_winning_reply=int(q_second_with_win),
        num_trump_waste=int(q_waste),
        num_second_hand_trump_wins=int(q_trump_wins),
        num_trump_overkill=int(q_trump_overkill),
        num_second_hand_trump_wins_low_lead_points=int(q_trump_wins_low),
        num_trump_overkill_low_lead_points=int(q_trump_overkill_low),
    )


def collect_a2c_trajectory_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_name: str,
    game_seed: int,
    policy_seat: int,
    policy_belief_w1: np.ndarray | None = None,
    policy_belief_b1: np.ndarray | None = None,
    policy_belief_w2: np.ndarray | None = None,
    policy_belief_b2: np.ndarray | None = None,
    opponent_w1: np.ndarray | None = None,
    opponent_b1: np.ndarray | None = None,
    opponent_w2: np.ndarray | None = None,
    opponent_b2: np.ndarray | None = None,
    opponent_overkill_guard: bool = False,
    overkill_penalty_beta: float = 0.0,
    overkill_low_lead_points_max: int = 2,
    overkill_penalty_mode: str = "flat",
) -> NumbaA2CTrajectory:
    """
    Raccoglie una traiettoria A2C full-JIT per il trainer.

    Il wrapper valida i tensori e restituisce solo le righe effettivamente popolate.
    """
    if policy_seat not in (0, 1):
        raise ValueError(f"policy_seat fuori range: {policy_seat}")
    if float(overkill_penalty_beta) < 0.0:
        raise ValueError("overkill_penalty_beta deve essere >= 0")
    if int(overkill_low_lead_points_max) < 0:
        raise ValueError("overkill_low_lead_points_max deve essere >= 0")
    mode_code = _overkill_penalty_mode_code(overkill_penalty_mode)
    (
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
    ) = _prepare_policy_belief_arrays(
        _as_float32_matrix("w1", w1), policy_belief_w1, policy_belief_b1, policy_belief_w2, policy_belief_b2
    )
    prepared = _prepare_a2c_numba_inputs(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        opponent_name=opponent_name,
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
    )

    (
        policy_points,
        opponent_points,
        winner,
        step_count,
        avg_entropy,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    ) = _collect_mlp_policy_game_numba(
        prepared.w1,
        prepared.b1,
        prepared.w2,
        prepared.b2,
        prepared.wv,
        float(bv),
        prepared.opponent_code,
        prepared.opponent_model_enabled,
        prepared.opponent_w1,
        prepared.opponent_b1,
        prepared.opponent_w2,
        prepared.opponent_b2,
        bool(opponent_overkill_guard),
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
        float(overkill_penalty_beta),
        int(overkill_low_lead_points_max),
        int(mode_code),
        int(game_seed),
        int(policy_seat),
    )
    count = int(step_count)
    return NumbaA2CTrajectory(
        policy_points=int(policy_points),
        opponent_points=int(opponent_points),
        winner=int(winner),
        avg_entropy=float(avg_entropy),
        xs=xs[:count].copy(),
        z1s=z1s[:count].copy(),
        hs=hs[:count].copy(),
        action_masks=masks[:count].copy(),
        probs=probs[:count].copy(),
        action_ids=action_ids[:count].copy(),
        value_preds=value_preds[:count].copy(),
        rewards=rewards[:count].copy(),
    )


def collect_a2c_batch_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_name: str,
    game_seeds: np.ndarray,
    policy_seats: np.ndarray,
    policy_belief_w1: np.ndarray | None = None,
    policy_belief_b1: np.ndarray | None = None,
    policy_belief_w2: np.ndarray | None = None,
    policy_belief_b2: np.ndarray | None = None,
    opponent_w1: np.ndarray | None = None,
    opponent_b1: np.ndarray | None = None,
    opponent_w2: np.ndarray | None = None,
    opponent_b2: np.ndarray | None = None,
    opponent_overkill_guard: bool = False,
    opponent_codes: np.ndarray | None = None,
    opponent_model_enabled_flags: np.ndarray | None = None,
    overkill_penalty_beta: float = 0.0,
    overkill_low_lead_points_max: int = 2,
    overkill_penalty_mode: str = "flat",
) -> NumbaA2CBatch:
    """
    Raccoglie un batch di traiettorie A2C full-JIT per il trainer.

    A differenza di `collect_a2c_trajectory_numba_2p`, questo wrapper valida i tensori una
    volta sola e restituisce buffer `(batch, max_steps, ...)`: il trainer usa `step_counts`
    per considerare solo le righe valide.
    """
    seeds_arr = np.asarray(game_seeds, dtype=np.int64)
    seats_arr = np.asarray(policy_seats, dtype=np.int64)
    if seeds_arr.ndim != 1:
        raise ValueError(f"game_seeds deve essere 1D, ottenuto shape={seeds_arr.shape}")
    if seats_arr.ndim != 1:
        raise ValueError(f"policy_seats deve essere 1D, ottenuto shape={seats_arr.shape}")
    if seats_arr.shape != seeds_arr.shape:
        raise ValueError(f"Shape mismatch: game_seeds={seeds_arr.shape} policy_seats={seats_arr.shape}")
    if not np.all((seats_arr == 0) | (seats_arr == 1)):
        raise ValueError("policy_seats deve contenere solo 0/1")
    if float(overkill_penalty_beta) < 0.0:
        raise ValueError("overkill_penalty_beta deve essere >= 0")
    if int(overkill_low_lead_points_max) < 0:
        raise ValueError("overkill_low_lead_points_max deve essere >= 0")
    mode_code = _overkill_penalty_mode_code(overkill_penalty_mode)
    (
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
    ) = _prepare_policy_belief_arrays(
        _as_float32_matrix("w1", w1), policy_belief_w1, policy_belief_b1, policy_belief_w2, policy_belief_b2
    )

    prepared = _prepare_a2c_numba_inputs(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        opponent_name=opponent_name,
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
    )
    if opponent_codes is None:
        codes_arr = np.full(seeds_arr.shape, prepared.opponent_code, dtype=np.int64)
    else:
        codes_arr = np.asarray(opponent_codes, dtype=np.int64)
        if codes_arr.ndim != 1:
            raise ValueError(f"opponent_codes deve essere 1D, ottenuto shape={codes_arr.shape}")
        if codes_arr.shape != seeds_arr.shape:
            raise ValueError(f"Shape mismatch: opponent_codes={codes_arr.shape} game_seeds={seeds_arr.shape}")
        if not np.all((codes_arr >= 0) & (codes_arr <= NUMBA_AGENT_CODE_MAX)):
            raise ValueError("opponent_codes contiene codici Numba non supportati")
    if opponent_model_enabled_flags is None:
        model_flags_arr = np.full(seeds_arr.shape, bool(prepared.opponent_model_enabled), dtype=np.bool_)
    else:
        model_flags_arr = np.asarray(opponent_model_enabled_flags, dtype=np.bool_)
        if model_flags_arr.ndim != 1:
            raise ValueError(f"opponent_model_enabled_flags deve essere 1D, ottenuto shape={model_flags_arr.shape}")
        if model_flags_arr.shape != seeds_arr.shape:
            raise ValueError(
                f"Shape mismatch: opponent_model_enabled_flags={model_flags_arr.shape} game_seeds={seeds_arr.shape}"
            )
        if bool(np.any(model_flags_arr)) and not prepared.opponent_model_enabled:
            raise ValueError("opponent_model_enabled_flags richiede un opponent model `.npz` caricato.")

    (
        policy_points,
        opponent_points,
        winners,
        step_counts,
        avg_entropies,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    ) = _collect_mlp_policy_batch_numba(
        prepared.w1,
        prepared.b1,
        prepared.w2,
        prepared.b2,
        prepared.wv,
        float(bv),
        prepared.opponent_code,
        prepared.opponent_model_enabled,
        prepared.opponent_w1,
        prepared.opponent_b1,
        prepared.opponent_w2,
        prepared.opponent_b2,
        bool(opponent_overkill_guard),
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
        float(overkill_penalty_beta),
        int(overkill_low_lead_points_max),
        int(mode_code),
        np.ascontiguousarray(model_flags_arr),
        np.ascontiguousarray(codes_arr),
        np.ascontiguousarray(seeds_arr),
        np.ascontiguousarray(seats_arr),
    )
    return NumbaA2CBatch(
        policy_points=policy_points,
        opponent_points=opponent_points,
        winners=winners,
        step_counts=step_counts,
        avg_entropies=avg_entropies,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=masks,
        probs=probs,
        action_ids=action_ids,
        value_preds=value_preds,
        rewards=rewards,
    )


def warm_up_numba_mlp_rollout() -> None:
    """Compila il rollout MLP full-JIT con un modello minimale."""
    w1 = np.zeros((int(FEATURE_DIM_2P_V1), 4), dtype=np.float32)
    b1 = np.zeros((4,), dtype=np.float32)
    w2 = np.zeros((4, ACTION_DIM), dtype=np.float32)
    b2 = np.zeros((ACTION_DIM,), dtype=np.float32)
    opponent_w1 = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
    opponent_b1 = np.zeros((1,), dtype=np.float32)
    opponent_w2 = np.zeros((1, ACTION_DIM), dtype=np.float32)
    opponent_b2 = np.zeros((ACTION_DIM,), dtype=np.float32)
    _evaluate_mlp_policy_numba(
        w1,
        b1,
        w2,
        b2,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        np.asarray([0], dtype=np.int64),
        False,
        False,
        False,
    )
    _evaluate_mlp_policy_numba_parallel_plain(
        w1,
        b1,
        w2,
        b2,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        np.asarray([0], dtype=np.int64),
        False,
        False,
    )
    _evaluate_mlp_policy_numba_parallel_seat_fair(
        w1,
        b1,
        w2,
        b2,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        np.asarray([0], dtype=np.int64),
        False,
        False,
    )
    _evaluate_mlp_policy_quality_numba(
        w1,
        b1,
        w2,
        b2,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        np.asarray([0], dtype=np.int64),
        False,
    )
    _evaluate_mlp_policy_quality_numba_parallel(
        w1,
        b1,
        w2,
        b2,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        np.asarray([0], dtype=np.int64),
        False,
    )
    wv = np.zeros((4,), dtype=np.float32)
    # Belief dummy (disabilitata): compila la firma estesa dell'iterazione 1b.
    belief_dummy_w1 = np.zeros((1, 1), dtype=np.float32)
    belief_dummy_b1 = np.zeros(1, dtype=np.float32)
    belief_dummy_w2 = np.zeros((1, ACTION_DIM), dtype=np.float32)
    belief_dummy_b2 = np.zeros(ACTION_DIM, dtype=np.float32)
    _collect_mlp_policy_game_numba(
        w1,
        b1,
        w2,
        b2,
        wv,
        0.0,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        False,
        belief_dummy_w1,
        belief_dummy_b1,
        belief_dummy_w2,
        belief_dummy_b2,
        0.0,
        2,
        0,
        0,
        0,
    )
    _collect_mlp_policy_batch_numba(
        w1,
        b1,
        w2,
        b2,
        wv,
        0.0,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        False,
        belief_dummy_w1,
        belief_dummy_b1,
        belief_dummy_w2,
        belief_dummy_b2,
        0.0,
        2,
        0,
        np.asarray([False], dtype=np.bool_),
        np.asarray([numba_agent_code("random")], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
    )
