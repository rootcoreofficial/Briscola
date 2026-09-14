"""
Test di equivalenza per l'encoder osservazione Numba.

Il layout feature/mask deve restare identico a `encode_fast_observation_2p`: questo
ci permette di usare lo stesso modello `.npz` quando il rollout A2C passerà a stato
numerico/JIT.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, EncoderVersion
from briscola_ai.ai.fast.observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.fast.state_2p import new_fast_2p_state, step_fast_2p
from briscola_ai.ai.numba.core import numba_agent_code
from briscola_ai.ai.numba.mlp import (
    collect_a2c_batch_numba_2p,
    collect_a2c_trajectory_numba_2p,
    evaluate_mlp_policy_numba_2p,
    evaluate_mlp_policy_quality_numba_2p,
    warm_up_numba_mlp_rollout,
)
from briscola_ai.ai.numba.observation import (
    _trump_overkill_penalty_numba,
    encode_fast_observation_numba_2p,
    warm_up_numba_observation,
)

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba


def _assert_encoders_match(*, seed: int, steps: int, version: EncoderVersion) -> None:
    """Avanza una partita fast e confronta encoder Python vs Numba per entrambi i player."""
    state = new_fast_2p_state(seed=seed)
    seen = [0] * 40
    seen[state.trump_card] = 1
    rng = random.Random(seed ^ 0xA2C)

    for _ in range(steps):
        if state.game_over:
            break
        current = state.current_turn
        card_index = rng.randrange(len(state.hands[current]))
        result = step_fast_2p(state, player_index=current, card_index=card_index)
        seen[result.played_card] = 1

    for player_index in (0, 1):
        py_encoded = encode_fast_observation_2p(
            state,
            player_index=player_index,
            seen_cards_onehot=tuple(seen),
            version=version,
        )
        nb_encoded = encode_fast_observation_numba_2p(
            state,
            player_index=player_index,
            seen_cards_onehot=tuple(seen),
            version=version,
        )

        assert nb_encoded.action_mask == py_encoded.action_mask
        assert nb_encoded.features == pytest.approx(py_encoded.features)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("steps", [0, 1, 2, 7, 20])
def test_numba_fast_observation_matches_python_encoder(version: EncoderVersion, steps: int) -> None:
    """L'encoder JIT deve essere semanticamente equivalente al path fast Python."""
    warm_up_numba_observation()

    _assert_encoders_match(seed=42, steps=steps, version=version)


def test_numba_fast_observation_rejects_invalid_seen() -> None:
    """Il wrapper Python mantiene le stesse validazioni base sul vettore seen."""
    state = new_fast_2p_state(seed=0)

    with pytest.raises(ValueError, match="seen_cards_onehot len"):
        encode_fast_observation_numba_2p(state, player_index=0, seen_cards_onehot=(0,), version="v2")

    bad_seen = [0] * 40
    bad_seen[0] = 2
    with pytest.raises(ValueError, match="solo 0/1"):
        encode_fast_observation_numba_2p(state, player_index=0, seen_cards_onehot=tuple(bad_seen), version="v2")


@pytest.mark.parametrize("feature_dim", [int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)])
def test_numba_mlp_rollout_is_deterministic_and_valid(feature_dim: int) -> None:
    """Il rollout full-JIT MLP deve essere deterministico per seed e rispettare gli invarianti."""
    warm_up_numba_mlp_rollout()

    w1 = np.zeros((feature_dim, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w2 = np.zeros((8, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    first = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=99,
        seat_fair=True,
    )
    second = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=99,
        seat_fair=True,
    )

    assert first == second
    assert first.wins_policy + first.wins_opponent + first.draws == 40
    assert first.sum_policy + first.sum_opponent == 40 * 120
    stats = first.to_match_stats()
    assert stats.avg_points_agent0 + stats.avg_points_agent1 == pytest.approx(120.0)

    serial_argmax = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=99,
        seat_fair=True,
        deterministic=True,
        game_seeds=list(range(20)),
    )
    parallel_argmax = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=99,
        seat_fair=True,
        deterministic=True,
        parallel=True,
        game_seeds=list(range(20)),
    )
    assert parallel_argmax == serial_argmax


def test_numba_mlp_rollout_supports_mlp_opponent() -> None:
    """Il rollout evaluation Numba deve supportare anche opponent `.npz` MLP."""
    feature_dim = int(FEATURE_DIM_2P_V1)
    w1 = np.zeros((feature_dim, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w2 = np.zeros((8, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    summary = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="opponent_model",
        num_games=20,
        seed=11,
        seat_fair=True,
        deterministic=True,
        opponent_w1=w1,
        opponent_b1=b1,
        opponent_w2=w2,
        opponent_b2=b2,
        opponent_overkill_guard=False,
    )

    assert summary.wins_policy + summary.wins_opponent + summary.draws == 20
    assert summary.sum_policy + summary.sum_opponent == 20 * 120
    assert summary.to_seat_fair_stats().agent_b_name == "opponent_model"


def test_numba_quality_parallel_matches_serial() -> None:
    """Le metriche decision-quality Numba parallele devono combaciare col path seriale."""
    warm_up_numba_mlp_rollout()

    feature_dim = int(FEATURE_DIM_2P_V1)
    w1 = np.zeros((feature_dim, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w2 = np.zeros((8, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    serial = evaluate_mlp_policy_quality_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=17,
        game_seeds=list(range(20)),
        policy_overkill_guard=True,
    )
    parallel = evaluate_mlp_policy_quality_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=17,
        game_seeds=list(range(20)),
        policy_overkill_guard=True,
        parallel=True,
    )

    assert parallel == serial


def test_numba_mlp_rollout_rejects_bad_shapes() -> None:
    """Il wrapper Python deve intercettare modelli MLP non compatibili prima del JIT."""
    w1 = np.zeros((123, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w2 = np.zeros((8, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    with pytest.raises(ValueError, match="feature_dim"):
        evaluate_mlp_policy_numba_2p(
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            opponent_name="random",
            num_games=1,
            seed=0,
        )


@pytest.mark.parametrize("feature_dim", [int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)])
def test_numba_a2c_trajectory_shapes_and_rewards_are_valid(feature_dim: int) -> None:
    """Il collector JIT deve produrre buffer A2C coerenti con una partita completa."""
    warm_up_numba_mlp_rollout()

    hidden_dim = 8
    w1 = np.zeros((feature_dim, hidden_dim), dtype=np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = np.zeros((hidden_dim, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    wv = np.zeros((hidden_dim,), dtype=np.float32)

    traj = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="heuristic_v1",
        game_seed=123,
        policy_seat=0,
    )
    again = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="heuristic_v1",
        game_seed=123,
        policy_seat=0,
    )

    assert traj.policy_points + traj.opponent_points == 120
    assert traj.winner in (-1, 0, 1)
    assert 1 <= len(traj.rewards) <= 20
    assert traj.xs.shape == (len(traj.rewards), feature_dim)
    assert traj.z1s.shape == (len(traj.rewards), hidden_dim)
    assert traj.hs.shape == (len(traj.rewards), hidden_dim)
    assert traj.action_masks.shape == (len(traj.rewards), 40)
    assert traj.probs.shape == (len(traj.rewards), 40)
    assert traj.action_ids.shape == (len(traj.rewards),)
    assert traj.value_preds.shape == (len(traj.rewards),)
    assert float(np.sum(traj.rewards)) == pytest.approx((traj.policy_points - traj.opponent_points) / 120.0)
    assert np.allclose(traj.xs, again.xs)
    assert np.allclose(traj.probs, again.probs)
    assert np.array_equal(traj.action_ids, again.action_ids)
    assert np.allclose(traj.rewards, again.rewards)


def test_numba_a2c_trajectory_supports_mlp_opponent() -> None:
    """Il collector JIT deve poter usare un opponent MLP `.npz`-style con argmax mascherato."""
    warm_up_numba_mlp_rollout()

    policy_hidden = 8
    opponent_hidden = 6
    w1 = np.zeros((int(FEATURE_DIM_2P_V1), policy_hidden), dtype=np.float32)
    b1 = np.zeros((policy_hidden,), dtype=np.float32)
    w2 = np.zeros((policy_hidden, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    wv = np.zeros((policy_hidden,), dtype=np.float32)
    opponent_w1 = np.zeros((int(FEATURE_DIM_2P_V1), opponent_hidden), dtype=np.float32)
    opponent_b1 = np.zeros((opponent_hidden,), dtype=np.float32)
    opponent_w2 = np.zeros((opponent_hidden, 40), dtype=np.float32)
    opponent_b2 = np.zeros((40,), dtype=np.float32)

    traj = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="bc_model",
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
        opponent_overkill_guard=True,
        game_seed=456,
        policy_seat=1,
    )

    assert traj.policy_points + traj.opponent_points == 120
    assert 1 <= len(traj.rewards) <= 20
    assert float(np.sum(traj.rewards)) == pytest.approx((traj.policy_points - traj.opponent_points) / 120.0)


def test_numba_a2c_batch_matches_single_trajectory() -> None:
    """Il collector batch deve produrre gli stessi buffer validi del wrapper single-game."""
    warm_up_numba_mlp_rollout()

    feature_dim = int(FEATURE_DIM_2P_V1)
    hidden_dim = 8
    w1 = np.zeros((feature_dim, hidden_dim), dtype=np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = np.zeros((hidden_dim, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    wv = np.zeros((hidden_dim,), dtype=np.float32)
    seeds = np.asarray([123, 456], dtype=np.int64)
    seats = np.asarray([0, 1], dtype=np.int64)

    batch = collect_a2c_batch_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="heuristic_v1",
        game_seeds=seeds,
        policy_seats=seats,
    )

    assert batch.xs.shape == (2, 20, feature_dim)
    assert batch.hs.shape == (2, 20, hidden_dim)
    for i, (seed, seat) in enumerate(zip(seeds, seats, strict=True)):
        single = collect_a2c_trajectory_numba_2p(
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            wv=wv,
            bv=0.0,
            opponent_name="heuristic_v1",
            game_seed=int(seed),
            policy_seat=int(seat),
        )
        count = int(batch.step_counts[i])
        assert batch.policy_points[i] == single.policy_points
        assert batch.opponent_points[i] == single.opponent_points
        assert batch.winners[i] == single.winner
        assert count == len(single.rewards)
        assert batch.avg_entropies[i] == pytest.approx(single.avg_entropy)
        assert np.allclose(batch.xs[i, :count], single.xs)
        assert np.allclose(batch.z1s[i, :count], single.z1s)
        assert np.allclose(batch.hs[i, :count], single.hs)
        assert np.array_equal(batch.action_masks[i, :count], single.action_masks)
        assert np.allclose(batch.probs[i, :count], single.probs)
        assert np.array_equal(batch.action_ids[i, :count], single.action_ids)
        assert np.allclose(batch.value_preds[i, :count], single.value_preds)
        assert np.allclose(batch.rewards[i, :count], single.rewards)


def test_numba_a2c_batch_supports_per_game_opponent_codes() -> None:
    """Il batch collector deve poter variare opponent rule-based a ogni partita."""
    warm_up_numba_mlp_rollout()

    feature_dim = int(FEATURE_DIM_2P_V1)
    hidden_dim = 8
    w1 = np.zeros((feature_dim, hidden_dim), dtype=np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = np.zeros((hidden_dim, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    wv = np.zeros((hidden_dim,), dtype=np.float32)
    seeds = np.asarray([100, 101, 102], dtype=np.int64)
    seats = np.asarray([0, 1, 0], dtype=np.int64)
    opponent_names = ["random", "greedy_points", "heuristic_v1"]
    opponent_codes = np.asarray([numba_agent_code(name) for name in opponent_names], dtype=np.int64)

    batch = collect_a2c_batch_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="random",
        opponent_codes=opponent_codes,
        game_seeds=seeds,
        policy_seats=seats,
    )

    for i, opponent_name in enumerate(opponent_names):
        single = collect_a2c_trajectory_numba_2p(
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            wv=wv,
            bv=0.0,
            opponent_name=opponent_name,
            game_seed=int(seeds[i]),
            policy_seat=int(seats[i]),
        )
        count = int(batch.step_counts[i])
        assert batch.policy_points[i] == single.policy_points
        assert batch.opponent_points[i] == single.opponent_points
        assert count == len(single.rewards)
        assert np.allclose(batch.rewards[i, :count], single.rewards)


def test_numba_a2c_batch_supports_mixed_rule_and_model_opponents() -> None:
    """Il batch collector deve poter alternare rule-based e MLP `.npz` nello stesso batch."""
    warm_up_numba_mlp_rollout()

    feature_dim = int(FEATURE_DIM_2P_V1)
    policy_hidden = 8
    opponent_hidden = 6
    w1 = np.zeros((feature_dim, policy_hidden), dtype=np.float32)
    b1 = np.zeros((policy_hidden,), dtype=np.float32)
    w2 = np.zeros((policy_hidden, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    wv = np.zeros((policy_hidden,), dtype=np.float32)
    opponent_w1 = np.zeros((feature_dim, opponent_hidden), dtype=np.float32)
    opponent_b1 = np.zeros((opponent_hidden,), dtype=np.float32)
    opponent_w2 = np.zeros((opponent_hidden, 40), dtype=np.float32)
    opponent_b2 = np.zeros((40,), dtype=np.float32)
    seeds = np.asarray([222, 333], dtype=np.int64)
    seats = np.asarray([0, 1], dtype=np.int64)

    batch = collect_a2c_batch_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="random",
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
        opponent_codes=np.asarray([numba_agent_code("random"), 0], dtype=np.int64),
        opponent_model_enabled_flags=np.asarray([False, True], dtype=np.bool_),
        game_seeds=seeds,
        policy_seats=seats,
    )

    single_rule = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="random",
        game_seed=int(seeds[0]),
        policy_seat=int(seats[0]),
    )
    single_model = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="bc_model",
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
        game_seed=int(seeds[1]),
        policy_seat=int(seats[1]),
    )

    assert batch.policy_points[0] == single_rule.policy_points
    assert batch.opponent_points[0] == single_rule.opponent_points
    assert np.allclose(batch.rewards[0, : int(batch.step_counts[0])], single_rule.rewards)
    assert batch.policy_points[1] == single_model.policy_points
    assert batch.opponent_points[1] == single_model.opponent_points
    assert np.allclose(batch.rewards[1, : int(batch.step_counts[1])], single_model.rewards)


def test_numba_trump_overkill_penalty_matches_expected_flat_and_gap() -> None:
    """La penalità JIT deve replicare i casi base del reward shaping canonico."""
    hands = np.full((2, 3), -1, dtype=np.int64)
    hand_sizes = np.asarray([1, 2], dtype=np.int64)
    hands[0, 0] = 31  # swords two
    hands[1, 0] = 11  # cups two, trump winning minima
    hands[1, 1] = 10  # cups ace, overkill
    table_cards = np.asarray([31, -1], dtype=np.int64)
    table_players = np.asarray([0, -1], dtype=np.int64)
    trump_card = 12  # cups three

    flat = _trump_overkill_penalty_numba(
        hands,
        hand_sizes,
        table_cards,
        table_players,
        1,
        trump_card,
        1,
        1,
        0.005,
        2,
        0,
    )
    cheap = _trump_overkill_penalty_numba(
        hands,
        hand_sizes,
        table_cards,
        table_players,
        1,
        trump_card,
        1,
        0,
        0.005,
        2,
        0,
    )
    gap = _trump_overkill_penalty_numba(
        hands,
        hand_sizes,
        table_cards,
        table_players,
        1,
        trump_card,
        1,
        1,
        0.01,
        2,
        1,
    )

    assert flat == pytest.approx(-0.005)
    assert cheap == 0.0
    assert gap == pytest.approx(-0.019)
