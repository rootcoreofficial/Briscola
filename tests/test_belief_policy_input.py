"""
Iterazione 1b: la belief come input della policy (409 = encoder v4 + 40 probabilità).

Cosa proteggono questi test:
- il kernel JIT `_policy_input_v4_belief_numba` produce ESATTAMENTE encoder v4 + sigmoid
  della belief calcolata in NumPy (parità kernel ↔ path Python su stati di partite reali);
- il collector A2C con belief produce xs a 409 colonne con blocco belief in (0,1);
- l'artefatto self-contained (belief_* embedded) viene caricato e validato da `bc_model`
  e l'inferenza dominio usa lo stesso input combinato.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.fast.state_2p import new_fast_2p_state, step_fast_2p
from briscola_ai.ai.models.bc_model import load_bc_model_npz
from briscola_ai.ai.numba.observation import (
    _policy_input_v4_belief_numba,
    _state_to_numba_arrays,
    encode_fast_observation_numba_2p,
)

pytestmark = pytest.mark.numba

ACTION_DIM = 40


def _random_belief_weights(hidden: int = 16, seed: int = 5):
    rng = np.random.default_rng(seed)
    bw1 = rng.normal(0, 0.1, size=(FEATURE_DIM_2P_V4, hidden)).astype(np.float32)
    bb1 = rng.normal(0, 0.1, size=hidden).astype(np.float32)
    bw2 = rng.normal(0, 0.1, size=(hidden, ACTION_DIM)).astype(np.float32)
    bb2 = rng.normal(0, 0.1, size=ACTION_DIM).astype(np.float32)
    return bw1, bb1, bw2, bb2


def _numpy_belief_probs(features_v4: np.ndarray, bw1, bb1, bw2, bb2) -> np.ndarray:
    h = np.maximum(features_v4 @ bw1 + bb1, 0.0)
    return 1.0 / (1.0 + np.exp(-(h @ bw2 + bb2)))


@pytest.mark.parametrize("steps", [0, 9, 25])
def test_policy_input_kernel_matches_numpy_belief(steps: int) -> None:
    """Su stati di partita reali, il blocco belief del kernel coincide col calcolo NumPy."""
    state = new_fast_2p_state(seed=11)
    seen = [0] * ACTION_DIM
    seen[state.trump_card] = 1
    out_of_play = [0] * ACTION_DIM
    rng = random.Random(99)
    for _ in range(steps):
        if state.game_over:
            break
        current = state.current_turn
        result = step_fast_2p(state, player_index=current, card_index=rng.randrange(len(state.hands[current])))
        seen[result.played_card] = 1
        out_of_play[result.played_card] = 1

    bw1, bb1, bw2, bb2 = _random_belief_weights()

    # Riferimento: encoder v4 (wrapper già coperto dalla parità a tre motori) + belief NumPy.
    ref_enc = encode_fast_observation_numba_2p(
        state,
        player_index=0,
        seen_cards_onehot=tuple(seen),
        out_of_play_cards_onehot=tuple(out_of_play),
        version="v4",
    )
    ref_v4 = np.asarray(ref_enc.features, dtype=np.float32)
    ref_belief = _numpy_belief_probs(ref_v4, bw1, bb1, bw2, bb2)

    # Kernel: stesso stato in forma array + storia numerica.
    hands, hand_sizes, points, table_cards = _state_to_numba_arrays(state)
    trick_hist = np.zeros((20, 5), dtype=np.int64)
    for i, record in enumerate(state.trick_history):
        trick_hist[i] = record
    features, action_mask = _policy_input_v4_belief_numba(
        hands,
        hand_sizes,
        points,
        table_cards,
        len(state.table_cards),
        len(state.deck),
        int(state.current_turn),
        int(state.trump_card),
        0,
        np.asarray(seen, dtype=np.int64),
        np.asarray(out_of_play, dtype=np.int64),
        trick_hist,
        len(state.trick_history),
        bw1,
        bb1,
        bw2,
        bb2,
    )

    assert features.shape == (FEATURE_DIM_2P_V4 + ACTION_DIM,)
    assert features[:FEATURE_DIM_2P_V4] == pytest.approx(ref_v4, abs=1e-6)
    assert features[FEATURE_DIM_2P_V4:] == pytest.approx(ref_belief, abs=1e-5)
    assert list(action_mask) == ref_enc.action_mask


def _write_belief_policy_npz(path, *, coherent: bool = True) -> None:
    """Scrive un artefatto policy+belief minimo (409 = 369 + 40)."""
    rng = np.random.default_rng(0)
    bw1, bb1, bw2, bb2 = _random_belief_weights(hidden=8, seed=1)
    policy_dim = FEATURE_DIM_2P_V4 + (ACTION_DIM if coherent else 0)
    np.savez(
        path,
        w1=rng.normal(0, 0.05, size=(policy_dim, 4)).astype(np.float32),
        b1=np.zeros(4, dtype=np.float32),
        w2=rng.normal(0, 0.05, size=(4, ACTION_DIM)).astype(np.float32),
        b2=np.zeros(ACTION_DIM, dtype=np.float32),
        belief_w1=bw1,
        belief_b1=bb1,
        belief_w2=bw2,
        belief_b2=bb2,
        metadata_json=json.dumps({"format": "mlp_bc_v1", "encoder_version": "v4"}),
    )


def test_loader_accepts_selfcontained_belief_artifact(tmp_path) -> None:
    """Il loader riconosce belief_* embedded e l'input combinato ha la dimensione giusta."""
    path = tmp_path / "belief_policy.npz"
    _write_belief_policy_npz(path)
    model = load_bc_model_npz(path)
    assert model.has_belief_input
    assert model.feature_dim == FEATURE_DIM_2P_V4 + ACTION_DIM

    features_v4 = np.random.default_rng(2).normal(0, 1, size=FEATURE_DIM_2P_V4).astype(np.float32)
    x = model.policy_input(features_v4)
    assert x.shape == (FEATURE_DIM_2P_V4 + ACTION_DIM,)
    assert x[:FEATURE_DIM_2P_V4] == pytest.approx(features_v4)
    assert np.all((x[FEATURE_DIM_2P_V4:] > 0.0) & (x[FEATURE_DIM_2P_V4:] < 1.0))


def test_loader_rejects_belief_with_wrong_policy_dim(tmp_path) -> None:
    """Policy 369 con belief embedded è incoerente (attesa 409): errore esplicito."""
    path = tmp_path / "belief_policy_bad.npz"
    _write_belief_policy_npz(path, coherent=False)
    with pytest.raises(ValueError, match="encoder \\+ 40"):
        load_bc_model_npz(path)
