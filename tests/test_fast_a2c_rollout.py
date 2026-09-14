"""
Smoke test per il rollout A2C fast.

Il test non valuta la qualità del modello: verifica solo che il trainer possa usare
`--rollout-engine fast`, salvare un modello e dichiarare il metadato corretto.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.models import get_models_dir_from_env
from briscola_ai.ai.training.policy_regularization import (
    cross_entropy_from_probs,
    grad_ce_wrt_logits_from_probs,
)

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.slow

# L'opponent mix `best_a2c:...` carica `best_a2c.npz` dalla directory modelli: è un artefatto
# locale gitignored, assente su CI/cloni puliti, quindi il test che lo usa viene saltato.
requires_best_a2c_npz = pytest.mark.skipif(
    not (get_models_dir_from_env() / "best_a2c.npz").exists(),
    reason="richiede l'artefatto locale best_a2c.npz (gitignored, assente in CI)",
)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"
_SPEC = importlib.util.spec_from_file_location("train_a2c_for_tests", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Impossibile caricare {_SCRIPT_PATH}")
_TRAIN_A2C = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN_A2C
_SPEC.loader.exec_module(_TRAIN_A2C)

A2CPolicy = _TRAIN_A2C.A2CPolicy
_accumulate_numba_trajectory_grads = _TRAIN_A2C._accumulate_numba_trajectory_grads
_masked_logits_1d = _TRAIN_A2C._masked_logits_1d
_softmax_1d = _TRAIN_A2C._softmax_1d


def _write_dummy_mlp_model(path: Path, *, feature_dim: int = 248, hidden_dim: int = 4) -> None:
    """Scrive un piccolo modello MLP compatibile con `BCModelAgent`."""
    encoder_version = "v1"
    if feature_dim == 288:
        encoder_version = "v2"
    elif feature_dim == 310:
        encoder_version = "v3"
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": feature_dim,
        "encoder_version": encoder_version,
        "inference_overkill_guard": True,
    }
    np.savez(
        path,
        w1=np.zeros((feature_dim, hidden_dim), dtype=np.float32),
        b1=np.zeros((hidden_dim,), dtype=np.float32),
        w2=np.zeros((hidden_dim, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps(metadata),
    )


def _write_dummy_value_model(path: Path, *, feature_dim: int = 310, hidden_dim: int = 4) -> None:
    """Scrive un piccolo value model MLP compatibile con `load_value_model_npz`."""
    encoder_version = "v1"
    if feature_dim == 288:
        encoder_version = "v2"
    elif feature_dim == 310:
        encoder_version = "v3"
    metadata = {
        "format": "value_mlp_v1",
        "feature_dim": feature_dim,
        "encoder_version": encoder_version,
        "target": "residual",
        "target_scale": 120.0,
    }
    np.savez(
        path,
        w1=np.zeros((feature_dim, hidden_dim), dtype=np.float32),
        b1=np.zeros((hidden_dim,), dtype=np.float32),
        w2=np.zeros((hidden_dim,), dtype=np.float32),
        b2=np.asarray([0.0], dtype=np.float32),
        metadata_json=json.dumps(metadata),
    )


@pytest.mark.parametrize(
    ("fast_rollout", "fast_encoder"),
    [
        ("python", "python"),
        ("python", "numba"),
        ("numba", "python"),
    ],
)
def test_train_a2c_fast_rollout_smoke(tmp_path: Path, fast_rollout: str, fast_encoder: str) -> None:
    """Esegue pochissime partite A2C con rollout fast e verifica il modello salvato."""
    out_path = tmp_path / f"a2c_fast_{fast_rollout}_{fast_encoder}_smoke.npz"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            fast_rollout,
            "--fast-encoder",
            fast_encoder,
            "--opponent",
            "random",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == fast_rollout
    assert metadata["fast_encoder"] == fast_encoder
    assert metadata["train"]["num_games"] == 4


def test_train_a2c_fast_numba_rollout_supports_bc_model_opponent(tmp_path: Path) -> None:
    """Il rollout Numba deve poter usare un opponent `.npz` senza tornare al dominio canonico."""
    out_path = tmp_path / "a2c_fast_numba_bc_opponent_smoke.npz"
    opponent_path = tmp_path / "opponent.npz"
    _write_dummy_mlp_model(opponent_path)
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent",
            "bc_model",
            "--opponent-model",
            str(opponent_path),
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == "numba"
    assert metadata["opponent"] == "bc_model"
    assert metadata["opponent_model"] == str(opponent_path)


def test_train_a2c_fast_numba_rollout_supports_opponent_mix(tmp_path: Path) -> None:
    """Il rollout Numba batch deve supportare opponent mix rule-based senza tornare al path per-game."""
    out_path = tmp_path / "a2c_fast_numba_mix_smoke.npz"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent-mix",
            "random:0.5,greedy_points:0.5",
            "--overkill-penalty-beta",
            "0.01",
            "--overkill-penalty-mode",
            "gap",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == "numba"
    assert metadata["reward_shaping_overkill_penalty_beta"] == 0.01
    assert metadata["reward_shaping_overkill_penalty_mode"] == "gap"
    assert metadata["opponent"] is None
    assert sorted(metadata["opponent_mix"], key=lambda item: item["name"]) == [
        {"name": "greedy_points", "prob": 0.5},
        {"name": "random", "prob": 0.5},
    ]


@requires_best_a2c_npz
def test_train_a2c_fast_numba_rollout_supports_best_a2c_in_opponent_mix(tmp_path: Path) -> None:
    """Il rollout Numba batch deve poter mischiare baseline rule-based e best A2C congelato."""
    out_path = tmp_path / "a2c_fast_numba_best_mix_smoke.npz"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent-mix",
            "best_a2c:0.5,random:0.5",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == "numba"
    assert metadata["opponent"] is None
    assert sorted(metadata["opponent_mix"], key=lambda item: item["name"]) == [
        {"name": "best_a2c", "prob": 0.5},
        {"name": "random", "prob": 0.5},
    ]


def test_train_a2c_fast_numba_rollout_supports_bc_model_in_opponent_mix(tmp_path: Path) -> None:
    """Il rollout Numba batch deve poter mischiare un opponent `.npz` esplicito con baseline rule-based."""
    out_path = tmp_path / "a2c_fast_numba_bc_mix_smoke.npz"
    opponent_path = tmp_path / "opponent_mix_model.npz"
    _write_dummy_mlp_model(opponent_path)
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent-mix",
            "bc_model:0.5,random:0.5",
            "--opponent-model",
            str(opponent_path),
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == "numba"
    assert metadata["opponent"] is None
    assert metadata["opponent_model"] == str(opponent_path)
    assert sorted(metadata["opponent_mix"], key=lambda item: item["name"]) == [
        {"name": "bc_model", "prob": 0.5},
        {"name": "random", "prob": 0.5},
    ]


def test_train_a2c_fast_numba_rollout_supports_v3_bc_model_in_opponent_mix(tmp_path: Path) -> None:
    """Il league training v3 deve poter usare un opponent `.npz` v3 nel mix fast Numba."""
    out_path = tmp_path / "a2c_fast_numba_bc_v3_mix_smoke.npz"
    opponent_path = tmp_path / "opponent_mix_model_v3.npz"
    _write_dummy_mlp_model(opponent_path, feature_dim=310)
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--encoder-version",
            "v3",
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent-mix",
            "bc_model:0.5,random:0.5",
            "--opponent-model",
            str(opponent_path),
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["encoder_version"] == "v3"
    assert metadata["feature_dim"] == 310
    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == "numba"
    assert metadata["opponent"] is None
    assert metadata["opponent_model"] == str(opponent_path)
    assert sorted(metadata["opponent_mix"], key=lambda item: item["name"]) == [
        {"name": "bc_model", "prob": 0.5},
        {"name": "random", "prob": 0.5},
    ]


def test_train_a2c_fast_numba_rollout_supports_value_lookahead_opponent(tmp_path: Path) -> None:
    """Il trainer A2C può usare il value-lookahead Numba come opponent fast determinizzato."""
    out_path = tmp_path / "a2c_fast_numba_value_lookahead_smoke.npz"
    opponent_path = tmp_path / "opponent_v3.npz"
    value_path = tmp_path / "value_v3.npz"
    _write_dummy_mlp_model(opponent_path, feature_dim=310)
    _write_dummy_value_model(value_path, feature_dim=310)
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--encoder-version",
            "v3",
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent",
            "bc_model_value_lookahead_8x8",
            "--opponent-model",
            str(opponent_path),
            "--opponent-value-model",
            str(value_path),
            "--opponent-value-max-unknown-cards",
            "8",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["encoder_version"] == "v3"
    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == "numba"
    assert metadata["opponent"] == "bc_model_value_lookahead_8x8"
    assert metadata["opponent_model"] == str(opponent_path)
    assert metadata["opponent_value_model"] == str(value_path)
    assert metadata["opponent_value_max_unknown_cards"] == 8
    assert metadata["opponent_value_lookahead_rollout"] == "fast_numba_determinized"


def test_train_a2c_saves_update_aligned_checkpoints(tmp_path: Path) -> None:
    """I run lunghi possono salvare checkpoint confrontabili dopo update completi."""
    out_path = tmp_path / "a2c_final.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent",
            "random",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
            "--checkpoint-games",
            "2,4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-prefix",
            "scale_test",
            "--metrics-mode",
            "summary",
            "--inference-overkill-guard",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    checkpoint_2 = checkpoint_dir / "scale_test_2.npz"
    checkpoint_4 = checkpoint_dir / "scale_test_4.npz"
    assert out_path.exists()
    assert checkpoint_2.exists()
    assert checkpoint_4.exists()

    with np.load(checkpoint_2, allow_pickle=False) as data:
        metadata_2 = json.loads(str(data["metadata_json"]))
    with np.load(out_path, allow_pickle=False) as data:
        final_metadata = json.loads(str(data["metadata_json"]))

    assert metadata_2["checkpoint"] == {"games": 2, "final_num_games": 4}
    assert metadata_2["train"]["num_games"] == 2
    assert metadata_2["train"]["requested_num_games"] == 4
    assert metadata_2["inference_overkill_guard"] is True
    assert "metrics" not in metadata_2
    assert metadata_2["metrics_summary"]["count"] == 1

    assert "checkpoint" not in final_metadata
    assert final_metadata["train"]["num_games"] == 4
    assert final_metadata["train"]["requested_num_games"] == 4
    assert final_metadata["inference_overkill_guard"] is True
    assert "metrics" not in final_metadata
    assert final_metadata["metrics_summary"]["count"] == 2


class _LinearAnchor:
    """Anchor minimale per testare la regolarizzazione nel batch backprop."""

    def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
        self.w = w
        self.b = b

    @property
    def metadata(self) -> dict[str, str]:
        """Metadati fittizi, necessari per rispettare il protocollo del modello."""
        return {}

    @property
    def feature_dim(self) -> int:
        """Dimensione feature accettata dall'anchor."""
        return int(self.w.shape[0])

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Forward lineare usato solo dal test."""
        return x @ self.w + self.b


def test_numba_trajectory_vectorized_backprop_matches_step_loop() -> None:
    """Il nuovo accumulo batch deve restare equivalente al loop `np.outer` originale."""
    rng = np.random.default_rng(7)
    steps = 6
    feature_dim = 9
    hidden_dim = 5

    policy = A2CPolicy(
        w1=rng.normal(0.0, 0.05, size=(feature_dim, hidden_dim)).astype(np.float32),
        b1=rng.normal(0.0, 0.05, size=(hidden_dim,)).astype(np.float32),
        w2=rng.normal(0.0, 0.05, size=(hidden_dim, 40)).astype(np.float32),
        b2=rng.normal(0.0, 0.05, size=(40,)).astype(np.float32),
        wv=rng.normal(0.0, 0.05, size=(hidden_dim,)).astype(np.float32),
        bv=0.0,
    )
    xs = rng.normal(0.0, 1.0, size=(steps, feature_dim)).astype(np.float32)
    z1s = xs @ policy.w1 + policy.b1
    hs = np.maximum(z1s, 0.0).astype(np.float32)
    action_masks = rng.random((steps, 40)) > 0.25
    action_ids = np.zeros((steps,), dtype=np.int64)
    probs = np.zeros((steps, 40), dtype=np.float32)
    for i in range(steps):
        valid_ids = np.flatnonzero(action_masks[i])
        action_ids[i] = int(valid_ids[i % len(valid_ids)])
        raw = rng.random(len(valid_ids)).astype(np.float32)
        raw /= np.sum(raw, dtype=np.float32)
        probs[i, valid_ids] = raw
    value_preds = rng.normal(0.0, 0.2, size=(steps,)).astype(np.float32)
    returns_to_go = rng.normal(0.0, 0.2, size=(steps,)).astype(np.float32)
    anchor = _LinearAnchor(
        rng.normal(0.0, 0.05, size=(feature_dim, 40)).astype(np.float32),
        rng.normal(0.0, 0.05, size=(40,)).astype(np.float32),
    )

    ref_gw1 = np.zeros_like(policy.w1)
    ref_gb1 = np.zeros_like(policy.b1)
    ref_gw2 = np.zeros_like(policy.w2)
    ref_gb2 = np.zeros_like(policy.b2)
    ref_gwv = np.zeros_like(policy.wv)
    ref_gbv = 0.0
    ref_value_loss_sum = 0.0
    ref_anchor_ce_sum = 0.0
    for i, g in enumerate(returns_to_go):
        x = xs[i]
        z1 = z1s[i]
        h = hs[i]
        mask = action_masks[i]
        p = probs[i]
        v = float(value_preds[i])
        dlogits = p.copy()
        dlogits[int(action_ids[i])] -= 1.0
        dlogits *= float(g - v)

        logp = np.log(p + 1e-12)
        s = float(np.sum(p * (logp + 1.0)))
        dlogits += np.float32(0.0005) * (p * (logp + 1.0 - s)).astype(np.float32)

        anchor_logits = anchor.logits(x)
        anchor_probs = _softmax_1d(_masked_logits_1d(anchor_logits, mask))
        ref_anchor_ce_sum += cross_entropy_from_probs(target_probs=anchor_probs, pred_probs=p)
        grad_anchor = grad_ce_wrt_logits_from_probs(
            pred_probs=p,
            target_probs=anchor_probs,
            action_mask=mask,
        )
        dlogits += np.float32(0.03) * grad_anchor.astype(np.float32)

        ref_gw2 += np.outer(h, dlogits).astype(np.float32)
        ref_gb2 += dlogits.astype(np.float32)
        dh_policy = policy.w2 @ dlogits

        dv = 0.5 * (v - float(g))
        ref_value_loss_sum += 0.5 * 0.5 * (v - float(g)) ** 2
        ref_gwv += (h * dv).astype(np.float32)
        ref_gbv += dv
        dh_value = policy.wv * dv

        dz1 = (dh_policy + dh_value) * (z1 > 0.0)
        ref_gw1 += np.outer(x, dz1).astype(np.float32)
        ref_gb1 += dz1.astype(np.float32)

    gw1 = np.zeros_like(policy.w1)
    gb1 = np.zeros_like(policy.b1)
    gw2 = np.zeros_like(policy.w2)
    gb2 = np.zeros_like(policy.b2)
    gwv = np.zeros_like(policy.wv)
    stats = _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=action_masks,
        probs=probs,
        action_ids=action_ids,
        value_preds=value_preds,
        returns_to_go=returns_to_go,
        entropy_beta=0.0005,
        value_coef=0.5,
        bc_anchor=anchor,
        bc_anchor_beta=0.03,
        gw1=gw1,
        gb1=gb1,
        gw2=gw2,
        gb2=gb2,
        gwv=gwv,
    )

    assert stats.steps == steps
    assert stats.gbv == pytest.approx(ref_gbv)
    assert stats.value_loss_sum == pytest.approx(ref_value_loss_sum)
    assert stats.anchor_ce_sum == pytest.approx(ref_anchor_ce_sum)
    assert stats.anchor_ce_count == steps
    assert gw1 == pytest.approx(ref_gw1, abs=1e-6)
    assert gb1 == pytest.approx(ref_gb1, abs=1e-6)
    assert gw2 == pytest.approx(ref_gw2, abs=1e-6)
    assert gb2 == pytest.approx(ref_gb2, abs=1e-6)
    assert gwv == pytest.approx(ref_gwv, abs=1e-6)
