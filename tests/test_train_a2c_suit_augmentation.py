"""Test del wiring A2C per l'augmentation paired dei semi."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.evaluation.suit_symmetry import IDENTITY_SUIT_PERMUTATION, all_suit_permutations
from briscola_ai.ai.training.suit_augmentation import (
    permute_action_masks,
    permute_action_vectors,
    permute_encoded_features,
)

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _ROOT / "scripts" / "train_a2c.py"
_SPEC = importlib.util.spec_from_file_location("train_a2c_suit_augmentation_tests", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Impossibile caricare {_SCRIPT_PATH}")
_TRAIN_A2C = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN_A2C
_SPEC.loader.exec_module(_TRAIN_A2C)

A2CPolicy = _TRAIN_A2C.A2CPolicy
_accumulate_numba_trajectory_grads = _TRAIN_A2C._accumulate_numba_trajectory_grads
_accumulate_paired_suit_trajectory_grads = _TRAIN_A2C._accumulate_paired_suit_trajectory_grads
_accumulate_suit_consistency_grads = _TRAIN_A2C._accumulate_suit_consistency_grads
_accumulate_suit_margin_grads = _TRAIN_A2C._accumulate_suit_margin_grads
_forward_policy_batch = _TRAIN_A2C._forward_policy_batch


def _policy_and_trajectory() -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crea policy e traiettoria numerica deterministiche con tre azioni legali per step."""
    rng = np.random.default_rng(77)
    steps = 5
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden_dim = 7
    policy = A2CPolicy(
        w1=rng.normal(0.0, 0.03, size=(feature_dim, hidden_dim)).astype(np.float32),
        b1=rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32),
        w2=rng.normal(0.0, 0.03, size=(hidden_dim, 40)).astype(np.float32),
        b2=rng.normal(0.0, 0.03, size=40).astype(np.float32),
        wv=rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32),
        bv=0.01,
    )
    xs = rng.normal(0.0, 0.5, size=(steps, feature_dim)).astype(np.float32)
    masks = np.zeros((steps, 40), dtype=bool)
    ids = np.empty(steps, dtype=np.int64)
    for step_index in range(steps):
        legal = np.asarray([step_index, 10 + step_index, 30 + step_index], dtype=np.int64)
        masks[step_index, legal] = True
        ids[step_index] = legal[step_index % len(legal)]
    returns = rng.normal(0.0, 0.2, size=steps).astype(np.float32)
    return policy, xs, masks, ids, returns


def _zero_grads(policy) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alloca i cinque tensori gradiente mutati dagli helper vettoriali."""
    return (
        np.zeros_like(policy.w1),
        np.zeros_like(policy.b1),
        np.zeros_like(policy.w2),
        np.zeros_like(policy.b2),
        np.zeros_like(policy.wv),
    )


def test_identity_paired_copy_preserves_mean_gradient_and_doubles_step_count() -> None:
    """Originale+copia identica, mediati su 2N, devono uguagliare il gradiente originale su N."""
    policy, xs, masks, ids, returns = _policy_and_trajectory()
    z1s, hs, probs, values = _forward_policy_batch(policy, xs=xs, action_masks=masks)

    original_grads = _zero_grads(policy)
    original_stats = _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=masks,
        probs=probs,
        action_ids=ids,
        value_preds=values,
        returns_to_go=returns,
        entropy_beta=0.0005,
        value_coef=0.5,
        bc_anchor=None,
        bc_anchor_beta=0.0,
        gw1=original_grads[0],
        gb1=original_grads[1],
        gw2=original_grads[2],
        gb2=original_grads[3],
        gwv=original_grads[4],
    )

    paired_grads = _zero_grads(policy)
    first_stats = _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=masks,
        probs=probs,
        action_ids=ids,
        value_preds=values,
        returns_to_go=returns,
        entropy_beta=0.0005,
        value_coef=0.5,
        bc_anchor=None,
        bc_anchor_beta=0.0,
        gw1=paired_grads[0],
        gb1=paired_grads[1],
        gw2=paired_grads[2],
        gb2=paired_grads[3],
        gwv=paired_grads[4],
    )
    copy_stats = _accumulate_paired_suit_trajectory_grads(
        policy=policy,
        xs=xs,
        action_masks=masks,
        action_ids=ids,
        returns_to_go=returns,
        encoder_version="v4",
        permutation=IDENTITY_SUIT_PERMUTATION,
        entropy_beta=0.0005,
        value_coef=0.5,
        bc_anchor=None,
        bc_anchor_beta=0.0,
        gw1=paired_grads[0],
        gb1=paired_grads[1],
        gw2=paired_grads[2],
        gb2=paired_grads[3],
        gwv=paired_grads[4],
    )

    assert first_stats.steps + copy_stats.steps == 2 * original_stats.steps
    for original, paired in zip(original_grads, paired_grads, strict=True):
        np.testing.assert_allclose(
            original / np.float32(original_stats.steps),
            paired / np.float32(first_stats.steps + copy_stats.steps),
            rtol=1e-6,
            atol=1e-7,
        )
    assert (first_stats.gbv + copy_stats.gbv) / (first_stats.steps + copy_stats.steps) == pytest.approx(
        original_stats.gbv / original_stats.steps
    )


def _forward_kl(target: np.ndarray, prediction: np.ndarray) -> float:
    """KL media per riga con clipping coerente col trainer."""
    epsilon = np.float32(1e-12)
    values = target * (np.log(np.maximum(target, epsilon)) - np.log(np.maximum(prediction, epsilon)))
    return float(np.mean(np.sum(values, axis=1, dtype=np.float64)))


def test_consistency_gradient_reduces_frozen_target_kl_without_critic_grads() -> None:
    """Un piccolo update ausiliario deve avvicinare la copia al target originale congelato."""
    policy, xs, masks, _, _ = _policy_and_trajectory()
    _, _, original_probs, _ = _forward_policy_batch(policy, xs=xs, action_masks=masks)
    permutation = all_suit_permutations()[13]
    paired_xs = permute_encoded_features(xs, version="v4", permutation=permutation)
    paired_masks = permute_action_masks(masks, permutation=permutation)
    target_probs = permute_action_vectors(original_probs, permutation=permutation)
    _, _, before_probs, _ = _forward_policy_batch(policy, xs=paired_xs, action_masks=paired_masks)
    before_kl = _forward_kl(target_probs, before_probs)

    grads = _zero_grads(policy)
    stats = _accumulate_suit_consistency_grads(
        policy=policy,
        xs=xs,
        action_masks=masks,
        original_probs=original_probs,
        encoder_version="v4",
        permutation=permutation,
        beta=1.0,
        gw1=grads[0],
        gb1=grads[1],
        gw2=grads[2],
        gb2=grads[3],
    )
    learning_rate = np.float32(0.01 / stats.suit_consistency_count)
    policy.w1 -= learning_rate * grads[0]
    policy.b1 -= learning_rate * grads[1]
    policy.w2 -= learning_rate * grads[2]
    policy.b2 -= learning_rate * grads[3]

    _, _, after_probs, _ = _forward_policy_batch(policy, xs=paired_xs, action_masks=paired_masks)
    after_kl = _forward_kl(target_probs, after_probs)
    assert stats.steps == 0
    assert stats.gbv == 0.0
    assert stats.suit_consistency_count == len(xs)
    assert stats.suit_consistency_kl_sum / stats.suit_consistency_count == pytest.approx(before_kl)
    assert after_kl < before_kl
    np.testing.assert_array_equal(grads[4], np.zeros_like(grads[4]))


def test_margin_gradient_increases_student_margin_and_reduces_hinge() -> None:
    """La hinge deve agire sulla scelta rinominata senza introdurre gradienti del critic."""
    policy, xs, masks, _, _ = _policy_and_trajectory()
    _, _, original_probs, _ = _forward_policy_batch(policy, xs=xs, action_masks=masks)
    permutation = all_suit_permutations()[13]

    grads = _zero_grads(policy)
    before = _accumulate_suit_margin_grads(
        policy=policy,
        xs=xs,
        action_masks=masks,
        original_probs=original_probs,
        encoder_version="v4",
        permutation=permutation,
        beta=1.0,
        margin_cap=2.0,
        gw1=grads[0],
        gb1=grads[1],
        gw2=grads[2],
        gb2=grads[3],
    )
    learning_rate = np.float32(0.01 / before.suit_margin_count)
    policy.w1 -= learning_rate * grads[0]
    policy.b1 -= learning_rate * grads[1]
    policy.w2 -= learning_rate * grads[2]
    policy.b2 -= learning_rate * grads[3]

    after_grads = _zero_grads(policy)
    after = _accumulate_suit_margin_grads(
        policy=policy,
        xs=xs,
        action_masks=masks,
        original_probs=original_probs,
        encoder_version="v4",
        permutation=permutation,
        beta=1.0,
        margin_cap=2.0,
        gw1=after_grads[0],
        gb1=after_grads[1],
        gw2=after_grads[2],
        gb2=after_grads[3],
    )

    assert before.steps == 0
    assert before.gbv == 0.0
    assert before.suit_margin_count == len(xs)
    assert before.suit_margin_violation_count > 0
    assert after.suit_margin_loss_sum < before.suit_margin_loss_sum
    assert after.suit_margin_student_sum > before.suit_margin_student_sum
    assert after.suit_margin_teacher_sum == pytest.approx(before.suit_margin_teacher_sum)
    np.testing.assert_array_equal(grads[4], np.zeros_like(grads[4]))


def _write_synthetic_v4_model(path: Path) -> None:
    """Modello v4 piccolo ma caricabile come init e BC-anchor dello smoke."""
    rng = np.random.default_rng(11)
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden_dim = 8
    np.savez(
        path,
        w1=rng.normal(0.0, 0.02, size=(feature_dim, hidden_dim)).astype(np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=rng.normal(0.0, 0.02, size=(hidden_dim, 40)).astype(np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "mlp_bc_v1",
                "feature_dim": feature_dim,
                "encoder_version": "v4",
                "inference_overkill_guard": False,
            }
        ),
    )


def _run_cli_smoke(
    *,
    init_path: Path,
    output_path: Path,
    suit_augmentation: str | None,
    suit_consistency_beta: str | None = None,
    suit_margin_beta: str | None = None,
    suit_margin_cap: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Esegue un training minimo; ``None`` lascia al parser il default storico."""
    command = [
        sys.executable,
        str(_SCRIPT_PATH),
        "--init",
        str(init_path),
        "--encoder-version",
        "v4",
        "--rollout-engine",
        "fast",
        "--fast-rollout",
        "numba",
        "--opponent",
        "heuristic_v1",
        "--num-games",
        "20",
        "--update-every",
        "10",
        "--seed",
        "123",
        "--metrics-mode",
        "summary",
        "--out",
        str(output_path),
    ]
    if suit_augmentation is not None:
        command.extend(["--suit-augmentation", suit_augmentation])
    if suit_consistency_beta is not None:
        command.extend(["--suit-consistency-beta", suit_consistency_beta])
    if suit_margin_beta is not None:
        command.extend(["--suit-margin-beta", suit_margin_beta])
    if suit_margin_cap is not None:
        command.extend(["--suit-margin-cap", suit_margin_cap])
    return subprocess.run(
        command,
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
def test_train_a2c_explicit_off_is_identical_to_historical_default(tmp_path: Path) -> None:
    """Il nuovo flag, se spento, non deve cambiare pesi, bias o metadata del trainer."""
    init_path = tmp_path / "init_v4.npz"
    default_path = tmp_path / "default_v4.npz"
    explicit_off_path = tmp_path / "explicit_off_v4.npz"
    _write_synthetic_v4_model(init_path)

    default_result = _run_cli_smoke(
        init_path=init_path,
        output_path=default_path,
        suit_augmentation=None,
    )
    off_result = _run_cli_smoke(
        init_path=init_path,
        output_path=explicit_off_path,
        suit_augmentation="off",
        suit_consistency_beta="0",
        suit_margin_beta="0",
    )

    assert default_result.returncode == 0, default_result.stderr
    assert off_result.returncode == 0, off_result.stderr
    with (
        np.load(default_path, allow_pickle=False) as default_data,
        np.load(explicit_off_path, allow_pickle=False) as off_data,
    ):
        assert set(default_data.files) == set(off_data.files)
        for key in default_data.files:
            np.testing.assert_array_equal(default_data[key], off_data[key])
        metadata = json.loads(str(default_data["metadata_json"].item()))
    assert "suit_augmentation" not in metadata
    assert "suit_consistency" not in metadata
    assert "suit_margin_consistency" not in metadata


@pytest.mark.slow
def test_train_a2c_margin_cli_smoke_and_metadata(tmp_path: Path) -> None:
    """Il trainer deve applicare la hinge e serializzarne contratto e diagnostica."""
    init_path = tmp_path / "init_v4.npz"
    output_path = tmp_path / "margin_v4.npz"
    _write_synthetic_v4_model(init_path)

    result = _run_cli_smoke(
        init_path=init_path,
        output_path=output_path,
        suit_augmentation="off",
        suit_margin_beta="0.01",
        suit_margin_cap="2.0",
    )

    assert result.returncode == 0, f"trainer fallito:\n{result.stdout}\n{result.stderr}"
    assert "suit_hinge" in result.stdout
    with np.load(output_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    assert metadata["suit_margin_consistency"] == {
        "mode": "teacher_argmax_hinge",
        "beta": 0.01,
        "margin_cap": 2.0,
        "teacher": "original_argmax_and_capped_logit_margin",
        "student": "nonidentity_suit_permutation",
        "forced_actions": "excluded",
        "permutation_scope": "whole_trajectory",
        "permutation_distribution": "uniform_nonidentity_23",
        "loss_normalization": "mean_over_original_on_policy_steps",
        "rng_seed": 123 ^ 0x0A461A9E,
    }


@pytest.mark.slow
def test_train_a2c_consistency_cli_smoke_and_metadata(tmp_path: Path) -> None:
    """Il trainer deve applicare e dichiarare la KL senza cambiare il conteggio A2C."""
    init_path = tmp_path / "init_v4.npz"
    output_path = tmp_path / "consistency_v4.npz"
    _write_synthetic_v4_model(init_path)

    result = _run_cli_smoke(
        init_path=init_path,
        output_path=output_path,
        suit_augmentation="off",
        suit_consistency_beta="0.01",
    )

    assert result.returncode == 0, f"trainer fallito:\n{result.stdout}\n{result.stderr}"
    assert "suit_kl" in result.stdout
    with np.load(output_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    assert metadata["suit_consistency"] == {
        "mode": "forward_kl_stop_gradient",
        "beta": 0.01,
        "target": "original_policy_distribution",
        "student": "nonidentity_suit_permutation",
        "permutation_scope": "whole_trajectory",
        "permutation_distribution": "uniform_nonidentity_23",
        "loss_normalization": "mean_over_original_on_policy_steps",
        "rng_seed": 123 ^ 0x0C05157E,
    }


@pytest.mark.slow
def test_train_a2c_paired_cli_smoke_and_metadata(tmp_path: Path) -> None:
    """Il path Numba batch deve allenare e dichiarare il contratto paired nell'artefatto."""
    init_path = tmp_path / "init_v4.npz"
    output_path = tmp_path / "paired_v4.npz"
    _write_synthetic_v4_model(init_path)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--init",
            str(init_path),
            "--bc-anchor",
            str(init_path),
            "--bc-anchor-beta",
            "0.01",
            "--encoder-version",
            "v4",
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent",
            "heuristic_v1",
            "--num-games",
            "20",
            "--update-every",
            "10",
            "--suit-augmentation",
            "paired",
            "--seed",
            "123",
            "--metrics-mode",
            "summary",
            "--out",
            str(output_path),
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"trainer fallito:\n{result.stdout}\n{result.stderr}"
    with np.load(output_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    assert metadata["suit_augmentation"] == {
        "mode": "paired",
        "copies_per_trajectory": 1,
        "permutation_scope": "whole_trajectory",
        "permutation_distribution": "uniform_nonidentity_23",
        "loss_normalization": "mean_over_original_and_copy",
        "rng_seed": 123 ^ 0x51A17A9E,
    }
