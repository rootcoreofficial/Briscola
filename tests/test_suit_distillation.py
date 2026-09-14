"""Test del dataset e del trainer di distillazione della simmetria dei semi."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.models import BCModelAgent, MLPBCModel
from briscola_ai.ai.training.suit_distillation import (
    DATASET_FORMAT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SuitDistillationDataset,
    load_suit_distillation_dataset,
    make_game_split_ids,
    masked_softmax_batch,
    train_suit_distillation,
)

ROOT = Path(__file__).resolve().parents[1]


def _constant_target_dataset() -> SuitDistillationDataset:
    """Dataset piccolo in cui anche una MLP inizialmente nulla può imparare dai bias."""
    num_examples = 24
    rng = np.random.default_rng(11)
    features = rng.normal(0.0, 0.2, size=(num_examples, FEATURE_DIM_2P_V4)).astype(np.float32)
    masks = np.zeros((num_examples, 40), dtype=bool)
    masks[:, [0, 1, 2]] = True
    targets = np.zeros((num_examples, 40), dtype=np.float32)
    targets[:, [0, 1, 2]] = (0.80, 0.15, 0.05)
    game_ids = np.repeat(np.arange(12, dtype=np.int32), 2)
    split_by_game = np.asarray(
        [SPLIT_TRAIN] * 8 + [SPLIT_VALIDATION] * 2 + [SPLIT_TEST] * 2,
        dtype=np.uint8,
    )
    return SuitDistillationDataset(
        features=features,
        action_masks=masks,
        target_probs=targets,
        target_action_ids=np.zeros(num_examples, dtype=np.int16),
        game_ids=game_ids,
        split_ids=split_by_game[game_ids],
        metadata={"format": DATASET_FORMAT, "encoder_version": "v4"},
    )


def _zero_model() -> MLPBCModel:
    """MLP v4 con distribuzione iniziale uniforme sulle azioni legali."""
    hidden_dim = 4
    return MLPBCModel(
        w1=np.zeros((FEATURE_DIM_2P_V4, hidden_dim), dtype=np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=np.zeros((hidden_dim, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata={"format": "mlp_bc_v1", "feature_dim": FEATURE_DIM_2P_V4, "encoder_version": "v4"},
    )


def test_game_split_is_deterministic_disjoint_and_has_three_nonempty_parts() -> None:
    """Lo split 8/1/1 deve avvenire sull'identità della partita, mai sulle singole righe."""
    first = make_game_split_ids(10_000, seed=20260711)
    second = make_game_split_ids(10_000, seed=20260711)

    np.testing.assert_array_equal(first, second)
    assert np.bincount(first, minlength=3).tolist() == [8_000, 1_000, 1_000]
    assert make_game_split_ids(3, seed=1).tolist().count(int(SPLIT_TRAIN)) == 1
    assert set(make_game_split_ids(3, seed=1).tolist()) == {0, 1, 2}


def test_dataset_roundtrip_validates_probabilities_and_rejects_game_leakage(tmp_path: Path) -> None:
    """Il file numerico deve preservare gli array e intercettare una partita in due split."""
    dataset = _constant_target_dataset()
    path = tmp_path / "teacher_dataset.npz"
    dataset.save(path)
    restored = load_suit_distillation_dataset(path)

    np.testing.assert_array_equal(restored.features, dataset.features)
    np.testing.assert_array_equal(restored.target_probs, dataset.target_probs)
    assert restored.indices("train").size == 16
    assert restored.indices("validation").size == 4
    assert restored.indices("test").size == 4

    bad_splits = dataset.split_ids.copy()
    bad_splits[1] = SPLIT_VALIDATION
    bad = SuitDistillationDataset(
        features=dataset.features,
        action_masks=dataset.action_masks,
        target_probs=dataset.target_probs,
        target_action_ids=dataset.target_action_ids,
        game_ids=dataset.game_ids,
        split_ids=bad_splits,
        metadata=dataset.metadata,
    )
    with pytest.raises(ValueError, match="Leakage"):
        bad.validate()


def test_masked_teacher_softmax_ignores_illegal_logits_and_validates_temperature() -> None:
    """Logits illegali enormi non devono ricevere massa nel target distillato."""
    logits = np.zeros((2, 40), dtype=np.float64)
    logits[:, 39] = 1_000_000.0
    masks = np.zeros((2, 40), dtype=bool)
    masks[0, [0, 1]] = True
    masks[1, [10, 20, 30]] = True

    probs = masked_softmax_batch(logits, masks, temperature=1.0)

    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)
    assert np.all(probs[~masks] == 0.0)
    with pytest.raises(ValueError, match="temperature"):
        masked_softmax_batch(logits, masks, temperature=0.0)


def test_supervised_update_reduces_holdout_kl_from_the_warm_start() -> None:
    """Il trainer deve avvicinare la policy ai target su validation e test non usati negli update."""
    dataset = _constant_target_dataset()
    result = train_suit_distillation(
        dataset,
        _zero_model(),
        epochs=20,
        batch_size=4,
        learning_rate=0.03,
        weight_decay=0.0,
        seed=7,
        paired_augmentation=False,
    )

    assert result.best_epoch > 0
    assert result.best_validation.kl_divergence < result.before_validation.kl_divergence * 0.1
    assert result.test.kl_divergence < result.before_test.kl_divergence * 0.1
    assert result.test.argmax_agreement == pytest.approx(1.0)


@pytest.mark.slow
def test_generation_and_training_cli_roundtrip_with_real_v13(tmp_path: Path) -> None:
    """Sei partite attraversano teacher 24x, file compresso, training e loader runtime."""
    dataset_path = tmp_path / "suit_teacher_6games.npz"
    model_path = tmp_path / "distilled.npz"
    report_path = tmp_path / "report.json"
    generate = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_suit_distillation_dataset.py"),
            "--model",
            str(ROOT / "data" / "models" / "best_a2c_v13.npz"),
            "--out",
            str(dataset_path),
            "--num-games",
            "6",
            "--seed",
            "17",
            "--opponent-mix",
            "mirror:1",
            "--progress-every",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert generate.returncode == 0, f"generator fallito:\n{generate.stdout}\n{generate.stderr}"
    dataset = load_suit_distillation_dataset(dataset_path)
    assert dataset.features.shape == (6 * 38, FEATURE_DIM_2P_V4)
    assert np.all(dataset.target_probs[~dataset.action_masks] == 0.0)

    train = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_suit_distillation.py"),
            "--data",
            str(dataset_path),
            "--init",
            str(ROOT / "data" / "models" / "best_a2c_v13.npz"),
            "--out",
            str(model_path),
            "--report-json",
            str(report_path),
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--lr",
            "0.0001",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert train.returncode == 0, f"trainer fallito:\n{train.stdout}\n{train.stderr}"
    agent = BCModelAgent.from_npz(model_path)
    assert agent.encoder_version == "v4"
    assert agent.overkill_guard_enabled is False
    assert report_path.exists()
