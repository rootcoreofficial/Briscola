"""Test del formato sharded, del resume e del trainer streaming di distillazione."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.models import BCModelAgent, MLPBCModel
from briscola_ai.ai.training.suit_distillation import (
    DATASET_FORMAT,
    SuitDistillationDataset,
    make_game_split_ids,
)
from briscola_ai.ai.training.suit_distillation_shards import (
    SHARDED_MANIFEST_SCHEMA,
    SHARDED_MANIFEST_STATUS_COMPLETE,
    SuitDistillationShard,
    derive_shard_seed,
    load_sharded_suit_distillation_dataset,
    sha256_file,
    train_suit_distillation_sharded,
)

ROOT = Path(__file__).resolve().parents[1]


def _zero_model() -> MLPBCModel:
    """MLP piccola con distribuzione iniziale uniforme sulle azioni legali."""
    hidden_dim = 4
    return MLPBCModel(
        w1=np.zeros((FEATURE_DIM_2P_V4, hidden_dim), dtype=np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=np.zeros((hidden_dim, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata={"format": "mlp_bc_v1", "feature_dim": FEATURE_DIM_2P_V4, "encoder_version": "v4"},
    )


def _write_synthetic_manifest(tmp_path: Path) -> Path:
    """Crea tre shard realistici da 38 decisioni per partita e split globali disgiunti."""
    manifest_path = tmp_path / "dataset" / "manifest.json"
    shard_dir = manifest_path.parent / "shards"
    shard_dir.mkdir(parents=True)
    fingerprint = "a" * 64
    games_per_shard = 4
    total_games = 12
    split_seed = 123
    train_fraction = 0.5
    validation_fraction = 0.25
    split_by_game = make_game_split_ids(
        total_games,
        seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    records: list[SuitDistillationShard] = []

    for shard_index in range(3):
        game_start = shard_index * games_per_shard
        game_stop = game_start + games_per_shard
        game_ids = np.repeat(np.arange(game_start, game_stop, dtype=np.int32), 38)
        num_examples = int(game_ids.size)
        rng = np.random.default_rng(100 + shard_index)
        features = rng.normal(0.0, 0.2, size=(num_examples, FEATURE_DIM_2P_V4)).astype(np.float32)
        masks = np.zeros((num_examples, 40), dtype=bool)
        masks[:, [0, 1, 2]] = True
        targets = np.zeros((num_examples, 40), dtype=np.float32)
        targets[:, [0, 1, 2]] = (0.80, 0.15, 0.05)
        shard_splits = split_by_game[game_ids]
        split_counts = {
            "train": int(np.sum(split_by_game[game_start:game_stop] == 0)),
            "validation": int(np.sum(split_by_game[game_start:game_stop] == 1)),
            "test": int(np.sum(split_by_game[game_start:game_stop] == 2)),
        }
        dataset = SuitDistillationDataset(
            features=features,
            action_masks=masks,
            target_probs=targets,
            target_action_ids=np.zeros(num_examples, dtype=np.int16),
            game_ids=game_ids,
            split_ids=shard_splits,
            metadata={
                "format": DATASET_FORMAT,
                "encoder_version": "v4",
                "manifest_config_fingerprint": fingerprint,
                "shard_index": shard_index,
                "game_id_start": game_start,
                "game_id_stop": game_stop,
                "num_games": games_per_shard,
                "num_examples": num_examples,
                "split_game_counts": split_counts,
                "opponent_game_counts": {"mirror": games_per_shard},
            },
        )
        shard_path = shard_dir / f"shard-{shard_index:05d}-of-00003.npz"
        dataset.save(shard_path)
        records.append(
            SuitDistillationShard(
                index=shard_index,
                path=f"shards/{shard_path.name}",
                sha256=sha256_file(shard_path),
                size_bytes=shard_path.stat().st_size,
                seed=derive_shard_seed(77, shard_index),
                game_id_start=game_start,
                game_id_stop=game_stop,
                num_games=games_per_shard,
                num_examples=num_examples,
                split_game_counts=split_counts,
                opponent_game_counts={"mirror": games_per_shard},
            )
        )

    payload = {
        "schema": SHARDED_MANIFEST_SCHEMA,
        "status": SHARDED_MANIFEST_STATUS_COMPLETE,
        "config_fingerprint": fingerprint,
        "dataset": {
            "format": DATASET_FORMAT,
            "encoder_version": "v4",
            "feature_dim": FEATURE_DIM_2P_V4,
            "action_dim": 40,
            "num_games": total_games,
            "num_examples": total_games * 38,
            "num_shards": 3,
            "games_per_shard": games_per_shard,
            "seed": 77,
            "split_seed": split_seed,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "split_game_counts": {
                "train": int(np.sum(split_by_game == 0)),
                "validation": int(np.sum(split_by_game == 1)),
                "test": int(np.sum(split_by_game == 2)),
            },
        },
        "teacher_model": {"path": "teacher.npz", "sha256": "b" * 64, "size_bytes": 1},
        "shards": [record.to_payload() for record in records],
        "completed": {
            "num_shards": 3,
            "num_games": total_games,
            "num_examples": total_games * 38,
            "split_game_counts": {
                "train": int(np.sum(split_by_game == 0)),
                "validation": int(np.sum(split_by_game == 1)),
                "test": int(np.sum(split_by_game == 2)),
            },
            "opponent_game_counts": {"mirror": total_games},
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def test_manifest_validates_hashes_ranges_and_lazy_shard_content(tmp_path: Path) -> None:
    """Il loader deve coprire game_id 0..N senza unire gli array o fidarsi del solo JSON."""
    manifest_path = _write_synthetic_manifest(tmp_path)
    corpus = load_sharded_suit_distillation_dataset(manifest_path, verify_hashes=True)

    assert not hasattr(corpus, "features")
    assert [item.game_id_start for item in corpus.shards] == [0, 4, 8]
    assert [item.game_id_stop for item in corpus.shards] == [4, 8, 12]
    last = corpus.load_shard(corpus.shards[-1])
    assert np.unique(last.game_ids).tolist() == [8, 9, 10, 11]
    assert last.indices("test").size % 38 == 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["shards"][1]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_sharded_suit_distillation_dataset(manifest_path, verify_hashes=True)


def test_streaming_training_is_deterministic_and_reduces_global_holdout_kl(tmp_path: Path) -> None:
    """Stesso seed e manifest devono produrre gli stessi pesi e imparare il target soft."""
    corpus = load_sharded_suit_distillation_dataset(_write_synthetic_manifest(tmp_path), verify_hashes=True)
    kwargs = {
        "epochs": 12,
        "batch_size": 32,
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "seed": 91,
        "paired_augmentation": False,
    }

    first = train_suit_distillation_sharded(corpus, _zero_model(), **kwargs)
    second = train_suit_distillation_sharded(corpus, _zero_model(), **kwargs)

    assert first.best_epoch > 0
    assert first.best_validation.kl_divergence < first.before_validation.kl_divergence * 0.1
    assert first.test.kl_divergence < first.before_test.kl_divergence * 0.1
    np.testing.assert_array_equal(first.w1, second.w1)
    np.testing.assert_array_equal(first.b2, second.b2)
    assert first.epochs == second.epochs


@pytest.mark.slow
def test_sharded_cli_resume_verify_and_training_roundtrip(tmp_path: Path) -> None:
    """Un'interruzione dopo il primo shard deve riprendere senza riscriverlo e produrre una policy caricabile."""
    data_dir = tmp_path / "data"
    common_generate = [
        sys.executable,
        str(ROOT / "scripts/generate_suit_distillation_shards.py"),
        "--model",
        str(ROOT / "data/models/best_a2c_v13.npz"),
        "--out-dir",
        str(data_dir),
        "--num-games",
        "9",
        "--games-per-shard",
        "3",
        "--seed",
        "17",
        "--opponent-mix",
        "mirror:1",
        "--progress-every",
        "0",
    ]
    first = subprocess.run(
        [*common_generate, "--stop-after-shards", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, f"prima tranche fallita:\n{first.stdout}\n{first.stderr}"
    first_shard = data_dir / "shards/shard-00000-of-00003.npz"
    first_digest = sha256_file(first_shard)

    resumed = subprocess.run([*common_generate, "--resume"], cwd=ROOT, capture_output=True, text=True)
    assert resumed.returncode == 0, f"resume fallito:\n{resumed.stdout}\n{resumed.stderr}"
    assert sha256_file(first_shard) == first_digest
    manifest_path = data_dir / "manifest.json"
    corpus = load_sharded_suit_distillation_dataset(manifest_path, verify_hashes=True)
    assert len(corpus.shards) == 3

    model_path = tmp_path / "student.npz"
    report_path = tmp_path / "train.json"
    trained = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_suit_distillation_shards.py"),
            "--manifest",
            str(manifest_path),
            "--init",
            str(ROOT / "data/models/best_a2c_v13.npz"),
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
            "--seed",
            "19",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert trained.returncode == 0, f"trainer fallito:\n{trained.stdout}\n{trained.stderr}"
    agent = BCModelAgent.from_npz(model_path)
    assert agent.encoder_version == "v4"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "briscola.suit_distillation_sharded_train.v1"
    assert report["dataset"]["num_games"] == 9
