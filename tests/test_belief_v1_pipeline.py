"""Pipeline belief v1: roster mirror, split ermetico e metriche calibrate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from briscola_ai.ai.models.belief_model import load_belief_model_npz

_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, filename: str):
    path = _ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


_generator = _load_script_module("generate_belief_dataset_v1_test", "generate_belief_dataset.py")
_trainer = _load_script_module("train_belief_v1_test", "train_belief.py")
_summarizer = _load_script_module("summarize_belief_folds_test", "summarize_belief_folds.py")
_runner = _load_script_module("run_belief_v1_gate_test", "run_belief_v1_gate.py")


def test_weighted_schedule_is_balanced_and_reproducible() -> None:
    """Due blocchi completi rispettano esattamente i pesi congelati."""
    roster_path = _ROOT / "docs/plans/belief-v1-roster-2026-07-14.json"
    entries, _metadata = _generator.load_roster(
        roster_path,
        legacy_policy_model=_ROOT / "data/models/best_a2c_v14.npz",
    )
    total_weight = sum(entry.weight for entry in entries)
    first = _generator.build_weighted_schedule(entries, num_games=2 * total_weight, seed=17)
    second = _generator.build_weighted_schedule(entries, num_games=2 * total_weight, seed=17)

    assert first == second
    assert {index: first.count(index) for index in range(len(entries))} == {
        index: 2 * entry.weight for index, entry in enumerate(entries)
    }


def test_leave_one_out_rejects_games_crossing_opponent_styles() -> None:
    """Una partita non puo' finire contemporaneamente in train e validation."""
    game_index = np.asarray([0, 0, 1, 1], dtype=np.int64)
    opponent_id = np.asarray([0, 1, 0, 0], dtype=np.int64)

    try:
        _trainer.build_split(
            game_index,
            opponent_id,
            ("a", "b"),
            holdout_opponent="b",
            val_frac=0.1,
        )
    except ValueError as exc:
        assert "attraversa piu' opponent_id" in str(exc)
    else:
        raise AssertionError("Lo split deve rifiutare leakage tra stili nella stessa partita")


def test_small_multi_style_training_reports_brier_ece_and_baseline(tmp_path: Path) -> None:
    """Smoke completo: genera dataset v2 e allena un fold leave-one-out."""
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(
        json.dumps(
            {
                "schema": "briscola.belief_roster.v1",
                "items": [
                    {
                        "id": "v14",
                        "agent": "bc_model",
                        "model_path": str(_ROOT / "data/models/best_a2c_v14.npz"),
                        "weight": 1,
                    },
                    {"id": "heuristic_v1", "agent": "heuristic_v1", "weight": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "belief_dataset.npz"
    candidate_path = tmp_path / "belief_candidate.npz"

    subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/generate_belief_dataset.py"),
            "--roster",
            str(roster_path),
            "--num-games",
            "20",
            "--seed",
            "3",
            "--log-every",
            "0",
            "--out",
            str(dataset_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(dataset_path) as dataset:
        game_index = np.asarray(dataset["game_index"], dtype=np.int64)
        opponent_id = np.asarray(dataset["opponent_id"], dtype=np.int64)
        metadata = json.loads(str(dataset["metadata_json"]))
    assert metadata["format"] == "belief_dataset_v2"
    for game in np.unique(game_index):
        assert np.unique(opponent_id[game_index == game]).size == 1

    subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/train_belief.py"),
            "--data",
            str(dataset_path),
            "--out",
            str(candidate_path),
            "--hidden-dim",
            "8",
            "--epochs",
            "2",
            "--batch-size",
            "32",
            "--seed",
            "4",
            "--holdout-opponent",
            "heuristic_v1",
            "--baseline-model",
            str(_ROOT / "data/models/belief_v0_h128_50k_seed20260702.npz"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidate = load_belief_model_npz(candidate_path)
    train_meta = candidate.metadata["train"]

    assert train_meta["split"]["strategy"] == "leave_one_opponent_out"
    assert train_meta["split"]["val_opponents"] == ["heuristic_v1"]
    assert {"bce", "topk_recall", "brier", "ece"} <= set(train_meta["validation"]["candidate"])
    assert train_meta["validation"]["baseline"]["artifact"]["sha256"]


def test_fold_summary_requires_complete_roster_and_applies_gate(tmp_path: Path) -> None:
    """L'aggregatore usa macro-medie per stile e non accetta fold mancanti."""
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps(
            {
                "schema": "briscola.belief_roster.v1",
                "items": [
                    {"id": "style_a", "agent": "heuristic_v1"},
                    {"id": "style_b", "agent": "heuristic_v2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"dataset")
    baseline = _ROOT / "data/models/belief_v0_h128_50k_seed20260702.npz"
    dataset_artifact = {"path": str(dataset), "sha256": "dataset-sha", "size_bytes": 7}
    baseline_artifact = {"path": str(baseline), "sha256": "baseline-sha", "size_bytes": baseline.stat().st_size}

    paths: list[Path] = []
    for index, holdout in enumerate(("style_a", "style_b")):
        path = tmp_path / f"fold_{holdout}.npz"
        candidate_bce = 0.48 + index * 0.01
        metadata = {
            "format": "belief_mlp_v1",
            "encoder_version": "v4",
            "feature_dim": 369,
            "dataset_artifact": dataset_artifact,
            "train": {
                "split": {"strategy": "leave_one_opponent_out", "holdout_opponent": holdout},
                "validation": {
                    "candidate": {"bce": candidate_bce, "topk_recall": 0.61, "brier": 0.17, "ece": 0.02},
                    "baseline": {
                        "artifact": baseline_artifact,
                        "metrics": {"bce": 0.50, "topk_recall": 0.60, "brier": 0.18, "ece": 0.03},
                    },
                },
            },
        }
        np.savez(
            path,
            w1=np.zeros((369, 2), dtype=np.float32),
            b1=np.zeros(2, dtype=np.float32),
            w2=np.zeros((2, 40), dtype=np.float32),
            b2=np.zeros(40, dtype=np.float32),
            metadata_json=json.dumps(metadata),
        )
        paths.append(path)

    report = _summarizer.summarize_folds(
        tuple(paths),
        roster_path=roster,
        gates=_summarizer.FoldGates(),
        pilot=False,
    )

    assert report["decision"]["verdict"] == "go_train_all_styles_candidate"
    assert np.isclose(report["macro"]["bce_relative_improvement"], 0.03)


def test_gate_runner_uses_complete_weight_blocks_and_stable_paths(tmp_path: Path) -> None:
    """Il comando lungo richiede blocchi completi e nomi riprendibili senza ambiguita'."""
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps(
            {
                "schema": "briscola.belief_roster.v1",
                "items": [
                    {"id": "policy/a", "agent": "heuristic_v1", "weight": 2},
                    {"id": "policy-b", "agent": "heuristic_v2", "weight": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    opponent_ids, total_weight = _runner.load_roster_ids(roster)
    paths = _runner.build_protocol_paths(
        tmp_path / "work",
        opponent_ids,
        num_games=30,
        seed=17,
        hidden_dim=128,
    )

    assert opponent_ids == ("policy/a", "policy-b")
    assert total_weight == 3
    assert paths.dataset.name == "belief_v1_multistyle_g30_seed17.npz"
    assert [path.name for path in paths.folds] == [
        "belief_v1_holdout_policy_a_g30_seed17.npz",
        "belief_v1_holdout_policy-b_g30_seed17.npz",
    ]
    assert paths.candidate.name == "belief_v1_all_styles_h128_g30_seed17.npz"
