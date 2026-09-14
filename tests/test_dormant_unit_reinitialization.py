"""Test dello screening controllato per riattivare unità dormienti di v14."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.training.suit_distillation import (
    DATASET_FORMAT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SuitDistillationDataset,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """Carica la CLI per testare separatamente selezione e gate."""
    path = ROOT / "scripts/run_dormant_reinitialization_screen.py"
    spec = importlib.util.spec_from_file_location("run_dormant_reinitialization_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _write_evidence_pair(tmp_path: Path, *, model_sha256: str, zero_units: list[int]) -> tuple[Path, Path]:
    """Crea due evidenze minime con legame SHA equivalente al protocollo reale."""
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "schema": "briscola.hidden_unit_diagnostic.v1",
                "artifacts": {"primary": {"sha256": model_sha256}},
                "thresholds": {"activation_epsilon": 1e-8},
                "models": {
                    "primary": {
                        "units": [
                            {"unit": unit, "activation_rate": 0.0 if unit in zero_units else 0.5}
                            for unit in range(max(zero_units) + 2)
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    diagnostic_sha256 = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
    holdout = tmp_path / "holdout.json"
    holdout.write_text(
        json.dumps(
            {
                "schema": "briscola.dormant_unit_ablation.v1",
                "artifacts": {
                    "source_model": {"sha256": model_sha256},
                    "source_evidence": {"sha256": diagnostic_sha256},
                },
                "policy_agreement": {"selected_units_never_active": zero_units},
            }
        ),
        encoding="utf-8",
    )
    return diagnostic, holdout


def test_pool_requires_zero_activity_in_both_evidences_and_ranking_is_stable(tmp_path: Path) -> None:
    """La selezione non deve includere unità attive né dipendere dall'ordine dei file."""
    diagnostic, holdout = _write_evidence_pair(tmp_path, model_sha256="model", zero_units=[1, 3, 5])
    holdout_payload = json.loads(holdout.read_text(encoding="utf-8"))
    holdout_payload["policy_agreement"]["selected_units_never_active"] = [1, 5, 7]
    holdout.write_text(json.dumps(holdout_payload), encoding="utf-8")

    pool, provenance = module._selection_pool(diagnostic, holdout, source_model_sha256="model")
    first = module._rank_units(pool, seed=17)
    second = module._rank_units(list(reversed(pool)), seed=17)

    assert pool == [1, 5]
    assert provenance["intersection_count"] == 2
    assert first == second
    assert sorted(first[:2]) == sorted(second[:2])


def test_decision_selects_only_by_validation_and_requires_every_gate() -> None:
    """Un miglior test non può compensare KL validation o attività insufficienti."""

    def row(key: str, count: int, validation_kl: float, validation_agreement: float) -> dict:
        return {
            "key": key,
            "reinitialized_unit_count": count,
            "initial_policy_agreement": {
                "validation": {"agreement_rate": 1.0},
                "test": {"agreement_rate": 1.0},
            },
            "best_validation": {
                "kl_divergence": validation_kl,
                "argmax_agreement": validation_agreement,
            },
            "test": {"kl_divergence": validation_kl, "argmax_agreement": validation_agreement},
            "post_training_reinitialized_activity": {
                "active_at_least_one_percent_fraction": 1.0,
                "learned_outgoing_fraction": 1.0,
            },
        }

    variants = {
        "control": row("control", 0, 0.100, 0.9500),
        "reset_8": row("reset_8", 8, 0.098, 0.9505),
        "reset_16": row("reset_16", 16, 0.099, 0.9600),
    }
    variants["reset_16"]["test"] = {"kl_divergence": 0.001, "argmax_agreement": 1.0}
    decision = module._decision(variants)

    assert decision["selected_variant"] == "reset_8"
    assert decision["verdict"] == "go_evaluate_reinitialized_candidate"
    variants["reset_8"]["post_training_reinitialized_activity"]["learned_outgoing_fraction"] = 0.5
    assert module._decision(variants)["verdict"] == "stop_dormant_reinitialization"


@pytest.mark.slow
def test_cli_runs_control_and_nested_reset_variants(tmp_path: Path) -> None:
    """Lo smoke attraversa evidenze, training comune, modelli locali e decisione JSON."""
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden_dim = 4
    rng = np.random.default_rng(41)
    source_path = tmp_path / "source.npz"
    np.savez(
        source_path,
        w1=rng.normal(0.0, 0.05, size=(feature_dim, hidden_dim)).astype(np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=rng.normal(0.0, 0.05, size=(hidden_dim, 40)).astype(np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": feature_dim, "encoder_version": "v4"}),
    )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    diagnostic, holdout = _write_evidence_pair(tmp_path, model_sha256=source_sha256, zero_units=[1, 2])

    num_examples = 24
    features = rng.normal(0.0, 0.2, size=(num_examples, feature_dim)).astype(np.float32)
    masks = np.zeros((num_examples, 40), dtype=bool)
    masks[:, :3] = True
    targets = np.zeros((num_examples, 40), dtype=np.float32)
    targets[:, :3] = (0.8, 0.15, 0.05)
    game_ids = np.repeat(np.arange(12, dtype=np.int32), 2)
    split_by_game = np.asarray(
        [SPLIT_TRAIN] * 8 + [SPLIT_VALIDATION] * 2 + [SPLIT_TEST] * 2,
        dtype=np.uint8,
    )
    dataset = SuitDistillationDataset(
        features=features,
        action_masks=masks,
        target_probs=targets,
        target_action_ids=np.zeros(num_examples, dtype=np.int16),
        game_ids=game_ids,
        split_ids=split_by_game[game_ids],
        metadata={"format": DATASET_FORMAT, "encoder_version": "v4"},
    )
    data_path = tmp_path / "dataset.npz"
    dataset.save(data_path)
    out_dir = tmp_path / "models"
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_dormant_reinitialization_screen.py"),
            "--data",
            str(data_path),
            "--init",
            str(source_path),
            "--diagnostic-evidence",
            str(diagnostic),
            "--holdout-evidence",
            str(holdout),
            "--variants",
            "0",
            "1",
            "2",
            "--expected-pool-size",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--out-dir",
            str(out_dir),
            "--out-json",
            str(report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "briscola.dormant_reinitialization_screen.v1"
    assert set(report["variants"]) == {"control", "reset_1", "reset_2"}
    assert report["selection"]["intersection_count"] == 2
    assert len(list(out_dir.glob("*.npz"))) == 3
