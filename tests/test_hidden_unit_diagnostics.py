"""Test della diagnostica causale sulle unità ReLU delle policy MLP."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.evaluation.hidden_units import (
    HiddenUnitThresholds,
    ablate_mlp_hidden_units,
    analyze_hidden_unit_arrays,
    analyze_suit_ablation_arrays,
    reinitialize_mlp_hidden_units,
)
from briscola_ai.ai.models.bc_model import MLPBCModel


def _model(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> MLPBCModel:
    """Costruisce una MLP sintetica con metadati minimi coerenti."""
    return MLPBCModel(
        w1=np.asarray(w1, dtype=np.float32),
        b1=np.asarray(b1, dtype=np.float32),
        w2=np.asarray(w2, dtype=np.float32),
        b2=np.asarray(b2, dtype=np.float32),
        metadata={"format": "mlp_bc_v1", "feature_dim": int(w1.shape[0])},
    )


def test_hidden_unit_analysis_finds_dead_and_causally_influential_units() -> None:
    """Un'unità spenta e una che decide l'argmax devono avere firme diverse e leggibili."""
    w1 = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    b1 = np.asarray([0.0, -1.0, 1.0], dtype=np.float32)
    w2 = np.zeros((3, 40), dtype=np.float32)
    w2[0, 0] = 2.0  # unità 0 rende sempre vincente action 0
    w2[2, 1] = 1.0  # unità 2 è il fallback quando la 0 viene spenta
    model = _model(w1, b1, w2, np.zeros(40, dtype=np.float32))
    inputs = np.asarray([[0.6, 0.0], [1.0, 1.0], [2.0, -1.0], [3.0, 0.5]], dtype=np.float32)
    masks = np.zeros((4, 40), dtype=bool)
    masks[:, :2] = True

    report = analyze_hidden_unit_arrays(model, inputs, masks)
    units = {row["unit"]: row for row in report["units"]}

    assert report["utilization"]["dead_units"] == [1]
    assert units[1]["activation_rate"] == 0.0
    assert units[1]["ablation_action_flip_rate"] == 0.0
    assert units[0]["ablation_action_flip_rate"] == 1.0
    assert report["influence"]["top_by_action_flip"][0]["unit"] == 0
    assert report["influence"]["dominant_units"] == [0]


def test_suit_ablation_remaps_actions_and_can_localize_one_asymmetric_unit() -> None:
    """Se una sola unità crea il flip, azzerarla deve portare il tasso esattamente a zero."""
    w1 = np.asarray([[1.0]], dtype=np.float32)
    b1 = np.asarray([0.0], dtype=np.float32)
    w2 = np.zeros((1, 40), dtype=np.float32)
    w2[0, 0] = 1.0
    b2 = np.zeros(40, dtype=np.float32)
    b2[1] = 0.5
    model = _model(w1, b1, w2, b2)

    orbit_inputs = np.ones((1, 24, 1), dtype=np.float32)
    orbit_inputs[0, 1, 0] = 0.0  # soltanto questa ristampa spegne l'unità
    orbit_masks = np.zeros((1, 24, 40), dtype=bool)
    orbit_masks[:, :, :2] = True
    remap = np.tile(np.arange(40, dtype=np.int16), (24, 1))

    report = analyze_suit_ablation_arrays(model, orbit_inputs, orbit_masks, remap)

    assert report["baseline_flip_rate"] == pytest.approx(1.0 / 23.0)
    assert report["best_single_unit_removal"]["unit"] == 0
    assert report["best_single_unit_removal"]["ablation_flip_rate"] == 0.0
    assert report["best_single_unit_removal"]["delta_vs_baseline"] == pytest.approx(-1.0 / 23.0)


def test_threshold_validation_rejects_overlapping_dead_and_always_active_bands() -> None:
    """Etichette incompatibili devono fallire prima dell'analisi."""
    with pytest.raises(ValueError, match="dead"):
        HiddenUnitThresholds(dead_activation_rate_max=0.8, always_active_rate_min=0.7).validate()


def test_joint_ablation_zeros_only_selected_output_rows() -> None:
    """La copia deve conservare ogni peso tranne le righe w2 causalmente rimosse."""
    rng = np.random.default_rng(42)
    model = _model(
        rng.normal(size=(3, 4)),
        rng.normal(size=4),
        rng.normal(size=(4, 40)),
        rng.normal(size=40),
    )
    ablated = ablate_mlp_hidden_units(model, (1, 3), metadata={"label": "ablated"})

    assert np.array_equal(ablated.w1, model.w1)
    assert np.array_equal(ablated.b1, model.b1)
    assert np.array_equal(ablated.b2, model.b2)
    assert np.array_equal(ablated.w2[[0, 2]], model.w2[[0, 2]])
    assert np.count_nonzero(ablated.w2[[1, 3]]) == 0
    assert np.count_nonzero(model.w2[[1, 3]]) > 0
    assert ablated.metadata == {"label": "ablated"}

    with pytest.raises(ValueError, match="unici"):
        ablate_mlp_hidden_units(model, (1, 1))
    with pytest.raises(ValueError, match="fuori range"):
        ablate_mlp_hidden_units(model, (4,))


def test_reinitialization_is_nested_deterministic_and_matches_ablation_logits() -> None:
    """La stessa unità deve ricevere gli stessi ingressi e partire con contributo nullo."""
    rng = np.random.default_rng(7)
    model = _model(
        rng.normal(size=(3, 4)),
        rng.normal(size=4),
        rng.normal(size=(4, 40)),
        rng.normal(size=40),
    )
    original_w2 = model.w2.copy()
    reset_one = reinitialize_mlp_hidden_units(model, (1,), seed=17)
    reset_nested = reinitialize_mlp_hidden_units(model, (1, 3), seed=17)
    ablated_nested = ablate_mlp_hidden_units(model, (1, 3))
    inputs = rng.normal(size=(12, 3)).astype(np.float32)

    np.testing.assert_array_equal(reset_one.w1[:, 1], reset_nested.w1[:, 1])
    np.testing.assert_array_equal(reset_nested.w1[:, [0, 2]], model.w1[:, [0, 2]])
    assert not np.array_equal(reset_nested.w1[:, 1], model.w1[:, 1])
    assert np.count_nonzero(reset_nested.w2[[1, 3]]) == 0
    np.testing.assert_array_equal(reset_nested.logits(inputs), ablated_nested.logits(inputs))
    np.testing.assert_array_equal(model.w2, original_w2)


@pytest.mark.slow
def test_hidden_unit_cli_runs_end_to_end_and_writes_finite_json(tmp_path: Path) -> None:
    """Lo smoke attraversa raccolta bilanciata, due modelli, ablation e serializzazione."""
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden_dim = 4
    model_path = tmp_path / "synthetic_v4.npz"
    np.savez(
        model_path,
        w1=np.zeros((feature_dim, hidden_dim), dtype=np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=np.zeros((hidden_dim, 40), dtype=np.float32),
        b2=np.asarray([float(action_id % 10) for action_id in range(40)], dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "mlp_bc_v1",
                "feature_dim": feature_dim,
                "encoder_version": "v4",
                "inference_overkill_guard": False,
            }
        ),
    )
    root = Path(__file__).resolve().parent.parent
    output = tmp_path / "hidden.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "diagnose_hidden_units.py"),
            "--model",
            str(model_path),
            "--reference-model",
            str(model_path),
            "--seed-count",
            "1",
            "--samples-per-cell",
            "1",
            "--suit-samples-per-cell",
            "1",
            "--opponents",
            "random",
            "--out-json",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"), parse_constant=lambda value: pytest.fail(value))
    assert payload["schema"] == "briscola.hidden_unit_diagnostic.v1"
    assert payload["coverage"]["selected_observations"] == 4
    assert payload["models"]["primary"]["hidden_dim"] == hidden_dim
    assert payload["models"]["primary"]["suit_ablation"]["nonidentity_comparisons"] == 4 * 23
    assert "Decisione:" in completed.stdout
