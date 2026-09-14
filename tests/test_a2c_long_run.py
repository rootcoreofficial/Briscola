"""Regressioni end-to-end per training A2C streaming e resume esatto."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.training.a2c_checkpoint import A2C_RESUME_SCHEMA, parse_resume_json

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts/train_a2c.py"


def _run_training(*, out: Path, diagnostics: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
    """Esegue una minuscola ricetta Numba identica nei segmenti del test."""
    command = [
        sys.executable,
        str(_SCRIPT),
        "--out",
        str(out),
        "--rollout-engine",
        "fast",
        "--fast-rollout",
        "numba",
        "--opponent",
        "random",
        "--encoder-version",
        "v4",
        "--hidden-dim",
        "8",
        "--num-games",
        "8",
        "--update-every",
        "2",
        "--log-every",
        "1",
        "--metrics-mode",
        "summary",
        "--diagnostics-json",
        str(diagnostics),
        "--diagnostics-every",
        "2",
        "--seat-fair",
        "--seed",
        "123",
        *extra,
    ]
    return subprocess.run(command, cwd=_ROOT, check=True, capture_output=True, text=True)


@pytest.mark.slow
@pytest.mark.numba
def test_interrupted_resume_is_bit_identical_to_continuous_training(tmp_path: Path) -> None:
    """Pesi, critic, metriche e digest devono ignorare il confine tra processi."""
    continuous = tmp_path / "continuous.npz"
    continuous_diagnostics = tmp_path / "continuous.json"
    segmented = tmp_path / "segment_4.npz"
    segmented_diagnostics = tmp_path / "segment_4.json"
    checkpoint_dir = tmp_path / "checkpoints"
    resumed = tmp_path / "resumed.npz"
    resumed_diagnostics = tmp_path / "resumed.json"

    _run_training(out=continuous, diagnostics=continuous_diagnostics, extra=[])
    _run_training(
        out=segmented,
        diagnostics=segmented_diagnostics,
        extra=[
            "--stop-after-games",
            "4",
            "--checkpoint-games",
            "4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-prefix",
            "longrun",
        ],
    )
    checkpoint = checkpoint_dir / "longrun_4.npz"
    _run_training(
        out=resumed,
        diagnostics=resumed_diagnostics,
        extra=["--resume", str(checkpoint)],
    )

    with np.load(checkpoint, allow_pickle=False) as archive:
        state = parse_resume_json(archive["resume_state_json"])
        assert state["schema"] == A2C_RESUME_SCHEMA
        assert state["games_completed"] == 4
        assert state["optimizer_updates"] == 2
        assert archive["resume_st_w1_m"].shape == archive["w1"].shape

    with np.load(continuous, allow_pickle=False) as expected, np.load(resumed, allow_pickle=False) as actual:
        for name in ("w1", "b1", "w2", "b2", "wv", "bv"):
            np.testing.assert_array_equal(actual[name], expected[name])
        expected_metadata = json.loads(str(expected["metadata_json"]))
        actual_metadata = json.loads(str(actual["metadata_json"]))

    assert actual_metadata["training_config_fingerprint"] == expected_metadata["training_config_fingerprint"]
    assert actual_metadata["train"]["training_schedule"] == expected_metadata["train"]["training_schedule"]
    assert actual_metadata["metrics_summary"] == expected_metadata["metrics_summary"]
    assert actual_metadata["train"]["run_complete"] is True

    expected_report = json.loads(continuous_diagnostics.read_text(encoding="utf-8"))
    actual_report = json.loads(resumed_diagnostics.read_text(encoding="utf-8"))
    assert [row["iteration"] for row in actual_report["updates"]] == [1, 2, 4]
    assert actual_report["updates"] == expected_report["updates"]
    assert actual_report["summary"] == expected_report["summary"]


@pytest.mark.slow
@pytest.mark.numba
def test_resume_rejects_changed_training_configuration(tmp_path: Path) -> None:
    """Un flag comportamentale diverso non deve creare una continuazione solo apparente."""
    checkpoint_dir = tmp_path / "checkpoints"
    _run_training(
        out=tmp_path / "segment.npz",
        diagnostics=tmp_path / "segment.json",
        extra=[
            "--stop-after-games",
            "4",
            "--checkpoint-games",
            "4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-prefix",
            "longrun",
        ],
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_training(
            out=tmp_path / "invalid.npz",
            diagnostics=tmp_path / "invalid.json",
            extra=["--resume", str(checkpoint_dir / "longrun_4.npz"), "--lr", "0.001"],
        )

    assert "Configurazione del resume diversa" in exc_info.value.stderr
