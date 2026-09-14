"""Test del protocollo che confronta schedule A2C seriale e paired."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4


def _load_module():
    """Carica lo script CLI come modulo senza eseguire ``main``."""
    path = Path(__file__).resolve().parents[1] / "scripts/run_a2c_paired_schedule_probe.py"
    spec = importlib.util.spec_from_file_location("run_a2c_paired_schedule_probe_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _write_zero_value_model(path: Path) -> None:
    """Crea l'asset v4 minimo richiesto dall'opponent value-lookahead dello smoke."""
    feature_dim = int(FEATURE_DIM_2P_V4)
    np.savez(
        path,
        w1=np.zeros((feature_dim, 1), dtype=np.float32),
        b1=np.zeros(1, dtype=np.float32),
        w2=np.zeros(1, dtype=np.float32),
        b2=np.zeros(1, dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "value_mlp_v1",
                "feature_dim": feature_dim,
                "hidden_dim": 1,
                "encoder_version": "v4",
                "target": "residual",
                "target_scale": 120.0,
            }
        ),
    )


def _run_cli(command: list[str], *, root: Path) -> None:
    """Esegue lo smoke mostrando l'output del figlio quando il comando fallisce."""
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"Comando terminato con exit {completed.returncode}: {' '.join(command)}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )


def _regime_summaries(
    *,
    serial_strength_std: float,
    paired_strength_std: float,
    serial_gradient_cv: float,
    paired_gradient_cv: float,
) -> dict[str, dict[str, Any]]:
    """Costruisce il sottoinsieme aggregato letto dalla decisione."""
    return {
        "serial_same_games": {
            "vs_v14_point_diff": {"sample_std": serial_strength_std},
            "global_gradient_cv": {"median": serial_gradient_cv},
        },
        "paired_same_games": {
            "vs_v14_point_diff": {"sample_std": paired_strength_std},
            "global_gradient_cv": {"median": paired_gradient_cv},
        },
    }


def _direct(*values: float) -> list[dict[str, float]]:
    """Crea risultati diretti paired-minus-serial per i tre seed."""
    return [{"avg_point_diff_agent_a_minus_agent_b": value} for value in values]


def test_environment_alignment_matches_prefix_and_full_serial_pool() -> None:
    """Il paired a pari mazzi deve usare lo stesso pool; a pari game il suo prefisso."""
    schedules = {
        regime.name: module._expected_schedule(
            seed=17,
            regime=regime,
            base_games=8,
            update_every=4,
            opponent_mix="random:0.5,heuristic_v1:0.5",
        )
        for regime in module.REGIMES
    }

    alignment = module.validate_environment_alignment(schedules)

    assert alignment["paired_same_games_uses_serial_environment_prefix"] is True
    assert alignment["paired_same_decks_matches_serial_environments"] is True
    assert alignment["serial_environment_count"] == 8
    assert alignment["paired_same_games_environment_count"] == 4
    assert alignment["paired_same_decks_environment_count"] == 8


def test_decision_go_requires_nonregression_and_both_variance_signals() -> None:
    """Forza favorevole da sola non basta: devono calare entrambe le dispersioni."""
    summaries = _regime_summaries(
        serial_strength_std=0.4,
        paired_strength_std=0.3,
        serial_gradient_cv=0.5,
        paired_gradient_cv=0.4,
    )
    decision = module.decide(
        regime_summaries=summaries,
        direct_same_games=_direct(0.2, 0.0, -0.1),
        thresholds=module.DecisionThresholds(),
    )
    assert decision["verdict"] == "go_longer_paired_screen"

    summaries["paired_same_games"]["global_gradient_cv"]["median"] = 0.6
    decision = module.decide(
        regime_summaries=summaries,
        direct_same_games=_direct(0.2, 0.0, -0.1),
        thresholds=module.DecisionThresholds(),
    )
    assert decision["verdict"] == "inconclusive_keep_serial"


def test_decision_stops_on_material_strength_regression() -> None:
    """Due seed negativi e mediana sotto -0,25 devono chiudere la pista."""
    decision = module.decide(
        regime_summaries=_regime_summaries(
            serial_strength_std=0.4,
            paired_strength_std=0.3,
            serial_gradient_cv=0.5,
            paired_gradient_cv=0.4,
        ),
        direct_same_games=_direct(-0.4, -0.3, 0.1),
        thresholds=module.DecisionThresholds(),
    )
    assert decision["verdict"] == "stop_paired_schedule"
    assert decision["stop_strength"] is True


@pytest.mark.slow
@pytest.mark.numba
def test_cli_smoke_runs_all_regimes_and_direct_matches(tmp_path: Path) -> None:
    """Lo smoke attraversa nove training, suite comune, resume metadata e aggregazione."""
    root = Path(__file__).resolve().parents[1]
    value_model = tmp_path / "value_v4_zero.npz"
    _write_zero_value_model(value_model)
    command = [
        sys.executable,
        str(root / "scripts/run_a2c_paired_schedule_probe.py"),
        "--work-dir",
        str(tmp_path),
        "--base-games",
        "4",
        "--update-every",
        "2",
        "--eval-games",
        "4",
        "--seeds",
        "17,18,19",
        "--value-model",
        str(value_model),
    ]
    _run_cli(command, root=root)
    _run_cli([*command, "--resume"], root=root)

    report_path = tmp_path / "a2c_paired_schedule_g4_3seeds.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == module.SCHEMA
    assert len(report["runs"]) == 9
    assert len(report["reference_evaluations"]) == 9
    assert len(report["direct_evaluations"]["paired_same_games_vs_serial"]) == 3
    assert len(report["direct_evaluations"]["paired_same_decks_vs_serial"]) == 3
    assert all(
        alignment["paired_same_decks_matches_serial_environments"]
        for alignment in report["environment_alignment"].values()
    )
