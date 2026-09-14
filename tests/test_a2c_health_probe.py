"""Test del protocollo multi-seed che instrada la prossima ablation A2C."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module():
    """Carica lo script CLI come modulo senza eseguire ``main``."""
    path = Path(__file__).resolve().parents[1] / "scripts/run_a2c_health_probe.py"
    spec = importlib.util.spec_from_file_location("run_a2c_health_probe_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _healthy_report(*, seed: int = 7) -> dict[str, Any]:
    """Costruisce quattro update sintetici che superano tutte le soglie."""
    updates = []
    for iteration in range(1, 5):
        updates.append(
            {
                "iteration": iteration,
                "games": iteration * 20,
                "signals": {
                    "critic_explained_variance": 0.20 + iteration * 0.01,
                    "critic_mean_squared_error": 0.10,
                    "advantage_mean": 0.10,
                    "advantage_std": 1.0,
                    "hidden_units_never_active": 10,
                    "hidden_activation_rate_mean": 0.50,
                },
                "global_gradient_l2": 2.0,
                "trunk_relative_update": 0.001,
                "actor_head_relative_update": 0.002,
            }
        )
    return {
        "schema": module.DIAGNOSTIC_SCHEMA,
        "method": {"passive": True},
        "config": {"seed": seed, "num_games": 80, "update_every": 20, "hidden_dim": 100},
        "artifacts": {"init": {"sha256": "init-sha"}},
        "initialization": {
            "critic_mode": "reset_zero",
            "init_critic_used": False,
        },
        "updates": updates,
    }


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Applica alla fixture i parametri congelati del protocollo."""
    return module.summarize_run(
        report,
        expected_seed=7,
        expected_num_games=80,
        expected_update_every=20,
        expected_init_sha256="init-sha",
        thresholds=module.HealthThresholds(),
    )


def test_healthy_run_passes_every_gate_and_routes_to_paired_schedule() -> None:
    """Segnali sani su ogni controllo non devono inventare una correzione numerica."""
    summary = _summarize(_healthy_report())

    assert all(summary["gates"].values())
    assert summary["metrics"]["late_updates"] == 2
    assert summary["metrics"]["critic_explained_variance_median"] == pytest.approx(0.235)
    assert summary["metrics"]["global_gradient_p95_over_median"] == 1.0
    assert module.route_next_experiment([summary])["next_experiment"] == "paired_training_schedule"


def test_critic_failure_has_priority_over_other_numerical_ablation() -> None:
    """Un critic insufficiente deve essere isolato prima di normalizzazione e clipping."""
    report = _healthy_report()
    for update in report["updates"][2:]:
        update["signals"]["critic_explained_variance"] = -0.5
        update["signals"]["advantage_mean"] = 1.0
        update["global_gradient_l2"] = 100.0
    summary = _summarize(report)

    assert summary["gates"]["critic"] is False
    assert summary["gates"]["advantage"] is False
    assert module.route_next_experiment([summary])["next_experiment"] == "critic_reset_vs_reuse"


def test_every_seed_must_pass_instead_of_hiding_failure_in_average() -> None:
    """Un solo seed negativo deve bloccare il gate critic aggregato."""
    healthy = _summarize(_healthy_report())
    failing_report = _healthy_report()
    for update in failing_report["updates"][2:]:
        update["signals"]["critic_explained_variance"] = -0.1
    failing = _summarize(failing_report)

    gates = module._aggregate_gates([healthy, failing])

    assert gates["critic"] is False
    assert gates["advantage"] is True
