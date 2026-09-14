"""Test delle metriche passive usate per diagnosticare il trainer A2C."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.training.a2c_diagnostics import (
    A2CArrayGroups,
    A2CSignalAccumulator,
    build_update_diagnostics,
    summarize_update_diagnostics,
)


def test_signal_accumulator_summarizes_returns_advantages_and_hidden_units() -> None:
    """I momenti aggregati devono coincidere con il calcolo diretto sui singoli step."""
    accumulator = A2CSignalAccumulator(hidden_dim=4)
    accumulator.observe(
        returns_to_go=np.asarray([1.0], dtype=np.float32),
        value_preds=np.asarray([0.5], dtype=np.float32),
        hidden=np.asarray([[0.0, 2.0, 0.0, 0.0]], dtype=np.float32),
    )
    accumulator.observe(
        returns_to_go=np.asarray([3.0], dtype=np.float32),
        value_preds=np.asarray([1.0], dtype=np.float32),
        hidden=np.asarray([[1.0, 0.0, 3.0, 0.0]], dtype=np.float32),
    )

    snapshot = accumulator.snapshot()

    assert snapshot.steps == 2
    assert snapshot.return_mean == pytest.approx(2.0)
    assert snapshot.return_std == pytest.approx(1.0)
    assert snapshot.value_mean == pytest.approx(0.75)
    assert snapshot.value_std == pytest.approx(0.25)
    assert snapshot.advantage_mean == pytest.approx(1.25)
    assert snapshot.advantage_std == pytest.approx(0.75)
    assert snapshot.advantage_rms == pytest.approx(np.sqrt(2.125))
    assert snapshot.critic_mean_squared_error == pytest.approx(2.125)
    assert snapshot.advantage_abs_mean == pytest.approx(1.25)
    assert snapshot.advantage_min == pytest.approx(0.5)
    assert snapshot.advantage_max == pytest.approx(2.0)
    assert snapshot.advantage_positive_fraction == pytest.approx(1.0)
    assert snapshot.critic_explained_variance == pytest.approx(0.4375)
    assert snapshot.hidden_activation_rate_mean == pytest.approx(0.375)
    assert snapshot.hidden_activation_rate_p10 == pytest.approx(0.15)
    assert snapshot.hidden_activation_rate_p50 == pytest.approx(0.5)
    assert snapshot.hidden_activation_rate_p90 == pytest.approx(0.5)
    assert snapshot.hidden_units_never_active == 1
    assert snapshot.hidden_mean_activation == pytest.approx(0.75)


def test_update_diagnostics_separates_gradient_and_parameter_groups() -> None:
    """Norme e passi relativi devono rispettare trunk, actor e critic separati."""
    accumulator = A2CSignalAccumulator(hidden_dim=1)
    accumulator.observe(
        returns_to_go=np.asarray([1.0], dtype=np.float32),
        value_preds=np.asarray([0.0], dtype=np.float32),
        hidden=np.asarray([[1.0]], dtype=np.float32),
    )
    gradients = A2CArrayGroups(
        trunk=(np.asarray([3.0]), np.asarray([4.0])),
        actor_head=(np.asarray([12.0]),),
        critic_head=(np.asarray([0.0]), np.asarray([5.0])),
    )
    before = A2CArrayGroups(
        trunk=(np.asarray([3.0]), np.asarray([4.0])),
        actor_head=(np.asarray([12.0]),),
        critic_head=(np.asarray([0.0]), np.asarray([0.0])),
    )
    after = A2CArrayGroups(
        trunk=(np.asarray([6.0]), np.asarray([8.0])),
        actor_head=(np.asarray([9.0]),),
        critic_head=(np.asarray([3.0]), np.asarray([4.0])),
    )

    update = build_update_diagnostics(
        iteration=1,
        games=20,
        signals=accumulator.snapshot(),
        gradients=gradients,
        parameters_before=before,
        parameters_after=after,
    )

    assert update.trunk_gradient_l2 == pytest.approx(5.0)
    assert update.actor_head_gradient_l2 == pytest.approx(12.0)
    assert update.critic_head_gradient_l2 == pytest.approx(5.0)
    assert update.global_gradient_l2 == pytest.approx(np.sqrt(194.0))
    assert update.global_gradient_max_abs == pytest.approx(12.0)
    assert update.trunk_update_l2 == pytest.approx(5.0)
    assert update.actor_head_update_l2 == pytest.approx(3.0)
    assert update.critic_head_update_l2 == pytest.approx(5.0)
    assert update.global_update_l2 == pytest.approx(np.sqrt(59.0))
    assert update.trunk_relative_update == pytest.approx(1.0)
    assert update.actor_head_relative_update == pytest.approx(0.25)
    # Il critic parte nullo: un rapporto con denominatore zero sarebbe fuorviante.
    assert update.critic_head_relative_update is None

    summary = summarize_update_diagnostics([update])
    assert summary["count"] == 1
    assert summary["distributions"]["critic_head_relative_update"] is None


@pytest.mark.slow
@pytest.mark.numba
def test_diagnostics_do_not_change_numba_training_weights(tmp_path: Path) -> None:
    """Attivare la sonda non deve consumare RNG né cambiare un singolo peso finale."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/train_a2c.py"
    plain_model = tmp_path / "plain.npz"
    observed_model = tmp_path / "observed.npz"
    diagnostics = tmp_path / "observed.json"
    common = [
        sys.executable,
        str(script),
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
        "4",
        "--update-every",
        "2",
        "--log-every",
        "1",
        "--seed",
        "123",
    ]
    subprocess.run([*common, "--out", str(plain_model)], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        [*common, "--out", str(observed_model), "--diagnostics-json", str(diagnostics)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    with np.load(plain_model, allow_pickle=False) as plain, np.load(observed_model, allow_pickle=False) as observed:
        for name in ("w1", "b1", "w2", "b2", "wv", "bv"):
            np.testing.assert_array_equal(observed[name], plain[name])

    report = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert report["schema"] == "briscola.a2c_training_diagnostics.v1"
    assert report["method"]["passive"] is True
    assert report["initialization"]["critic_mode"] == "reset_zero"
    assert report["summary"]["count"] == 2
    assert len(report["updates"]) == 2
    assert all(update["signals"]["steps"] > 0 for update in report["updates"])
