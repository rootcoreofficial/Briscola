"""Test del confronto PIMC simmetrico fra due budget di determinizzazione."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_pimc_supports_distinct_policies_and_latency_distribution(tmp_path: Path) -> None:
    """Il JSON prova policy distinte, belief simmetrica e distribuzioni di costo per search."""
    model = ROOT / "data/models/best_a2c_v14.npz"
    opponent_model = ROOT / "data/models/best_a2c_v13.npz"
    belief = ROOT / "data/models/belief_v0_h128_50k_seed20260702.npz"
    output = tmp_path / "pimc.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_pimc.py"),
            "--model",
            str(model),
            "--num-games",
            "2",
            "--seed",
            "17",
            "--determinizations",
            "1",
            "--max-unknown-cards",
            "8",
            "--opponent",
            "pimc",
            "--opponent-model",
            str(opponent_model),
            "--opponent-determinizations",
            "1",
            "--opponent-max-unknown-cards",
            "8",
            "--belief-model",
            str(belief),
            "--opponent-belief-model",
            str(belief),
            "--out-json",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    belief_sha256 = hashlib.sha256(belief.read_bytes()).hexdigest()
    opponent_model_sha256 = hashlib.sha256(opponent_model.read_bytes()).hexdigest()
    assert payload["schema"] == "briscola.pimc_evaluation.v1"
    assert payload["opponent_model"] == str(opponent_model)
    assert payload["belief_model"] == str(belief)
    assert payload["opponent_belief_model"] == str(belief)
    assert payload["artifacts"]["belief_model"]["sha256"] == belief_sha256
    assert payload["artifacts"]["opponent_belief_model"]["sha256"] == belief_sha256
    assert payload["artifacts"]["opponent_model"]["sha256"] == opponent_model_sha256
    assert "belief_v0_h128_50k_seed20260702.npz" in payload["stats"]["agent_a_name"]
    assert "belief_v0_h128_50k_seed20260702.npz" in payload["stats"]["agent_b_name"]
    assert "best_a2c_v14.npz" in payload["stats"]["agent_a_name"]
    assert "best_a2c_v13.npz" in payload["stats"]["agent_b_name"]
    assert payload["pimc_metrics"]["successful_determinizations"] > 0
    assert payload["opponent_metrics"]["successful_determinizations"] > 0
    for key in ("pimc_search_latency_seconds", "opponent_search_latency_seconds"):
        latency = payload[key]
        assert latency["count"] > 0
        assert 0.0 < latency["p50"] <= latency["p95"] <= latency["max"]
