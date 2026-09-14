"""Test del controllo holdout sulle unità ReLU classificate dormienti."""

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
from briscola_ai.ai.evaluation.hidden_units import ablate_mlp_hidden_units
from briscola_ai.ai.models.bc_model import MLPBCModel


def _load_module():
    """Carica lo script mantenendo disponibili gli altri moduli CLI fratelli."""
    root = Path(__file__).resolve().parents[1]
    scripts_dir = str(root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    path = root / "scripts/evaluate_dormant_unit_ablation.py"
    spec = importlib.util.spec_from_file_location("evaluate_dormant_unit_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _tiny_model() -> MLPBCModel:
    """Policy a due unità: la seconda è sempre spenta e quindi rimovibile esattamente."""
    w1 = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    b1 = np.asarray([0.0, -1.0], dtype=np.float32)
    w2 = np.zeros((2, 40), dtype=np.float32)
    w2[0, 0] = 1.0
    w2[1, 1] = 3.0
    return MLPBCModel(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=np.zeros(40, dtype=np.float32),
        metadata={"format": "mlp_bc_v1", "feature_dim": 2},
    )


def test_policy_agreement_is_exact_when_selected_unit_stays_inactive() -> None:
    """Una unità ReLU spenta sull'intero holdout non può cambiare logits o action."""
    original = _tiny_model()
    ablated = ablate_mlp_hidden_units(original, (1,))
    inputs = np.asarray([[1.0, 0.0], [2.0, 1.0], [0.5, -1.0]], dtype=np.float32)
    masks = np.zeros((3, 40), dtype=bool)
    masks[:, :2] = True

    report = module._policy_agreement(
        original,
        ablated,
        inputs,
        masks,
        (1,),
        activation_epsilon=1e-8,
    )

    assert report["action_agreement_rate"] == 1.0
    assert report["changed_action_count"] == 0
    assert report["states_with_any_selected_unit_active"] == 0
    assert report["max_abs_logit_delta"]["max"] == 0.0


def test_decision_requires_match_and_every_preregistered_gate() -> None:
    """Un GO non deve essere possibile prima del match né con un solo limite fallito."""
    agreement = {"action_agreement_rate": 1.0}
    suit = {"flip_rate_delta": 0.0}
    assert module._decision(agreement, suit, None)["verdict"] == "pending_direct_match"

    match = {
        "stats": {"avg_point_diff_agent_a_minus_agent_b": 0.0},
        "avg_point_diff_ci95": {"low": -0.1, "high": 0.1},
    }
    assert module._decision(agreement, suit, match)["verdict"] == "go_reinitialize_small_dormant_subset"

    suit["flip_rate_delta"] = 0.006
    assert module._decision(agreement, suit, match)["verdict"] == "stop_dormant_capacity_reuse"


def test_match_loader_rejects_different_agents_and_records_hash(tmp_path: Path) -> None:
    """Un risultato di altri modelli non deve poter soddisfare accidentalmente i gate."""
    match_path = tmp_path / "match.json"
    stats = {
        "num_games": 10_000,
        "agent_a_name": "candidate",
        "agent_b_name": "v14",
        "wins_agent_a": 4_800,
        "wins_agent_b": 4_800,
        "draws": 400,
        "avg_points_agent_a": 60.0,
        "avg_points_agent_b": 60.0,
        "avg_point_diff_agent_a_minus_agent_b": 0.0,
        "sum_sq_point_diff_agent_a_minus_agent_b": 1_000_000.0,
        "sum_sq_pair_point_diff_agent_a_minus_agent_b": 500_000.0,
        "sum_sq_pair_score_agent_a": 1_250.0,
    }
    match_path.write_text(
        json.dumps(
            {
                "engine": "numba",
                "mode": "seat_fair",
                "seed_suite": {"range_start": 1_000_000, "range_step": 1},
                "agents": {"agent0": "candidate", "agent1": "v14"},
                "stats": stats,
            }
        ),
        encoding="utf-8",
    )

    report = module._load_match(
        match_path,
        root=tmp_path,
        expected_agent_a="candidate",
        expected_agent_b="v14",
    )

    assert report["source"] == "match.json"
    assert report["source_sha256"] == hashlib.sha256(match_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Agenti del match inattesi"):
        module._load_match(
            match_path,
            root=tmp_path,
            expected_agent_a="other-candidate",
            expected_agent_b="v14",
        )


@pytest.mark.slow
def test_cli_creates_candidate_and_pending_holdout_report(tmp_path: Path) -> None:
    """Lo smoke attraversa unit list congelata, raccolta, 24 rinomine e salvataggio `.npz`."""
    root = Path(__file__).resolve().parents[1]
    feature_dim = int(FEATURE_DIM_2P_V4)
    model_path = tmp_path / "source.npz"
    w1 = np.zeros((feature_dim, 2), dtype=np.float32)
    b1 = np.asarray([0.0, -1.0], dtype=np.float32)
    w2 = np.zeros((2, 40), dtype=np.float32)
    b2 = np.asarray([float(action_id % 10) for action_id in range(40)], dtype=np.float32)
    np.savez(
        model_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        metadata_json=json.dumps(
            {
                "format": "mlp_bc_v1",
                "feature_dim": feature_dim,
                "encoder_version": "v4",
                "inference_overkill_guard": False,
            }
        ),
    )
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    source_evidence = tmp_path / "hidden.json"
    source_evidence.write_text(
        json.dumps(
            {
                "schema": "briscola.hidden_unit_diagnostic.v1",
                "artifacts": {"primary": {"sha256": model_sha256}},
                "thresholds": {"activation_epsilon": 1e-8},
                "models": {"primary": {"utilization": {"dead_units": [1]}}},
            }
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.npz"
    output = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_dormant_unit_ablation.py"),
            "--model",
            str(model_path),
            "--source-evidence",
            str(source_evidence),
            "--seed-count",
            "1",
            "--samples-per-cell",
            "1",
            "--opponents",
            "random",
            "--suit-chunk-size",
            "2",
            "--candidate-out",
            str(candidate),
            "--out-json",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "briscola.dormant_unit_ablation.v1"
    assert payload["coverage"]["selected_observations"] == 4
    assert payload["policy_agreement"]["action_agreement_rate"] == 1.0
    assert payload["suit_symmetry"]["nonidentity_comparisons"] == 4 * 23
    assert payload["decision"]["verdict"] == "pending_direct_match"
    with np.load(candidate, allow_pickle=False) as data:
        assert np.count_nonzero(data["w2"][1]) == 0
