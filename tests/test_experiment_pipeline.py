"""
Test per la pipeline esperimenti (helper core).

Nota:
non testiamo qui l'esecuzione reale di training/eval (sarebbe lenta e flaky),
ma le parti “pure” e riproducibili: naming, estrazione metrica da JSON e costruzione
dei comandi pipeline senza lanciare training reali.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.experiment_pipeline import build_experiment_name, extract_best_metric_from_matrix_json

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.slow


def _run_experiment_script() -> Path:
    """Restituisce il path assoluto dello script pipeline usato dai test CLI."""
    return Path(__file__).resolve().parent.parent / "scripts" / "run_experiment.py"


def _load_run_experiment_module():
    """Carica `scripts/run_experiment.py` come modulo per poter monkeypatchare `_run`."""
    spec = importlib.util.spec_from_file_location("_run_experiment_under_test", _run_experiment_script())
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_experiment_name_is_deterministic_and_safe() -> None:
    """A parità di input il nome esperimento deve essere deterministico e filesystem-safe
    (niente spazi/slash), con i token attesi (prefisso algo, "200k", "seed5")."""
    name1 = build_experiment_name(
        algo="a2c",
        num_games=200_000,
        seed=5,
        opponent=None,
        opponent_mix="heuristic_v1:0.7,random:0.2,greedy_points:0.1",
        tag=None,
    )
    name2 = build_experiment_name(
        algo="a2c",
        num_games=200_000,
        seed=5,
        opponent=None,
        opponent_mix="heuristic_v1:0.7,random:0.2,greedy_points:0.1",
        tag=None,
    )
    assert name1 == name2
    assert " " not in name1
    assert "/" not in name1
    assert name1.startswith("a2c_")
    assert "200k" in name1
    assert "seed5" in name1


def test_extract_best_metric_from_matrix_json_reads_holdout_diff() -> None:
    """L'estrazione metrica deve selezionare la riga della suite richiesta (holdout) per l'opponent
    dato e leggerne l'`avg_point_diff` corretto."""
    matrix_json = {
        "rows": [
            {
                "suite": {"name": "standard", "range_start": 0, "range_step": 1, "num_seeds": 2},
                "opponent": "heuristic_v1",
                "stats": {"avg_point_diff_agent_a_minus_agent_b": 1.0},
            },
            {
                "suite": {"name": "holdout", "range_start": 1_000_000, "range_step": 1, "num_seeds": 2},
                "opponent": "heuristic_v1",
                "stats": {"avg_point_diff_agent_a_minus_agent_b": 7.06},
            },
        ]
    }
    metric = extract_best_metric_from_matrix_json(
        matrix_json, benchmark="big", opponent="heuristic_v1", suite="holdout"
    )
    assert metric.benchmark == "big"
    assert metric.avg_diff == pytest.approx(7.06)


def test_extract_best_metric_from_matrix_json_raises_when_missing() -> None:
    """Se la matrix non contiene righe (nessuna metrica estraibile) l'estrazione deve sollevare
    ValueError invece di restituire un valore silenzioso."""
    with pytest.raises(ValueError):
        extract_best_metric_from_matrix_json({"rows": []}, benchmark="big")


def test_copy_best_model_compacts_training_metrics(tmp_path: Path) -> None:
    """
    Il best runtime deve restare leggero: le metriche complete vivono nel modello
    dell'esperimento, mentre `best_a2c.npz` conserva solo un riassunto.
    """
    module = _load_run_experiment_module()
    source_model = tmp_path / "experiment_model.npz"
    best_model = tmp_path / "best_a2c.npz"
    best_meta = tmp_path / "best_a2c.json"

    w1 = np.arange(6, dtype=np.float32).reshape(2, 3)
    b1 = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    metrics = [
        {"iter": 1, "games": 20, "avg_return": 0.1},
        {"iter": 2, "games": 40, "avg_return": 0.2},
    ]
    metadata = {
        "format": "mlp_a2c_shaped_v1",
        "label": "A2C shaped smoke",
        "inference_overkill_guard": True,
        "metrics": metrics,
    }
    np.savez(
        source_model,
        w1=w1,
        b1=b1,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )

    updated = module._copy_best_model(
        model_path=source_model,
        best_path=best_model,
        score=12.34,
        meta_path=best_meta,
        manifest={"name": "compact_test"},
    )

    assert updated is True
    with np.load(best_model, allow_pickle=False) as data:
        compacted_metadata = json.loads(str(data["metadata_json"]))
        np.testing.assert_array_equal(data["w1"], w1)
        np.testing.assert_array_equal(data["b1"], b1)

    assert "metrics" not in compacted_metadata
    assert compacted_metadata["metrics_summary"] == {
        "stripped_for_best_model": True,
        "num_records": 2,
        "first_record": metrics[0],
        "last_record": metrics[-1],
        "full_metrics_source_model_path": str(source_model),
    }
    assert compacted_metadata["inference_overkill_guard"] is True

    sidecar = json.loads(best_meta.read_text(encoding="utf-8"))
    assert sidecar["score"] == pytest.approx(12.34)
    assert sidecar["inference_overkill_guard"] is True
    assert sidecar["metadata_compaction"] == {
        "best_model_metrics_stripped": True,
        "num_training_metric_records": 2,
        "full_metrics_source_model_path": str(source_model),
    }


def test_run_experiment_help_exposes_fast_rollout_flags() -> None:
    """La pipeline deve pubblicare i flag Numba senza obbligare a usare `--train-extra`."""
    proc = subprocess.run(
        [sys.executable, str(_run_experiment_script()), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--rollout-engine" in proc.stdout
    assert "--fast-encoder" in proc.stdout
    assert "--fast-rollout" in proc.stdout


def test_run_experiment_rejects_fast_rollout_for_pg() -> None:
    """I flag fast/Numba sono specifici di A2C: PG non ha quel collector."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_run_experiment_script()),
            "--algo",
            "pg",
            "--rollout-engine",
            "fast",
            "--num-games",
            "1",
            "--benchmarks",
            "small",
            "--no-update-best",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "supportati solo con --algo a2c" in proc.stderr


def test_run_experiment_forwards_fast_rollout_flags_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Verifica la parte importante senza allenare davvero:
    il comando A2C generato contiene il path Numba e il manifest lo registra.
    """
    module = _load_run_experiment_module()
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, log_path: Path, env_overrides: dict[str, str] | None = None) -> None:
        commands.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + " ".join(cmd) + "\n", encoding="utf-8")

        if any(part.endswith("train_a2c.py") for part in cmd):
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake-npz-for-pipeline-test")
            return

        if any(part.endswith("evaluate_matrix.py") for part in cmd):
            out_json = Path(cmd[cmd.index("--out-json") + 1])
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "suite": {"name": "holdout", "range_start": 1_000_000, "range_step": 1},
                                "opponent": "heuristic_v1",
                                "stats": {"avg_point_diff_agent_a_minus_agent_b": 1.23},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return

        raise AssertionError(f"Comando inatteso: {cmd}")

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--algo",
            "a2c",
            "--name",
            "numba_pipeline_test",
            "--models-dir",
            str(tmp_path / "models"),
            "--experiments-dir",
            str(tmp_path / "experiments"),
            "--opponent-mix",
            "heuristic_v1:1.0",
            "--num-games",
            "10",
            "--train-seed",
            "3",
            "--benchmarks",
            "small",
            "--eval-engine",
            "numba",
            "--eval-workers",
            "4",
            "--no-update-best",
            "--seat-fair",
            "--rollout-engine",
            "fast",
            "--fast-encoder",
            "numba",
            "--fast-rollout",
            "numba",
        ],
    )

    assert module.main() == 0
    train_cmd = commands[0]
    assert train_cmd[train_cmd.index("--rollout-engine") + 1] == "fast"
    assert train_cmd[train_cmd.index("--fast-encoder") + 1] == "numba"
    assert train_cmd[train_cmd.index("--fast-rollout") + 1] == "numba"
    eval_cmd = commands[1]
    assert eval_cmd[eval_cmd.index("--engine") + 1] == "numba"
    assert eval_cmd[eval_cmd.index("--workers") + 1] == "1"

    manifest_path = tmp_path / "experiments" / "numba_pipeline_test" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["train"]["rollout"] == {
        "engine": "fast",
        "fast_encoder": "numba",
        "fast_rollout": "numba",
    }
    assert manifest["train"]["cmd"] == train_cmd
    assert manifest["eval"][0]["engine"] == "numba"
    assert manifest["eval"][0]["workers"] == 1
    assert manifest["eval"][0]["requested_workers"] == 4
    assert manifest["eval"][0]["parallelism"] == "numba_threads"
