"""
Test per evaluation matrix.

Obiettivo:
- garantire che la generazione seed suite sia stabile e con dimensioni corrette
- sanity check end-to-end su una valutazione minuscola (pochi game) non necessaria:
  qui testiamo solo la parte di costruzione/serializzazione senza costi eccessivi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V1
from briscola_ai.ai.evaluation import matrix as evaluation_matrix
from briscola_ai.ai.evaluation.matrix import (
    EvaluationMatrix,
    build_suites_for_benchmark,
    evaluate_model_matrix,
    make_range_seed_suite,
)


def test_make_range_seed_suite_length_and_32bit_normalization() -> None:
    """La seed suite deve avere la lunghezza richiesta e i semi devono wrappare a 32 bit
    (0xFFFFFFFF -> 0 -> 1) per restare deterministici tra engine."""
    seeds = make_range_seed_suite(start=0xFFFFFFFF, step=1, count=3)
    assert len(seeds) == 3
    assert seeds[0] == 0xFFFFFFFF
    # wrap 32-bit
    assert seeds[1] == 0
    assert seeds[2] == 1


def test_build_suites_for_benchmark_returns_two_suites_with_expected_counts() -> None:
    """Un benchmark deve generare le due suite [standard, holdout] con il numero di semi seat-fair
    atteso (medium: 10k game -> 5k semi ciascuna)."""
    suites = build_suites_for_benchmark(benchmark="medium", standard_start=0, holdout_start=1_000_000, step=1)
    assert [s.name for s in suites] == ["standard", "holdout"]
    # medium seat-fair: 10k games -> 5k seeds
    assert suites[0].num_seeds == 5000
    assert suites[1].num_seeds == 5000


def test_evaluation_matrix_json_serialization_is_stable(tmp_path: Path) -> None:
    """`to_json_text()` deve produrre JSON valido che preserva i campi (model_path, benchmark)
    e applica il default `engine="domain"`."""
    # Creiamo un oggetto minimale e verifichiamo che `to_json_text()` sia JSON valido.
    matrix = EvaluationMatrix(model_path="data/model.npz", benchmark="small", num_games=2000, seed=0, rows=[])
    text = matrix.to_json_text()
    parsed = json.loads(text)
    assert parsed["model_path"] == "data/model.npz"
    assert parsed["engine"] == "domain"
    assert parsed["benchmark"] == "small"


def test_evaluation_matrix_parallel_matches_serial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """La matrix parallela deve aggregare le stesse righe della versione seriale."""
    monkeypatch.setattr(evaluation_matrix, "benchmark_num_games", lambda benchmark: 20)

    d = int(FEATURE_DIM_2P_V1)
    h = 4
    model_path = tmp_path / "dummy.npz"
    np.savez(
        model_path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": d}, ensure_ascii=False),
    )

    serial = evaluate_model_matrix(
        model_path=model_path,
        opponents=["heuristic_v1"],
        benchmark="small",
        seed=7,
        workers=1,
    )
    parallel = evaluate_model_matrix(
        model_path=model_path,
        opponents=["heuristic_v1"],
        benchmark="small",
        seed=7,
        workers=2,
    )

    assert parallel.to_json_dict() == serial.to_json_dict()


def test_evaluation_matrix_numba_engine_returns_consistent_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Il path matrix Numba deve produrre righe seat-fair standard per modelli MLP."""
    monkeypatch.setattr(evaluation_matrix, "benchmark_num_games", lambda benchmark: 20)

    d = int(FEATURE_DIM_2P_V1)
    h = 4
    model_path = tmp_path / "dummy_numba.npz"
    np.savez(
        model_path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": d}, ensure_ascii=False),
    )

    matrix = evaluate_model_matrix(
        model_path=model_path,
        opponents=["heuristic_v1"],
        benchmark="small",
        seed=7,
        workers=1,
        engine="numba",
    )

    assert matrix.engine == "numba"
    assert len(matrix.rows) == 2
    for row in matrix.rows:
        assert row.stats.num_games == 20
        assert row.stats.wins_agent_a + row.stats.wins_agent_b + row.stats.draws == 20
        assert row.stats.agent_b_name == "heuristic_v1"

    workers_requested = evaluate_model_matrix(
        model_path=model_path,
        opponents=["heuristic_v1"],
        benchmark="small",
        seed=7,
        workers=2,
        engine="numba",
    )
    assert workers_requested.to_json_dict() == matrix.to_json_dict()


def test_evaluation_matrix_numba_engine_supports_model_opponent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """La matrix Numba deve supportare anche head-to-head contro un opponent `.npz` MLP."""
    monkeypatch.setattr(evaluation_matrix, "benchmark_num_games", lambda benchmark: 20)

    d = int(FEATURE_DIM_2P_V1)
    h = 4
    model_path = tmp_path / "candidate_numba.npz"
    opponent_path = tmp_path / "previous_best_numba.npz"
    for path in (model_path, opponent_path):
        np.savez(
            path,
            w1=np.zeros((d, h), dtype=np.float32),
            b1=np.zeros((h,), dtype=np.float32),
            w2=np.zeros((h, 40), dtype=np.float32),
            b2=np.zeros((40,), dtype=np.float32),
            metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": d}, ensure_ascii=False),
        )

    with pytest.raises(ValueError, match="opponent_model_path"):
        evaluate_model_matrix(
            model_path=model_path,
            opponents=["bc_model"],
            benchmark="small",
            seed=7,
            workers=1,
            engine="numba",
        )

    matrix = evaluate_model_matrix(
        model_path=model_path,
        opponents=["bc_model"],
        benchmark="small",
        seed=7,
        workers=1,
        engine="numba",
        opponent_model_path=opponent_path,
    )

    assert matrix.engine == "numba"
    assert len(matrix.rows) == 2
    for row in matrix.rows:
        assert row.stats.num_games == 20
        assert row.stats.wins_agent_a + row.stats.wins_agent_b + row.stats.draws == 20
        assert row.opponent == row.stats.agent_b_name
        assert row.stats.agent_b_name.startswith("bc_model(previous_best_numba.npz")
