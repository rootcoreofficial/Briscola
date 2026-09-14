"""Test delle primitive di checkpoint A2C senza eseguire rollout."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from briscola_ai.ai.training.a2c_checkpoint import (
    A2C_RESUME_SCHEMA,
    atomic_savez,
    canonical_json,
    config_fingerprint,
    parse_resume_json,
    tuple_tree,
)


def test_rng_states_round_trip_through_strict_json() -> None:
    """Gli RNG ripresi devono produrre esattamente gli stessi valori successivi."""
    numpy_rng = np.random.default_rng(123)
    python_rng = random.Random(456)
    numpy_rng.random(5)
    for _ in range(5):
        python_rng.random()

    payload = {
        "numpy": numpy_rng.bit_generator.state,
        "python": python_rng.getstate(),
    }
    restored = json.loads(canonical_json(payload))
    resumed_numpy = np.random.default_rng()
    resumed_numpy.bit_generator.state = restored["numpy"]
    resumed_python = random.Random()
    resumed_python.setstate(tuple_tree(restored["python"]))

    np.testing.assert_array_equal(resumed_numpy.random(10), numpy_rng.random(10))
    assert [resumed_python.random() for _ in range(10)] == [python_rng.random() for _ in range(10)]


def test_config_fingerprint_is_order_independent_and_sensitive() -> None:
    """L'ordine delle chiavi non conta, un parametro comportamentale diverso sì."""
    assert config_fingerprint({"seed": 1, "lr": 0.1}) == config_fingerprint({"lr": 0.1, "seed": 1})
    assert config_fingerprint({"seed": 1, "lr": 0.1}) != config_fingerprint({"seed": 2, "lr": 0.1})


def test_atomic_npz_contains_parseable_resume_state(tmp_path: Path) -> None:
    """Il file pubblicato deve essere completo e privo del temporaneo di scrittura."""
    path = tmp_path / "checkpoint.npz"
    resume = {"schema": A2C_RESUME_SCHEMA, "games_completed": 20}

    atomic_savez(path, weights=np.asarray([1.0], dtype=np.float32), resume_state_json=canonical_json(resume))

    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["weights"], np.asarray([1.0], dtype=np.float32))
        assert parse_resume_json(archive["resume_state_json"])["games_completed"] == 20
    assert list(tmp_path.glob(".*.tmp.npz")) == []
