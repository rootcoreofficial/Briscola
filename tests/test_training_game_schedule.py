"""Test della schedule seriale e realmente paired usata dal training A2C."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.training.game_schedule import (
    TrainingGameScheduleStream,
    build_training_game_schedule,
    training_schedule_sha256,
)
from briscola_ai.ai.training.opponent_mix import parse_opponent_mix


def _schedule(*, mode: str, num_games: int = 8, update_every: int = 4):
    """Costruisce una fixture con RNG distinti per mazzi e opponent."""
    return build_training_game_schedule(
        num_games=num_games,
        update_every=update_every,
        mode=mode,
        seat_fair=True,
        default_opponent_name="random",
        opponent_mix=parse_opponent_mix("random:1,heuristic_v1:1"),
        rng_game=np.random.default_rng(11),
        rng_opponent=np.random.default_rng(12),
    )


def test_serial_schedule_is_reproducible_and_keeps_one_environment_per_game() -> None:
    """La modalità storica deve continuare a campionare ogni partita separatamente."""
    first = _schedule(mode="serial")
    second = _schedule(mode="serial")

    assert first == second
    assert [game.ordinal for game in first] == list(range(1, 9))
    assert [game.policy_seat for game in first] == [1, 0, 1, 0, 1, 0, 1, 0]
    assert all(game.pair_index is None for game in first)
    assert training_schedule_sha256(first) == training_schedule_sha256(second)


def test_paired_schedule_reuses_seed_and_opponent_then_swaps_seat() -> None:
    """Ogni coppia adiacente deve vedere lo stesso ambiente con seat `{0,1}`."""
    schedule = _schedule(mode="paired")

    for pair_index in range(4):
        left, right = schedule[2 * pair_index : 2 * pair_index + 2]
        assert left.game_seed == right.game_seed
        assert left.opponent_name == right.opponent_name
        assert (left.policy_seat, right.policy_seat) == (0, 1)
        assert left.pair_index == pair_index
        assert right.pair_index == pair_index

    # Con update da quattro partite, ogni chunk contiene esattamente due coppie intere.
    for start in range(0, len(schedule), 4):
        pair_ids = [game.pair_index for game in schedule[start : start + 4]]
        assert pair_ids == [start // 2, start // 2, start // 2 + 1, start // 2 + 1]


@pytest.mark.parametrize("mode", ["serial", "paired"])
def test_streamed_batches_equal_materialized_schedule(mode: str) -> None:
    """La versione O(1) deve emettere le stesse righe e lo stesso digest storico."""
    expected = _schedule(mode=mode)
    stream = TrainingGameScheduleStream(
        mode=mode,
        seat_fair=True,
        default_opponent_name="random",
        opponent_mix=parse_opponent_mix("random:1,heuristic_v1:1"),
        rng_game=np.random.default_rng(11),
        rng_opponent=np.random.default_rng(12),
    )

    actual = stream.take(4) + stream.take(4)

    assert actual == expected
    assert stream.consumed_games == 8
    assert stream.sha256 == training_schedule_sha256(expected)


def test_stream_resume_extends_digest_without_replaying_prefix() -> None:
    """RNG, cursore e 32 byte di digest devono bastare per continuare la schedule."""
    game_rng = np.random.default_rng(11)
    opponent_rng = np.random.default_rng(12)
    first = TrainingGameScheduleStream(
        mode="serial",
        seat_fair=True,
        default_opponent_name="random",
        opponent_mix=parse_opponent_mix("random:1,heuristic_v1:1"),
        rng_game=game_rng,
        rng_opponent=opponent_rng,
    )
    prefix = first.take(4)
    prefix_digest = first.sha256
    game_state = game_rng.bit_generator.state
    opponent_state = opponent_rng.bit_generator.state

    resumed_game_rng = np.random.default_rng()
    resumed_game_rng.bit_generator.state = game_state
    resumed_opponent_rng = np.random.default_rng()
    resumed_opponent_rng.bit_generator.state = opponent_state
    resumed = TrainingGameScheduleStream(
        mode="serial",
        seat_fair=True,
        default_opponent_name="random",
        opponent_mix=parse_opponent_mix("random:1,heuristic_v1:1"),
        rng_game=resumed_game_rng,
        rng_opponent=resumed_opponent_rng,
        consumed_games=4,
        digest_hex=prefix_digest,
    )

    suffix = resumed.take(4)

    assert prefix + suffix == _schedule(mode="serial")
    assert resumed.sha256 == training_schedule_sha256(prefix + suffix)


@pytest.mark.parametrize(
    ("num_games", "update_every", "message"),
    [
        (7, 4, "--num-games pari"),
        (8, 3, "--update-every pari"),
        (10, 4, "multiplo di --update-every"),
    ],
)
def test_paired_schedule_rejects_shapes_that_cannot_form_complete_updates(
    num_games: int,
    update_every: int,
    message: str,
) -> None:
    """Nessuna coppia deve attraversare un update o finire in un resto non applicato."""
    with pytest.raises(ValueError, match=message):
        _schedule(mode="paired", num_games=num_games, update_every=update_every)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("rollout_engine", "fast_rollout"),
    [("domain", "python"), ("fast", "python"), ("fast", "numba")],
)
def test_train_a2c_paired_schedule_smoke(
    tmp_path: Path,
    rollout_engine: str,
    fast_rollout: str,
) -> None:
    """Dominio, fast Python e Numba devono consumare la stessa schedule condivisa."""
    root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / f"paired_{rollout_engine}_{fast_rollout}.npz"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/train_a2c.py"),
            "--out",
            str(out_path),
            "--rollout-engine",
            rollout_engine,
            "--fast-rollout",
            fast_rollout,
            "--opponent-mix",
            "random:0.5,heuristic_v1:0.5",
            "--training-schedule",
            "paired",
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
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    with np.load(out_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    schedule = metadata["train"]["training_schedule"]
    assert metadata["train"]["seat_fair"] is True
    assert metadata["train"]["seat_fair_requested"] is False
    assert schedule["mode"] == "paired"
    assert schedule["pair_size"] == 2
    assert schedule["scheduled_environment_draws"] == 2
    assert schedule["opponent_sampling_scope"] == "pair"
    assert schedule["seat_order"] == [0, 1]
    assert len(schedule["sha256"]) == 64


@pytest.mark.slow
@pytest.mark.numba
def test_serial_default_and_explicit_mode_produce_identical_weights(tmp_path: Path) -> None:
    """L'aggiunta del flag non deve cambiare il training storico quando resta seriale."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/train_a2c.py"
    default_model = tmp_path / "default.npz"
    explicit_model = tmp_path / "explicit.npz"
    common = [
        sys.executable,
        str(script),
        "--rollout-engine",
        "fast",
        "--fast-rollout",
        "numba",
        "--opponent-mix",
        "random:0.5,heuristic_v1:0.5",
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
        "--seat-fair",
        "--seed",
        "123",
    ]
    subprocess.run([*common, "--out", str(default_model)], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        [*common, "--training-schedule", "serial", "--out", str(explicit_model)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    with np.load(default_model, allow_pickle=False) as default, np.load(explicit_model, allow_pickle=False) as explicit:
        for name in ("w1", "b1", "w2", "b2", "wv", "bv"):
            np.testing.assert_array_equal(default[name], explicit[name])
