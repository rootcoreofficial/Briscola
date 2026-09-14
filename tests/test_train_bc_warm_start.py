"""Smoke test per warm-start e anchor del training BC."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.state import new_game_state

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.slow

_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_v3_jsonl(path: Path) -> None:
    """Scrive un JSONL minimo compatibile con `train_bc --encoder-version v3`."""
    rows: list[str] = []
    for seed in range(8):
        state = new_game_state(num_players=2, seed=seed)
        obs = build_observation_dto(state, player_index=0, server_version=seed).model_dump(mode="json")
        rows.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "game_id": f"tiny_{seed}",
                    "event_id": seed,
                    "server_version": seed,
                    "player_index": 0,
                    "is_ai": True,
                    "observation": obs,
                    "action": {"card_index": 0},
                    "reward": 0,
                    "next_observation": None,
                    "done": None,
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_tiny_mlp(path: Path) -> None:
    """Crea un modello MLP v3 minimale, usabile sia come init sia come anchor."""
    rng = np.random.default_rng(7)
    d = int(FEATURE_DIM_2P_V3)
    h = 8
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "encoder_version": "v3",
        "inference_overkill_guard": True,
    }
    np.savez(
        path,
        w1=rng.normal(0.0, 0.02, size=(d, h)).astype(np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=rng.normal(0.0, 0.02, size=(h, 40)).astype(np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )


class _AlwaysFirstCardAgent:
    """Agente fake per testare il filtro di disaccordo senza dipendere da un `.npz`."""

    name = "always_first"

    def choose_card_index(self, observation, *, rng) -> int:
        return 0


def test_build_training_examples_can_filter_disagreements(tmp_path: Path) -> None:
    """Il filtro deve rimuovere esempi dove il target coincide con la mossa baseline."""
    train_bc = _load_train_bc_module()
    data_path = tmp_path / "tiny_v3.jsonl"
    state = new_game_state(num_players=2, seed=42)
    obs = build_observation_dto(state, player_index=0, server_version=1).model_dump(mode="json")
    rows = []
    for card_index in (0, 1):
        rows.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "game_id": f"disagree_{card_index}",
                    "event_id": card_index,
                    "server_version": 1,
                    "player_index": 0,
                    "is_ai": True,
                    "observation": obs,
                    "action": {"card_index": card_index},
                },
                ensure_ascii=False,
            )
        )
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = train_bc._build_training_examples(
        data_path,
        encoder_version="v3",
        disagreement_agent=_AlwaysFirstCardAgent(),
    )

    assert dataset.x.shape[0] == 1
    assert dataset.mask.shape == (1, 40)
    # Senza campo sample_weight nel JSONL, il peso di default è 1.0 (training uniforme).
    assert dataset.weights.shape == (1,)
    assert float(dataset.weights[0]) == 1.0
    assert dataset.target_probs is None
    assert dataset.game_ids.tolist() == ["disagree_1"]
    kept_card = obs["my_hand"][1]
    assert int(dataset.y[0]) == train_bc.card_dto_to_action_id(kept_card)


def test_build_training_examples_rejects_missing_game_id(tmp_path: Path) -> None:
    """Un esempio BC senza partita sorgente non deve ricadere nello split per record."""
    train_bc = _load_train_bc_module()
    state = new_game_state(num_players=2, seed=43)
    obs = build_observation_dto(state, player_index=0, server_version=1).model_dump(mode="json")
    data_path = tmp_path / "missing_game_id.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation": obs,
                "action": {"card_index": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split per partita e' obbligatorio"):
        train_bc._build_training_examples(data_path, encoder_version="v3")


def _load_train_bc_module():
    """Carica `scripts/train_bc.py` per testare helper non esportati."""
    import importlib.util

    path = _ROOT / "scripts" / "train_bc.py"
    spec = importlib.util.spec_from_file_location("train_bc_for_warm_start_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_train_bc_mlp_supports_init_and_anchor(tmp_path: Path) -> None:
    """Il trainer BC deve poter fare fine-tuning da un MLP v3 e ancorarsi allo stesso modello."""
    data_path = tmp_path / "tiny_v3.jsonl"
    init_path = tmp_path / "init_v3.npz"
    out_path = tmp_path / "bc_pimc_distilled.npz"
    _write_tiny_v3_jsonl(data_path)
    _write_tiny_mlp(init_path)

    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "train_bc.py"),
            "--data",
            str(data_path),
            "--out",
            str(out_path),
            "--encoder-version",
            "v3",
            "--model",
            "mlp",
            "--init",
            str(init_path),
            "--bc-anchor",
            str(init_path),
            "--bc-anchor-beta",
            "0.01",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--val-frac",
            "0.25",
            "--inference-overkill-guard",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )

    assert result.returncode == 0, f"train_bc fallito:\n{result.stdout}\n{result.stderr}"
    agent = BCModelAgent.from_npz(out_path)
    assert agent.encoder_version == "v3"
    assert agent.overkill_guard_enabled is True
    assert int(agent.model.feature_dim) == int(FEATURE_DIM_2P_V3)

    with np.load(out_path) as data:
        metadata = json.loads(str(data["metadata_json"]))
    assert metadata["init"] == str(init_path)
    assert metadata["bc_anchor_path"] == str(init_path)
    assert metadata["bc_anchor_beta"] == 0.01
    assert metadata["dataset_split"]["unit"] == "game"
    assert metadata["dataset_split"]["group_counts"] == {
        "total": 8,
        "train": 5,
        "validation": 2,
        "test": 1,
    }
    assert len(metadata["dataset_split"]["assignment_sha256"]) == 64


def test_train_bc_upgrade_init_v3_to_v4(tmp_path: Path) -> None:
    """
    L'upgrade v3 -> v4 estende `w1` con righe a zero: il warm start parte ESATTAMENTE
    dal modello v3 (le 59 feature nuove sono inerti) e il training v4 procede senza errori.
    """
    data_path = tmp_path / "tiny_v3.jsonl"
    init_path = tmp_path / "init_v3.npz"
    out_path = tmp_path / "bc_v4_warm.npz"
    _write_tiny_v3_jsonl(data_path)
    _write_tiny_mlp(init_path)

    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "train_bc.py"),
            "--data",
            str(data_path),
            "--out",
            str(out_path),
            "--encoder-version",
            "v4",
            "--model",
            "mlp",
            "--init",
            str(init_path),
            "--upgrade-init-v3-to-v4",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--val-frac",
            "0.25",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )

    assert result.returncode == 0, f"train_bc v4 fallito:\n{result.stdout}\n{result.stderr}"
    agent = BCModelAgent.from_npz(out_path)
    assert agent.encoder_version == "v4"
    assert int(agent.model.feature_dim) == int(FEATURE_DIM_2P_V4)

    # Senza il flag di upgrade il mismatch deve restare un errore esplicito.
    result_no_flag = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "train_bc.py"),
            "--data",
            str(data_path),
            "--out",
            str(tmp_path / "no_flag.npz"),
            "--encoder-version",
            "v4",
            "--model",
            "mlp",
            "--init",
            str(init_path),
            "--epochs",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result_no_flag.returncode != 0
    assert "upgrade-init-v3-to-v4" in (result_no_flag.stderr + result_no_flag.stdout)
