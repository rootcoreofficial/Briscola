"""
Smoke test della pipeline v3 (Fase 5G, step 5): self-play -> export -> BC v3.

Obiettivo: dimostrare che la pipeline v3 è *sana end-to-end* prima dei run "veri".
NON valutiamo forza/qualità qui: verifichiamo solo plumbing e contratto dati.

In particolare (guard pratici concordati):
- l'export preserva `out_of_play_cards_onehot` e nei record di endgame il campo è NON banale
  (non basta che la chiave esista);
- l'encoder v3 attraverso il data path BC produce feature_dim=310;
- `train_bc --encoder-version v3` salva un `.npz` caricabile con encoder_version="v3".
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V3
from briscola_ai.ai.models import BCModelAgent

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.slow

_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str) -> Any:
    """Carica uno script di `scripts/` come modulo (non è un package installato)."""
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_teacher_jsonl(tmp_path: Path, *, num_games: int = 4) -> Path:
    """Genera un mini-dataset (self-play -> SQLite -> JSONL) e ne ritorna il path JSONL.

    Usa `heuristic_v1` in entrambi i seat: è model-free e veloce, e per gli smoke test la
    presenza di `out_of_play` non dipende dall'agente ma da `build_observation_dto`.
    """
    self_play = _load_script_module("self_play_to_db")
    export = _load_script_module("export_dataset")

    db_path = tmp_path / "smoke.sqlite3"
    self_play.simulate_self_play_to_db(
        self_play.SelfPlayConfig(
            db_path=db_path,
            num_games=num_games,
            seed=0,
            num_players=2,
            agent_names=("heuristic_v1", "heuristic_v1"),
        )
    )

    jsonl_path = tmp_path / "smoke.jsonl"
    export.export_dataset(
        export.ExportConfig(
            db_path=db_path,
            out_path=jsonl_path,
            player_index=None,  # entrambi i seat (teacher-vs-teacher): tutte le azioni sono del teacher
            include_ai=True,
            include_next_state=False,
            only_completed_games=True,
        )
    )
    return jsonl_path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_export_preserves_nontrivial_out_of_play_in_endgame(tmp_path: Path) -> None:
    """Nei record di endgame (mazzo vuoto) `out_of_play_cards_onehot` deve essere popolato."""
    jsonl_path = _make_teacher_jsonl(tmp_path)
    records = _read_jsonl(jsonl_path)
    assert records, "Export vuoto: pipeline self-play/export non ha prodotto record"

    endgame_out_of_play_sums: list[int] = []
    for record in records:
        obs = record.get("observation")
        if not isinstance(obs, dict):
            continue
        # invariante: out_of_play ⊆ seen
        seen = obs.get("seen_cards_onehot") or []
        oop = obs.get("out_of_play_cards_onehot")
        assert isinstance(oop, list) and len(oop) == 40, "out_of_play assente/shape errata nel JSONL"
        if seen:
            for cid in range(40):
                if oop[cid] == 1:
                    assert seen[cid] == 1
        if obs.get("cards_remaining_in_deck") == 0:
            endgame_out_of_play_sums.append(sum(oop))

    assert endgame_out_of_play_sums, "Nessun record endgame trovato nel mini-dataset"
    # A mazzo vuoto restano in gioco al più 6 carte: almeno 34 dovrebbero essere fuori gioco.
    assert max(endgame_out_of_play_sums) >= 30


def test_bc_data_path_v3_has_310_features(tmp_path: Path) -> None:
    """Il data path BC con encoder v3 produce esempi a 310 feature."""
    jsonl_path = _make_teacher_jsonl(tmp_path)
    train_bc = _load_script_module("train_bc")

    dataset = train_bc._build_training_examples(jsonl_path, encoder_version="v3")
    assert dataset.x.shape[0] > 0
    assert dataset.x.shape[1] == int(FEATURE_DIM_2P_V3) == 310
    assert dataset.mask.shape[1] == 40
    assert dataset.x.shape[0] == dataset.y.shape[0] == dataset.mask.shape[0] == dataset.weights.shape[0]
    assert dataset.target_probs is None
    assert dataset.game_ids.shape == (dataset.x.shape[0],)


def test_train_bc_v3_cli_roundtrip(tmp_path: Path) -> None:
    """`train_bc --encoder-version v3` salva un .npz caricabile come modello v3 (310)."""
    jsonl_path = _make_teacher_jsonl(tmp_path, num_games=6)
    out_path = tmp_path / "bc_v3.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "train_bc.py"),
            "--data",
            str(jsonl_path),
            "--out",
            str(out_path),
            "--encoder-version",
            "v3",
            "--model",
            "linear",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--val-frac",
            "0.0",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, f"train_bc fallito:\n{result.stdout}\n{result.stderr}"
    assert out_path.exists()

    agent = BCModelAgent.from_npz(out_path)
    assert agent.encoder_version == "v3"
    assert int(agent.model.feature_dim) == 310


def test_train_a2c_v3_fast_numba_smoke(tmp_path: Path) -> None:
    """Smoke: A2C v3 con rollout fast+numba gira e salva un modello v3 (310).

    Esercita end-to-end il collector numba con encoder v3 (costruzione out_of_play nel kernel).
    """
    out_path = tmp_path / "a2c_v3_fastnumba.npz"
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "train_a2c.py"),
            "--encoder-version",
            "v3",
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent",
            "heuristic_v1",
            "--num-games",
            "200",
            "--hidden-dim",
            "16",
            "--seed",
            "0",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, f"train_a2c v3 fast/numba fallito:\n{result.stdout}\n{result.stderr}"
    assert out_path.exists()

    agent = BCModelAgent.from_npz(out_path)
    assert agent.encoder_version == "v3"
    assert int(agent.model.feature_dim) == 310
