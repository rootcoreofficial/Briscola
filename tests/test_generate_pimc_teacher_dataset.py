"""Test per il generatore dataset teacher PIMC."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from briscola_ai.ai.agents import HeuristicAgentV2, PIMCAgent
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V3

_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str) -> Any:
    """Carica uno script da `scripts/` come modulo testabile."""
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generate_pimc_teacher_dataset_is_bc_compatible(tmp_path: Path) -> None:
    """Il JSONL teacher deve essere leggibile dal data path BC v3 senza conversioni ad hoc."""
    generator = _load_script_module("generate_pimc_teacher_dataset")
    train_bc = _load_script_module("train_bc")

    out_path = tmp_path / "pimc_teacher.jsonl"
    base_agent = HeuristicAgentV2()
    teacher = PIMCAgent(
        rollout_agent=base_agent,
        fallback=base_agent,
        num_determinizations=2,
        max_unknown_cards=6,
    )

    counters = generator.generate_pimc_teacher_dataset(
        generator.PIMCTeacherDatasetConfig(
            out_path=out_path,
            num_examples=8,
            max_games=20,
            seed=123,
            max_unknown_cards=6,
            include_fallback_examples=False,
        ),
        teacher=teacher,
        play_agent=base_agent,
    )

    assert counters["records_written"] == 8
    assert "records_written_search_strong_reliable_disagree" in counters
    records = _read_jsonl(out_path)
    assert len(records) == 8

    for record in records:
        assert record["schema_version"] == 1
        assert record["dataset_kind"] == "pimc_teacher"
        assert record["is_ai"] is True
        assert record["next_observation"] is None
        assert record["generation"]["unknown_live_cards"] <= 6
        assert record["generation"]["strong_margin_min"] == 2.0
        assert record["generation"]["reliable_margin_ci_low_min"] == 0.0
        assert record["reference"]["agent"] == base_agent.name
        assert isinstance(record["reference"]["card_index"], int)
        assert isinstance(record["reference"]["disagrees_with_teacher"], bool)
        assert record["teacher"]["decision_type"] in {"search", "endgame_solver", "fallback"}
        if record["teacher"]["decision_type"] == "search":
            diagnostics = record["teacher"]["search_diagnostics"]
            assert isinstance(diagnostics, dict)
            assert diagnostics["best_card_index"] == record["action"]["card_index"]
            assert isinstance(diagnostics["action_values"], list)
            assert diagnostics["paired_margin_sample_count"] == len(diagnostics["paired_margin_samples"])
        else:
            assert record["teacher"]["search_diagnostics"] is None

        obs = record["observation"]
        action = record["action"]["card_index"]
        assert obs["type"] == "observation"
        assert obs["num_players"] == 2
        assert obs["my_turn"] is True
        assert action in obs["valid_actions"]
        assert 0 <= action < len(obs["my_hand"])
        assert isinstance(obs["out_of_play_cards_onehot"], list)
        assert len(obs["out_of_play_cards_onehot"]) == 40

    dataset = train_bc._build_training_examples(out_path, encoder_version="v3")
    assert dataset.x.shape == (8, int(FEATURE_DIM_2P_V3))
    assert dataset.mask.shape == (8, 40)
    assert dataset.y.shape == (8,)
    assert dataset.weights.shape == (8,)
    assert dataset.target_probs is None
    assert dataset.game_ids.shape == (8,)


def test_generate_pimc_teacher_dataset_respects_player_filter_and_threshold(tmp_path: Path) -> None:
    """I filtri dichiarati nel config devono riflettersi in ogni record scritto."""
    generator = _load_script_module("generate_pimc_teacher_dataset")

    out_path = tmp_path / "pimc_teacher_p0.jsonl"
    base_agent = HeuristicAgentV2()
    teacher = PIMCAgent(
        rollout_agent=base_agent,
        fallback=base_agent,
        num_determinizations=1,
        max_unknown_cards=4,
    )

    counters = generator.generate_pimc_teacher_dataset(
        generator.PIMCTeacherDatasetConfig(
            out_path=out_path,
            num_examples=5,
            max_games=30,
            seed=321,
            max_unknown_cards=4,
            player_index=0,
            include_fallback_examples=False,
        ),
        teacher=teacher,
        play_agent=base_agent,
    )

    assert counters["records_written"] == 5
    records = _read_jsonl(out_path)
    assert len(records) == 5
    assert {record["player_index"] for record in records} == {0}
    assert all(record["observation"]["my_index"] == 0 for record in records)
    assert all(record["generation"]["unknown_live_cards"] <= 4 for record in records)
    assert counters["records_skipped_player"] > 0


def test_generate_pimc_teacher_dataset_includes_fallback_examples_by_default(tmp_path: Path) -> None:
    """Il default deve produrre anche esempi di apertura etichettati dal fallback v6/teacher."""
    generator = _load_script_module("generate_pimc_teacher_dataset")

    out_path = tmp_path / "pimc_teacher_full_policy.jsonl"
    base_agent = HeuristicAgentV2()
    teacher = PIMCAgent(
        rollout_agent=base_agent,
        fallback=base_agent,
        num_determinizations=1,
        max_unknown_cards=0,
    )

    counters = generator.generate_pimc_teacher_dataset(
        generator.PIMCTeacherDatasetConfig(
            out_path=out_path,
            num_examples=6,
            max_games=2,
            seed=456,
            max_unknown_cards=0,
        ),
        teacher=teacher,
        play_agent=base_agent,
    )

    records = _read_jsonl(out_path)
    assert counters["records_written"] == 6
    assert counters["fallback_window_positions"] > 0
    assert counters["records_written_fallback"] > 0
    assert any(record["generation"]["unknown_live_cards"] > 0 for record in records)
    assert all(record["generation"]["include_fallback_examples"] is True for record in records)
    assert any(record["teacher"]["decision_type"] == "fallback" for record in records)
