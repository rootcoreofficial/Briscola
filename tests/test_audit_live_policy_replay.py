"""Test del replay v13-v14 sulle osservazioni pubbliche esportate dal live."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.backend.dto import CardDTO
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "scripts" / "audit_live_policy_replay.py"
_spec = importlib.util.spec_from_file_location("audit_live_policy_replay", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[misc]

ReplayConfig = _mod.ReplayConfig
observation_from_live_dto = _mod.observation_from_live_dto
run_audit = _mod.run_audit


def _advance_until_deck_empty(seed: int = 7):
    """Crea uno stato decisionale endgame con storia v4 non vuota."""
    state = new_game_state(2, seed=seed)
    while state.deck:
        current = state.current_turn
        state, result = step(state, PlayCardAction(player_index=current, card_index=0))
        assert result.error is None
    assert not state.game_over
    return state


def _write_policy(path: Path, *, prefer_high_card_id: bool) -> None:
    """Scrive una MLP v4 sintetica deterministica."""
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden = 4
    biases = np.arange(40, dtype=np.float32)
    if not prefer_high_card_id:
        biases = -biases
    np.savez(
        path,
        w1=np.zeros((feature_dim, hidden), dtype=np.float32),
        b1=np.zeros(hidden, dtype=np.float32),
        w2=np.zeros((hidden, 40), dtype=np.float32),
        b2=biases,
        metadata_json=json.dumps(
            {
                "format": "mlp_bc_v1",
                "encoder_version": "v4",
                "feature_dim": feature_dim,
            }
        ),
    )


def _write_belief(path: Path) -> None:
    """Scrive una belief uniforme valida per lo smoke end-to-end."""
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden = 4
    np.savez(
        path,
        w1=np.zeros((feature_dim, hidden), dtype=np.float32),
        b1=np.zeros(hidden, dtype=np.float32),
        w2=np.zeros((hidden, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "belief_mlp_v1",
                "encoder_version": "v4",
                "feature_dim": feature_dim,
            }
        ),
    )


def _write_complete_live_game(path: Path) -> None:
    """Genera un export live minimo ma completo, senza passare dal backend."""
    state = new_game_state(2, seed=11)
    rows: list[dict] = []
    event_id = 1
    while not state.game_over:
        current = state.current_turn
        observation = build_observation_dto(state, current, server_version=event_id).model_dump(mode="json")
        played_card = state.players[current].hand[0]
        rows.append(
            {
                "game_id": "private-test-game-id",
                "event_id": event_id,
                "actor": "human" if current == 0 else "ai",
                "player_index": current,
                "action": {"card_index": 0, "card": CardDTO.from_domain(played_card).model_dump(mode="json")},
                "ai": {"decision_type": "fallback"} if current == 1 else None,
                "observation": observation,
                "metadata": {},
            }
        )
        state, result = step(state, PlayCardAction(player_index=current, card_index=0))
        assert result.error is None
        event_id += 1

    final_points = [player.points for player in state.players]
    for row in rows:
        row["metadata"] = {
            "code_version": "test",
            "ai_model_id": "model_a.npz",
            "final_points_by_player_index": final_points,
            "consent_to_data_collection": True,
        }
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_reconstructs_endgame_observation_with_public_trump_and_history() -> None:
    """Il DTO endgame omette trump_card, ma il replay ricostruisce il dominio esatto."""
    state = _advance_until_deck_empty()
    current = state.current_turn
    expected = make_player_observation(state, current)
    dto = build_observation_dto(state, current, server_version=3).model_dump(mode="json")

    assert dto["trump_card"] is None
    actual = observation_from_live_dto(dto, exposed_trump_card=state.trump_card)

    assert actual == expected
    assert len(actual.trick_history) == len(state.trick_history) == 17


def test_run_audit_is_end_to_end_and_does_not_emit_game_ids(tmp_path: Path) -> None:
    """Lo smoke copre policy, PIMC/solver, aggregazione e confine privacy."""
    input_path = tmp_path / "live.jsonl"
    model_a = tmp_path / "model_a.npz"
    model_b = tmp_path / "model_b.npz"
    belief = tmp_path / "belief.npz"
    _write_complete_live_game(input_path)
    _write_policy(model_a, prefer_high_card_id=True)
    _write_policy(model_b, prefer_high_card_id=False)
    _write_belief(belief)

    report = run_audit(
        ReplayConfig(
            input_paths=(input_path,),
            model_a_path=model_a,
            model_b_path=model_b,
            belief_model_path=belief,
            out_json_path=tmp_path / "report.json",
            seed=5,
            runtime_repeats=1,
            determinizations=1,
            max_unknown_cards=8,
            bootstrap_samples=20,
        )
    )

    assert report["schema"] == "briscola.live_policy_replay.v1"
    assert report["scope"]["games"] == 1
    assert report["scope"]["records"] == 40
    assert report["scope"]["eligible_nonforced_decisions"] == 38
    assert report["comparison"]["policy"]["overall"]["disagreements"] > 0
    assert report["choice_transitions_a_to_b"]["policy"]["overall"]["disagreements"] > 0
    assert set(report["choice_quality"]["by_runtime_branch"]["runtime_a"]) == {"fallback", "search", "solver"}
    assert report["runtime_metrics"]["a"]["coerced_moves"] == 0
    assert "private-test-game-id" not in json.dumps(report)
