"""
Smoke test per `scripts/lead_load_guard_probe.py`.

Verifica solo contratto CLI/output JSON. I valori numerici della Fase 0 dipendono dal
modello reale e non sono parte del test.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _load_probe_module() -> ModuleType:
    script = Path(__file__).resolve().parent.parent / "scripts" / "lead_load_guard_probe.py"
    spec = importlib.util.spec_from_file_location("lead_load_guard_probe_for_test", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossibile caricare lead_load_guard_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PROBE = _load_probe_module()
LeadLoadGuardConfig = _PROBE.LeadLoadGuardConfig
apply_lead_load_guard_index = _PROBE.apply_lead_load_guard_index


def _write_synthetic_v4_model(path: Path) -> None:
    """MLP BC sintetico encoder v4: pesi a zero, sufficiente per smoke end-to-end."""
    d, h = int(FEATURE_DIM_2P_V4), 8
    np.savez(
        path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": d}, ensure_ascii=False),
    )


def _make_lead_load_observation(*, seen_all_trumps: bool = False) -> PlayerObservation:
    """Osservazione minimale: il player guida Asso non-briscola e ha una liscia alternativa."""
    trump_card = Card(Suit.CUPS, Rank.THREE)
    hand = (
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.SWORDS, Rank.TWO),
        Card(Suit.CLUBS, Rank.TWO),
    )
    seen = [0] * 40
    if seen_all_trumps:
        for rank in Rank:
            seen[card_to_id(Card(Suit.CUPS, rank))] = 1
    seen[card_to_id(trump_card)] = 1
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="P0",
        hand=hand,
        trump_card=trump_card,
        deck_size=6,
        table_cards=(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(20, 18),
        players_hand_sizes=(3, 3),
        seen_cards_onehot=tuple(seen),
    )


def test_lead_load_guard_replaces_non_master_load_with_smooth_card() -> None:
    obs = _make_lead_load_observation(seen_all_trumps=False)
    guarded, reason = apply_lead_load_guard_index(
        obs,
        chosen_card_index=0,
        config=LeadLoadGuardConfig(trigger="not_master"),
    )

    assert reason == "adjusted"
    assert guarded == 1


def test_lead_load_guard_keeps_master_load() -> None:
    obs = _make_lead_load_observation(seen_all_trumps=True)
    guarded, reason = apply_lead_load_guard_index(
        obs,
        chosen_card_index=0,
        config=LeadLoadGuardConfig(trigger="not_master"),
    )

    assert reason == "safe_by_trigger"
    assert guarded == 0


@pytest.mark.slow
def test_lead_load_guard_probe_smoke(tmp_path: Path) -> None:
    model_path = tmp_path / "synthetic_v4.npz"
    _write_synthetic_v4_model(model_path)
    out_json = tmp_path / "lead_load_probe.json"

    script = Path(__file__).resolve().parent.parent / "scripts" / "lead_load_guard_probe.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--num-games",
            "4",
            "--opponents",
            "mirror,heuristic_trump_saver",
            "--model",
            str(model_path),
            "--seed",
            "0",
            "--out-json",
            str(out_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["meta"]["seed"] == 0
    assert payload["meta"]["mode"] == "phase0"
    assert payload["meta"]["load_points_min"] == 10
    assert payload["meta"]["thin_unknown_same_suit_max"] == 1
    assert set(payload["profiles"]) == {"mirror", "heuristic_trump_saver"}
    summary = payload["profiles"]["mirror"]["summary"]
    assert "lead_load_pct" in summary
    assert "lead_load_thin_cut_pct" in summary
    assert "match" in payload["profiles"]["mirror"]
