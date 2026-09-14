"""
Smoke test per `scripts/trump_play_probe.py`.

Verifica la logica dei trattamenti controfattuali (unità pure, deterministiche) e il
contratto CLI/JSON. I valori numerici della Fase 0 dipendono dal modello reale e non sono
parte del test.
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
    script = Path(__file__).resolve().parent.parent / "scripts" / "trump_play_probe.py"
    spec = importlib.util.spec_from_file_location("trump_play_probe_for_test", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossibile caricare trump_play_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PROBE = _load_probe_module()
TrumpTreatmentConfig = _PROBE.TrumpTreatmentConfig
apply_trump_treatment_index = _PROBE.apply_trump_treatment_index

# Briscola = COPPE in tutti gli scenari sotto.
_TRUMP = Suit.CUPS


def _obs(hand: tuple[Card, ...], *, deck_size: int, table_cards: tuple = ()) -> PlayerObservation:
    """Osservazione minimale di un leader (player 0) con briscola COPPE nota."""
    seen = [0] * 40
    seen[card_to_id(Card(_TRUMP, Rank.THREE))] = 1  # briscola scoperta qualsiasi
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="P0",
        hand=hand,
        trump_card=Card(_TRUMP, Rank.THREE),
        deck_size=deck_size,
        table_cards=table_cards,
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(20, 18),
        players_hand_sizes=(3, 3),
        seen_cards_onehot=tuple(seen),
    )


def test_ace_hold_replaces_early_led_trump_ace_with_non_trump() -> None:
    # Guido l'asso di briscola presto (deck alto): il trattamento deve trattenerlo.
    hand = (Card(_TRUMP, Rank.ACE), Card(Suit.COINS, Rank.TWO), Card(Suit.SWORDS, Rank.TWO))
    idx, reason = apply_trump_treatment_index(
        _obs(hand, deck_size=12), chosen_card_index=0, config=TrumpTreatmentConfig(treatment="ace_hold")
    )
    assert reason == "adjusted"
    assert hand[idx].suit != _TRUMP  # sostituito con una liscia non-briscola


def test_ace_hold_no_op_when_deck_low() -> None:
    # Asso guidato ma tardi (deck basso): sotto la soglia "presto", non intervenire.
    hand = (Card(_TRUMP, Rank.ACE), Card(Suit.COINS, Rank.TWO), Card(Suit.SWORDS, Rank.TWO))
    idx, reason = apply_trump_treatment_index(
        _obs(hand, deck_size=4), chosen_card_index=0, config=TrumpTreatmentConfig(treatment="ace_hold")
    )
    assert reason == "not_early"
    assert idx == 0


def test_pull_more_forces_cheapest_non_load_trump() -> None:
    # Mano lunga di briscola (2), ma apro liscio: forza una briscolina non carico.
    hand = (Card(Suit.COINS, Rank.TWO), Card(_TRUMP, Rank.FOUR), Card(_TRUMP, Rank.TWO))
    idx, reason = apply_trump_treatment_index(
        _obs(hand, deck_size=12), chosen_card_index=0, config=TrumpTreatmentConfig(treatment="pull_more")
    )
    assert reason == "adjusted"
    assert hand[idx].suit == _TRUMP
    assert hand[idx].rank.points < 10  # non è un carico di briscola


def test_pull_more_skips_when_only_load_trumps() -> None:
    # Le uniche briscole sono carichi (Asso/Tre): non dumpare un carico per "pullare".
    hand = (Card(Suit.COINS, Rank.TWO), Card(_TRUMP, Rank.ACE), Card(_TRUMP, Rank.THREE))
    idx, reason = apply_trump_treatment_index(
        _obs(hand, deck_size=12), chosen_card_index=0, config=TrumpTreatmentConfig(treatment="pull_more")
    )
    assert reason == "no_cheap_trump"
    assert idx == 0


def test_pull_less_replaces_trump_lead_with_non_trump() -> None:
    # Mano lunga di briscola e apro in briscola: il trattamento gioca invece una liscia.
    hand = (Card(_TRUMP, Rank.TWO), Card(_TRUMP, Rank.FOUR), Card(Suit.COINS, Rank.TWO))
    idx, reason = apply_trump_treatment_index(
        _obs(hand, deck_size=12), chosen_card_index=0, config=TrumpTreatmentConfig(treatment="pull_less")
    )
    assert reason == "adjusted"
    assert hand[idx].suit != _TRUMP


def test_treatment_inactive_when_not_lead() -> None:
    # Con una carta sul tavolo non siamo leader: nessun trattamento agisce.
    hand = (Card(_TRUMP, Rank.ACE), Card(Suit.COINS, Rank.TWO), Card(Suit.SWORDS, Rank.TWO))
    table = ((Card(Suit.SWORDS, Rank.KING), 1),)
    idx, reason = apply_trump_treatment_index(
        _obs(hand, deck_size=12, table_cards=table),
        chosen_card_index=0,
        config=TrumpTreatmentConfig(treatment="ace_hold"),
    )
    assert reason == "not_lead"
    assert idx == 0


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


@pytest.mark.slow
def test_trump_play_probe_smoke(tmp_path: Path) -> None:
    model_path = tmp_path / "synthetic_v4.npz"
    _write_synthetic_v4_model(model_path)
    out_json = tmp_path / "trump_probe.json"

    script = Path(__file__).resolve().parent.parent / "scripts" / "trump_play_probe.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "both",
            "--num-games",
            "4",
            "--opponents",
            "mirror,heuristic_trump_saver",
            "--treatment",
            "ace_hold",
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
    assert payload["meta"]["mode"] == "both"
    assert payload["meta"]["ace_early_deck_min"] == 9
    assert payload["meta"]["long_trumps_min"] == 2
    assert set(payload["profiles"]) == {"mirror", "heuristic_trump_saver"}
    summary = payload["profiles"]["mirror"]["phase0"]["summary"]
    assert "ace_plays" in summary
    assert "endgame_decisions" in summary
    assert "lead_trump_ge2_pct" in summary
    ablation = payload["profiles"]["mirror"]["treatments"]["ace_hold"]
    assert "delta" in ablation
    assert "avg_point_diff" in ablation["delta"]
