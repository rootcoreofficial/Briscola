"""
Test per encoder observation v2 (storia pubblica `seen_cards_onehot`).

Obiettivo didattico
-------------------
La v2 aggiunge 40 feature che rappresentano *solo* informazione pubblica:
- briscola scoperta
- carte sul tavolo
- carte già uscite (ricostruite dalle prese/captured)

Questo abilita un "card counting" lecito e permette al modello di apprendere
comportamenti più strategici (es. non sprecare briscole alte per scarti).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from briscola_ai.ai.encoding.card_action_space import action_id_from_suit_number
from briscola_ai.ai.encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    encode_player_observation_2p,
)
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _make_minimal_observation_2p(*, seen_cards_onehot: tuple[int, ...]) -> PlayerObservation:
    """Costruisce una `PlayerObservation` minimale ma valida per l'encoder (2-player)."""
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="P0",
        hand=(Card(Suit.CUPS, Rank.ACE), Card(Suit.CLUBS, Rank.TWO)),
        trump_card=Card(Suit.CUPS, Rank.THREE),
        deck_size=10,
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(2, 2),
        seen_cards_onehot=seen_cards_onehot,
    )


def test_encode_player_observation_2p_v2_appends_seen_cards() -> None:
    """
    v2 = v1 + 40 feature in coda.

    Verifichiamo:
    - dimensione (248 -> 288)
    - le ultime 40 feature coincidono con `seen_cards_onehot`.
    """
    seen = tuple([0] * 40)
    obs = _make_minimal_observation_2p(seen_cards_onehot=seen)

    enc_v1 = encode_player_observation_2p(obs, version="v1")
    assert len(enc_v1.features) == FEATURE_DIM_2P_V1

    enc_v2 = encode_player_observation_2p(obs, version="v2")
    assert len(enc_v2.features) == FEATURE_DIM_2P_V2
    assert enc_v2.features[-40:] == [0.0] * 40


def test_bc_model_agent_supports_v2_models(tmp_path: Path) -> None:
    """
    Un modello con feature_dim=288 deve:
    - essere caricato dall'agente,
    - usare automaticamente l'encoder v2,
    - scegliere una carta valida in mano.
    """
    d = int(FEATURE_DIM_2P_V2)
    h = 8
    w1 = np.zeros((d, h), dtype=np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = np.zeros((h, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "action_dim": 40,
        "encoder": "encode_observation_2p:v2",
    }

    model_path = tmp_path / "dummy_v2.npz"
    np.savez(model_path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(metadata, ensure_ascii=False))

    agent = BCModelAgent.from_npz(model_path)
    assert agent.encoder_version == "v2"

    # Observation con seen_cards presente (v2).
    seen = tuple([0] * 40)
    obs = _make_minimal_observation_2p(seen_cards_onehot=seen)

    idx = agent.choose_card_index(obs, rng=random.Random(0))
    assert idx in (0, 1)

    # Con logits tutti uguali (0), dopo la mask l'argmax sceglie l'action_id valido più piccolo.
    c0, c1 = obs.hand
    a0 = action_id_from_suit_number(suit=c0.suit.value, number=c0.rank.number)
    a1 = action_id_from_suit_number(suit=c1.suit.value, number=c1.rank.number)
    expected = 0 if a0 < a1 else 1
    assert idx == expected
