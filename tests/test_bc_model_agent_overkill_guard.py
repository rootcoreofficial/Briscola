"""
Test per il post-processing anti-overkill in `BCModelAgent`.

Obiettivo:
quando il modello (logits) sceglierebbe una briscola "alta" per vincere una presa,
ma esiste una briscola vincente più economica in mano, l'agente deve giocare quella minima
se `inference_overkill_guard` è abilitato (metadati).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from briscola_ai.ai.encoding.card_action_space import action_id_from_suit_number
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _make_second_hand_observation_with_two_trumps() -> PlayerObservation:
    """
    Costruisce una `PlayerObservation` minimale:
    - briscola = cups
    - sul tavolo: swords TWO (player 0)
    - in mano (player 1): cups TWO (briscola bassa) e cups ACE (briscola alta)
    """
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=1,
        player_name="P1",
        hand=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CUPS, Rank.ACE)),
        trump_card=Card(Suit.CUPS, Rank.THREE),
        deck_size=10,
        table_cards=((Card(Suit.SWORDS, Rank.TWO), 0),),
        current_turn=1,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(1, 2),
        seen_cards_onehot=tuple([0] * 40),
    )


def test_overkill_guard_chooses_min_winning_trump(tmp_path: Path) -> None:
    """
    Modello dummy: logits spingono verso cups ACE.
    Guard abilitato: deve scegliere cups TWO (briscola vincente minima).
    """
    feature_dim = 248
    w = np.zeros((feature_dim, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)

    ace_id = action_id_from_suit_number(suit="cups", number=1)
    two_id = action_id_from_suit_number(suit="cups", number=2)
    b[ace_id] = 10.0
    b[two_id] = 0.0

    metadata = {
        "format": "linear_softmax_bc_v1",
        "feature_dim": feature_dim,
        "action_dim": 40,
        "encoder": "encode_observation_2p:v1",
        "inference_overkill_guard": True,
    }

    path = tmp_path / "dummy_linear_guard.npz"
    np.savez(path, w=w, b=b, metadata_json=json.dumps(metadata, ensure_ascii=False))

    agent = BCModelAgent.from_npz(path)
    assert agent.overkill_guard_enabled is True

    obs = _make_second_hand_observation_with_two_trumps()
    chosen_idx = agent.choose_card_index(obs, rng=random.Random(0))

    # hand[0] = cups TWO, hand[1] = cups ACE
    assert chosen_idx == 0
