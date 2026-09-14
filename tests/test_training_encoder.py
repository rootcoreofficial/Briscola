"""
Test per encoder ML (Behavior Cloning): 40 carte + action mask.

Obiettivo:
- garantire che la mappatura carta -> action_id sia stabile e bijettiva
- garantire che l'encoder produca dimensioni e mask coerenti
"""

from __future__ import annotations

import random

from briscola_ai.ai.encoding.card_action_space import (
    action_id_from_suit_number,
    action_mask_from_hand,
    suit_number_from_action_id,
)
from briscola_ai.ai.encoding.observation_encoder import (
    encode_observation_2p,
    encode_observation_2p_with_version,
    encode_player_observation_2p,
)
from briscola_ai.ai.fast.observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.fast.state_2p import new_fast_2p_state, step_fast_2p
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import new_game_state


def test_action_id_mapping_is_bijective() -> None:
    """(suit, number) <-> action_id deve essere invertibile per tutte le 40 carte."""
    seen = set()
    for suit in ("clubs", "cups", "coins", "swords"):
        for number in range(1, 11):
            action_id = action_id_from_suit_number(suit=suit, number=number)
            assert 0 <= action_id < 40
            assert action_id not in seen
            seen.add(action_id)

            suit2, number2 = suit_number_from_action_id(action_id)
            assert suit2 == suit
            assert number2 == number

    assert len(seen) == 40


def test_action_mask_from_hand_marks_only_cards_in_hand() -> None:
    """La mask deve attivare esattamente le carte presenti in mano."""
    hand = [
        {"suit": "cups", "number": 1, "rank": "ACE", "points": 11},
        {"suit": "coins", "number": 3, "rank": "THREE", "points": 10},
    ]
    mask = action_mask_from_hand(hand)
    assert len(mask) == 40
    assert sum(mask) == 2
    assert mask[action_id_from_suit_number(suit="cups", number=1)] is True
    assert mask[action_id_from_suit_number(suit="coins", number=3)] is True


def test_encode_observation_2p_shapes_and_mask() -> None:
    """Encoder: feature dimension fissa e mask coerente con my_hand."""
    obs = {
        "type": "observation",
        "server_version": 0,
        "my_index": 0,
        "my_hand": [
            {"suit": "cups", "rank": "ACE", "number": 1, "points": 11},
            {"suit": "clubs", "rank": "TWO", "number": 2, "points": 0},
        ],
        "my_points": 0,
        "my_turn": True,
        "trump_card": {"suit": "coins", "rank": "KING", "number": 10, "points": 4},
        "trump_suit": "coins",
        "table_cards": [{"card": {"suit": "swords", "rank": "THREE", "number": 3, "points": 10}, "player_index": 1}],
        "cards_remaining_in_deck": 20,
        "valid_actions": [0, 1],
        "game_over": False,
        "num_players": 2,
        "is_team_game": False,
        "players": [
            {"index": 0, "name": "A", "points": 0, "hand_size": 2},
            {"index": 1, "name": "B", "points": 0, "hand_size": 2},
        ],
    }

    encoded = encode_observation_2p(obs)
    assert len(encoded.action_mask) == 40
    assert sum(encoded.action_mask) == 2
    assert len(encoded.features) == (40 * 6 + 4 + 4)


def test_encode_player_observation_matches_dto_encoder() -> None:
    """
    Il path veloce `PlayerObservation -> feature` deve restare equivalente al path DTO.

    Questo protegge il refactor performance: training/evaluation usano il path diretto,
    mentre dataset/API possono ancora passare dal dict JSON.
    """
    seen = [0] * 40
    seen[action_id_from_suit_number(suit="coins", number=10)] = 1
    seen[action_id_from_suit_number(suit="swords", number=3)] = 1

    obs = PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="A",
        hand=(Card(Suit.CUPS, Rank.ACE), Card(Suit.CLUBS, Rank.TWO)),
        trump_card=Card(Suit.COINS, Rank.KING),
        deck_size=20,
        table_cards=((Card(Suit.SWORDS, Rank.THREE), 1),),
        current_turn=0,
        first_player=1,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(7, 12),
        players_hand_sizes=(2, 2),
        seen_cards_onehot=tuple(seen),
    )
    dto_like = {
        "num_players": 2,
        "my_index": 0,
        "my_hand": [
            {"suit": "cups", "rank": "ACE", "number": 1, "points": 11},
            {"suit": "clubs", "rank": "TWO", "number": 2, "points": 0},
        ],
        "my_points": 7,
        "my_turn": True,
        "trump_suit": "coins",
        "table_cards": [{"card": {"suit": "swords", "rank": "THREE", "number": 3, "points": 10}, "player_index": 1}],
        "cards_remaining_in_deck": 20,
        "players": [
            {"index": 0, "name": "A", "points": 7, "hand_size": 2},
            {"index": 1, "name": "B", "points": 12, "hand_size": 2},
        ],
        "seen_cards_onehot": seen,
    }

    for version in ("v1", "v2"):
        direct = encode_player_observation_2p(obs, version=version)
        generic = encode_observation_2p_with_version(dto_like, version=version)

        assert direct.action_mask == generic.action_mask
        assert direct.features == generic.features


def test_encode_fast_observation_matches_player_observation_encoder() -> None:
    """
    L'encoder `Fast2PState -> feature` deve restare equivalente al path canonico.

    Questo protegge il rollout A2C fast: la policy neurale deve vedere le stesse feature
    che vedrebbe usando `make_player_observation` + encoder canonico.
    """
    canonical = new_game_state(num_players=2, seed=17)
    fast = new_fast_2p_state(seed=17)
    rng = random.Random(99)

    while not canonical.game_over:
        current = canonical.current_turn
        obs = make_player_observation(canonical, current)

        for version in ("v1", "v2", "v3"):
            direct = encode_player_observation_2p(obs, version=version)
            fast_encoded = encode_fast_observation_2p(
                fast,
                player_index=current,
                seen_cards_onehot=obs.seen_cards_onehot,
                out_of_play_cards_onehot=obs.out_of_play_cards_onehot,
                version=version,
            )
            assert fast_encoded.action_mask == direct.action_mask
            assert fast_encoded.features == direct.features

        card_index = rng.randrange(len(canonical.players[current].hand))
        canonical, result = step(canonical, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None
        step_fast_2p(fast, player_index=current, card_index=card_index)
