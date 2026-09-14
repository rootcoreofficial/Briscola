"""Encoding delle osservazioni e dello spazio azioni per i modelli."""

from .card_action_space import action_id_from_suit_number, build_card_features, card_dto_to_action_id
from .observation_encoder import (
    ACTION_DIM,
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    FEATURE_DIM_2P_V3,
    EncoderVersion,
    encode_observation_2p,
    encode_observation_2p_with_version,
    encode_player_observation_2p,
    feature_dim_for_encoder_version,
)

__all__ = [
    "ACTION_DIM",
    "FEATURE_DIM_2P_V1",
    "FEATURE_DIM_2P_V2",
    "FEATURE_DIM_2P_V3",
    "EncoderVersion",
    "action_id_from_suit_number",
    "build_card_features",
    "card_dto_to_action_id",
    "encode_observation_2p",
    "encode_observation_2p_with_version",
    "encode_player_observation_2p",
    "feature_dim_for_encoder_version",
]
