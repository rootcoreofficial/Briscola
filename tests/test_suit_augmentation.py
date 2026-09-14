"""
Test del contratto numerico usato dall'augmentation paired dei semi.

Il riferimento non è una tabella di indici duplicata nel test: trasformiamo la
``PlayerObservation`` semanticamente e rieseguiamo l'encoder canonico. La mappa veloce è
corretta soltanto se produce lo stesso vettore per v1, v2, v3 e v4.
"""

from __future__ import annotations

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import EncoderVersion, encode_player_observation_2p
from briscola_ai.ai.evaluation.suit_symmetry import (
    IDENTITY_SUIT_PERMUTATION,
    all_suit_permutations,
    inverse_suit_permutation,
    permute_card_id,
    permute_player_observation,
)
from briscola_ai.ai.training.suit_augmentation import (
    encoded_feature_index_mapping,
    permute_action_ids,
    permute_action_masks,
    permute_encoded_features,
    permute_encoded_trajectory,
    sample_nonidentity_suit_permutation,
)
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import new_game_state


def _trajectory_observations() -> list[PlayerObservation]:
    """Raccoglie stati reali con tavolo, storia e fasi differenti, senza agenti ML."""
    state = new_game_state(2, seed=20260711)
    observations: list[PlayerObservation] = []
    action_number = 0
    while not state.game_over:
        observation = make_player_observation(state, state.current_turn)
        if action_number in {0, 1, 10, 21, 30, 35, 37}:
            observations.append(observation)
        card_index = (action_number + state.current_turn) % len(observation.hand)
        state, result = step(
            state,
            PlayCardAction(player_index=observation.player_index, card_index=card_index),
        )
        assert result.error is None
        action_number += 1
    return observations


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_encoded_feature_permutation_matches_semantic_reference(version: EncoderVersion) -> None:
    """Ogni coordinata numerica deve coincidere col percorso semantico su tutte le 24 rinomine."""
    for observation in _trajectory_observations():
        encoded = encode_player_observation_2p(observation, version=version)
        features = np.asarray(encoded.features, dtype=np.float32)
        mask = np.asarray(encoded.action_mask, dtype=bool)
        for permutation in all_suit_permutations():
            semantic = encode_player_observation_2p(
                permute_player_observation(observation, permutation),
                version=version,
            )
            numeric_features = permute_encoded_features(features, version=version, permutation=permutation)
            numeric_mask = permute_action_masks(mask, permutation=permutation)

            np.testing.assert_array_equal(numeric_features, np.asarray(semantic.features, dtype=np.float32))
            np.testing.assert_array_equal(numeric_mask, np.asarray(semantic.action_mask, dtype=bool))


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_feature_mapping_is_bijective_and_roundtrips_batches(version: EncoderVersion) -> None:
    """La mappa deve coprire ogni feature una volta e l'inversa deve annullarla anche in batch."""
    permutation = all_suit_permutations()[9]
    inverse = inverse_suit_permutation(permutation)
    mapping = encoded_feature_index_mapping(version, permutation)
    assert len(set(mapping)) == len(mapping)

    observation = _trajectory_observations()[3]
    row = np.asarray(encode_player_observation_2p(observation, version=version).features, dtype=np.float32)
    batch = np.stack([row, row * np.float32(0.5)])
    transformed = permute_encoded_features(batch, version=version, permutation=permutation)
    restored = permute_encoded_features(transformed, version=version, permutation=inverse)
    np.testing.assert_array_equal(restored, batch)


def test_paired_trajectory_uses_one_permutation_for_features_masks_and_actions() -> None:
    """Tutti gli step della copia devono seguire la stessa rinomina, inclusa l'azione scelta."""
    observations = _trajectory_observations()[:5]
    permutation = all_suit_permutations()[18]
    xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    action_ids: list[int] = []
    for observation in observations:
        encoded = encode_player_observation_2p(observation, version="v4")
        xs.append(np.asarray(encoded.features, dtype=np.float32))
        masks.append(np.asarray(encoded.action_mask, dtype=bool))
        action_ids.append(card_to_id(observation.hand[0]))

    paired = permute_encoded_trajectory(
        xs=np.stack(xs),
        action_masks=np.stack(masks),
        action_ids=np.asarray(action_ids, dtype=np.int64),
        version="v4",
        permutation=permutation,
    )

    assert paired.permutation == permutation
    for index, observation in enumerate(observations):
        semantic = encode_player_observation_2p(
            permute_player_observation(observation, permutation),
            version="v4",
        )
        np.testing.assert_array_equal(paired.xs[index], np.asarray(semantic.features, dtype=np.float32))
        np.testing.assert_array_equal(paired.action_masks[index], np.asarray(semantic.action_mask, dtype=bool))
        assert int(paired.action_ids[index]) == permute_card_id(action_ids[index], permutation)


def test_action_helpers_validate_shapes_ranges_and_dtype() -> None:
    """Input malformati devono fallire prima di contaminare un optimizer update."""
    permutation = all_suit_permutations()[1]
    with pytest.raises(ValueError, match="shape"):
        permute_action_masks(np.zeros((2, 39), dtype=bool), permutation=permutation)
    with pytest.raises(ValueError, match="dtype intero"):
        permute_action_ids(np.asarray([1.0], dtype=np.float32), permutation=permutation)
    with pytest.raises(ValueError, match="fuori"):
        permute_action_ids(np.asarray([40], dtype=np.int64), permutation=permutation)
    with pytest.raises(ValueError, match="Numero step"):
        permute_encoded_trajectory(
            xs=np.zeros((2, 369), dtype=np.float32),
            action_masks=np.zeros((1, 40), dtype=bool),
            action_ids=np.zeros(2, dtype=np.int64),
            version="v4",
            permutation=permutation,
        )


def test_nonidentity_sampler_is_deterministic_and_covers_all_23_choices() -> None:
    """L'RNG dedicato non deve mai restituire l'identità e deve raggiungere tutta l'orbita."""
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)
    first = [sample_nonidentity_suit_permutation(first_rng) for _ in range(2_300)]
    second = [sample_nonidentity_suit_permutation(second_rng) for _ in range(2_300)]

    assert first == second
    assert IDENTITY_SUIT_PERMUTATION not in first
    assert set(first) == set(all_suit_permutations()[1:])
