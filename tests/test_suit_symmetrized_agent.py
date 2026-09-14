"""Test della policy ottenuta mediando l'orbita completa delle rinomine dei semi."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.agents.suit_symmetrized import SuitSymmetrizedBCModelAgent
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4, encode_player_observation_2p
from briscola_ai.ai.evaluation.suit_symmetry import (
    all_suit_permutations,
    inverse_suit_permutation,
    permute_action_vector,
    permute_player_observation,
)
from briscola_ai.ai.models.bc_model import BCModelAgent, MLPBCModel
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import new_game_state

_FLOAT32_RTOL = 1e-5
_FLOAT32_ATOL = 1e-7


def _observation_with_history() -> PlayerObservation:
    """Costruisce uno stato reale avanzato, così il test esercita tutto l'encoder v4."""
    state = new_game_state(2, seed=20260711)
    for action_number in range(17):
        observation = make_player_observation(state, state.current_turn)
        state, result = step(
            state,
            PlayCardAction(
                player_index=state.current_turn,
                card_index=(action_number + state.current_turn) % len(observation.hand),
            ),
        )
        assert result.error is None
    return make_player_observation(state, state.current_turn)


def _random_agent(*, overkill_guard: bool = False) -> BCModelAgent:
    """Crea una MLP deterministica non equivariant, utile a rendere causale il test."""
    generator = np.random.default_rng(20260711)
    hidden_dim = 11
    model = MLPBCModel(
        w1=generator.normal(0.0, 0.08, size=(FEATURE_DIM_2P_V4, hidden_dim)).astype(np.float32),
        b1=generator.normal(0.0, 0.03, size=hidden_dim).astype(np.float32),
        w2=generator.normal(0.0, 0.08, size=(hidden_dim, 40)).astype(np.float32),
        b2=generator.normal(0.0, 0.03, size=40).astype(np.float32),
        metadata={"format": "mlp_bc_v1", "feature_dim": FEATURE_DIM_2P_V4, "encoder_version": "v4"},
    )
    return BCModelAgent(
        model=model,
        model_path=Path("synthetic_non_equivariant_v4.npz"),
        encoder_version="v4",
        overkill_guard_enabled=overkill_guard,
    )


def test_batched_orbit_matches_24_semantic_forwards_and_identity() -> None:
    """La scorciatoia batch deve coincidere numericamente con i 24 forward semantici."""
    observation = _observation_with_history()
    base_agent = _random_agent()
    agent = SuitSymmetrizedBCModelAgent(base_agent)
    orbit = agent.orbit_logits(observation)

    encoded = encode_player_observation_2p(observation, version="v4")
    identity_logits = base_agent.model.logits(np.asarray(encoded.features, dtype=np.float32))
    # BLAS può accumulare i prodotti float32 in un ordine diverso fra un vettore e
    # un batch. Il contratto è l'equivalenza numerica, non l'identità bit per bit.
    np.testing.assert_allclose(orbit[0], identity_logits, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)

    for row, permutation in zip(orbit, all_suit_permutations(), strict=True):
        transformed = permute_player_observation(observation, permutation)
        transformed_features = np.asarray(
            encode_player_observation_2p(transformed, version="v4").features,
            dtype=np.float32,
        )
        transformed_logits = base_agent.model.logits(transformed_features)
        expected = permute_action_vector(transformed_logits, inverse_suit_permutation(permutation))
        np.testing.assert_allclose(
            row,
            np.asarray(expected),
            rtol=_FLOAT32_RTOL,
            atol=_FLOAT32_ATOL,
        )


def test_group_average_is_numerically_equivariant_for_all_24_renamings() -> None:
    """Rinominare prima dell'agente deve soltanto rinominare logits e carta scelta."""
    observation = _observation_with_history()
    agent = SuitSymmetrizedBCModelAgent(_random_agent())
    baseline_logits = agent.symmetrized_logits(observation)
    baseline_choice = agent.choose_card_index(observation, rng=random.Random(0))

    for permutation in all_suit_permutations():
        transformed = permute_player_observation(observation, permutation)
        expected_logits = np.asarray(permute_action_vector(baseline_logits, permutation))
        actual_logits = agent.symmetrized_logits(transformed)

        np.testing.assert_allclose(actual_logits, expected_logits, rtol=0.0, atol=2e-8)
        assert agent.choose_card_index(transformed, rng=random.Random(0)) == baseline_choice


def test_wrapper_rejects_postprocessing_that_would_confuse_the_policy_ablation() -> None:
    """Il confronto causale deve isolare la media, senza un guard aggiunto dopo l'argmax."""
    with pytest.raises(ValueError, match="overkill_guard disabilitato"):
        SuitSymmetrizedBCModelAgent(_random_agent(overkill_guard=True))


def test_empty_hand_fails_before_inference() -> None:
    """Una osservazione terminale non deve produrre un argmax artificiale."""
    observation = replace(_observation_with_history(), hand=())
    agent = SuitSymmetrizedBCModelAgent(_random_agent())

    with pytest.raises(ValueError, match="Mano vuota"):
        agent.choose_card_index(observation, rng=random.Random(0))
