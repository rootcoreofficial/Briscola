"""
Pipeline belief (Fase 2): determinizzazioni pesate, loader, pesi dalla rete.

Cosa proteggono questi test:
- il campionamento pesato rispetta i vincoli anti-cheat/di coerenza (mai carte note,
  cardinalità giusta, briscola pescata per ultima) ed è davvero sbilanciato dai pesi;
- `card_weights=None` mantiene il comportamento storico (percorso uniforme intatto);
- `belief_card_weights` calcola pesi solo sulle carte ignote, con il mix uniforme;
- il loader `.npz` valida formato/shape/encoder;
- lo scenario "carta impossibile": una belief a zero su una carta NON la rende
  incampionabile grazie al pavimento uniforme (anti punto-cieco).
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from briscola_ai.ai.agents.pimc import (
    _weighted_sample_without_replacement,
    belief_card_weights,
    determinize_observation,
)
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.models.belief_model import MLPBeliefModel, load_belief_model_npz
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state


def _observation_in_search_window(seed: int = 3):
    """Porta una partita reale nella finestra di search (mazzo corto) e ritorna l'osservazione."""
    state = new_game_state(2, seed=seed)
    while len(state.deck) > 6:
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None
    return make_player_observation(state, state.current_turn), state


def _uniform_belief_model(bias: np.ndarray | None = None) -> MLPBeliefModel:
    """Belief giocattolo: pesi nulli => logits = b2 (controllabili dal test)."""
    return MLPBeliefModel(
        w1=np.zeros((FEATURE_DIM_2P_V4, 4), dtype=np.float32),
        b1=np.zeros(4, dtype=np.float32),
        w2=np.zeros((4, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32) if bias is None else bias.astype(np.float32),
        metadata={"format": "belief_mlp_v1", "encoder_version": "v4", "feature_dim": FEATURE_DIM_2P_V4},
    )


def test_weighted_sample_respects_weights_and_no_replacement() -> None:
    """Il campione è senza rimpiazzo e fortemente sbilanciato dai pesi."""
    rng = random.Random(0)
    pool = [0, 1, 2, 3]
    weights = {0: 100.0, 1: 1.0, 2: 1.0, 3: 1.0}
    heavy_first = 0
    for _ in range(500):
        sample = _weighted_sample_without_replacement(pool, 2, weights, rng)
        assert len(sample) == len(set(sample)) == 2
        assert set(sample) <= set(pool)
        if sample[0] == 0:
            heavy_first += 1
    # P(prima estratta = carta pesante) = 100/103 ~ 0.97.
    assert heavy_first > 450


def test_weighted_sample_falls_back_to_uniform_on_zero_weights() -> None:
    """Pesi tutti nulli: degradazione all'uniforme, mai un crash."""
    rng = random.Random(1)
    seen = set()
    for _ in range(200):
        seen.update(_weighted_sample_without_replacement([5, 6, 7], 1, {}, rng))
    assert seen == {5, 6, 7}


def test_determinize_with_weights_keeps_anticheat_invariants() -> None:
    """Con i pesi, la determinizzazione resta coerente: mai carte note, conteggi giusti."""
    observation, _state = _observation_in_search_window()
    my_ids = {card_to_id(c) for c in observation.hand}
    out_ids = {i for i, f in enumerate(observation.out_of_play_cards_onehot) if f}
    opp_size = observation.players_hand_sizes[1 - observation.player_index]

    weights = {card_id: float(card_id + 1) for card_id in range(40)}
    for trial in range(30):
        sampled = determinize_observation(observation, rng=random.Random(trial), card_weights=weights)
        opp_hand_ids = {card_to_id(c) for c in sampled.players[1 - observation.player_index].hand}
        assert len(opp_hand_ids) == opp_size
        assert not (opp_hand_ids & my_ids)
        assert not (opp_hand_ids & out_ids)
        assert len(sampled.deck) == observation.deck_size
        # La briscola pescabile resta l'ultima pescata (in testa al deck).
        if observation.trump_card is not None and card_to_id(observation.trump_card) not in out_ids | my_ids:
            assert sampled.deck[0] == observation.trump_card


def test_determinize_weights_bias_opponent_hand() -> None:
    """Pesare una carta ~tutto il pool la mette in mano avversaria quasi sempre."""
    observation, _state = _observation_in_search_window()
    my_ids = {card_to_id(c) for c in observation.hand}
    out_ids = {i for i, f in enumerate(observation.out_of_play_cards_onehot) if f}
    pool = sorted(set(range(40)) - my_ids - out_ids)
    trump_id = card_to_id(observation.trump_card) if observation.trump_card else None
    # La briscola pescabile è forzata nel deck: scegli una carta pool diversa da lei.
    heavy = next(cid for cid in pool if cid != trump_id)

    weights = {cid: (1000.0 if cid == heavy else 1.0) for cid in pool}
    hits = 0
    trials = 200
    for trial in range(trials):
        sampled = determinize_observation(observation, rng=random.Random(trial), card_weights=weights)
        opp_ids = {card_to_id(c) for c in sampled.players[1 - observation.player_index].hand}
        if heavy in opp_ids:
            hits += 1
    assert hits > trials * 0.9


def test_determinize_without_weights_is_unchanged_and_deterministic() -> None:
    """`card_weights=None` percorre il path storico: stesso rng => stesso stato."""
    observation, _state = _observation_in_search_window()
    a = determinize_observation(observation, rng=random.Random(7))
    b = determinize_observation(observation, rng=random.Random(7))
    assert a == b


def test_belief_card_weights_only_on_unknown_and_mixed() -> None:
    """I pesi coprono SOLO le carte ignote e includono il pavimento uniforme."""
    observation, _state = _observation_in_search_window()
    my_ids = {card_to_id(c) for c in observation.hand}
    out_ids = {i for i, f in enumerate(observation.out_of_play_cards_onehot) if f}
    unknown = set(range(40)) - my_ids - out_ids

    # Belief che spara tutto su una sola carta ignota.
    target = sorted(unknown)[0]
    bias = np.full(40, -20.0)
    bias[target] = 20.0
    weights = belief_card_weights(_uniform_belief_model(bias), observation, uniform_mix=0.10)

    assert set(weights) == unknown
    # Pavimento anti punto-cieco: anche le carte a belief ~0 restano campionabili.
    n = len(unknown)
    for card_id in unknown - {target}:
        assert weights[card_id] >= 0.10 / n * 0.99
    assert weights[target] == max(weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_belief_model_loader_validates(tmp_path) -> None:
    """Il loader rifiuta formato sbagliato, shape errate ed encoder incoerente."""
    good_meta = {"format": "belief_mlp_v1", "encoder_version": "v4", "feature_dim": FEATURE_DIM_2P_V4}

    path = tmp_path / "ok.npz"
    np.savez(
        path,
        w1=np.zeros((FEATURE_DIM_2P_V4, 8), dtype=np.float32),
        b1=np.zeros(8, dtype=np.float32),
        w2=np.zeros((8, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps(good_meta),
    )
    model = load_belief_model_npz(path)
    assert model.feature_dim == FEATURE_DIM_2P_V4
    probs = model.predict_probs(np.zeros(FEATURE_DIM_2P_V4, dtype=np.float32))
    assert probs.shape == (40,)
    assert np.all((probs > 0) & (probs < 1))

    bad_format = tmp_path / "bad_format.npz"
    np.savez(
        bad_format,
        w1=np.zeros((FEATURE_DIM_2P_V4, 8), dtype=np.float32),
        b1=np.zeros(8, dtype=np.float32),
        w2=np.zeros((8, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps({**good_meta, "format": "altro"}),
    )
    with pytest.raises(ValueError, match="Formato belief"):
        load_belief_model_npz(bad_format)

    bad_encoder = tmp_path / "bad_encoder.npz"
    np.savez(
        bad_encoder,
        w1=np.zeros((100, 8), dtype=np.float32),
        b1=np.zeros(8, dtype=np.float32),
        w2=np.zeros((8, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps({**good_meta, "feature_dim": 100}),
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_belief_model_npz(bad_encoder)
