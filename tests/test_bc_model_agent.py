"""
Test per integrazione del modello BC come agente.

Obiettivo didattico:
- verificare che l'agente scelga una carta valida in mano
- verificare che la action mask impedisca selezioni "impossibili"
- garantire che il caricamento `.npz` sia robusto (shape/metadata)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.agents import ValueLookaheadAgent, build_agent
from briscola_ai.ai.encoding.card_action_space import action_id_from_suit_number
from briscola_ai.ai.encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V3,
    encode_player_observation_2p,
)
from briscola_ai.ai.models import BCModelAgent, load_bc_model_npz
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _make_2p_observation(*, hand: tuple[Card, ...]) -> PlayerObservation:
    """Crea una `PlayerObservation` minimale (2-player) per test."""
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="A",
        hand=hand,
        trump_card=Card(Suit.CUPS, Rank.ACE),
        deck_size=20,
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(len(hand), len(hand)),
    )


def test_bc_model_agent_picks_valid_card_and_respects_mask(tmp_path: Path) -> None:
    """
    L'agente deve scegliere una carta in mano anche se il modello "preferirebbe" una carta non valida.

    Nota:
    Inseriamo un bias enorme su un'azione NON presente in mano per verificare che la mask la azzeri.
    """
    hand = (
        Card(Suit.CUPS, Rank.THREE),  # in mano
        Card(Suit.CLUBS, Rank.TWO),  # in mano
    )
    obs = _make_2p_observation(hand=hand)
    encoded = encode_player_observation_2p(obs)
    d = len(encoded.features)

    # Costruiamo un modello che preferisce la prima carta in mano (CUPS 3).
    action_a = action_id_from_suit_number(suit=hand[0].suit.value, number=hand[0].rank.number)
    action_b = action_id_from_suit_number(suit=hand[1].suit.value, number=hand[1].rank.number)

    w = np.zeros((d, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)

    # Le prime 40 feature sono `my_hand_onehot`, quindi:
    # - feature[action_id] = 1 se quella carta è in mano
    w[action_a, action_a] = 2.0
    w[action_b, action_b] = 1.0

    # Bias enorme su una carta non in mano: la mask deve comunque bloccarla.
    invalid_action = action_id_from_suit_number(suit="coins", number=1)
    assert invalid_action not in (action_a, action_b)
    b[invalid_action] = 10_000.0

    model_path = tmp_path / "bc_model.npz"
    np.savez(model_path, w=w, b=b, metadata_json=f'{{"format":"linear_softmax_bc_v1","feature_dim":{d}}}')

    agent = BCModelAgent.from_npz(model_path)
    idx = agent.choose_card_index(obs, rng=random.Random(0))

    assert 0 <= idx < len(hand)
    assert hand[idx] == hand[0]


def test_load_bc_model_npz_validates_metadata_feature_dim(tmp_path: Path) -> None:
    """Se `metadata_json.feature_dim` non coincide con `w.shape[0]`, il loader deve fallire."""
    w = np.zeros((7, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)
    model_path = tmp_path / "bad_model.npz"
    np.savez(model_path, w=w, b=b, metadata_json='{"feature_dim":8}')

    with pytest.raises(ValueError):
        load_bc_model_npz(model_path)


def test_bc_model_agent_supports_mlp_format(tmp_path: Path) -> None:
    """L'agente deve supportare anche il formato MLP (w1/b1/w2/b2) e rispettare la mask."""
    hand = (
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CLUBS, Rank.TWO),
    )
    obs = _make_2p_observation(hand=hand)
    encoded = encode_player_observation_2p(obs)
    d = len(encoded.features)

    action_a = action_id_from_suit_number(suit=hand[0].suit.value, number=hand[0].rank.number)
    action_b = action_id_from_suit_number(suit=hand[1].suit.value, number=hand[1].rank.number)

    hidden_dim = 2
    w1 = np.zeros((d, hidden_dim), dtype=np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = np.zeros((hidden_dim, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    # Proiettiamo due feature della mano su due unità hidden (ReLU pass-through perché x>=0).
    w1[action_a, 0] = 1.0
    w1[action_b, 1] = 1.0
    # Poi preferiamo action_a rispetto ad action_b.
    w2[0, action_a] = 2.0
    w2[1, action_b] = 1.0

    invalid_action = action_id_from_suit_number(suit="coins", number=1)
    assert invalid_action not in (action_a, action_b)
    b2[invalid_action] = 10_000.0

    model_path = tmp_path / "bc_model_mlp.npz"
    np.savez(
        model_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        metadata_json=f'{{"format":"mlp_bc_v1","feature_dim":{d},"hidden_dim":{hidden_dim}}}',
    )

    agent = BCModelAgent.from_npz(model_path)
    idx = agent.choose_card_index(obs, rng=random.Random(0))
    assert idx == 0


def test_build_agent_best_a2c_loads_from_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `best_a2c` è un alias comodo per league training: carica `best_a2c.npz` dalla directory modelli.

    Questo test verifica che:
    - la risoluzione avvenga tramite `BRISCOLA_MODELS_DIR`
    - il file venga caricato come policy compatibile (feature_dim = encoder 2p v1)
    """
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))

    model_path = tmp_path / "best_a2c.npz"
    w1 = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
    b1 = np.zeros((1,), dtype=np.float32)
    w2 = np.zeros((1, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    np.savez(model_path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json='{"format":"mlp_a2c_shaped_v1"}')

    agent = build_agent("best_a2c")
    assert isinstance(agent, BCModelAgent)
    assert agent.model_path.name == "best_a2c.npz"


def test_build_agent_best_a2c_errors_if_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Se manca il file `best_a2c.npz`, l'errore deve essere chiaro e user-friendly."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match=r"best_a2c\.npz"):
        build_agent("best_a2c")


def _write_linear_model(path: Path, *, bias_action: int = 0) -> None:
    """Salva un modello lineare minimale (feature_dim v1) per test di cache."""
    d = int(FEATURE_DIM_2P_V1)
    w = np.zeros((d, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)
    b[bias_action] = 1.0
    np.savez(path, w=w, b=b, metadata_json=f'{{"format":"linear_softmax_bc_v1","feature_dim":{d}}}')


def _write_zero_value_model(path: Path) -> None:
    """Salva un value model minimale v3 per testare la factory del lookahead."""
    d = int(FEATURE_DIM_2P_V3)
    h = 4
    metadata = {
        "format": "value_mlp_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "encoder_version": "v3",
        "target": "residual",
        "target_scale": 120.0,
    }
    np.savez(
        path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h,), dtype=np.float32),
        b2=np.asarray([0.0], dtype=np.float32),
        metadata_json=json.dumps(metadata),
    )


def test_build_agent_caches_bc_model_by_path(tmp_path: Path) -> None:
    """Lo stesso `.npz` non viene riletto: il modello caricato è condiviso dalla cache in-process."""
    model_path = tmp_path / "m.npz"
    _write_linear_model(model_path)

    a1 = build_agent("bc_model", model_path=model_path)
    a2 = build_agent("bc_model", model_path=model_path)
    # La cache è a livello di payload `.npz`: il modello caricato è la stessa istanza riusata
    # (l'agente wrapper può essere ricreato, ma il file non viene riletto).
    assert a1.model is a2.model


def test_bc_model_cache_invalidated_when_file_changes(tmp_path: Path) -> None:
    """Se il file cambia (mtime/size), la cache si invalida e viene ricaricato."""
    model_path = tmp_path / "m.npz"
    _write_linear_model(model_path)
    a1 = build_agent("bc_model", model_path=model_path)

    # Forziamo un mtime diverso (simula una nuova promozione/sovrascrittura del modello).
    st = os.stat(model_path)
    bumped = st.st_mtime_ns + 1_000_000_000
    os.utime(model_path, ns=(bumped, bumped))

    a2 = build_agent("bc_model", model_path=model_path)
    assert a2.model is not a1.model  # file cambiato => riletto


def test_catalog_excludes_belief_and_value_assets(tmp_path: Path) -> None:
    """
    Regressione v0.23.0: la belief network (369->128->40) ha le stesse shape di una policy
    v4 e senza filtro appariva SELEZIONABILE nel catalogo modelli della UI. Value e belief
    sono asset interni degli agenti search: mai nel menu.
    """
    import json

    import numpy as np

    from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
    from briscola_ai.ai.models.catalog import list_local_models

    rng = np.random.default_rng(0)
    for fname, fmt in (("belief.npz", "belief_mlp_v1"), ("value.npz", "value_mlp_v1")):
        np.savez(
            tmp_path / fname,
            w1=rng.normal(0, 0.05, size=(int(FEATURE_DIM_2P_V4), 4)).astype(np.float32),
            b1=np.zeros(4, dtype=np.float32),
            w2=rng.normal(0, 0.05, size=(4, 40)).astype(np.float32),
            b2=np.zeros(40, dtype=np.float32),
            metadata_json=json.dumps({"format": fmt, "encoder_version": "v4"}),
        )

    assert list_local_models(tmp_path) == []


def test_official_v15_asset_is_compact_compatible_and_has_public_metadata() -> None:
    """L'asset della release deve essere giocabile, immutabile e privo di path locali."""
    from briscola_ai.ai.models.catalog import list_local_models, validate_model_compatible_for_ui

    root = Path(__file__).resolve().parents[1]
    model_path = root / "data/models/best_a2c_v15.npz"

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == (
        "2f2dca3d4e77a363783124feeb30f482a85a740077222936b025b37b865f2eb6"
    )
    assert model_path.stat().st_size < 500_000
    validate_model_compatible_for_ui(model_path)

    spec = next(model for model in list_local_models(model_path.parent) if model.id == model_path.name)
    assert spec.is_compatible is True
    assert spec.metadata["label"] == "Briscola AI v15"
    assert spec.metadata["release"]["version"] == "0.38.0"
    assert spec.metadata["release"]["runtime_agent"] == "bc_model_pimc_belief_12x8"
    assert str(root) not in json.dumps(spec.metadata)


def test_validate_for_ui_accepts_v4_and_v4_belief_models(tmp_path: Path) -> None:
    """
    Regressione promozione v8: il validatore UI deve accettare encoder v4 (369) e
    policy con belief embedded (409). Il bug originale whitelist-ava solo v1-v3 e il
    modello promosso appariva "NON COMPATIBILE" nel menu della UI.
    """
    import json

    import numpy as np

    from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
    from briscola_ai.ai.models import validate_model_compatible_for_ui

    rng = np.random.default_rng(0)

    def _write_mlp(path: Path, feature_dim: int, with_belief: bool) -> None:
        arrays = {
            "w1": rng.normal(0, 0.05, size=(feature_dim, 4)).astype(np.float32),
            "b1": np.zeros(4, dtype=np.float32),
            "w2": rng.normal(0, 0.05, size=(4, 40)).astype(np.float32),
            "b2": np.zeros(40, dtype=np.float32),
        }
        if with_belief:
            arrays.update(
                belief_w1=rng.normal(0, 0.05, size=(FEATURE_DIM_2P_V4, 4)).astype(np.float32),
                belief_b1=np.zeros(4, dtype=np.float32),
                belief_w2=rng.normal(0, 0.05, size=(4, 40)).astype(np.float32),
                belief_b2=np.zeros(40, dtype=np.float32),
            )
        np.savez(path, **arrays, metadata_json=json.dumps({"format": "mlp_bc_v1", "encoder_version": "v4"}))

    v4_path = tmp_path / "v4.npz"
    _write_mlp(v4_path, int(FEATURE_DIM_2P_V4), with_belief=False)
    validate_model_compatible_for_ui(v4_path)  # non deve sollevare

    v4b_path = tmp_path / "v4_belief.npz"
    _write_mlp(v4b_path, int(FEATURE_DIM_2P_V4) + 40, with_belief=True)
    validate_model_compatible_for_ui(v4b_path)  # non deve sollevare


def test_validate_for_ui_shares_cache_with_from_npz(tmp_path: Path) -> None:
    """La validazione UI e `from_npz` usano lo stesso `load_bc_model_npz` cacheato (no doppia lettura)."""
    from briscola_ai.ai.models import load_bc_model_npz, validate_model_compatible_for_ui

    model_path = tmp_path / "m.npz"
    _write_linear_model(model_path)

    m1 = load_bc_model_npz(model_path)
    validate_model_compatible_for_ui(model_path)  # non deve rileggere il file
    agent = build_agent("bc_model", model_path=model_path)
    assert agent.model is m1  # stesso payload condiviso tra validazione e costruzione agente


def test_build_agent_value_lookahead_uses_selected_policy_and_required_value_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La variante value-lookahead usa la policy selezionata dalla UI più il value model fisso."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    policy_path = tmp_path / "policy.npz"
    _write_linear_model(policy_path)
    _write_zero_value_model(tmp_path / "value_v0_h128_clean50k_seed20260701.npz")

    agent = build_agent("bc_model_value_lookahead_8x8", model_path=policy_path)

    assert isinstance(agent, ValueLookaheadAgent)
    assert agent.name == "bc_model_value_lookahead_8x8"
    assert agent.num_determinizations == 8
    assert agent.max_unknown_cards == 8
    assert agent.overkill_guard_enabled is True


def test_build_agent_value_lookahead_errors_if_value_model_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Senza asset value model l'agente avanzato deve fallire subito, non a partita iniziata."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    policy_path = tmp_path / "policy.npz"
    _write_linear_model(policy_path)

    with pytest.raises(ValueError, match="value_v0_h128_clean50k_seed20260701"):
        build_agent("bc_model_value_lookahead_8x8", model_path=policy_path)
