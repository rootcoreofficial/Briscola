"""
Test per l'encoder v3 (Fase 5G, step 4) — domain-first.

v3 = v2 (288) + 22 feature strategiche aggregate (totale 310). Focus dei test:
- contratto dimensione e prefisso v2 invariato;
- parità tra path dict (DTO) e path oggetto (PlayerObservation);
- definizione "ignota" che esclude la briscola scoperta pubblica;
- feature `*_out_of_play` legate alle prese;
- guard espliciti: v3 non supportato su fast/numba.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import (
    FEATURE_DIM_2P_V2,
    FEATURE_DIM_2P_V3,
    encode_observation_2p_with_version,
    encode_player_observation_2p,
    feature_dim_for_encoder_version,
)
from briscola_ai.ai.fast.observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.fast.state_2p import new_fast_2p_state
from briscola_ai.ai.models import BCModelAgent, validate_model_compatible_for_ui
from briscola_ai.ai.numba.observation import encode_fast_observation_numba_2p
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, PlayerState, new_game_state

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba


def _endgame_like_state() -> GameState:
    """Stato a mazzo vuoto con prese non banali (per attivare le feature out_of_play)."""
    trump = Card(Suit.COINS, Rank.KING)
    captured0 = (Card(Suit.SWORDS, Rank.ACE), Card(Suit.CUPS, Rank.THREE))
    captured1 = (Card(Suit.CLUBS, Rank.ACE),)
    players = (
        PlayerState("P0", (Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)), captured0, 21),
        PlayerState("P1", (Card(Suit.COINS, Rank.THREE), Card(Suit.CLUBS, Rank.TWO)), captured1, 11),
    )
    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=players,
        deck=(),
        trump_card=trump,
        table_cards=(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def test_v3_feature_dim_contract() -> None:
    """v3 = v2 + 22 = 310, sia come costante sia come output reale (dict e oggetto)."""
    assert FEATURE_DIM_2P_V3 == FEATURE_DIM_2P_V2 + 22 == 310
    assert feature_dim_for_encoder_version("v3") == 310

    state = new_game_state(2, seed=3)
    obs = make_player_observation(state, player_index=0)
    dto = build_observation_dto(state, player_index=0, server_version=1).model_dump()

    assert len(encode_player_observation_2p(obs, version="v3").features) == 310
    assert len(encode_observation_2p_with_version(dto, version="v3").features) == 310


def test_v3_keeps_v2_prefix() -> None:
    """Le prime 288 feature di v3 coincidono con v2 (estensione additiva)."""
    state = new_game_state(2, seed=11)
    obs = make_player_observation(state, player_index=0)

    v2 = encode_player_observation_2p(obs, version="v2").features
    v3 = encode_player_observation_2p(obs, version="v3").features

    assert v3[:FEATURE_DIM_2P_V2] == v2


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_v3_dict_object_parity_midgame(seed: int) -> None:
    """Path dict (DTO) e path oggetto producono lo stesso vettore v3 in mid-game."""
    state = new_game_state(2, seed=seed)
    # Gioca qualche mossa per popolare prese/tavolo (out_of_play non banale).
    for _ in range(5):
        if state.game_over:
            break
        state, _ = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))

    player = state.current_turn
    obs = make_player_observation(state, player_index=player)
    dto = build_observation_dto(state, player_index=player, server_version=1).model_dump()

    from_obj = encode_player_observation_2p(obs, version="v3")
    from_dict = encode_observation_2p_with_version(dto, version="v3")

    assert from_obj.action_mask == from_dict.action_mask
    assert from_obj.features == pytest.approx(from_dict.features)


def test_v3_dict_object_parity_endgame() -> None:
    """Parità anche a mazzo vuoto, dove il DTO azzera trump_card ma mantiene trump_suit."""
    state = new_game_state(2, seed=123)
    while len(state.deck) > 0 and not state.game_over:
        state, _ = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))

    player = state.current_turn
    obs = make_player_observation(state, player_index=player)
    dto = build_observation_dto(state, player_index=player, server_version=1).model_dump()
    assert dto["trump_card"] is None  # a mazzo vuoto il builder non mostra la carta
    assert dto["trump_suit"] is not None

    from_obj = encode_player_observation_2p(obs, version="v3")
    from_dict = encode_observation_2p_with_version(dto, version="v3")

    assert from_obj.action_mask == from_dict.action_mask
    assert from_obj.features == pytest.approx(from_dict.features)


def test_public_trump_is_not_counted_as_unknown() -> None:
    """
    La briscola scoperta è "nota" e non deve contare tra le briscole alte ignote.

    Costruiamo uno stato a inizio partita dove l'Asso di briscola è proprio la briscola scoperta
    (ancora nel mazzo): unknown_high_trump_ace deve essere 0, mentre un'altra briscola alta non
    vista (es. il Tre) resta ignota = 1.
    """
    trump = Card(Suit.COINS, Rank.ACE)  # Asso di briscola scoperto
    players = (
        PlayerState("P0", (Card(Suit.CUPS, Rank.TWO), Card(Suit.CLUBS, Rank.FOUR)), (), 0),
        PlayerState("P1", (Card(Suit.SWORDS, Rank.FIVE), Card(Suit.CUPS, Rank.SIX)), (), 0),
    )
    # Deck non vuoto con la briscola "sotto": basta che deck_size > 0 e trump in deck.
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=players,
        deck=(trump, Card(Suit.SWORDS, Rank.KING)),
        trump_card=trump,
        table_cards=(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )
    obs = make_player_observation(state, player_index=0)
    features = encode_player_observation_2p(obs, version="v3").features

    # Layout v3 extra: indice 288 = unknown_trumps_norm, 289 = ace, 290 = three, 291 = king.
    unknown_high_trump_ace = features[FEATURE_DIM_2P_V2 + 1]
    unknown_high_trump_three = features[FEATURE_DIM_2P_V2 + 2]

    assert unknown_high_trump_ace == 0.0  # è la briscola scoperta (pubblica): non ignota
    assert unknown_high_trump_three == 1.0  # Tre di briscola non visto: ignota


def test_out_of_play_features_reflect_captures() -> None:
    """`ace_out_of_play`/`three_out_of_play` riflettono le carte finite nelle prese."""
    state = _endgame_like_state()
    obs = make_player_observation(state, player_index=0)
    features = encode_player_observation_2p(obs, version="v3").features

    # Blocco per-seme parte all'offset FEATURE_DIM_2P_V2 + 4; ogni seme occupa 3 valori
    # [ace_out_of_play, three_out_of_play, unknown_load_norm], ordine semi clubs,cups,coins,swords.
    block = FEATURE_DIM_2P_V2 + 4
    cups_ace_out = features[block + 1 * 3 + 0]  # Asso di coppe... non catturato qui
    swords_ace_out = features[block + 3 * 3 + 0]  # Asso di spade -> catturato (captured0)
    cups_three_out = features[block + 1 * 3 + 1]  # Tre di coppe -> catturato (captured0)

    assert swords_ace_out == 1.0
    assert cups_three_out == 1.0
    assert cups_ace_out == 0.0


def test_v3_endgame_flag_set_when_deck_empty() -> None:
    """`is_endgame` (deck_size==0) attivo a mazzo vuoto."""
    state = _endgame_like_state()
    obs = make_player_observation(state, player_index=0)
    features = encode_player_observation_2p(obs, version="v3").features

    # Blocco fase: [deck_size_norm, my_hand_size_norm, is_endgame] dopo briscole(4) + semi(12).
    is_endgame = features[FEATURE_DIM_2P_V2 + 4 + 12 + 2]
    assert is_endgame == 1.0


def test_fast_v3_requires_out_of_play() -> None:
    """Il fast encoder ora supporta v3, ma senza `out_of_play` deve fallire chiaro (no fallback)."""
    state = new_fast_2p_state(seed=1)
    seen = tuple(0 for _ in range(40))

    with pytest.raises(ValueError, match="richiede `out_of_play_cards_onehot`"):
        encode_fast_observation_2p(state, player_index=0, seen_cards_onehot=seen, version="v3")


def test_numba_v3_requires_out_of_play() -> None:
    """Il numba encoder supporta v3, ma senza `out_of_play` deve fallire chiaro (no fallback)."""
    state = new_fast_2p_state(seed=1)
    seen = tuple(0 for _ in range(40))

    with pytest.raises(ValueError, match="richiede `out_of_play_cards_onehot`"):
        encode_fast_observation_numba_2p(state, player_index=0, seen_cards_onehot=seen, version="v3")


def test_v3_model_roundtrip_loads_and_is_catalog_compatible(tmp_path: Path) -> None:
    """Un modello con feature_dim=310 viene inferito come v3, è giocabile e accettato dal catalogo."""
    d = int(FEATURE_DIM_2P_V3)
    w = np.zeros((d, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)
    model_path = tmp_path / "model_v3.npz"
    # Senza encoder dichiarato: la versione deve essere inferita da feature_dim (310 -> v3).
    np.savez(model_path, w=w, b=b, metadata_json=f'{{"format":"linear_softmax_bc_v1","feature_dim":{d}}}')

    agent = BCModelAgent.from_npz(model_path)
    assert agent.encoder_version == "v3"

    state = new_game_state(2, seed=5)
    obs = make_player_observation(state, player_index=0)
    idx = agent.choose_card_index(obs, rng=random.Random(0))
    assert 0 <= idx < len(obs.hand)

    # Il catalogo UI non deve sollevare per un modello v3.
    validate_model_compatible_for_ui(model_path)


def test_v3_model_metadata_feature_dim_mismatch_is_rejected(tmp_path: Path) -> None:
    """Se i metadati dichiarano encoder=v3 ma feature_dim non è 310, il load deve fallire."""
    w = np.zeros((int(FEATURE_DIM_2P_V2), 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)
    model_path = tmp_path / "bad_v3.npz"
    np.savez(
        model_path,
        w=w,
        b=b,
        metadata_json=f'{{"format":"linear_softmax_bc_v1","feature_dim":{int(FEATURE_DIM_2P_V2)},"encoder_version":"v3"}}',
    )

    with pytest.raises(ValueError, match="Modello incoerente"):
        BCModelAgent.from_npz(model_path)
