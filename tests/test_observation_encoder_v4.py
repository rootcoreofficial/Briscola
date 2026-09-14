"""
Encoder v4: storia ordinata/attribuita delle prese (opponent modeling).

Cosa proteggono questi test:
- contratto dimensionale (v4 = v3 + 59) e proprietà di prefisso (i primi 310 valori
  coincidono con v3: gli encoder sono cumulativi per costruzione);
- parità dict (ObservationDTO) / oggetto (PlayerObservation) lungo partite reali;
- semantica delle feature comportamentali su uno scenario costruito a mano
  (taglio, risposta a seme, scarto: il cuore dell'opponent modeling);
- backward compatibility: dataset/payload senza `trick_history` degradano a zero;
- anti-cheat: la storia è identica per entrambi gli osservatori (è pubblica).
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.encoding.observation_encoder import (
    _V4_EXTRA_DIM,
    FEATURE_DIM_2P_V3,
    FEATURE_DIM_2P_V4,
    encode_observation_2p_v4,
    encode_player_observation_2p,
)
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, PlayerState, TrickRecord, new_game_state


def test_v4_dimension_contract_and_v3_prefix() -> None:
    """v4 = v3 + 59, e il prefisso v3 resta identico (encoder cumulativi)."""
    assert FEATURE_DIM_2P_V4 == FEATURE_DIM_2P_V3 + _V4_EXTRA_DIM == 369

    state = new_game_state(2, seed=11)
    for _ in range(6):
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None

    obs = make_player_observation(state, state.current_turn)
    enc_v3 = encode_player_observation_2p(obs, version="v3")
    enc_v4 = encode_player_observation_2p(obs, version="v4")
    assert len(enc_v4.features) == FEATURE_DIM_2P_V4
    assert enc_v4.features[:FEATURE_DIM_2P_V3] == enc_v3.features
    assert enc_v4.action_mask == enc_v3.action_mask


@pytest.mark.parametrize("seed", [0, 3, 7, 42])
def test_v4_dict_object_parity_on_real_games(seed: int) -> None:
    """Path dict (DTO) e path oggetto (PlayerObservation) devono coincidere a ogni decisione."""
    state = new_game_state(2, seed=seed)
    while not state.game_over:
        for player_index in range(2):
            obs = make_player_observation(state, player_index)
            enc_obj = encode_player_observation_2p(obs, version="v4")
            dto = build_observation_dto(state, player_index, 0).model_dump()
            enc_dict = encode_observation_2p_v4(dto)
            assert enc_obj.features == enc_dict.features
            assert enc_obj.action_mask == enc_dict.action_mask
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None


def _observation_with_history(
    trick_history: tuple[TrickRecord, ...],
    *,
    observer: int = 0,
    trump_card: Card = Card(Suit.COINS, Rank.SEVEN),  # noqa: B008 (Card è frozen: default sicuro)
):
    """Costruisce una PlayerObservation minimale con la storia data (solo per il blocco v4)."""
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState("P0", (Card(Suit.CLUBS, Rank.TWO),), (), 0),
            PlayerState("P1", (Card(Suit.CUPS, Rank.FOUR),), (), 0),
        ),
        deck=(),
        trump_card=trump_card,
        table_cards=(),
        current_turn=observer,
        first_player=observer,
        game_over=False,
        winner_index=None,
        winning_team=None,
        trick_history=trick_history,
    )
    return make_player_observation(state, observer)


def test_v4_behavior_features_hand_computed() -> None:
    """
    Scenario costruito a mano (briscola = denari, osservatore = P0):

    - presa 1: P0 lede Re di coppe, P1 risponde Asso di DENARI (briscola) e vince -> "taglio";
    - presa 2: P1 lede Due di bastoni, P0 risponde Tre di bastoni e vince;
    - presa 3: P0 lede Sette di spade, P1 risponde Cinque di coppe ("scarto"), vince P0.

    Attesi (blocco comportamento avversario, offset 310..320):
      per-seme P1: bastoni=1 (lead presa 2), coppe=1 (scarto), denari=1 (taglio), spade=0;
      lead=1, tagli=1, risposte a seme=0, scarti=1, lead briscola=0, lead carichi=0, prese=3.
    """
    trump = Card(Suit.COINS, Rank.SEVEN)
    history = (
        TrickRecord(
            cards=((Card(Suit.CUPS, Rank.KING), 0), (Card(Suit.COINS, Rank.ACE), 1)),
            winner_index=1,
            points=15,
        ),
        TrickRecord(
            cards=((Card(Suit.CLUBS, Rank.TWO), 1), (Card(Suit.CLUBS, Rank.THREE), 0)),
            winner_index=0,
            points=10,
        ),
        TrickRecord(
            cards=((Card(Suit.SWORDS, Rank.SEVEN), 0), (Card(Suit.CUPS, Rank.FIVE), 1)),
            winner_index=0,
            points=0,
        ),
    )
    obs = _observation_with_history(history, observer=0, trump_card=trump)
    features = encode_player_observation_2p(obs, version="v4").features
    behavior = features[FEATURE_DIM_2P_V3 : FEATURE_DIM_2P_V3 + 11]

    assert behavior == pytest.approx(
        [
            1 / 10,  # bastoni giocati da P1 (lead presa 2)
            1 / 10,  # coppe (scarto presa 3)
            1 / 10,  # denari (taglio presa 1)
            0 / 10,  # spade
            1 / 10,  # lead di P1
            1 / 10,  # tagli
            0 / 10,  # risposte a seme
            1 / 10,  # scarti
            0 / 10,  # lead di briscola
            0 / 10,  # lead carichi
            3 / 20,  # prese completate
        ]
    )

    # Blocco "ultime prese": lo slot 0 è la presa più recente (la 3).
    slot0 = features[FEATURE_DIM_2P_V3 + 11 : FEATURE_DIM_2P_V3 + 11 + 12]
    assert slot0 == pytest.approx(
        [
            1.0,  # slot presente
            1.0,  # ero io di mano
            0.0,
            0.0,
            0.0,
            1.0,  # lead spade
            5 / 10,  # forza del Sette
            0 / 11,  # punti del Sette
            0.0,  # la risposta NON ha seguito il seme
            0.0,  # la risposta non era briscola
            1.0,  # ho vinto io
            0 / 21,  # punti della presa
        ]
    )


def test_v4_degrades_to_zeros_without_history() -> None:
    """Payload/dataset senza `trick_history`: blocco v4 a zero (backward compatibility)."""
    obs = _observation_with_history(())
    features = encode_player_observation_2p(obs, version="v4").features
    assert features[FEATURE_DIM_2P_V3:] == [0.0] * _V4_EXTRA_DIM

    # Path dict con campo assente (dataset pre-v4).
    state = new_game_state(2, seed=1)
    dto = build_observation_dto(state, 0, 0).model_dump()
    dto.pop("trick_history", None)
    enc = encode_observation_2p_v4(dto)
    assert enc.features[FEATURE_DIM_2P_V3:] == [0.0] * _V4_EXTRA_DIM


def test_v4_history_is_public_and_symmetric() -> None:
    """
    Anti-cheat: la storia è informazione pubblica, quindi i due osservatori vedono le
    STESSE prese (cambia solo la prospettiva my/opp nelle feature derivate).
    """
    state = new_game_state(2, seed=5)
    for _ in range(8):
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None

    obs0 = make_player_observation(state, 0)
    obs1 = make_player_observation(state, 1)
    assert obs0.trick_history == obs1.trick_history
    assert len(obs0.trick_history) == 4

    f0 = encode_player_observation_2p(obs0, version="v4").features
    f1 = encode_player_observation_2p(obs1, version="v4").features
    base = FEATURE_DIM_2P_V3
    # "prese completate" coincide; "lead avversario" è complementare (ogni presa ha un solo lead).
    assert f0[base + 10] == f1[base + 10]
    leads_opp_seen_by_0 = f0[base + 4] * 10
    leads_opp_seen_by_1 = f1[base + 4] * 10
    assert leads_opp_seen_by_0 + leads_opp_seen_by_1 == len(obs0.trick_history)
