"""Test del kernel Numba V-lookahead per stati numerici determinizzati."""

from __future__ import annotations

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V3
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.ai.numba.core import numba_agent_code
from briscola_ai.ai.numba.value_lookahead import (
    OPPONENT_MODE_MODEL,
    OPPONENT_MODE_RULE,
    OPPONENT_MODE_VALUE_LOOKAHEAD,
    arrays_from_game_state_for_value_lookahead,
    choose_value_lookahead_card_numba,
    choose_value_lookahead_card_numba_arrays,
    collect_a2c_batch_numba_value_lookahead_2p,
    collect_a2c_trajectory_numba_value_lookahead_2p,
    warm_up_numba_value_lookahead,
)
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.rules import trick_points
from briscola_ai.domain.state import GameState, PlayerState

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba

# Default condivisi a livello di modulo: `Card` e' frozen, quindi riusarli e' sicuro
# (B008 vieta le chiamate nei default perche' non distingue i costruttori immutabili).
_DEFAULT_TRUMP_CLUBS = Card(Suit.CLUBS, Rank.SEVEN)
_DEFAULT_TRUMP_COINS = Card(Suit.COINS, Rank.SEVEN)


def _state(
    *,
    hand0: tuple[Card, ...],
    hand1: tuple[Card, ...],
    deck: tuple[Card, ...],
    table_cards: tuple[tuple[Card, int], ...],
    current_turn: int,
    trump_card: Card = _DEFAULT_TRUMP_CLUBS,
    points0: int = 0,
    points1: int = 0,
) -> GameState:
    """Costruisce uno stato 2-player minimale per testare il kernel."""
    first_player = table_cards[0][1] if table_cards else current_turn
    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState("P0", hand0, tuple(), points0),
            PlayerState("P1", hand1, tuple(), points1),
        ),
        deck=deck,
        trump_card=trump_card,
        table_cards=table_cards,
        current_turn=current_turn,
        first_player=first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def _endgame_state(
    hand0: tuple[Card, ...],
    hand1: tuple[Card, ...],
    *,
    current_turn: int,
    table_cards: tuple[tuple[Card, int], ...] = (),
    trump_card: Card = _DEFAULT_TRUMP_COINS,
) -> GameState:
    """Costruisce uno stato endgame con punti coerenti."""
    return _state(
        hand0=hand0,
        hand1=hand1,
        deck=tuple(),
        table_cards=table_cards,
        current_turn=current_turn,
        trump_card=trump_card,
    )


def _weights() -> tuple[np.ndarray, ...]:
    """Pesi sintetici: policy/value nulli, dimensioni uguali ai modelli v3 reali."""
    feature_dim = int(FEATURE_DIM_2P_V3)
    policy_w1 = np.zeros((feature_dim, 1), dtype=np.float32)
    policy_b1 = np.zeros(1, dtype=np.float32)
    policy_w2 = np.zeros((1, 40), dtype=np.float32)
    policy_b2 = np.zeros(40, dtype=np.float32)
    value_w1 = np.zeros((feature_dim, 1), dtype=np.float32)
    value_b1 = np.zeros(1, dtype=np.float32)
    value_w2 = np.zeros(1, dtype=np.float32)
    return policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2


def _choose_from_state(state: GameState) -> int:
    """Wrapper test: chiama il kernel con value model residuale nullo."""
    policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2 = _weights()
    card_index, _score = choose_value_lookahead_card_numba(
        policy_w1=policy_w1,
        policy_b1=policy_b1,
        policy_w2=policy_w2,
        policy_b2=policy_b2,
        value_w1=value_w1,
        value_b1=value_b1,
        value_w2=value_w2,
        value_b2=0.0,
        value_target_scale=120.0,
        value_target_is_residual=True,
        state=state,
        overkill_guard_enabled=True,
    )
    return int(card_index)


def test_warm_up_numba_value_lookahead_compiles_kernel() -> None:
    """Il warm-up deve compilare il kernel prima di loop lunghi di training."""
    warm_up_numba_value_lookahead()


def test_value_lookahead_arrays_from_game_state_matches_public_observation_bits() -> None:
    """La conversione conserva le maschere pubbliche usate dall'encoder v3."""
    state = _state(
        hand0=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CLUBS, Rank.TWO), Card(Suit.SWORDS, Rank.TWO)),
        hand1=(Card(Suit.COINS, Rank.TWO), Card(Suit.COINS, Rank.FOUR)),
        deck=(Card(Suit.SWORDS, Rank.THREE), Card(Suit.COINS, Rank.SEVEN)),
        table_cards=((Card(Suit.CUPS, Rank.ACE), 1),),
        current_turn=0,
        trump_card=Card(Suit.CLUBS, Rank.SEVEN),
        points1=trick_points((Card(Suit.SWORDS, Rank.KING),)),
    )

    *_, seen_cards, out_of_play_cards, _trick_hist, _num_tricks = arrays_from_game_state_for_value_lookahead(state)
    observation = make_player_observation(state, 0)

    assert seen_cards.tolist() == list(observation.seen_cards_onehot)
    assert out_of_play_cards.tolist() == list(observation.out_of_play_cards_onehot)


def test_value_lookahead_uses_numba_endgame_solver_when_deck_is_empty() -> None:
    """A mazzo vuoto il kernel deve scegliere come il solver canonico."""
    state = _endgame_state(
        hand0=(Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)),
        hand1=(Card(Suit.COINS, Rank.KING), Card(Suit.CLUBS, Rank.TWO)),
        current_turn=0,
    )

    assert _choose_from_state(state) == solve_endgame(state).best_card_index


def test_value_lookahead_depth1_prefers_immediate_point_gain() -> None:
    """Su presa gia' aperta, il value residuale nullo deve preferire la carta che vince punti."""
    trump_winning_card = Card(Suit.CLUBS, Rank.TWO)
    state = _state(
        hand0=(Card(Suit.CUPS, Rank.TWO), trump_winning_card, Card(Suit.SWORDS, Rank.TWO)),
        hand1=(Card(Suit.COINS, Rank.TWO), Card(Suit.COINS, Rank.FOUR)),
        deck=(Card(Suit.SWORDS, Rank.THREE), Card(Suit.COINS, Rank.SEVEN)),
        table_cards=((Card(Suit.CUPS, Rank.ACE), 1),),
        current_turn=0,
        trump_card=Card(Suit.CLUBS, Rank.SEVEN),
    )

    assert state.players[0].hand[_choose_from_state(state)] == trump_winning_card


def test_value_lookahead_array_entrypoint_returns_valid_card_index() -> None:
    """L'entrypoint da array e' quello destinato ai futuri loop training full-JIT."""
    state = _state(
        hand0=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CLUBS, Rank.TWO), Card(Suit.SWORDS, Rank.TWO)),
        hand1=(Card(Suit.COINS, Rank.TWO), Card(Suit.COINS, Rank.FOUR)),
        deck=(Card(Suit.SWORDS, Rank.THREE), Card(Suit.COINS, Rank.SEVEN)),
        table_cards=((Card(Suit.CUPS, Rank.ACE), 1),),
        current_turn=0,
        trump_card=Card(Suit.CLUBS, Rank.SEVEN),
    )
    policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2 = _weights()

    card_index, score = choose_value_lookahead_card_numba_arrays(
        policy_w1,
        policy_b1,
        policy_w2,
        policy_b2,
        value_w1,
        value_b1,
        value_w2,
        0.0,
        120.0,
        True,
        *arrays_from_game_state_for_value_lookahead(state),
        True,
    )

    assert 0 <= card_index < len(state.players[state.current_turn].hand)
    assert np.isfinite(score)
    assert state.players[state.current_turn].hand[int(card_index)] == Card(Suit.CLUBS, Rank.TWO)
    assert card_to_id(state.players[state.current_turn].hand[int(card_index)]) == 1


def test_value_lookahead_a2c_trajectory_collector_shapes_are_valid() -> None:
    """Il collector A2C value-aware deve produrre gli stessi buffer del rollout Numba standard."""
    policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2 = _weights()
    wv = np.zeros((policy_w1.shape[1],), dtype=np.float32)

    traj = collect_a2c_trajectory_numba_value_lookahead_2p(
        w1=policy_w1,
        b1=policy_b1,
        w2=policy_w2,
        b2=policy_b2,
        wv=wv,
        bv=0.0,
        opponent_mode=OPPONENT_MODE_VALUE_LOOKAHEAD,
        opponent_code=0,
        opponent_w1=policy_w1,
        opponent_b1=policy_b1,
        opponent_w2=policy_w2,
        opponent_b2=policy_b2,
        opponent_overkill_guard=True,
        value_w1=value_w1,
        value_b1=value_b1,
        value_w2=value_w2,
        value_b2=0.0,
        value_target_scale=120.0,
        value_target_is_residual=True,
        value_max_unknown_cards=8,
        game_seed=20260701,
        policy_seat=0,
    )

    step_count = int(traj.rewards.shape[0])
    assert 0 < step_count <= 20
    assert traj.xs.shape == (step_count, int(FEATURE_DIM_2P_V3))
    assert traj.probs.shape == (step_count, 40)
    assert np.all(traj.action_ids >= 0)


def test_value_lookahead_a2c_batch_supports_rule_model_and_value_modes() -> None:
    """Il batch può mescolare baseline, MLP e value-lookahead usando un solo set di pesi opponent."""
    policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2 = _weights()
    wv = np.zeros((policy_w1.shape[1],), dtype=np.float32)
    modes = np.asarray([OPPONENT_MODE_RULE, OPPONENT_MODE_MODEL, OPPONENT_MODE_VALUE_LOOKAHEAD], dtype=np.int64)
    codes = np.asarray([numba_agent_code("random"), 0, 0], dtype=np.int64)
    seeds = np.asarray([101, 102, 103], dtype=np.int64)
    seats = np.asarray([0, 1, 0], dtype=np.int64)

    batch = collect_a2c_batch_numba_value_lookahead_2p(
        w1=policy_w1,
        b1=policy_b1,
        w2=policy_w2,
        b2=policy_b2,
        wv=wv,
        bv=0.0,
        opponent_modes=modes,
        opponent_codes=codes,
        opponent_w1=policy_w1,
        opponent_b1=policy_b1,
        opponent_w2=policy_w2,
        opponent_b2=policy_b2,
        opponent_overkill_guard=True,
        value_w1=value_w1,
        value_b1=value_b1,
        value_w2=value_w2,
        value_b2=0.0,
        value_target_scale=120.0,
        value_target_is_residual=True,
        value_max_unknown_cards=8,
        game_seeds=seeds,
        policy_seats=seats,
    )

    assert batch.step_counts.shape == (3,)
    assert np.all(batch.step_counts > 0)
    assert batch.xs.shape[0] == 3
    for index in range(3):
        single = collect_a2c_trajectory_numba_value_lookahead_2p(
            w1=policy_w1,
            b1=policy_b1,
            w2=policy_w2,
            b2=policy_b2,
            wv=wv,
            bv=0.0,
            opponent_mode=int(modes[index]),
            opponent_code=int(codes[index]),
            opponent_w1=policy_w1,
            opponent_b1=policy_b1,
            opponent_w2=policy_w2,
            opponent_b2=policy_b2,
            opponent_overkill_guard=True,
            value_w1=value_w1,
            value_b1=value_b1,
            value_w2=value_w2,
            value_b2=0.0,
            value_target_scale=120.0,
            value_target_is_residual=True,
            value_max_unknown_cards=8,
            game_seed=int(seeds[index]),
            policy_seat=int(seats[index]),
        )
        count = int(batch.step_counts[index])
        assert count == int(single.rewards.shape[0])
        np.testing.assert_array_equal(batch.action_ids[index, :count], single.action_ids)
        np.testing.assert_allclose(batch.rewards[index, :count], single.rewards)
