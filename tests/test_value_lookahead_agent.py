"""Test per l'agente V-lookahead depth-1."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents import HeuristicAgentV1, HeuristicAgentV2, HybridEndgameAgent, ValueLookaheadAgent
from briscola_ai.ai.agents.hybrid_endgame import reconstruct_endgame_state
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V3
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.ai.models import MLPValueModel
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import GameState, new_game_state

_ROOT = Path(__file__).resolve().parents[1]


class FixedFallbackAgent:
    """Fallback controllabile per verificare quando V-lookahead delega."""

    name = "fixed_fallback"

    def __init__(self, card_index: int) -> None:
        self.card_index = card_index
        self.calls = 0

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        self.calls += 1
        return min(self.card_index, len(observation.hand) - 1)


class FixedLookaheadAgent(ValueLookaheadAgent):
    """ValueLookaheadAgent che forza una scelta search, utile per testare il post-processing."""

    forced_choice: int = 0

    def _choose_with_lookahead(self, observation: PlayerObservation, *, rng: random.Random) -> int | None:
        self.metrics.lookahead_decisions += 1
        return self.forced_choice


def _load_script_module(name: str) -> Any:
    """Carica uno script da `scripts/` come modulo testabile."""
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _zero_value_model() -> MLPValueModel:
    """Value model minimale v3 che predice residuo zero."""
    d = int(FEATURE_DIM_2P_V3)
    h = 4
    return MLPValueModel(
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h,), dtype=np.float32),
        b2=0.0,
        metadata={
            "format": "value_mlp_v1",
            "feature_dim": d,
            "hidden_dim": h,
            "encoder_version": "v3",
            "target": "residual",
            "target_scale": 120.0,
        },
    )


def _write_zero_value_model(path: Path) -> None:
    """Salva su disco il value model minimale usato dagli smoke CLI."""
    model = _zero_value_model()
    import json

    np.savez(
        path,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=np.asarray([model.b2], dtype=np.float32),
        metadata_json=json.dumps(model.metadata),
    )


def _write_linear_bc_model(path: Path) -> None:
    """Salva un modello BC lineare minimale per lo smoke dello script."""
    d = int(FEATURE_DIM_2P_V1)
    np.savez(
        path,
        w=np.zeros((d, 40), dtype=np.float32),
        b=np.zeros((40,), dtype=np.float32),
        metadata_json=f'{{"format":"linear_softmax_bc_v1","feature_dim":{d}}}',
    )


def _play_with_heuristics_until(*, seed: int, max_deck_size: int) -> GameState:
    """Avanza una partita deterministica fino alla soglia mazzo richiesta."""
    state = new_game_state(num_players=2, seed=seed)
    agents = (HeuristicAgentV2(), HeuristicAgentV1())
    rng = random.Random(seed ^ 0xC0FFEE)
    safety = 200
    while not state.game_over and len(state.deck) > max_deck_size and safety > 0:
        safety -= 1
        current = state.current_turn
        observation = make_player_observation(state, current)
        card_index = agents[current].choose_card_index(observation, rng=rng)
        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None
    assert safety > 0
    assert not state.game_over
    return state


def _play_with_heuristics_until_deck_empty(*, seed: int) -> GameState:
    """Avanza fino al primo stato non terminale con mazzo vuoto."""
    state = new_game_state(num_players=2, seed=seed)
    agents = (HeuristicAgentV2(), HeuristicAgentV1())
    rng = random.Random(seed ^ 0xBAD5EED)
    safety = 200
    while not state.game_over and len(state.deck) > 0 and safety > 0:
        safety -= 1
        current = state.current_turn
        observation = make_player_observation(state, current)
        card_index = agents[current].choose_card_index(observation, rng=rng)
        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None
    assert safety > 0
    assert not state.game_over
    assert len(state.deck) == 0
    return state


def test_value_lookahead_delegates_outside_validated_window() -> None:
    """Con troppe carte ignote l'agente deve usare la baseline `v6 + solver`, non V."""
    state = new_game_state(num_players=2, seed=1)
    observation = make_player_observation(state, state.current_turn)
    fallback = FixedFallbackAgent(card_index=1)
    agent = ValueLookaheadAgent(
        value_model=_zero_value_model(),
        fallback=fallback,
        continuation_agent=fallback,
        max_unknown_cards=0,
    )

    choice = agent.choose_card_index(observation, rng=random.Random(1))

    assert choice == 1
    assert fallback.calls == 1
    assert agent.metrics.fallback_decisions == 1
    assert agent.metrics.lookahead_decisions == 0


def test_value_lookahead_uses_exact_solver_at_empty_deck() -> None:
    """A mazzo vuoto V-lookahead deve coincidere col solver endgame ricostruito."""
    state = _play_with_heuristics_until_deck_empty(seed=12)
    observation = make_player_observation(state, state.current_turn)
    expected = solve_endgame(reconstruct_endgame_state(observation)).best_card_index
    fallback = FixedFallbackAgent(card_index=0)
    agent = ValueLookaheadAgent(
        value_model=_zero_value_model(),
        fallback=fallback,
        continuation_agent=fallback,
    )

    choice = agent.choose_card_index(observation, rng=random.Random(12))

    assert choice == expected
    assert fallback.calls == 0
    assert agent.metrics.endgame_solver_decisions == 1


def test_value_lookahead_evaluates_leaves_inside_window() -> None:
    """Nella finestra validata deve campionare determinizzazioni e produrre una carta valida."""
    state = _play_with_heuristics_until(seed=22, max_deck_size=4)
    observation = make_player_observation(state, state.current_turn)
    control = HybridEndgameAgent(fallback=HeuristicAgentV2(), name="control")
    agent = ValueLookaheadAgent(
        value_model=_zero_value_model(),
        fallback=control,
        continuation_agent=control,
        num_determinizations=2,
        max_unknown_cards=8,
    )

    choice = agent.choose_card_index(observation, rng=random.Random(22))

    assert 0 <= choice < len(observation.hand)
    assert agent.metrics.lookahead_decisions == 1
    assert agent.metrics.successful_determinizations > 0
    assert agent.metrics.completed_leaf_evaluations > 0
    assert agent.metrics.fallback_decisions == 0


def test_value_lookahead_overkill_guard_can_adjust_search_choice() -> None:
    """Il guard anti-overkill deve proteggere le scelte V-lookahead come protegge il BCModelAgent."""
    from briscola_ai.domain.models import Card, Rank, Suit
    from briscola_ai.domain.rules import trick_points
    from briscola_ai.domain.state import PlayerState

    trump = Card(Suit.COINS, Rank.SEVEN)
    hand0 = (Card(Suit.COINS, Rank.ACE), Card(Suit.COINS, Rank.TWO))
    hand1 = (Card(Suit.CUPS, Rank.KING),)
    table = ((Card(Suit.SWORDS, Rank.FIVE), 1),)
    deck = (trump,)
    visible = set(hand0 + hand1 + tuple(card for card, _player in table) + deck)
    captured = tuple(Card(suit, rank) for suit in Suit for rank in Rank if Card(suit, rank) not in visible)
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState("P0", hand0, captured, trick_points(captured)),
            PlayerState("P1", hand1, tuple(), 0),
        ),
        deck=deck,
        trump_card=trump,
        table_cards=table,
        current_turn=0,
        first_player=1,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )
    observation = make_player_observation(state, player_index=0)
    fallback = FixedFallbackAgent(card_index=0)
    agent = FixedLookaheadAgent(
        value_model=_zero_value_model(),
        fallback=fallback,
        continuation_agent=fallback,
        max_unknown_cards=8,
    )
    agent.forced_choice = 0

    guarded = agent.choose_card_index(observation, rng=random.Random(1))

    assert guarded == 1
    assert agent.metrics.overkill_guard_adjustments == 1


def test_evaluate_value_lookahead_script_smoke(tmp_path: Path) -> None:
    """Lo script Stage 1 deve girare end-to-end con modelli temporanei minimali."""
    script = _load_script_module("evaluate_value_lookahead")
    policy_path = tmp_path / "policy.npz"
    value_path = tmp_path / "value.npz"
    out_path = tmp_path / "result.json"
    _write_linear_bc_model(policy_path)
    _write_zero_value_model(value_path)

    old_argv = sys.argv
    try:
        sys.argv = [
            "evaluate_value_lookahead.py",
            "--policy-model",
            str(policy_path),
            "--value-model",
            str(value_path),
            "--num-games",
            "4",
            "--seed",
            "123",
            "--determinizations",
            "1",
            "--max-unknown-cards",
            "8",
            "--out-json",
            str(out_path),
        ]
        assert script.main() == 0
    finally:
        sys.argv = old_argv

    assert out_path.exists()


def test_evaluate_value_lookahead_pair_script_smoke(tmp_path: Path) -> None:
    """Lo script pair deve confrontare due value model nello stesso harness."""
    script = _load_script_module("evaluate_value_lookahead_pair")
    policy_path = tmp_path / "policy.npz"
    value_a_path = tmp_path / "value_a.npz"
    value_b_path = tmp_path / "value_b.npz"
    out_path = tmp_path / "pair.json"
    _write_linear_bc_model(policy_path)
    _write_zero_value_model(value_a_path)
    _write_zero_value_model(value_b_path)

    old_argv = sys.argv
    try:
        sys.argv = [
            "evaluate_value_lookahead_pair.py",
            "--policy-model",
            str(policy_path),
            "--value-model-a",
            str(value_a_path),
            "--value-model-b",
            str(value_b_path),
            "--num-games",
            "4",
            "--seed",
            "456",
            "--determinizations",
            "1",
            "--max-unknown-cards",
            "8",
            "--out-json",
            str(out_path),
        ]
        assert script.main() == 0
    finally:
        sys.argv = old_argv

    assert out_path.exists()


def test_evaluate_value_lookahead_quality_script_smoke(tmp_path: Path) -> None:
    """Lo script quality deve confrontare candidato e baseline senza passare dal registry."""
    script = _load_script_module("evaluate_value_lookahead_quality")
    policy_path = tmp_path / "policy.npz"
    value_path = tmp_path / "value.npz"
    out_path = tmp_path / "quality.json"
    _write_linear_bc_model(policy_path)
    _write_zero_value_model(value_path)

    old_argv = sys.argv
    try:
        sys.argv = [
            "evaluate_value_lookahead_quality.py",
            "--policy-model",
            str(policy_path),
            "--value-model",
            str(value_path),
            "--num-games",
            "4",
            "--seed",
            "321",
            "--determinizations",
            "1",
            "--max-unknown-cards",
            "8",
            "--out-json",
            str(out_path),
        ]
        assert script.main() == 0
    finally:
        sys.argv = old_argv

    assert out_path.exists()
