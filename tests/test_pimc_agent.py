"""
Test del prototipo PIMC.

Questi test non cercano di dimostrare che PIMC sia forte: bloccano gli invarianti importanti
per un agente a informazione imperfetta, cioè determinizzazione senza leak, fallback fuori scope
e scelta valida quando la search è attiva.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.agents import HeuristicAgentV1, HeuristicAgentV2, build_agent, list_agent_specs
from briscola_ai.ai.agents.hybrid_endgame import reconstruct_endgame_state
from briscola_ai.ai.agents.pimc import (
    PIMCAgent,
    PIMCSearchStats,
    determinize_observation,
    rollout_to_terminal,
    unknown_live_card_count,
)
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V1
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state


class _InvalidIndexAgent:
    """Agente volutamente rotto: serve a testare il fallback difensivo del rollout."""

    name = "invalid_index"

    def choose_card_index(self, observation, *, rng: random.Random) -> int:
        return 999


def _write_linear_bc_model(path: Path, *, bias_action: int = 0) -> None:
    """Salva un modello lineare minimale per testare varianti PIMC con modello selezionato."""
    d = int(FEATURE_DIM_2P_V1)
    w = np.zeros((d, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)
    b[bias_action] = 1.0
    np.savez(path, w=w, b=b, metadata_json=f'{{"format":"linear_softmax_bc_v1","feature_dim":{d}}}')


def test_bc_model_pimc_16x8_variant_uses_selected_model(tmp_path: Path) -> None:
    """La variante UI PIMC usa il `.npz` selezionato e blocca la config Pareto 16×8."""
    model_path = tmp_path / "selected_model.npz"
    _write_linear_bc_model(model_path)

    assert "bc_model_pimc_16x8" in {spec.name for spec in list_agent_specs()}

    agent = build_agent("bc_model_pimc_16x8", model_path=model_path)

    assert isinstance(agent, PIMCAgent)
    assert agent.name == "bc_model_pimc_16x8"
    assert agent.num_determinizations == 16
    assert agent.max_unknown_cards == 8
    assert agent.use_endgame_solver is True
    assert isinstance(agent.fallback, BCModelAgent)
    assert isinstance(agent.rollout_agent, BCModelAgent)
    assert agent.fallback.model_path == model_path
    assert agent.rollout_agent.model_path == model_path


def test_bc_model_pimc_belief_64x10_variant_builds_with_belief(tmp_path: Path, monkeypatch) -> None:
    """
    La variante belief usa il `.npz` selezionato, blocca la config 64x10 e aggancia la
    belief network dalla directory modelli (regola di provisioning identica al value model).
    Senza belief nella directory: errore esplicito (opzione non disponibile).
    """
    import json

    from briscola_ai.ai.models.provisioning import PIMC_BELIEF_MODEL_ID

    model_path = tmp_path / "selected_model.npz"
    _write_linear_bc_model(model_path)
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))

    assert "bc_model_pimc_belief_64x10" in {spec.name for spec in list_agent_specs()}

    # Belief assente -> errore chiaro.
    with pytest.raises(ValueError, match="belief"):
        build_agent("bc_model_pimc_belief_64x10", model_path=model_path)
    with pytest.raises(ValueError, match="belief"):
        build_agent("bc_model_pimc_belief_16x10", model_path=model_path)
    with pytest.raises(ValueError, match="belief"):
        build_agent("bc_model_pimc_belief_12x8", model_path=model_path)

    # Belief minimale valida (formato belief_mlp_v1, encoder v4 = 369 input).
    from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4

    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / PIMC_BELIEF_MODEL_ID,
        w1=rng.normal(0, 0.05, size=(int(FEATURE_DIM_2P_V4), 4)).astype(np.float32),
        b1=np.zeros(4, dtype=np.float32),
        w2=rng.normal(0, 0.05, size=(4, 40)).astype(np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps({"format": "belief_mlp_v1", "encoder_version": "v4"}),
    )

    agent = build_agent("bc_model_pimc_belief_64x10", model_path=model_path)
    assert isinstance(agent, PIMCAgent)
    assert agent.name == "bc_model_pimc_belief_64x10"
    assert agent.num_determinizations == 64
    assert agent.max_unknown_cards == 10
    assert agent.belief_model is not None
    assert agent.use_endgame_solver is True

    eval_agent = build_agent("bc_model_pimc_belief_16x10", model_path=model_path)
    assert "bc_model_pimc_belief_16x10" not in {spec.name for spec in list_agent_specs()}
    assert isinstance(eval_agent, PIMCAgent)
    assert eval_agent.name == "bc_model_pimc_belief_16x10"
    assert eval_agent.num_determinizations == 16
    assert eval_agent.max_unknown_cards == 10
    assert eval_agent.belief_model is not None
    assert eval_agent.use_endgame_solver is True

    recommended_agent = build_agent("bc_model_pimc_belief_12x8", model_path=model_path)
    public_specs = {spec.name for spec in list_agent_specs()}
    assert "bc_model_pimc_belief_12x8" in public_specs
    assert isinstance(recommended_agent, PIMCAgent)
    assert recommended_agent.name == "bc_model_pimc_belief_12x8"
    assert recommended_agent.num_determinizations == 12
    assert recommended_agent.max_unknown_cards == 8
    assert recommended_agent.belief_uniform_mix == pytest.approx(0.10)
    assert recommended_agent.belief_model is not None
    assert recommended_agent.use_endgame_solver is True
    assert recommended_agent.use_numba_search is False


def _play_with_heuristics_until(*, seed: int, max_deck_size: int) -> GameState:
    """Avanza una partita deterministica finché il mazzo arriva sotto soglia."""
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


def _state_card_ids(state: GameState) -> list[int]:
    """Tutte le carte presenti nello stato completo, incluse prese e tavolo."""
    ids: list[int] = []
    for player in state.players:
        ids.extend(card_to_id(card) for card in player.hand)
        ids.extend(card_to_id(card) for card in player.captured_cards)
    ids.extend(card_to_id(card) for card in state.deck)
    ids.extend(card_to_id(card) for card, _player in state.table_cards)
    return ids


def test_determinize_observation_preserves_public_invariants() -> None:
    """Una determinizzazione deve rispettare mano nota, punteggi pubblici, deck size e unicità carte."""
    state = _play_with_heuristics_until(seed=11, max_deck_size=6)
    observation = make_player_observation(state, state.current_turn)

    determinized = determinize_observation(observation, rng=random.Random(123))

    assert determinized.current_turn == observation.current_turn
    assert determinized.players[observation.player_index].hand == observation.hand
    assert (
        len(determinized.players[1 - observation.player_index].hand)
        == observation.players_hand_sizes[1 - observation.player_index]
    )
    assert len(determinized.deck) == observation.deck_size
    assert determinized.players[0].points == observation.players_points[0]
    assert determinized.players[1].points == observation.players_points[1]

    all_ids = _state_card_ids(determinized)
    assert len(all_ids) == 40
    assert len(set(all_ids)) == 40


def test_pimc_delegates_to_fallback_when_too_many_unknown_cards() -> None:
    """Fuori soglia PIMC deve comportarsi come il fallback."""
    state = new_game_state(num_players=2, seed=7)
    observation = make_player_observation(state, state.current_turn)
    fallback = HeuristicAgentV2()
    agent = PIMCAgent(
        rollout_agent=HeuristicAgentV2(),
        fallback=fallback,
        num_determinizations=2,
        max_unknown_cards=0,
    )

    assert unknown_live_card_count(observation) > 0
    assert agent.choose_card_index(observation, rng=random.Random(5)) == fallback.choose_card_index(
        observation,
        rng=random.Random(5),
    )
    assert agent.metrics.total_decisions == 1
    assert agent.metrics.search_decisions == 0
    assert agent.metrics.fallback_decisions == 1


def test_pimc_returns_valid_card_when_search_is_active() -> None:
    """Nel semi-finale il prototipo deve riuscire a campionare e scegliere una carta valida."""
    state = _play_with_heuristics_until(seed=19, max_deck_size=4)
    observation = make_player_observation(state, state.current_turn)
    agent = PIMCAgent(
        rollout_agent=HeuristicAgentV2(),
        fallback=HeuristicAgentV2(),
        num_determinizations=3,
        max_unknown_cards=10,
    )

    choice = agent.choose_card_index(observation, rng=random.Random(9))

    assert 0 <= choice < len(observation.hand)
    assert agent.metrics.search_decisions == 1
    assert agent.metrics.successful_determinizations > 0
    assert agent.metrics.completed_rollouts > 0
    assert agent.metrics.seconds_per_search_decision > 0.0
    assert agent.last_search_diagnostics is not None
    diagnostics = agent.last_search_diagnostics
    assert diagnostics.best_card_index == choice
    assert len(diagnostics.action_values) == len(observation.hand)
    assert diagnostics.successful_determinizations == agent.num_determinizations
    assert diagnostics.completed_rollouts == agent.metrics.completed_rollouts
    assert diagnostics.margin is not None
    assert diagnostics.paired_margin_sample_count == len(diagnostics.paired_margin_samples)
    if diagnostics.margin_standard_error is not None:
        assert diagnostics.margin_ci95_low is not None
        assert diagnostics.margin_ci95_high is not None


def test_pimc_uses_exact_solver_when_deck_is_empty() -> None:
    """A mazzo vuoto PIMC deve coincidere col solver endgame ricostruito dall'osservazione."""
    state = _play_with_heuristics_until_deck_empty(seed=23)
    observation = make_player_observation(state, state.current_turn)
    expected = solve_endgame(reconstruct_endgame_state(observation)).best_card_index
    agent = PIMCAgent(
        rollout_agent=HeuristicAgentV2(),
        fallback=HeuristicAgentV1(),
        num_determinizations=1,
        max_unknown_cards=10,
        use_endgame_solver=True,
    )

    assert agent.choose_card_index(observation, rng=random.Random(4)) == expected
    assert agent.metrics.endgame_solver_decisions == 1
    assert agent.last_search_diagnostics is None


def test_rollout_to_terminal_handles_invalid_rollout_index_defensively() -> None:
    """Un rollout agent difettoso non deve abortire: l'indice viene normalizzato a una mossa valida."""
    state = _play_with_heuristics_until(seed=31, max_deck_size=4)
    metrics = PIMCSearchStats()

    final_state = rollout_to_terminal(
        state,
        rollout_agent=_InvalidIndexAgent(),
        rng=random.Random(7),
        use_endgame_solver=False,
        metrics=metrics,
    )

    assert final_state.game_over is True
    assert final_state.players[0].points + final_state.players[1].points == 120
    assert metrics.coerced_moves > 0


def test_endgame_reconstruction_matches_real_solver_on_seed_suite() -> None:
    """
    Stress anti-cheat: su molti endgame reali il solver da observation deve scegliere come il solver reale.

    Controlliamo sia lo stato a tavolo vuoto sia il caso "secondo di mano" ottenuto giocando la prima
    carta della presa successiva. Il delta assoluto differisce perché la ricostruzione azzera le prese,
    ma la mossa ottima e il delta futuro devono coincidere a meno del delta punti già acquisito.
    """
    for seed in range(100, 140):
        state = _play_with_heuristics_until_deck_empty(seed=seed)

        for cursor in (state, step(state, PlayCardAction(player_index=state.current_turn, card_index=0))[0]):
            observation = make_player_observation(cursor, cursor.current_turn)
            reconstructed = reconstruct_endgame_state(observation)

            real_solution = solve_endgame(cursor)
            reconstructed_solution = solve_endgame(reconstructed)
            base_delta = cursor.players[0].points - cursor.players[1].points

            assert reconstructed_solution.best_card_index == real_solution.best_card_index
            assert real_solution.final_delta_p0_p1 == base_delta + reconstructed_solution.final_delta_p0_p1


def test_determinize_observation_stress_preserves_public_invariants() -> None:
    """
    Stress della determinizzazione: ogni stato campionato deve rispettare informazione pubblica e unicità.

    Questo non può provare che la distribuzione campionata sia "la migliore" strategicamente, ma blocca i
    bug più pericolosi: leak di carte note, duplicati, deck/mani con size sbagliate e punteggi incoerenti.
    """
    for seed in range(200, 220):
        state = _play_with_heuristics_until(seed=seed, max_deck_size=6)
        observation = make_player_observation(state, state.current_turn)
        my_hand_ids = {card_to_id(card) for card in observation.hand}
        out_of_play_ids = {card_id for card_id, value in enumerate(observation.out_of_play_cards_onehot) if value}
        table_ids = {card_to_id(card) for card, _player_index in observation.table_cards}

        for sample in range(8):
            determinized = determinize_observation(observation, rng=random.Random(seed * 1000 + sample))
            opponent_index = 1 - observation.player_index
            hidden_ids = {
                *(card_to_id(card) for card in determinized.players[opponent_index].hand),
                *(card_to_id(card) for card in determinized.deck),
            }

            assert determinized.current_turn == observation.current_turn
            assert determinized.first_player == observation.first_player
            assert determinized.table_cards == observation.table_cards
            assert determinized.players[observation.player_index].hand == observation.hand
            assert len(determinized.players[opponent_index].hand) == observation.players_hand_sizes[opponent_index]
            assert len(determinized.deck) == observation.deck_size
            assert tuple(player.points for player in determinized.players) == tuple(observation.players_points)
            assert hidden_ids.isdisjoint(my_hand_ids)
            assert hidden_ids.isdisjoint(out_of_play_ids)
            assert table_ids.issubset(out_of_play_ids)

            all_ids = _state_card_ids(determinized)
            assert len(all_ids) == 40
            assert len(set(all_ids)) == 40


def test_pimc_search_stress_has_no_failed_or_coerced_moves() -> None:
    """
    Stress PIMC su stati semi-finali generati dal dominio.

    Usiamo agenti euristici deterministici come fallback/rollout per rendere il test stabile. La search deve
    attivarsi, scegliere sempre una carta valida e non accumulare determinizzazioni/rollout falliti né mosse
    normalizzate difensivamente.
    """
    metrics = PIMCSearchStats()
    agent = PIMCAgent(
        rollout_agent=HeuristicAgentV2(),
        fallback=HeuristicAgentV2(),
        num_determinizations=4,
        max_unknown_cards=10,
        metrics=metrics,
    )

    for seed in range(300, 312):
        state = _play_with_heuristics_until(seed=seed, max_deck_size=4)
        observation = make_player_observation(state, state.current_turn)

        choice = agent.choose_card_index(observation, rng=random.Random(seed ^ 0xA11CE))

        assert 0 <= choice < len(observation.hand)
        assert agent.last_search_diagnostics is not None
        diagnostics = agent.last_search_diagnostics
        assert diagnostics.best_card_index == choice
        assert diagnostics.successful_determinizations == agent.num_determinizations
        assert diagnostics.failed_determinizations == 0
        assert diagnostics.failed_rollouts == 0
        assert all(value.rollout_count == agent.num_determinizations for value in diagnostics.action_values)

    assert metrics.search_decisions == 12
    assert metrics.failed_determinizations == 0
    assert metrics.failed_rollouts == 0
    assert metrics.coerced_moves == 0
