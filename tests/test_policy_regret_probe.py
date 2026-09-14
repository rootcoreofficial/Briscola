"""Test della sonda automatica degli errori residui della policy."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from briscola_ai.ai.agents import HeuristicAgentV1, HeuristicAgentV2
from briscola_ai.ai.agents.hybrid_endgame import reconstruct_endgame_state
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V1
from briscola_ai.ai.endgame.fast_solver import solve_endgame_fast
from briscola_ai.ai.evaluation.policy_regret import (
    PolicyRegretConfig,
    _crossfit_estimate,
    classify_policy_regret,
    estimate_policy_regret,
)
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import GameState, new_game_state

_ROOT = Path(__file__).resolve().parents[1]


def _load_probe_script() -> Any:
    """Carica la CLI come modulo per uno smoke senza subprocess."""
    path = _ROOT / "scripts" / "probe_policy_regret.py"
    spec = importlib.util.spec_from_file_location("probe_policy_regret", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _initial_observation(*, seed: int = 11) -> PlayerObservation:
    state = new_game_state(num_players=2, seed=seed)
    return make_player_observation(state, state.current_turn)


def _play_until_deck_empty(seed: int) -> GameState:
    """Produce un endgame reale, quindi ricostruibile dalla sola osservazione."""
    state = new_game_state(num_players=2, seed=seed)
    agents = (HeuristicAgentV2(), HeuristicAgentV1())
    rng = random.Random(seed ^ 0xBAD5EED)
    while state.deck and not state.game_over:
        current = state.current_turn
        observation = make_player_observation(state, current)
        card_index = agents[current].choose_card_index(observation, rng=rng)
        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None
    assert not state.game_over
    assert state.deck == ()
    return state


def _public_observation(
    *,
    hand: tuple[Card, ...],
    trump_card: Card,
    table_cards: tuple[tuple[Card, int], ...],
) -> PlayerObservation:
    """Osservazione minimale per testare le etichette, senza stato completo."""
    player_index = 1 if table_cards else 0
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=player_index,
        player_name=f"P{player_index}",
        hand=hand,
        trump_card=trump_card,
        deck_size=4,
        table_cards=table_cards,
        current_turn=player_index,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(2, 2),
    )


def _write_linear_model(path: Path) -> None:
    """Policy minimale deterministica per lo smoke end-to-end."""
    feature_dim = int(FEATURE_DIM_2P_V1)
    np.savez(
        path,
        w=np.zeros((feature_dim, 40), dtype=np.float32),
        b=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "linear_softmax_bc_v1",
                "feature_dim": feature_dim,
                "encoder_version": "v1",
            }
        ),
    )


def test_crossfit_uses_disjoint_rows_to_confirm_candidate() -> None:
    """La carta scelta sul primo split deve essere giudicata soltanto sul secondo."""
    observation = _initial_observation()
    rows = (
        (0.0, 10.0, 1.0),
        (0.0, 8.0, 1.0),
        (0.0, 4.0, 2.0),
        (0.0, 6.0, 2.0),
    )
    estimate = _crossfit_estimate(
        observation,
        chosen_card_index=0,
        score_rows=rows,
        failed_determinizations=0,
        config=PolicyRegretConfig(determinizations=4, min_regret_points=1.0),
    )

    assert estimate.alternative_card_index == 1
    assert estimate.evaluation_best_card_index == 1
    assert estimate.selection_sample_count == 2
    assert estimate.evaluation_sample_count == 2
    assert estimate.regret_mean == pytest.approx(5.0)
    assert estimate.regret_confidence_low > 0.0
    assert estimate.reliable_error is True


def test_crossfit_rejects_candidate_that_fails_on_evaluation_rows() -> None:
    """Un vincitore fortunato nel selection split non deve diventare un errore dichiarato."""
    observation = _initial_observation(seed=12)
    rows = (
        (0.0, 10.0, 1.0),
        (0.0, 8.0, 1.0),
        (2.0, -4.0, 7.0),
        (2.0, -2.0, 7.0),
    )
    estimate = _crossfit_estimate(
        observation,
        chosen_card_index=0,
        score_rows=rows,
        failed_determinizations=0,
        config=PolicyRegretConfig(determinizations=4),
    )

    assert estimate.alternative_card_index == 1
    assert estimate.evaluation_best_card_index == 2
    assert estimate.candidate_confirmed_as_evaluation_best is False
    assert estimate.regret_mean < 0.0
    assert estimate.reliable_error is False


def test_sampled_regret_is_reproducible_from_observation_only() -> None:
    """Stessa osservazione e seed devono produrre lo stesso audit Monte Carlo."""
    observation = _initial_observation(seed=19)
    config = PolicyRegretConfig(determinizations=4, use_endgame_solver=True)
    kwargs = {
        "observation": observation,
        "chosen_card_index": 0,
        "rollout_agent": HeuristicAgentV2(),
        "config": config,
    }

    first = estimate_policy_regret(**kwargs, rng=random.Random(123))
    second = estimate_policy_regret(**kwargs, rng=random.Random(123))

    assert first == second
    assert first.method == "sampled_crossfit"
    assert first.successful_determinizations == 4
    assert first.failed_determinizations == 0


def test_endgame_regret_finds_an_exact_suboptimal_choice() -> None:
    """Nel finale la sonda deve coincidere col minimax dedotto dall'osservazione."""
    found = None
    for seed in range(20, 60):
        state = _play_until_deck_empty(seed)
        observation = make_player_observation(state, state.current_turn)
        best = solve_endgame_fast(reconstruct_endgame_state(observation)).best_card_index
        for chosen in range(len(observation.hand)):
            if chosen == best:
                continue
            estimate = estimate_policy_regret(
                observation,
                chosen_card_index=chosen,
                rollout_agent=HeuristicAgentV2(),
                rng=random.Random(0),
                config=PolicyRegretConfig(determinizations=4),
            )
            if estimate.regret_mean > 0.0:
                found = (best, estimate)
                break
        if found is not None:
            break

    assert found is not None
    best, estimate = found
    assert estimate.method == "exact_endgame"
    assert estimate.alternative_card_index == best
    assert estimate.regret_confidence_low == estimate.regret_mean
    assert estimate.regret_standard_error == 0.0
    assert estimate.reliable_error is True


def test_classification_marks_public_trump_overkill() -> None:
    """L'etichetta overkill deve dipendere solo da mano, tavolo e seme di briscola."""
    observation = _public_observation(
        hand=(Card(Suit.COINS, Rank.ACE), Card(Suit.COINS, Rank.FOUR)),
        trump_card=Card(Suit.COINS, Rank.SEVEN),
        table_cards=((Card(Suit.CUPS, Rank.TWO), 0),),
    )

    tags = classify_policy_regret(observation, chosen_card_index=0, alternative_card_index=1)

    assert "trump_overkill" in tags


def test_probe_builds_balanced_reproducible_report_without_hidden_state(tmp_path: Path) -> None:
    """Smoke completo: otto bucket, JSON deterministico e nessuna serializzazione del deck reale."""
    probe = _load_probe_script()
    model_path = tmp_path / "policy.npz"
    _write_linear_model(model_path)
    config = probe.ProbeConfig(
        model_path=model_path,
        belief_model_path=None,
        out_path=tmp_path / "report.json",
        num_observations=8,
        max_games=40,
        seed=77,
        opponents=("heuristic_v1",),
        determinizations=4,
        top_cases=8,
    )

    first = probe.run_policy_regret_probe(config)
    second = probe.run_policy_regret_probe(config)

    assert first == second
    assert first["schema"] == "briscola.policy_regret_probe.v1"
    assert first["anti_cheat"]["actual_hidden_state_used_by_estimator"] is False
    assert first["decision"]["verdict"] in {
        "actionable_policy_error_cluster",
        "unclassified_policy_error_cluster",
        "sparse_policy_error_signal",
        "no_policy_error_signal",
    }
    assert first["collection"]["records_collected"] == 8
    assert set(first["collection"]["bucket_counts"].values()) == {1}
    assert len(first["decisions"]) == 8
    assert all("game_seed" not in record["context"] for record in first["decisions"])
    assert all("game_seed" in record["provenance"] for record in first["decisions"])
    assert all("deck_size" in record["context"] for record in first["decisions"])
    assert {record["context"]["phase"] for record in first["decisions"]} == {
        "early",
        "mid",
        "pimc_window",
        "endgame",
    }


def test_cli_help_is_renderable() -> None:
    """Le percentuali nell'help argparse devono essere escape-ate, altrimenti la CLI non parte."""
    probe = _load_probe_script()

    help_text = probe._build_parser().format_help()

    assert "99%" in help_text
    assert "--confidence-z" in help_text


@pytest.mark.parametrize("determinizations", [1, 3, 5])
def test_config_rejects_non_pairable_determinizations(determinizations: int) -> None:
    """Selection ed evaluation richiedono un numero pari di campioni e almeno due per split."""
    with pytest.raises(ValueError, match="pari"):
        PolicyRegretConfig(determinizations=determinizations).validate()
