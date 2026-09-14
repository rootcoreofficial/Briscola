"""
Test per metriche di qualità decisionale.

Obiettivo:
- verificare che `trump_waste` venga rilevato correttamente in un caso semplice.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from briscola_ai.ai.agents import build_agent
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V1
from briscola_ai.ai.evaluation.decision_quality import (
    _is_trump_overkill_second_hand,
    _is_trump_waste_second_hand,
    evaluate_bc_model_seat_fair_match_2p_with_quality_numba,
    evaluate_seat_fair_match_2p_with_quality,
    evaluate_seat_fair_match_2p_with_quality_parallel,
)
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.state import GameState, PlayerState


def test_trump_waste_detected_when_non_trump_wins() -> None:
    """
    Caso:
    - briscola = cups
    - avversario gioca clubs TWO (scarto)
    - io (player 1) ho in mano clubs ACE (vince senza briscola) e cups ACE (briscola)
    - se gioco cups ACE -> spreco briscola (waste=True)
    """
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="P0", hand=tuple(), captured_cards=tuple(), points=0),
            PlayerState(
                name="P1",
                hand=(Card(Suit.CLUBS, Rank.ACE), Card(Suit.CUPS, Rank.ACE)),
                captured_cards=tuple(),
                points=0,
            ),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.THREE),
        table_cards=((Card(Suit.CLUBS, Rank.TWO), 0),),
        current_turn=1,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )

    waste = _is_trump_waste_second_hand(state=state, player_index=1, chosen_card_index=1)
    assert waste is True

    no_waste = _is_trump_waste_second_hand(state=state, player_index=1, chosen_card_index=0)
    assert no_waste is False


def test_trump_overkill_detected_when_cheaper_trump_wins() -> None:
    """
    Caso:
    - briscola = cups
    - avversario gioca swords TWO (scarto)
    - io (player 1) ho in mano cups TWO e cups ACE: entrambe vincono (sono briscole)
    - se gioco cups ACE -> overkill (potevo vincere con cups TWO)
    """
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="P0", hand=tuple(), captured_cards=tuple(), points=0),
            PlayerState(
                name="P1",
                hand=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CUPS, Rank.ACE)),
                captured_cards=tuple(),
                points=0,
            ),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.THREE),
        table_cards=((Card(Suit.SWORDS, Rank.TWO), 0),),
        current_turn=1,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )

    overkill = _is_trump_overkill_second_hand(state=state, player_index=1, chosen_card_index=1)
    assert overkill is True

    no_overkill = _is_trump_overkill_second_hand(state=state, player_index=1, chosen_card_index=0)
    assert no_overkill is False


def test_parallel_decision_quality_matches_serial_for_deterministic_agents() -> None:
    """
    La parallelizzazione divide solo la suite di seed: con agenti deterministici
    deve produrre gli stessi aggregati della valutazione seriale.
    """
    agent_a = build_agent("heuristic_v1")
    agent_b = build_agent("heuristic_v1")
    seeds = list(range(10))

    serial = evaluate_seat_fair_match_2p_with_quality(
        agent_a,
        agent_b,
        num_games=20,
        seed=123,
        game_seeds=seeds,
    )
    parallel = evaluate_seat_fair_match_2p_with_quality_parallel(
        agent_a,
        agent_b,
        num_games=20,
        seed=123,
        game_seeds=seeds,
        workers=2,
    )

    assert asdict(parallel.match) == asdict(serial.match)
    assert asdict(parallel.quality) == asdict(serial.quality)


def test_parallel_decision_quality_matches_serial_for_stochastic_agents() -> None:
    """
    Anche gli agenti casuali devono ricevere gli stessi stream RNG per coppia:
    cambiare il numero di worker deve influire solo sul tempo di esecuzione.

    Tre worker dividono 15 coppie in chunk non coincidenti con il caso seriale;
    il confronto protegge quindi anche il calcolo dell'offset globale della coppia.
    """
    agent_a = build_agent("random")
    agent_b = build_agent("random")
    seeds = list(range(15))

    serial = evaluate_seat_fair_match_2p_with_quality_parallel(
        agent_a,
        agent_b,
        num_games=30,
        seed=456,
        game_seeds=seeds,
        workers=1,
    )
    parallel = evaluate_seat_fair_match_2p_with_quality_parallel(
        agent_a,
        agent_b,
        num_games=30,
        seed=456,
        game_seeds=seeds,
        workers=3,
    )

    assert asdict(parallel.match) == asdict(serial.match)
    assert asdict(parallel.quality) == asdict(serial.quality)


def test_numba_decision_quality_returns_consistent_stats(tmp_path: Path) -> None:
    """Il path Numba decision-quality deve produrre DTO coerenti per un modello MLP."""
    d = int(FEATURE_DIM_2P_V1)
    h = 4
    model_path = tmp_path / "dummy_quality_numba.npz"
    np.savez(
        model_path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": d}, ensure_ascii=False),
    )

    agent = BCModelAgent.from_npz(model_path)
    out = evaluate_bc_model_seat_fair_match_2p_with_quality_numba(
        agent,
        "heuristic_v1",
        num_games=20,
        seed=0,
        game_seeds=list(range(10)),
    )

    assert out.match.num_games == 20
    assert out.match.wins_agent_a + out.match.wins_agent_b + out.match.draws == 20
    assert out.quality.num_second_hand_with_winning_reply <= out.quality.num_second_hand_decisions
    assert out.quality.num_trump_waste <= out.quality.num_second_hand_with_winning_reply
    assert out.quality.num_trump_overkill <= out.quality.num_second_hand_trump_wins

    head_to_head = evaluate_bc_model_seat_fair_match_2p_with_quality_numba(
        agent,
        agent.name,
        num_games=20,
        seed=0,
        game_seeds=list(range(10)),
        opponent_agent=agent,
    )
    assert head_to_head.match.num_games == 20
    assert head_to_head.match.wins_agent_a + head_to_head.match.wins_agent_b + head_to_head.match.draws == 20
    assert head_to_head.quality.num_trump_waste <= head_to_head.quality.num_second_hand_with_winning_reply
