"""
Kernel PIMC JIT: invarianti di campionamento, equivalenza col path python, fallback.

La search e' stocastica: la parita' col python non e' bit-a-bit ma SEMANTICA, protetta da
(a) invarianti sulle determinizzazioni, (b) partite intere legali via agente, (c) gate di
forza offline (benchmarks: +3.38 JIT vs +3.83 python contro lo stesso controllo, CI
sovrapposte — vedi benchmarks/experiments/fase3/pimc_numba_*_400.json).
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from briscola_ai.ai.numba.pimc import (
    _weighted_sample_without_replacement_numba,
    choose_pimc_card_numba_arrays,
    warm_up_numba_pimc,
)

pytestmark = pytest.mark.numba

ACTION_DIM = 40


def test_weighted_sampling_respects_weights_statistically() -> None:
    """Il successive sampling JIT campiona ~proporzionalmente ai pesi (k=1, legge esatta)."""
    warm_up_numba_pimc()
    np.random.seed(7)
    weights = np.zeros(ACTION_DIM, dtype=np.float64)
    weights[3] = 0.7
    weights[15] = 0.2
    weights[27] = 0.1
    counts: Counter[int] = Counter()
    out = np.empty(3, dtype=np.int64)
    for _ in range(4000):
        pool = np.asarray([3, 15, 27], dtype=np.int64)
        _weighted_sample_without_replacement_numba(pool, 3, 1, weights, out)
        counts[int(out[0])] += 1
    freq3 = counts[3] / 4000
    freq15 = counts[15] / 4000
    assert 0.65 < freq3 < 0.75, freq3
    assert 0.16 < freq15 < 0.24, freq15


def test_weighted_sampling_degrades_to_uniform_on_zero_weights() -> None:
    """Pesi tutti nulli: uniforme sul pool (mai crash, come nel python)."""
    np.random.seed(1)
    weights = np.zeros(ACTION_DIM, dtype=np.float64)
    out = np.empty(2, dtype=np.int64)
    seen: Counter[int] = Counter()
    for _ in range(600):
        pool = np.asarray([5, 6, 7], dtype=np.int64)
        _weighted_sample_without_replacement_numba(pool, 3, 2, weights, out)
        assert out[0] != out[1]
        seen.update(int(v) for v in out)
    assert set(seen) == {5, 6, 7}
    for card_id in (5, 6, 7):
        assert 0.55 < seen[card_id] / 600 < 0.80  # ~2/3 atteso


def test_kernel_returns_fallback_sentinel_on_inconsistent_or_endgame_state() -> None:
    """Stati non determinizzabili o endgame: -1 (delega al chiamante)."""
    w1 = np.zeros((248, 4), dtype=np.float32)
    b1 = np.zeros(4, dtype=np.float32)
    w2 = np.zeros((4, ACTION_DIM), dtype=np.float32)
    b2 = np.zeros(ACTION_DIM, dtype=np.float32)
    my_hand = np.asarray([0, 1, 2], dtype=np.int64)
    empty_table = np.full(2, -1, dtype=np.int64)

    # Conteggio incoerente: pool ignoto != opp_hand + deck.
    out_all_dead = np.ones(ACTION_DIM, dtype=np.int64)
    for live in (0, 1, 2):
        out_all_dead[live] = 0
    res = choose_pimc_card_numba_arrays(
        w1,
        b1,
        w2,
        b2,
        True,
        np.ones(ACTION_DIM, dtype=np.float64),
        my_hand,
        3,
        3,
        2,
        empty_table,
        empty_table,
        0,
        0,
        20,
        np.zeros(2, dtype=np.int64),
        out_all_dead,
        np.zeros(ACTION_DIM, dtype=np.int64),
        np.zeros((20, 5), dtype=np.int64),
        0,
        4,
        0,
    )
    assert res == -1

    # Endgame (deck 0): il chiamante deve usare il solver esatto.
    out_endgame = np.ones(ACTION_DIM, dtype=np.int64)
    for live in (0, 1, 2, 10, 11, 12):
        out_endgame[live] = 0
    res = choose_pimc_card_numba_arrays(
        w1,
        b1,
        w2,
        b2,
        True,
        np.ones(ACTION_DIM, dtype=np.float64),
        my_hand,
        3,
        3,
        0,
        empty_table,
        empty_table,
        0,
        0,
        20,
        np.zeros(2, dtype=np.int64),
        out_endgame,
        np.zeros(ACTION_DIM, dtype=np.int64),
        np.zeros((20, 5), dtype=np.int64),
        0,
        4,
        0,
    )
    assert res == -1


@pytest.mark.slow
def test_pimc_agent_with_numba_search_plays_full_games() -> None:
    """
    L'agente con search JIT gioca partite intere legali via dominio (path runtime UI)
    e usa davvero la search (metrica search_decisions > 0, zero determinizzazioni fallite).
    Richiede il modello locale best_a2c_v8 (skip su ambienti puliti/CI).
    """
    from pathlib import Path as _P

    model_path = _P("data/models/best_a2c_v8.npz")
    if not model_path.exists():
        pytest.skip("best_a2c_v8.npz assente (artefatto locale)")

    from briscola_ai.ai.agents.pimc import PIMCAgent
    from briscola_ai.ai.agents.registry import build_agent
    from briscola_ai.ai.models import BCModelAgent
    from briscola_ai.domain.engine import PlayCardAction, step
    from briscola_ai.domain.observation import make_player_observation
    from briscola_ai.domain.state import new_game_state

    policy = BCModelAgent.from_npz(model_path)
    agent = PIMCAgent(
        rollout_agent=policy,
        fallback=policy,
        num_determinizations=8,
        max_unknown_cards=10,
        use_numba_search=True,
    )
    opp = build_agent("heuristic_v2")
    rng = random.Random(5)
    for seed in (31, 32):
        s = new_game_state(2, seed=seed)
        moves = 0
        while not s.game_over and moves < 100:
            ag = agent if s.current_turn == 0 else opp
            idx = ag.choose_card_index(make_player_observation(s, s.current_turn), rng=rng)
            s, res = step(s, PlayCardAction(player_index=s.current_turn, card_index=idx))
            assert res.error is None
            moves += 1
        assert s.game_over
    assert agent.metrics.search_decisions > 0
    assert agent.metrics.failed_determinizations == 0
