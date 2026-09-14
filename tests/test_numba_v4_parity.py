"""
Parità dell'encoder v4 attraverso i tre motori: dominio ↔ fast Python ↔ kernel Numba.

Cosa proteggono questi test:
- la storia delle prese mantenuta nel fast state e nei kernel JIT produce ESATTAMENTE
  le stesse 59 feature v4 del path canonico (`_compute_v4_extra_features` su
  `PlayerObservation.trick_history`), a ogni decisione di partite reali;
- il collector A2C full-JIT con pesi v4 produce feature 369-dim con blocco storia
  popolato (il training dell'iterazione-1 dipende da questo);
- la valutazione numba accetta modelli v4 (gate veloci per l'iterazione-1).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import (
    FEATURE_DIM_2P_V3,
    FEATURE_DIM_2P_V4,
    encode_player_observation_2p,
)
from briscola_ai.ai.fast.observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.fast.state_2p import new_fast_2p_state, step_fast_2p
from briscola_ai.ai.numba.mlp import collect_a2c_trajectory_numba_2p, evaluate_mlp_policy_numba_2p
from briscola_ai.ai.numba.observation import encode_fast_observation_numba_2p, warm_up_numba_observation
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba


def _play_mirrored_games(seed: int, steps: int, action_seed: int):
    """
    Avanza in parallelo la partita di dominio e quella fast con le STESSE mosse.

    La parità del deal per costruzione (stesso `random.Random(seed).shuffle`) è già
    protetta dai test fast/dominio; qui la sfruttiamo per confrontare gli encoder.
    Ritorna (domain_state, fast_state, seen, out_of_play).
    """
    domain_state = new_game_state(2, seed=seed)
    fast_state = new_fast_2p_state(seed=seed)
    seen = [0] * 40
    seen[fast_state.trump_card] = 1
    out_of_play = [0] * 40
    rng = random.Random(action_seed)

    for _ in range(steps):
        if fast_state.game_over:
            break
        current = fast_state.current_turn
        assert current == domain_state.current_turn
        card_index = rng.randrange(len(fast_state.hands[current]))

        result = step_fast_2p(fast_state, player_index=current, card_index=card_index)
        seen[result.played_card] = 1
        out_of_play[result.played_card] = 1

        domain_state, domain_result = step(domain_state, PlayCardAction(player_index=current, card_index=card_index))
        assert domain_result.error is None

    return domain_state, fast_state, tuple(seen), tuple(out_of_play)


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("steps", [0, 5, 12, 26, 40])
def test_v4_parity_domain_fast_numba_on_mirrored_games(seed: int, steps: int) -> None:
    """A ogni profondità di partita, i tre encoder v4 devono coincidere per entrambi i player."""
    warm_up_numba_observation()
    domain_state, fast_state, seen, out_of_play = _play_mirrored_games(seed, steps, action_seed=seed ^ 0xA4)

    for player_index in (0, 1):
        domain_enc = encode_player_observation_2p(make_player_observation(domain_state, player_index), version="v4")
        fast_enc = encode_fast_observation_2p(
            fast_state,
            player_index=player_index,
            seen_cards_onehot=seen,
            out_of_play_cards_onehot=out_of_play,
            version="v4",
        )
        numba_enc = encode_fast_observation_numba_2p(
            fast_state,
            player_index=player_index,
            seen_cards_onehot=seen,
            out_of_play_cards_onehot=out_of_play,
            version="v4",
        )

        assert fast_enc.action_mask == domain_enc.action_mask == numba_enc.action_mask
        assert fast_enc.features == pytest.approx(domain_enc.features)
        assert numba_enc.features == pytest.approx(domain_enc.features)
        assert len(numba_enc.features) == FEATURE_DIM_2P_V4


def _dummy_v4_weights(hidden: int = 8):
    """Pesi MLP v4 minimi per esercitare collector/eval JIT."""
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 0.05, size=(FEATURE_DIM_2P_V4, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0, 0.05, size=(hidden, 40)).astype(np.float32)
    b2 = np.zeros(40, dtype=np.float32)
    wv = np.zeros(hidden, dtype=np.float32)
    return w1, b1, w2, b2, wv


def test_a2c_collector_produces_v4_features_with_history() -> None:
    """
    Il collector A2C full-JIT con pesi v4 registra feature 369-dim e, nelle decisioni
    avanzate della partita, il blocco storia (offset >= 310) è popolato.
    """
    w1, b1, w2, b2, wv = _dummy_v4_weights()
    trajectory = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="random",
        game_seed=3,
        policy_seat=0,
    )
    xs = trajectory.xs
    assert xs.shape[1] == FEATURE_DIM_2P_V4
    # Prima decisione: nessuna presa completata -> blocco v4 a zero.
    assert float(np.abs(xs[0, FEATURE_DIM_2P_V3:]).sum()) == 0.0
    # Ultima decisione: la storia deve essere popolata (contatore prese > 0).
    tricks_feature = xs[-1, FEATURE_DIM_2P_V3 + 10]  # "prese completate" / 20
    assert tricks_feature > 0.0
    # Lo slot piu' recente del blocco B esiste.
    assert xs[-1, FEATURE_DIM_2P_V3 + 11] == pytest.approx(1.0)


def test_numba_eval_accepts_v4_models() -> None:
    """La valutazione full-JIT accetta policy v4 (gate veloci per l'iterazione-1)."""
    w1, b1, w2, b2, _wv = _dummy_v4_weights()
    summary = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="random",
        num_games=8,
        seed=1,
        seat_fair=True,
        deterministic=True,
        parallel=True,
    )
    assert summary.num_games == 8
    assert summary.wins_policy + summary.wins_opponent + summary.draws == 8
