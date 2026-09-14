#!/usr/bin/env python3
"""
Training RL didattico: Actor-Critic (A2C minimale) + reward shaping "trick delta".

Perché A2C
----------
REINFORCE (policy gradient puro) aggiorna la policy usando un return spesso molto rumoroso.
Un modo semplice per ridurre la varianza è aggiungere un *critic* che stima `V(s)` e
usare l'**advantage**:

  A(s,a) = G_t - V(s)

dove `G_t` è il return-to-go (somma dei reward futuri).

Reward shaping: "trick delta"
-----------------------------
In Briscola i punti cambiano solo quando si chiude una mano (trick). Se usiamo solo il reward finale,
il segnale arriva tardi. Qui rendiamo il reward più denso senza barare:

- definiamo un "time-step" come: **una scelta della policy** (turno della policy)
- reward dello step = delta di `(punti_policy - punti_opp)` accumulato fino al prossimo turno della policy
  (include quindi l'azione dell'avversario che chiude la mano, se necessario).

Anti-cheat
----------
La policy vede solo `PlayerObservation` (osservazione parziale lecita).

Warm-start consigliato
----------------------
Come per REINFORCE, conviene partire da un BC MLP teacher-only:

  python scripts/train_a2c.py \\
    --init ./data/bc_model_teacher_mlp.npz \\
    --out ./data/a2c_shaped.npz \\
    --opponent-mix heuristic_v1:0.7,random:0.2,greedy_points:0.1 \\
    --num-games 200000 --seat-fair --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.encoding.card_action_space import action_id_from_suit_number
from briscola_ai.ai.encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    FEATURE_DIM_2P_V3,
    FEATURE_DIM_2P_V4,
    EncoderVersion,
    encode_player_observation_2p,
    feature_dim_for_encoder_version,
)
from briscola_ai.ai.evaluation.suit_symmetry import SuitPermutation
from briscola_ai.ai.fast.evaluation import FAST_EVALUATION_AGENT_NAMES, choose_fast_card_index
from briscola_ai.ai.fast.observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.fast.state_2p import Fast2PState, new_fast_2p_state, step_fast_2p
from briscola_ai.ai.models import BCModelAgent, LoadedBCModel, MLPBCModel, load_bc_model_npz
from briscola_ai.ai.models.belief_model import load_belief_model_npz
from briscola_ai.ai.models.provisioning import VALUE_LOOKAHEAD_MODEL_ID
from briscola_ai.ai.models.value_model import MLPValueModel, load_value_model_npz
from briscola_ai.ai.numba.core import numba_agent_code
from briscola_ai.ai.numba.mlp import collect_a2c_batch_numba_2p, collect_a2c_trajectory_numba_2p
from briscola_ai.ai.numba.observation import encode_fast_observation_numba_2p
from briscola_ai.ai.numba.types import NumbaA2CBatch, NumbaA2CTrajectory
from briscola_ai.ai.numba.value_lookahead import (
    OPPONENT_MODE_MODEL,
    OPPONENT_MODE_PIMC_BELIEF,
    OPPONENT_MODE_RULE,
    OPPONENT_MODE_VALUE_LOOKAHEAD,
    collect_a2c_batch_numba_value_lookahead_2p,
    collect_a2c_trajectory_numba_value_lookahead_2p,
)
from briscola_ai.ai.training.a2c_checkpoint import (
    A2C_RESUME_SCHEMA,
    atomic_savez,
    canonical_json,
    config_fingerprint,
    json_compatible,
    parse_resume_json,
    tuple_tree,
)
from briscola_ai.ai.training.a2c_diagnostics import (
    A2CArrayGroups,
    A2CSignalAccumulator,
    A2CUpdateDiagnostics,
    array_group_l2,
    build_update_diagnostics,
    summarize_update_diagnostics,
)
from briscola_ai.ai.training.game_schedule import (
    ScheduledTrainingGame,
    TrainingGameScheduleStream,
    TrainingScheduleMode,
)
from briscola_ai.ai.training.opponent_mix import OpponentMixItem, parse_opponent_mix, sample_opponent_name
from briscola_ai.ai.training.policy_regularization import cross_entropy_from_probs, grad_ce_wrt_logits_from_probs
from briscola_ai.ai.training.reward_shaping import trump_overkill_penalty, trump_overkill_penalty_gap
from briscola_ai.ai.training.streaming_history import HistoryMode, StreamingHistory
from briscola_ai.ai.training.suit_augmentation import (
    permute_action_ids,
    permute_action_masks,
    permute_action_vectors,
    permute_encoded_features,
    permute_encoded_trajectory,
    sample_nonidentity_suit_permutation,
)
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state
from briscola_ai.versioning import get_code_version, get_rules_version


def _sha256(path: Path) -> str:
    """SHA-256 streaming per rendere auditabili init, modello e report diagnostico."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str | int]:
    """Descrive un artefatto locale senza incorporarne i pesi."""
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _git_commit() -> str | None:
    """Commit corrente best-effort per la riproducibilita' del report."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError, subprocess.SubprocessError:
        return None


def _masked_logits_1d(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maschera logits 1D: azioni non valide -> numero molto negativo."""
    very_negative = -1e9
    out = logits.copy()
    out[~mask] = very_negative
    return out


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    """Softmax 1D numericamente stabile."""
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


def _entropy(probs: np.ndarray) -> float:
    """Entropia (Shannon) per una distribuzione discreta."""
    p = probs + 1e-12
    return float(-np.sum(p * np.log(p)))


@dataclass
class AdamState:
    """Stato Adam per un singolo tensore."""

    m: np.ndarray
    v: np.ndarray


def _adam_init(param: np.ndarray) -> AdamState:
    """Inizializza stato Adam (m,v) con zeri, stessa shape del parametro."""
    return AdamState(m=np.zeros_like(param), v=np.zeros_like(param))


def _adam_update(
    param: np.ndarray,
    grad: np.ndarray,
    *,
    state: AdamState,
    lr: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Aggiornamento Adam in-place."""
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * (grad * grad)
    m_hat = state.m / (1.0 - beta1**t)
    v_hat = state.v / (1.0 - beta2**t)
    param -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)


@dataclass(frozen=True, slots=True)
class OpponentPool:
    """Pool di avversari campionabili (opponent mix)."""

    items: list[OpponentMixItem]
    agents_by_name: dict[str, Agent]

    def sample(self, *, rng: np.random.Generator) -> Agent:
        """Campiona un avversario secondo la distribuzione."""
        name = sample_opponent_name(self.items, rng=rng)
        return self.agents_by_name[name]

    def to_metadata(self) -> list[dict[str, float | str]]:
        """Rappresentazione serializzabile (ordine stabile) per `metadata_json`."""
        return [{"name": item.name, "prob": float(item.prob)} for item in self.items]


@dataclass(frozen=True, slots=True)
class NamedAgentProxy:
    """Proxy leggero: conserva un nome canonico per logging delegando le mosse a un agente reale."""

    display_name: str
    inner: Agent

    @property
    def name(self) -> str:
        return self.display_name

    def choose_card_index(self, observation, *, rng: random.Random) -> int:
        return int(self.inner.choose_card_index(observation, rng=rng))


@dataclass(frozen=True, slots=True)
class FastNumbaModelOpponent:
    """Opponent MLP caricato da `.npz` per il rollout A2C Numba."""

    agent: BCModelAgent
    model: MLPBCModel


@dataclass(frozen=True, slots=True)
class FastNumbaValueLookaheadOpponent:
    """Opponent value-lookahead determinized per `--rollout-engine fast --fast-rollout numba`."""

    agent: BCModelAgent
    model: MLPBCModel
    value_model: MLPValueModel
    max_unknown_cards: int = 8


@dataclass
class A2CPolicy:
    """
    Policy + critic con trunk condiviso (MLP 1 hidden layer + ReLU).

    - trunk: w1/b1
    - actor head: w2/b2 (logits su 40 azioni)
    - critic head: wv/bv (valore scalare)
    """

    w1: np.ndarray  # (D, H)
    b1: np.ndarray  # (H,)
    w2: np.ndarray  # (H, 40)
    b2: np.ndarray  # (40,)
    wv: np.ndarray  # (H,)
    bv: float

    @property
    def feature_dim(self) -> int:
        return int(self.w1.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.w1.shape[1])

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Forward: ritorna (z1, h, logits, value)."""
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        logits = h @ self.w2 + self.b2
        value = float(h @ self.wv + self.bv)
        return z1, h, logits, value


@dataclass
class StepRecord:
    """Dati per un singolo step della policy (una decisione)."""

    x: np.ndarray  # (D,)
    z1: np.ndarray  # (H,)
    h: np.ndarray  # (H,)
    action_mask: np.ndarray  # (40,) bool
    probs: np.ndarray  # (40,)
    anchor_probs: np.ndarray | None
    anchor_ce: float
    action_id: int
    value_pred: float
    reward: float  # shaped reward (delta point diff / 120)


def _action_id_to_card_index(*, action_id: int, hand) -> int:
    """Converte action_id (carta canonica) in indice nella mano corrente."""
    for i, card in enumerate(hand):
        cid = action_id_from_suit_number(suit=card.suit.value, number=card.rank.number)
        if cid == action_id:
            return i
    raise ValueError(f"action_id {action_id} non corrisponde a nessuna carta nella mano (hand_size={len(hand)})")


def _points_diff(state: GameState, *, policy_seat: int) -> int:
    """Ritorna (punti_policy - punti_opp) dallo stato finale/corrente."""
    p0 = state.players[0].points
    p1 = state.players[1].points
    return int(p0 - p1) if policy_seat == 0 else int(p1 - p0)


def _play_one_game_2p_collect(
    *,
    policy: A2CPolicy,
    opponent: Agent,
    rng_opponent: random.Random,
    rng_action: np.random.Generator,
    game_seed: int,
    policy_seat: int,
    entropy_beta: float,
    encoder_version: EncoderVersion,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int | None,
    overkill_penalty_mode: str,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
) -> tuple[GameState, list[StepRecord], float]:
    """
    Simula una partita 2-player e colleziona la traiettoria vista come MDP "turno della policy".

    Ritorna:
    - stato finale
    - lista step della policy (uno per azione della policy)
    - entropia media (diagnostica)
    """
    state = new_game_state(num_players=2, seed=game_seed)
    traj: list[StepRecord] = []
    entropies: list[float] = []

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1

        # Se non è turno della policy, avanza con l'avversario finché lo diventa.
        while not state.game_over and state.current_turn != policy_seat:
            obs_opp = make_player_observation(state, state.current_turn)
            card_index = opponent.choose_card_index(obs_opp, rng=rng_opponent)
            state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

        if state.game_over:
            break

        # Ora tocca alla policy: definiamo uno step dell'MDP.
        diff_before = _points_diff(state, policy_seat=policy_seat)
        obs = make_player_observation(state, policy_seat)
        encoded = encode_player_observation_2p(obs, version=encoder_version)

        x = np.asarray(encoded.features, dtype=np.float32)
        mask = np.asarray(encoded.action_mask, dtype=bool)
        if x.shape[0] != policy.feature_dim:
            raise ValueError(f"Feature dim mismatch: got={x.shape[0]} expected={policy.feature_dim}")

        z1, h, logits, value_pred = policy.forward(x)
        masked = _masked_logits_1d(logits, mask)
        probs = _softmax_1d(masked)
        entropies.append(_entropy(probs))
        action_id = int(rng_action.choice(40, p=probs))
        card_index = _action_id_to_card_index(action_id=action_id, hand=obs.hand)

        # BC-anchor: regolarizzazione "stay-close-to-teacher" (senza barare).
        #
        # Idea:
        # - l'anchor è un modello BC fisso (teacher distillato) che non aggiorniamo.
        # - la policy RL viene penalizzata se si allontana troppo dall'anchor (cross-entropy).
        #
        # Questo termine di loss agisce durante training, non a inference-time: quindi
        # se vedi meno overkill nei benchmark senza guard, significa che la policy ha
        # interiorizzato (almeno in parte) la preferenza.
        anchor_probs: np.ndarray | None = None
        anchor_ce: float = 0.0
        if bc_anchor is not None and float(bc_anchor_beta) > 0.0:
            anchor_logits = bc_anchor.logits(x)
            anchor_masked = _masked_logits_1d(anchor_logits, mask)
            anchor_probs = _softmax_1d(anchor_masked)
            anchor_ce = cross_entropy_from_probs(target_probs=anchor_probs, pred_probs=probs)

        # Reward shaping opzionale: penalità "overkill briscola" (soft).
        #
        # Importante:
        # questa penalità è calcolata SOLO da `PlayerObservation` (anti-cheat),
        # quindi non introduce scorciatoie basate su informazione nascosta.
        if overkill_penalty_mode == "flat":
            extra_penalty = trump_overkill_penalty(
                obs,
                chosen_card_index=card_index,
                beta=float(overkill_penalty_beta),
                low_lead_points_max=overkill_low_lead_points_max,
            )
        elif overkill_penalty_mode == "gap":
            extra_penalty = trump_overkill_penalty_gap(
                obs,
                chosen_card_index=card_index,
                beta=float(overkill_penalty_beta),
                low_lead_points_max=overkill_low_lead_points_max,
            )
        else:
            raise ValueError(f"overkill_penalty_mode non supportato: {overkill_penalty_mode!r}")

        # Applica azione policy.
        state, result = step(state, PlayCardAction(player_index=policy_seat, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

        # Avanza con l'avversario fino al prossimo turno della policy (o fine partita).
        while not state.game_over and state.current_turn != policy_seat:
            obs_opp = make_player_observation(state, state.current_turn)
            opp_card_index = opponent.choose_card_index(obs_opp, rng=rng_opponent)
            state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=opp_card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

        diff_after = _points_diff(state, policy_seat=policy_seat)
        reward = float(diff_after - diff_before) / 120.0 + float(extra_penalty)

        traj.append(
            StepRecord(
                x=x,
                z1=z1,
                h=h,
                action_mask=mask,
                probs=probs,
                anchor_probs=anchor_probs,
                anchor_ce=float(anchor_ce),
                action_id=action_id,
                value_pred=float(value_pred),
                reward=reward,
            )
        )

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    return state, traj, avg_entropy


def _action_id_to_fast_card_index(*, action_id: int, hand: list[int]) -> int:
    """Converte action_id (che nel fast path coincide col card_id) in indice nella mano."""
    for i, card_id in enumerate(hand):
        if int(card_id) == int(action_id):
            return i
    raise ValueError(f"action_id {action_id} non corrisponde a nessuna carta fast nella mano (hand_size={len(hand)})")


def _points_diff_fast(state: Fast2PState, *, policy_seat: int) -> int:
    """Ritorna (punti_policy - punti_opp) dallo stato fast."""
    p0 = int(state.points[0])
    p1 = int(state.points[1])
    return p0 - p1 if policy_seat == 0 else p1 - p0


def _load_fast_numba_model_opponent(*, opponent_name: str, opponent_model_path: str) -> FastNumbaModelOpponent:
    """Carica un opponent `.npz` per `--rollout-engine fast --fast-rollout numba`."""
    if opponent_name == "best_a2c":
        agent = build_agent("best_a2c")
    elif opponent_name == "bc_model":
        if not opponent_model_path.strip():
            raise ValueError("`--opponent bc_model` richiede `--opponent-model <path.npz>`.")
        agent = build_agent("bc_model", model_path=Path(opponent_model_path.strip()))
    else:
        raise ValueError(f"Opponent modello non supportato nel fast rollout Numba: {opponent_name!r}")

    if not isinstance(agent, BCModelAgent):
        raise ValueError(f"Opponent {opponent_name!r} non ha prodotto un BCModelAgent.")
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("Il fast rollout Numba supporta per ora solo opponent `.npz` MLP (w1/b1/w2/b2).")
    if int(agent.model.feature_dim) not in (
        int(FEATURE_DIM_2P_V1),
        int(FEATURE_DIM_2P_V2),
        int(FEATURE_DIM_2P_V3),
        int(FEATURE_DIM_2P_V4),
    ):
        raise ValueError(
            "Opponent MLP non compatibile: "
            f"feature_dim={int(agent.model.feature_dim)} atteso "
            f"{int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, {int(FEATURE_DIM_2P_V3)} o {int(FEATURE_DIM_2P_V4)}."
        )
    return FastNumbaModelOpponent(agent=agent, model=agent.model)


def _load_fast_numba_value_lookahead_opponent(
    *,
    opponent_model_path: str,
    opponent_value_model_path: str,
    max_unknown_cards: int,
) -> FastNumbaValueLookaheadOpponent:
    """
    Carica l'opponent `bc_model_value_lookahead_8x8` per il rollout A2C Numba.

    Nota: nel fast rollout questo opponent usa lo stato numerico determinizzato della partita
    come singola determinizzazione. È pensato come avversario di training forte, non come replica
    bit-a-bit dell'agente UI che campiona information set da `PlayerObservation`.
    """
    if not opponent_model_path.strip():
        raise ValueError("`bc_model_value_lookahead_8x8` nel fast rollout richiede `--opponent-model <policy.npz>`.")
    value_path = opponent_value_model_path.strip()
    if not value_path:
        value_path = str(Path("data/models") / VALUE_LOOKAHEAD_MODEL_ID)

    agent = build_agent("bc_model", model_path=Path(opponent_model_path.strip()))
    if not isinstance(agent, BCModelAgent):
        raise ValueError("`bc_model_value_lookahead_8x8` richiede una policy base BCModelAgent.")
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("Il fast rollout Numba supporta solo policy base `.npz` MLP (w1/b1/w2/b2).")
    value_model = load_value_model_npz(Path(value_path))
    if int(value_model.feature_dim) != int(agent.model.feature_dim):
        raise ValueError(
            "Value model non compatibile con policy opponent: "
            f"value.feature_dim={int(value_model.feature_dim)} policy.feature_dim={int(agent.model.feature_dim)}."
        )
    if int(max_unknown_cards) < 0:
        raise ValueError("--opponent-value-max-unknown-cards deve essere >= 0")
    return FastNumbaValueLookaheadOpponent(
        agent=agent,
        model=agent.model,
        value_model=value_model,
        max_unknown_cards=int(max_unknown_cards),
    )


def _load_fast_numba_pimc_belief_opponent(
    *,
    opponent_model_path: str,
    opponent_belief_model_path: str,
    max_unknown_cards: int,
) -> FastNumbaValueLookaheadOpponent:
    """
    Carica il maestro `bc_model_pimc_belief` per il rollout A2C Numba (mode 3).

    Riusa il container del value-lookahead con un value model DUMMY (la modalita' PIMC
    non lo legge): policy base MLP + belief network per pesare le determinizzazioni.
    La finestra riusa `--opponent-value-max-unknown-cards`.
    """
    if not opponent_model_path.strip():
        raise ValueError("`bc_model_pimc_belief` nel fast rollout richiede `--opponent-model <policy.npz>`.")
    agent = build_agent("bc_model", model_path=Path(opponent_model_path.strip()))
    if not isinstance(agent, BCModelAgent) or not isinstance(agent.model, MLPBCModel):
        raise ValueError("`bc_model_pimc_belief` richiede una policy base `.npz` MLP.")
    import numpy as _np

    from briscola_ai.ai.models.value_model import MLPValueModel

    dummy_value = MLPValueModel(
        w1=_np.zeros((int(agent.model.feature_dim), 1), dtype=_np.float32),
        b1=_np.zeros(1, dtype=_np.float32),
        w2=_np.zeros(1, dtype=_np.float32),
        b2=0.0,
        metadata={"format": "value_mlp_v1", "note": "dummy per opponent PIMC (mode 3)"},
    )
    if int(max_unknown_cards) < 0:
        raise ValueError("--opponent-value-max-unknown-cards deve essere >= 0")
    return FastNumbaValueLookaheadOpponent(
        agent=agent,
        model=agent.model,
        value_model=dummy_value,
        max_unknown_cards=int(max_unknown_cards),
    )


def _fast_numba_opponent_mode_for_name(
    name: str, *, value_lookahead_name: str | None, pimc_belief_name: str | None = None
) -> int:
    """Codifica il tipo opponent per il collector A2C value-aware."""
    if pimc_belief_name is not None and name == pimc_belief_name:
        return OPPONENT_MODE_PIMC_BELIEF
    if value_lookahead_name is not None and name == value_lookahead_name:
        return OPPONENT_MODE_VALUE_LOOKAHEAD
    if name in {"best_a2c", "bc_model"}:
        return OPPONENT_MODE_MODEL
    return OPPONENT_MODE_RULE


def _numba_batch_trajectory_at(batch: NumbaA2CBatch, index: int) -> NumbaA2CTrajectory:
    """Estrae una traiettoria dal batch Numba usando view sulle righe valide."""
    count = int(batch.step_counts[index])
    return NumbaA2CTrajectory(
        policy_points=int(batch.policy_points[index]),
        opponent_points=int(batch.opponent_points[index]),
        winner=int(batch.winners[index]),
        avg_entropy=float(batch.avg_entropies[index]),
        xs=batch.xs[index, :count],
        z1s=batch.z1s[index, :count],
        hs=batch.hs[index, :count],
        action_masks=batch.action_masks[index, :count],
        probs=batch.probs[index, :count],
        action_ids=batch.action_ids[index, :count],
        value_preds=batch.value_preds[index, :count],
        rewards=batch.rewards[index, :count],
    )


def _play_one_fast_game_2p_collect(
    *,
    policy: A2CPolicy,
    opponent_name: str,
    rng_opponent: random.Random,
    rng_action: np.random.Generator,
    game_seed: int,
    policy_seat: int,
    encoder_version: EncoderVersion,
    fast_encoder: str,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
) -> tuple[Fast2PState, list[StepRecord], float]:
    """
    Simula una partita A2C usando il fast path 2-player (`ai.fast.state_2p`).

    Limitazioni intenzionali:
    - supporta solo avversari tradotti su card id (`random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`,
      `heuristic_trump_saver`);
    - non applica ancora reward shaping anti-overkill, perché quello oggi dipende da `PlayerObservation`.
    """
    if opponent_name not in FAST_EVALUATION_AGENT_NAMES:
        supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
        raise ValueError(f"`--rollout-engine fast` supporta solo avversari: {supported}. Ottenuto: {opponent_name!r}")

    state = new_fast_2p_state(seed=game_seed)
    traj: list[StepRecord] = []
    entropies: list[float] = []

    # Storia pubblica per encoder v2: briscola scoperta + ogni carta giocata.
    seen = [0] * 40
    seen[state.trump_card] = 1
    # Carte fuori gioco per encoder v3: SOLO carte giocate (no briscola iniziale).
    out_of_play = [0] * 40

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1

        while not state.game_over and state.current_turn != policy_seat:
            current = state.current_turn
            card_index = choose_fast_card_index(
                opponent_name,
                state,
                current,
                rng=rng_opponent,
                seen_cards_onehot=tuple(seen),
            )
            result = step_fast_2p(state, player_index=current, card_index=card_index)
            seen[result.played_card] = 1
            out_of_play[result.played_card] = 1

        if state.game_over:
            break

        diff_before = _points_diff_fast(state, policy_seat=policy_seat)
        if fast_encoder == "numba":
            encoded = encode_fast_observation_numba_2p(
                state,
                player_index=policy_seat,
                seen_cards_onehot=tuple(seen),
                out_of_play_cards_onehot=tuple(out_of_play),
                version=encoder_version,
            )
        elif fast_encoder == "python":
            encoded = encode_fast_observation_2p(
                state,
                player_index=policy_seat,
                seen_cards_onehot=tuple(seen),
                out_of_play_cards_onehot=tuple(out_of_play),
                version=encoder_version,
            )
        else:
            raise ValueError(f"fast_encoder non supportato: {fast_encoder!r}")
        x = np.asarray(encoded.features, dtype=np.float32)
        mask = np.asarray(encoded.action_mask, dtype=bool)
        if x.shape[0] != policy.feature_dim:
            raise ValueError(f"Feature dim mismatch: got={x.shape[0]} expected={policy.feature_dim}")

        z1, h, logits, value_pred = policy.forward(x)
        masked = _masked_logits_1d(logits, mask)
        probs = _softmax_1d(masked)
        entropies.append(_entropy(probs))
        action_id = int(rng_action.choice(40, p=probs))
        card_index = _action_id_to_fast_card_index(action_id=action_id, hand=state.hands[policy_seat])

        anchor_probs: np.ndarray | None = None
        anchor_ce = 0.0
        if bc_anchor is not None and float(bc_anchor_beta) > 0.0:
            anchor_logits = bc_anchor.logits(x)
            anchor_masked = _masked_logits_1d(anchor_logits, mask)
            anchor_probs = _softmax_1d(anchor_masked)
            anchor_ce = cross_entropy_from_probs(target_probs=anchor_probs, pred_probs=probs)

        result = step_fast_2p(state, player_index=policy_seat, card_index=card_index)
        seen[result.played_card] = 1
        out_of_play[result.played_card] = 1

        while not state.game_over and state.current_turn != policy_seat:
            current = state.current_turn
            opp_card_index = choose_fast_card_index(
                opponent_name,
                state,
                current,
                rng=rng_opponent,
                seen_cards_onehot=tuple(seen),
            )
            result = step_fast_2p(state, player_index=current, card_index=opp_card_index)
            seen[result.played_card] = 1
            out_of_play[result.played_card] = 1

        diff_after = _points_diff_fast(state, policy_seat=policy_seat)
        reward = float(diff_after - diff_before) / 120.0

        traj.append(
            StepRecord(
                x=x,
                z1=z1,
                h=h,
                action_mask=mask,
                probs=probs,
                anchor_probs=anchor_probs,
                anchor_ce=float(anchor_ce),
                action_id=action_id,
                value_pred=float(value_pred),
                reward=reward,
            )
        )

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita fast non termina")

    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    return state, traj, avg_entropy


def _compute_returns(rewards: list[float], *, gamma: float) -> list[float]:
    """Return-to-go (Monte Carlo) con sconto `gamma` (default tipico: 1.0)."""
    out = [0.0] * len(rewards)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out


def _compute_returns_array(rewards: np.ndarray, *, gamma: float) -> np.ndarray:
    """Return-to-go su array NumPy, usato dal rollout Numba senza passare da `StepRecord`."""
    out = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = float(rewards[i]) + gamma * g
        out[i] = g
    return out


@dataclass(frozen=True, slots=True)
class GradientStats:
    """Contatori prodotti dall'accumulo gradienti per il logging e la normalizzazione."""

    steps: int
    value_loss_sum: float
    anchor_ce_sum: float
    anchor_ce_count: int
    gbv: float
    suit_consistency_kl_sum: float = 0.0
    suit_consistency_count: int = 0
    suit_margin_loss_sum: float = 0.0
    suit_margin_count: int = 0
    suit_margin_violation_count: int = 0
    suit_margin_teacher_sum: float = 0.0
    suit_margin_student_sum: float = 0.0


def _add_gradient_stats(left: GradientStats, right: GradientStats) -> GradientStats:
    """Somma contatori di due accumuli che hanno già aggiornato gli stessi array gradiente."""
    return GradientStats(
        steps=left.steps + right.steps,
        value_loss_sum=left.value_loss_sum + right.value_loss_sum,
        anchor_ce_sum=left.anchor_ce_sum + right.anchor_ce_sum,
        anchor_ce_count=left.anchor_ce_count + right.anchor_ce_count,
        gbv=left.gbv + right.gbv,
        suit_consistency_kl_sum=left.suit_consistency_kl_sum + right.suit_consistency_kl_sum,
        suit_consistency_count=left.suit_consistency_count + right.suit_consistency_count,
        suit_margin_loss_sum=left.suit_margin_loss_sum + right.suit_margin_loss_sum,
        suit_margin_count=left.suit_margin_count + right.suit_margin_count,
        suit_margin_violation_count=left.suit_margin_violation_count + right.suit_margin_violation_count,
        suit_margin_teacher_sum=left.suit_margin_teacher_sum + right.suit_margin_teacher_sum,
        suit_margin_student_sum=left.suit_margin_student_sum + right.suit_margin_student_sum,
    )


def _forward_policy_batch(
    policy: A2CPolicy,
    *,
    xs: np.ndarray,
    action_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward vettoriale del trainer: ``z1, h, probs mascherate, value``."""
    if xs.ndim != 2 or action_masks.shape != (xs.shape[0], 40):
        raise ValueError(f"Shape batch invalide: xs={xs.shape}, masks={action_masks.shape}")
    if xs.shape[1] != policy.feature_dim:
        raise ValueError(f"Feature dim batch={xs.shape[1]} policy={policy.feature_dim}")
    if not bool(np.all(np.any(action_masks, axis=1))):
        raise ValueError("Action mask vuota nel batch")

    z1s = xs @ policy.w1 + policy.b1
    hs = np.maximum(z1s, np.float32(0.0))
    logits = hs @ policy.w2 + policy.b2
    masked_logits = np.where(action_masks, logits, np.float32(-1e9))
    shifted = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    value_preds = hs @ policy.wv + np.float32(policy.bv)
    return (
        z1s.astype(np.float32, copy=False),
        hs.astype(np.float32, copy=False),
        probs.astype(np.float32, copy=False),
        value_preds.astype(np.float32, copy=False),
    )


def _masked_model_probabilities(model: LoadedBCModel, *, xs: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
    """Distribuzioni batch di un anchor congelato sulle sole azioni legali."""
    logits = np.asarray(model.logits(xs), dtype=np.float32)
    if logits.shape != action_masks.shape:
        raise ValueError(f"Output anchor {logits.shape} incompatibile con mask {action_masks.shape}")
    masked = np.where(action_masks, logits, np.float32(-1e9))
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32, copy=False)


def _accumulate_numba_trajectory_grads(
    *,
    policy: A2CPolicy,
    xs: np.ndarray,
    z1s: np.ndarray,
    hs: np.ndarray,
    action_masks: np.ndarray,
    probs: np.ndarray,
    action_ids: np.ndarray,
    value_preds: np.ndarray,
    returns_to_go: np.ndarray,
    entropy_beta: float,
    value_coef: float,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
    anchor_target_probs: np.ndarray | None = None,
    gw1: np.ndarray,
    gb1: np.ndarray,
    gw2: np.ndarray,
    gb2: np.ndarray,
    gwv: np.ndarray,
    signal_diagnostics: A2CSignalAccumulator | None = None,
) -> GradientStats:
    """
    Accumula i gradienti di una traiettoria Numba usando batch matrix multiply.

    Il vecchio path faceva due `np.outer` per ogni decisione della policy. Qui costruiamo
    prima `dlogits` per tutti gli step della partita e poi accumuliamo:
    - `gw2 = H.T @ dlogits`
    - `gw1 = X.T @ dz1`

    La matematica resta la stessa del loop didattico per-step; cambia solo la forma
    computazionale, che riduce overhead Python e allocazioni temporanee nel path caldo.
    """
    steps = int(returns_to_go.shape[0])
    if steps == 0:
        return GradientStats(steps=0, value_loss_sum=0.0, anchor_ce_sum=0.0, anchor_ce_count=0, gbv=0.0)

    if signal_diagnostics is not None:
        signal_diagnostics.observe(
            returns_to_go=returns_to_go,
            value_preds=value_preds,
            hidden=hs,
        )

    adv = returns_to_go.astype(np.float32, copy=False) - value_preds.astype(np.float32, copy=False)
    dlogits = probs.astype(np.float32, copy=True)
    row_idx = np.arange(steps)
    dlogits[row_idx, action_ids.astype(np.int64, copy=False)] -= np.float32(1.0)
    dlogits *= adv[:, None]

    beta = float(entropy_beta)
    if beta > 0.0:
        logp = np.log(probs.astype(np.float32, copy=False) + np.float32(1e-12))
        entropy_center = np.sum(probs * (logp + np.float32(1.0)), axis=1, keepdims=True)
        dent = probs * (logp + np.float32(1.0) - entropy_center)
        dlogits += np.float32(beta) * dent.astype(np.float32, copy=False)

    anchor_ce_sum = 0.0
    anchor_ce_count = 0
    anchor_beta = float(bc_anchor_beta)
    if anchor_target_probs is not None:
        if anchor_target_probs.shape != probs.shape:
            raise ValueError(f"Target anchor paired {anchor_target_probs.shape} incompatibile con probs {probs.shape}")
        if bc_anchor is None:
            raise ValueError("Target anchor paired fornito senza bc_anchor")
    if anchor_beta > 0.0 and bc_anchor is not None:
        for i in range(steps):
            mask = action_masks[i]
            if anchor_target_probs is None:
                anchor_logits = bc_anchor.logits(xs[i])
                anchor_masked = _masked_logits_1d(anchor_logits, mask)
                anchor_probs = _softmax_1d(anchor_masked)
            else:
                anchor_probs = anchor_target_probs[i]
            anchor_ce_sum += cross_entropy_from_probs(target_probs=anchor_probs, pred_probs=probs[i])
            grad_anchor = grad_ce_wrt_logits_from_probs(
                pred_probs=probs[i],
                target_probs=anchor_probs,
                action_mask=mask,
            )
            dlogits[i] += np.float32(anchor_beta) * grad_anchor.astype(np.float32, copy=False)
            anchor_ce_count += 1

    # Actor head: somma degli outer product h x dlogits in una GEMM.
    gw2 += (hs.T @ dlogits).astype(np.float32, copy=False)
    gb2 += np.sum(dlogits, axis=0, dtype=np.float32)
    dh_policy = dlogits @ policy.w2.T

    # Critic head.
    value_error = value_preds.astype(np.float32, copy=False) - returns_to_go.astype(np.float32, copy=False)
    dv = (np.float32(value_coef) * value_error).astype(np.float32, copy=False)
    value_loss_sum = float(np.sum(0.5 * float(value_coef) * (value_error.astype(np.float64) ** 2)))

    gwv += (hs.T @ dv).astype(np.float32, copy=False)
    gbv = float(np.sum(dv, dtype=np.float64))
    dh_value = dv[:, None] * policy.wv[None, :]

    # Backprop sul trunk condiviso: somma degli outer product x x dz1 in una GEMM.
    dz1 = (dh_policy + dh_value) * (z1s > 0.0)
    gw1 += (xs.T @ dz1).astype(np.float32, copy=False)
    gb1 += np.sum(dz1, axis=0, dtype=np.float32)

    return GradientStats(
        steps=steps,
        value_loss_sum=value_loss_sum,
        anchor_ce_sum=anchor_ce_sum,
        anchor_ce_count=anchor_ce_count,
        gbv=gbv,
    )


def _accumulate_paired_suit_trajectory_grads(
    *,
    policy: A2CPolicy,
    xs: np.ndarray,
    action_masks: np.ndarray,
    action_ids: np.ndarray,
    returns_to_go: np.ndarray,
    encoder_version: EncoderVersion,
    permutation: SuitPermutation,
    entropy_beta: float,
    value_coef: float,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
    gw1: np.ndarray,
    gb1: np.ndarray,
    gw2: np.ndarray,
    gb2: np.ndarray,
    gwv: np.ndarray,
) -> GradientStats:
    """
    Accumula il gradiente della copia rinominata di una traiettoria.

    Return e advantage target restano quelli della traiettoria originale: una rinomina
    globale dei semi non cambia reward o valore strategico. L'anchor, se presente, viene
    anch'esso trasformato dall'orientamento originale, invece di interrogare il teacher
    asimmetrico sulla copia.
    """
    paired = permute_encoded_trajectory(
        xs=xs,
        action_masks=action_masks,
        action_ids=action_ids,
        version=encoder_version,
        permutation=permutation,
    )
    z1s, hs, probs, value_preds = _forward_policy_batch(
        policy,
        xs=paired.xs,
        action_masks=paired.action_masks,
    )

    anchor_targets: np.ndarray | None = None
    if bc_anchor is not None and float(bc_anchor_beta) > 0.0:
        original_targets = _masked_model_probabilities(
            bc_anchor,
            xs=xs,
            action_masks=action_masks,
        )
        anchor_targets = permute_action_vectors(original_targets, permutation=permutation)

    return _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=paired.xs,
        z1s=z1s,
        hs=hs,
        action_masks=paired.action_masks,
        probs=probs,
        action_ids=paired.action_ids,
        value_preds=value_preds,
        returns_to_go=returns_to_go,
        entropy_beta=entropy_beta,
        value_coef=value_coef,
        bc_anchor=bc_anchor,
        bc_anchor_beta=bc_anchor_beta,
        anchor_target_probs=anchor_targets,
        gw1=gw1,
        gb1=gb1,
        gw2=gw2,
        gb2=gb2,
        gwv=gwv,
    )


def _accumulate_suit_consistency_grads(
    *,
    policy: A2CPolicy,
    xs: np.ndarray,
    action_masks: np.ndarray,
    original_probs: np.ndarray,
    encoder_version: EncoderVersion,
    permutation: SuitPermutation,
    beta: float,
    gw1: np.ndarray,
    gb1: np.ndarray,
    gw2: np.ndarray,
    gb2: np.ndarray,
) -> GradientStats:
    """
    Aggiunge ``beta * KL(stopgrad(originale) || copia rinominata)`` al gradiente.

    Il loss A2C resta confinato alla traiettoria realmente campionata. Qui l'output
    originale è un teacher congelato per il solo update corrente; la copia riceve un
    gradiente di coerenza, senza action id, advantage o critic. I contatori ``steps``
    restano quindi a zero: la normalizzazione dell'A2C continua a usare i soli N step
    on-policy e il termine ausiliario viene mediato sugli stessi N step.
    """
    coefficient = float(beta)
    if coefficient <= 0.0:
        raise ValueError("beta della suit consistency deve essere > 0")
    if xs.ndim != 2 or action_masks.shape != (xs.shape[0], 40) or original_probs.shape != action_masks.shape:
        raise ValueError(
            f"Shape consistency invalide: xs={xs.shape}, masks={action_masks.shape}, probs={original_probs.shape}"
        )
    if xs.shape[0] == 0:
        return GradientStats(steps=0, value_loss_sum=0.0, anchor_ce_sum=0.0, anchor_ce_count=0, gbv=0.0)

    paired_xs = permute_encoded_features(xs, version=encoder_version, permutation=permutation)
    paired_masks = permute_action_masks(action_masks, permutation=permutation)
    target_probs = permute_action_vectors(original_probs, permutation=permutation).astype(np.float32, copy=False)
    z1s, hs, paired_probs, _ = _forward_policy_batch(
        policy,
        xs=paired_xs,
        action_masks=paired_masks,
    )

    # Gradiente della cross-entropy; con target stop-gradient coincide col gradiente
    # della forward KL, la metrica che registriamo qui sotto.
    dlogits = np.float32(coefficient) * (paired_probs - target_probs)
    gw2 += (hs.T @ dlogits).astype(np.float32, copy=False)
    gb2 += np.sum(dlogits, axis=0, dtype=np.float32)
    dh = dlogits @ policy.w2.T
    dz1 = dh * (z1s > 0.0)
    gw1 += (paired_xs.T @ dz1).astype(np.float32, copy=False)
    gb1 += np.sum(dz1, axis=0, dtype=np.float32)

    epsilon = np.float32(1e-12)
    kl_per_step = np.sum(
        target_probs
        * (
            np.log(np.maximum(target_probs, epsilon))
            - np.log(np.maximum(paired_probs.astype(np.float32, copy=False), epsilon))
        ),
        axis=1,
        dtype=np.float64,
    )
    return GradientStats(
        steps=0,
        value_loss_sum=0.0,
        anchor_ce_sum=0.0,
        anchor_ce_count=0,
        gbv=0.0,
        suit_consistency_kl_sum=float(np.sum(kl_per_step, dtype=np.float64)),
        suit_consistency_count=int(xs.shape[0]),
    )


def _accumulate_suit_margin_grads(
    *,
    policy: A2CPolicy,
    xs: np.ndarray,
    action_masks: np.ndarray,
    original_probs: np.ndarray,
    encoder_version: EncoderVersion,
    permutation: SuitPermutation,
    beta: float,
    margin_cap: float,
    gw1: np.ndarray,
    gb1: np.ndarray,
    gw2: np.ndarray,
    gb2: np.ndarray,
) -> GradientStats:
    """Aggiunge una hinge loss che preserva carta teacher e margine sotto rinomina."""
    coefficient = float(beta)
    cap = float(margin_cap)
    if coefficient <= 0.0:
        raise ValueError("beta della suit margin loss deve essere > 0")
    if cap <= 0.0:
        raise ValueError("margin_cap deve essere > 0")
    if xs.ndim != 2 or action_masks.shape != (xs.shape[0], 40) or original_probs.shape != action_masks.shape:
        raise ValueError(
            f"Shape margin consistency invalide: xs={xs.shape}, masks={action_masks.shape}, "
            f"probs={original_probs.shape}"
        )

    valid_rows = np.flatnonzero(np.sum(action_masks, axis=1) >= 2)
    if valid_rows.size == 0:
        return GradientStats(steps=0, value_loss_sum=0.0, anchor_ce_sum=0.0, anchor_ce_count=0, gbv=0.0)

    source_probs = original_probs[valid_rows].astype(np.float32, copy=False)
    source_masks = action_masks[valid_rows]
    teacher_ids = np.argmax(source_probs, axis=1).astype(np.int64, copy=False)
    epsilon = np.float32(1e-12)
    source_log_probs = np.log(np.maximum(source_probs, epsilon))
    source_other = np.where(source_masks, source_log_probs, np.float32(-1e9))
    source_other[np.arange(valid_rows.size), teacher_ids] = np.float32(-1e9)
    teacher_margins = source_log_probs[np.arange(valid_rows.size), teacher_ids] - np.max(source_other, axis=1)
    target_margins = np.minimum(teacher_margins, np.float32(cap)).astype(np.float32, copy=False)

    paired_xs = permute_encoded_features(xs[valid_rows], version=encoder_version, permutation=permutation)
    paired_masks = permute_action_masks(source_masks, permutation=permutation)
    paired_teacher_ids = permute_action_ids(teacher_ids, permutation=permutation).astype(np.int64, copy=False)
    z1s, hs, _, _ = _forward_policy_batch(policy, xs=paired_xs, action_masks=paired_masks)
    paired_logits = (hs @ policy.w2 + policy.b2).astype(np.float32, copy=False)
    paired_other = np.where(paired_masks, paired_logits, np.float32(-1e9))
    paired_other[np.arange(valid_rows.size), paired_teacher_ids] = np.float32(-1e9)
    best_other_ids = np.argmax(paired_other, axis=1).astype(np.int64, copy=False)
    student_margins = (
        paired_logits[np.arange(valid_rows.size), paired_teacher_ids]
        - paired_logits[np.arange(valid_rows.size), best_other_ids]
    )
    hinge = np.maximum(target_margins - student_margins, np.float32(0.0))
    violating = hinge > 0.0

    dlogits = np.zeros_like(paired_logits, dtype=np.float32)
    violating_rows = np.flatnonzero(violating)
    dlogits[violating_rows, paired_teacher_ids[violating_rows]] = np.float32(-coefficient)
    dlogits[violating_rows, best_other_ids[violating_rows]] = np.float32(coefficient)
    gw2 += (hs.T @ dlogits).astype(np.float32, copy=False)
    gb2 += np.sum(dlogits, axis=0, dtype=np.float32)
    dh = dlogits @ policy.w2.T
    dz1 = dh * (z1s > 0.0)
    gw1 += (paired_xs.T @ dz1).astype(np.float32, copy=False)
    gb1 += np.sum(dz1, axis=0, dtype=np.float32)

    return GradientStats(
        steps=0,
        value_loss_sum=0.0,
        anchor_ce_sum=0.0,
        anchor_ce_count=0,
        gbv=0.0,
        suit_margin_loss_sum=float(np.sum(hinge, dtype=np.float64)),
        suit_margin_count=int(valid_rows.size),
        suit_margin_violation_count=int(np.count_nonzero(violating)),
        suit_margin_teacher_sum=float(np.sum(target_margins, dtype=np.float64)),
        suit_margin_student_sum=float(np.sum(student_margins, dtype=np.float64)),
    )


def _accumulate_numba_batch_grads(
    *,
    policy: A2CPolicy,
    batch: NumbaA2CBatch,
    gamma: float,
    entropy_beta: float,
    value_coef: float,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
    gw1: np.ndarray,
    gb1: np.ndarray,
    gw2: np.ndarray,
    gb2: np.ndarray,
    gwv: np.ndarray,
    suit_augmentation_rng: np.random.Generator | None = None,
    suit_consistency_rng: np.random.Generator | None = None,
    suit_consistency_beta: float = 0.0,
    suit_margin_rng: np.random.Generator | None = None,
    suit_margin_beta: float = 0.0,
    suit_margin_cap: float = 2.0,
    encoder_version: EncoderVersion | None = None,
    signal_diagnostics: A2CSignalAccumulator | None = None,
) -> GradientStats:
    """
    Appiattisce un batch Numba e accumula i gradienti in una sola chiamata batch.

    `NumbaA2CBatch` conserva un rettangolo `(batch, 20, ...)`; `step_counts` indica
    quante righe sono valide per ogni partita. Qui compattiamo solo quelle righe e
    riusiamo il backprop vettoriale su matrici 2D.
    """
    total_steps = int(np.sum(batch.step_counts, dtype=np.int64))
    if total_steps == 0:
        return GradientStats(steps=0, value_loss_sum=0.0, anchor_ce_sum=0.0, anchor_ce_count=0, gbv=0.0)

    feature_dim = int(batch.xs.shape[2])
    hidden_dim = int(batch.hs.shape[2])
    xs = np.empty((total_steps, feature_dim), dtype=np.float32)
    z1s = np.empty((total_steps, hidden_dim), dtype=np.float32)
    hs = np.empty((total_steps, hidden_dim), dtype=np.float32)
    action_masks = np.empty((total_steps, 40), dtype=bool)
    probs = np.empty((total_steps, 40), dtype=np.float32)
    action_ids = np.empty((total_steps,), dtype=np.int64)
    value_preds = np.empty((total_steps,), dtype=np.float32)
    returns_to_go = np.empty((total_steps,), dtype=np.float32)

    trajectory_slices: list[slice] = []
    offset = 0
    for game_idx, raw_count in enumerate(batch.step_counts):
        count = int(raw_count)
        if count <= 0:
            continue
        sl = slice(offset, offset + count)
        xs[sl] = batch.xs[game_idx, :count]
        z1s[sl] = batch.z1s[game_idx, :count]
        hs[sl] = batch.hs[game_idx, :count]
        action_masks[sl] = batch.action_masks[game_idx, :count]
        probs[sl] = batch.probs[game_idx, :count]
        action_ids[sl] = batch.action_ids[game_idx, :count]
        value_preds[sl] = batch.value_preds[game_idx, :count]
        returns_to_go[sl] = _compute_returns_array(batch.rewards[game_idx, :count], gamma=gamma)
        trajectory_slices.append(sl)
        offset += count

    stats = _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=action_masks,
        probs=probs,
        action_ids=action_ids,
        value_preds=value_preds,
        returns_to_go=returns_to_go,
        entropy_beta=entropy_beta,
        value_coef=value_coef,
        bc_anchor=bc_anchor,
        bc_anchor_beta=bc_anchor_beta,
        gw1=gw1,
        gb1=gb1,
        gw2=gw2,
        gb2=gb2,
        gwv=gwv,
        signal_diagnostics=signal_diagnostics,
    )
    if suit_augmentation_rng is None and suit_consistency_rng is None and suit_margin_rng is None:
        return stats
    if encoder_version is None:
        raise ValueError("encoder_version obbligatorio con trasformazioni dei semi")

    for trajectory_slice in trajectory_slices:
        if suit_augmentation_rng is not None:
            paired_stats = _accumulate_paired_suit_trajectory_grads(
                policy=policy,
                xs=xs[trajectory_slice],
                action_masks=action_masks[trajectory_slice],
                action_ids=action_ids[trajectory_slice],
                returns_to_go=returns_to_go[trajectory_slice],
                encoder_version=encoder_version,
                permutation=sample_nonidentity_suit_permutation(suit_augmentation_rng),
                entropy_beta=entropy_beta,
                value_coef=value_coef,
                bc_anchor=bc_anchor,
                bc_anchor_beta=bc_anchor_beta,
                gw1=gw1,
                gb1=gb1,
                gw2=gw2,
                gb2=gb2,
                gwv=gwv,
            )
            stats = _add_gradient_stats(stats, paired_stats)
        if suit_consistency_rng is not None:
            consistency_stats = _accumulate_suit_consistency_grads(
                policy=policy,
                xs=xs[trajectory_slice],
                action_masks=action_masks[trajectory_slice],
                original_probs=probs[trajectory_slice],
                encoder_version=encoder_version,
                permutation=sample_nonidentity_suit_permutation(suit_consistency_rng),
                beta=suit_consistency_beta,
                gw1=gw1,
                gb1=gb1,
                gw2=gw2,
                gb2=gb2,
            )
            stats = _add_gradient_stats(stats, consistency_stats)
        if suit_margin_rng is not None:
            margin_stats = _accumulate_suit_margin_grads(
                policy=policy,
                xs=xs[trajectory_slice],
                action_masks=action_masks[trajectory_slice],
                original_probs=probs[trajectory_slice],
                encoder_version=encoder_version,
                permutation=sample_nonidentity_suit_permutation(suit_margin_rng),
                beta=suit_margin_beta,
                margin_cap=suit_margin_cap,
                gw1=gw1,
                gb1=gb1,
                gw2=gw2,
                gb2=gb2,
            )
            stats = _add_gradient_stats(stats, margin_stats)
    return stats


@dataclass
class TrainMetrics:
    """Metriche aggregate (logging)."""

    iter: int
    games: int
    avg_return: float
    win_rate: float
    draw_rate: float
    avg_entropy: float
    value_loss: float
    avg_anchor_ce: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RL A2C (MLP, 40 carte + action mask) con reward shaping")
    parser.add_argument("--out", required=True, help="Path output modello (.npz)")
    parser.add_argument("--init", default="", help="Warm-start da un modello `.npz` MLP (es. BC/RL).")
    parser.add_argument(
        "--resume",
        default="",
        help=(
            "Riprende esattamente un checkpoint A2C long-run: policy, critic, Adam, RNG, schedule e metriche. "
            "È mutuamente esclusivo con --init."
        ),
    )
    parser.add_argument(
        "--encoder-version",
        choices=["v1", "v2", "v3", "v4"],
        default="v1",
        help=(
            "Versione encoder per observation 2-player. "
            "v1=istantaneo (248 dim), v2=v1 + seen_cards_onehot[40] (288 dim, storia pubblica), "
            "v3=v2 + feature strategiche aggregate (310 dim, solo engine domain)."
        ),
    )
    parser.add_argument(
        "--upgrade-init-v1-to-v2",
        action="store_true",
        help=(
            "Se usi `--encoder-version v2` e `--init` è un modello v1, "
            "espande `w1` aggiungendo 40 righe a zero (warm-start compatibile)."
        ),
    )
    parser.add_argument(
        "--opponent",
        default="heuristic_v1",
        help=(
            "Nome avversario (se non usi --opponent-mix). "
            "Esempi: heuristic_v1, random, greedy_points, best_a2c "
            "(alias che carica `best_a2c.npz` dalla directory modelli)."
        ),
    )
    parser.add_argument(
        "--opponent-mix",
        default="",
        help=(
            "Miscela avversari: `name:weight,name:weight,...` "
            "(es. `heuristic_v1:0.7,random:0.2,greedy_points:0.1`). "
            "Se presente, sovrascrive `--opponent`."
        ),
    )
    parser.add_argument(
        "--opponent-model",
        default="",
        help=(
            "Path al modello `.npz` quando `--opponent bc_model` o `bc_model_value_lookahead_8x8` "
            "(supportato nel rollout domain e fast-rollout numba)."
        ),
    )
    parser.add_argument(
        "--policy-belief-model",
        default="",
        help=(
            "Belief model .npz (belief_mlp_v1, encoder v4) come INPUT della policy "
            "(iterazione 1b): input = v4 (369) + 40 probabilita' belief = 409. La belief "
            "resta CONGELATA (non allenata) e viene salvata dentro l'artefatto finale."
        ),
    )
    parser.add_argument(
        "--opponent-belief-model",
        default="",
        help=(
            "Belief network `.npz` quando `--opponent bc_model_pimc_belief` (maestro PIMC nel "
            "fast rollout Numba): pesa le determinizzazioni della search del maestro."
        ),
    )
    parser.add_argument(
        "--opponent-pimc-determinizations",
        type=int,
        default=32,
        help="Determinizzazioni della search del maestro PIMC (default 32).",
    )
    parser.add_argument(
        "--opponent-value-model",
        default=str(Path("data/models") / VALUE_LOOKAHEAD_MODEL_ID),
        help=("Path al value model `.npz` quando l'opponent è `bc_model_value_lookahead_8x8` nel fast rollout Numba."),
    )
    parser.add_argument(
        "--opponent-value-max-unknown-cards",
        type=int,
        default=8,
        help=(
            "Finestra dell'opponent value-lookahead nel fast rollout Numba: usa lookahead se "
            "mano avversaria + mazzo <= questo valore; altrimenti fallback MLP."
        ),
    )
    parser.add_argument("--num-games", type=int, default=20000, help="Numero partite di training (2-player).")
    parser.add_argument(
        "--stop-after-games",
        type=int,
        default=0,
        help=(
            "Ferma questo processo al conteggio cumulativo indicato, mantenendo --num-games come orizzonte totale. "
            "Serve per blocchi riprendibili bit-identici; 0 significa arrivare a --num-games."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità).")
    parser.add_argument(
        "--rollout-engine",
        choices=["domain", "fast"],
        default="domain",
        help=(
            "Motore rollout training. `domain` è canonico e supporta tutti gli agenti; `fast` è sperimentale "
            "e supporta solo avversari fast-compatible "
            "random/greedy_points/heuristic_v1/heuristic_v2/heuristic_trump_saver."
        ),
    )
    parser.add_argument(
        "--fast-encoder",
        choices=["python", "numba"],
        default="python",
        help=(
            "Encoder osservazione usato solo con `--rollout-engine fast --fast-rollout python`. "
            "`python` è il path stabile; `numba` usa il wrapper JIT sperimentale equivalente."
        ),
    )
    parser.add_argument(
        "--fast-rollout",
        choices=["python", "numba"],
        default="python",
        help=(
            "Loop rollout usato solo con `--rollout-engine fast`. "
            "`python` usa Fast2PState/list Python; `numba` raccoglie la traiettoria A2C in un core full-JIT."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dim (se non si usa --init).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay (solo pesi).")
    parser.add_argument("--entropy-beta", type=float, default=5e-4, help="Entropia bonus (>=0).")
    parser.add_argument(
        "--bc-anchor",
        default="",
        help=(
            "Path a un modello `.npz` usato come anchor fisso (tipicamente un BC teacher). "
            "Se valorizzato e `--bc-anchor-beta > 0`, aggiunge una regolarizzazione (cross-entropy) "
            "che mantiene la policy vicina all'anchor (utile per preservare stile anti-overkill senza guard)."
        ),
    )
    parser.add_argument(
        "--bc-anchor-beta",
        type=float,
        default=0.0,
        help=("Peso (>=0) della regolarizzazione verso l'anchor BC. Valori tipici: 0.005..0.05. Se 0, disattivata."),
    )
    parser.add_argument(
        "--overkill-penalty-mode",
        choices=["flat", "gap"],
        default="flat",
        help=(
            "Modalità penalità overkill briscola: "
            "`flat` aggiunge `-beta` quando overkill, `gap` aggiunge `-beta * gap_norm` (più informativa)."
        ),
    )
    parser.add_argument(
        "--overkill-penalty-beta",
        type=float,
        default=0.0,
        help=(
            "Penalità flat (>=0) per scoraggiare 'overkill briscola' da secondi di mano. "
            "Se >0 e la policy vince con una briscola pur avendo una briscola vincente più economica, "
            "aggiungiamo `-beta` al reward (soft shaping)."
        ),
    )
    parser.add_argument(
        "--overkill-low-lead-points-max",
        type=int,
        default=2,
        help=(
            "Applica la penalità overkill solo se la carta avversaria sul tavolo vale "
            "al massimo questo numero di punti. "
            "Default: 2 (scarti o quasi)."
        ),
    )
    parser.add_argument(
        "--inference-overkill-guard",
        action="store_true",
        help=(
            "Salva nei metadati del modello un flag per abilitare, a inference-time, "
            "un post-processing anti-overkill: se stiamo per vincere con una briscola da secondi di mano, "
            "giochiamo automaticamente la briscola vincente minima disponibile."
        ),
    )
    parser.add_argument("--value-coef", type=float, default=0.5, help="Peso loss critic (MSE).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Fattore di sconto per return-to-go (default: 1.0).")
    parser.add_argument(
        "--suit-augmentation",
        choices=["off", "paired"],
        default="off",
        help=(
            "Augmentation dei semi nel loss A2C. `paired` affianca a ogni traiettoria una copia "
            "con una rinomina non-identità coerente; il rollout e il costo inference non cambiano."
        ),
    )
    parser.add_argument(
        "--suit-consistency-beta",
        type=float,
        default=0.0,
        help=(
            "Peso (>=0) della forward-KL sui semi: l'output originale, con stop-gradient, "
            "diventa target della copia rinominata. Non duplica il policy gradient A2C."
        ),
    )
    parser.add_argument(
        "--suit-margin-beta",
        type=float,
        default=0.0,
        help=(
            "Peso (>=0) della hinge sui semi: la carta argmax originale deve conservare "
            "il proprio margine, limitato da --suit-margin-cap, nella copia rinominata."
        ),
    )
    parser.add_argument(
        "--suit-margin-cap",
        type=float,
        default=2.0,
        help="Massimo margine logit richiesto dalla suit margin loss (default 2.0).",
    )
    parser.add_argument("--update-every", type=int, default=20, help="Aggiorna i pesi ogni N partite (batch).")
    parser.add_argument("--log-every", type=int, default=200, help="Stampa metriche ogni N update.")
    parser.add_argument(
        "--checkpoint-games",
        default="",
        help=(
            "Lista separata da virgole di conteggi partita a cui salvare checkpoint `.npz` "
            "(es. `1000000,3000000,5000000`). Ogni valore deve essere multiplo di `--update-every`."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Directory per i checkpoint intermedi. Default: directory di `--out`.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        default="",
        help="Prefisso filename checkpoint. Default: stem di `--out`.",
    )
    parser.add_argument(
        "--metrics-mode",
        choices=["full", "summary"],
        default="full",
        help=(
            "Come salvare le metriche nel metadata_json. `full` mantiene ogni update; "
            "`summary` salva solo conteggio/prima/ultima riga, utile per run multi-milione."
        ),
    )
    parser.add_argument(
        "--diagnostics-json",
        default="",
        help=(
            "Report JSON passivo per update: advantage, critic, attivazioni, gradienti e passo Adam. "
            "Vuoto disabilita la sonda; non modifica il training."
        ),
    )
    parser.add_argument(
        "--diagnostics-every",
        type=int,
        default=1,
        help=(
            "Conserva una riga diagnostica ogni N optimizer update, includendo sempre primo, checkpoint e ultimo "
            "update del segmento (default 1)."
        ),
    )
    parser.add_argument(
        "--training-schedule",
        choices=["serial", "paired"],
        default="serial",
        help=(
            "Schedule degli ambienti. `serial` mantiene un seed/opponent per partita; "
            "`paired` ripete seed e opponent su seat 0/1 dentro lo stesso update."
        ),
    )
    parser.add_argument("--seat-fair", action="store_true", help="Alterna la seat della policy (riduce bias player0).")
    args = parser.parse_args()

    if args.num_games <= 0:
        raise ValueError("--num-games deve essere > 0")
    if str(args.init).strip() and str(args.resume).strip():
        raise ValueError("--init e --resume sono mutuamente esclusivi")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim deve essere > 0")
    if args.update_every <= 0:
        raise ValueError("--update-every deve essere > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every deve essere > 0")
    if args.diagnostics_every <= 0:
        raise ValueError("--diagnostics-every deve essere > 0")
    num_games = int(args.num_games)
    stop_after_games = int(args.stop_after_games) if int(args.stop_after_games) > 0 else num_games
    if stop_after_games <= 0 or stop_after_games > num_games:
        raise ValueError("--stop-after-games deve essere in (0, --num-games]")
    if stop_after_games < num_games and stop_after_games % int(args.update_every) != 0:
        raise ValueError("--stop-after-games intermedio deve essere multiplo di --update-every")
    training_schedule_mode: TrainingScheduleMode = str(args.training_schedule)
    if training_schedule_mode == "paired":
        if num_games % 2 != 0:
            raise ValueError("--training-schedule paired richiede --num-games pari")
        if int(args.update_every) % 2 != 0:
            raise ValueError("--training-schedule paired richiede --update-every pari")
        if num_games % int(args.update_every) != 0:
            raise ValueError(
                "--training-schedule paired richiede --num-games multiplo di --update-every, "
                "per evitare update parziali"
            )
    if float(args.gamma) <= 0.0 or float(args.gamma) > 1.0:
        raise ValueError("--gamma deve essere in (0,1]")
    if float(args.overkill_penalty_beta) < 0.0:
        raise ValueError("--overkill-penalty-beta deve essere >= 0")
    if int(args.overkill_low_lead_points_max) < 0:
        raise ValueError("--overkill-low-lead-points-max deve essere >= 0")
    if float(args.bc_anchor_beta) < 0.0:
        raise ValueError("--bc-anchor-beta deve essere >= 0")
    if float(args.suit_consistency_beta) < 0.0:
        raise ValueError("--suit-consistency-beta deve essere >= 0")
    if float(args.suit_margin_beta) < 0.0:
        raise ValueError("--suit-margin-beta deve essere >= 0")
    if float(args.suit_margin_cap) <= 0.0:
        raise ValueError("--suit-margin-cap deve essere > 0")
    active_suit_losses = sum(
        (
            str(args.suit_augmentation) == "paired",
            float(args.suit_consistency_beta) > 0.0,
            float(args.suit_margin_beta) > 0.0,
        )
    )
    if active_suit_losses > 1:
        raise ValueError("Provare separatamente paired, forward-KL e suit margin loss")
    if float(args.bc_anchor_beta) > 0.0 and not str(args.bc_anchor).strip():
        raise ValueError("Se `--bc-anchor-beta > 0` devi impostare anche `--bc-anchor <path.npz>`.")
    if int(args.opponent_value_max_unknown_cards) < 0:
        raise ValueError("--opponent-value-max-unknown-cards deve essere >= 0")
    rollout_engine = str(args.rollout_engine)
    fast_encoder = str(args.fast_encoder)
    fast_rollout = str(args.fast_rollout)
    if rollout_engine != "fast" and fast_encoder != "python":
        raise ValueError("`--fast-encoder numba` richiede `--rollout-engine fast`.")
    if rollout_engine != "fast" and fast_rollout != "python":
        raise ValueError("`--fast-rollout numba` richiede `--rollout-engine fast`.")
    if rollout_engine == "fast" and fast_rollout != "numba" and float(args.overkill_penalty_beta) > 0.0:
        raise ValueError(
            "`--rollout-engine fast --fast-rollout python` non supporta `--overkill-penalty-beta > 0`; "
            "usa `--fast-rollout numba` oppure `--rollout-engine domain`."
        )

    out_path = Path(args.out)
    resume_path = Path(str(args.resume).strip()) if str(args.resume).strip() else None
    resume_state: dict[str, object] | None = None
    resume_arrays: dict[str, np.ndarray] = {}
    if resume_path is not None:
        required_resume_arrays = {
            "w1",
            "b1",
            "w2",
            "b2",
            "wv",
            "bv",
            "resume_st_w1_m",
            "resume_st_w1_v",
            "resume_st_b1_m",
            "resume_st_b1_v",
            "resume_st_w2_m",
            "resume_st_w2_v",
            "resume_st_b2_m",
            "resume_st_b2_v",
            "resume_st_wv_m",
            "resume_st_wv_v",
            "resume_st_bv_m",
            "resume_st_bv_v",
            "resume_state_json",
        }
        with np.load(resume_path, allow_pickle=False) as archive:
            missing = sorted(required_resume_arrays.difference(archive.files))
            if missing:
                raise ValueError(f"Checkpoint non riprendibile, array mancanti: {missing}")
            resume_state = parse_resume_json(archive["resume_state_json"])
            resume_arrays = {
                name: np.asarray(archive[name]).copy() for name in required_resume_arrays if name != "resume_state_json"
            }
        completed = int(str(resume_state.get("games_completed", -1)))
        if completed <= 0 or completed % int(args.update_every) != 0:
            raise ValueError("Il checkpoint deve trovarsi dopo un optimizer update completo")
        if completed >= stop_after_games:
            raise ValueError(f"Il checkpoint contiene già {completed} partite; --stop-after-games deve essere maggiore")
    diagnostics_path = Path(str(args.diagnostics_json).strip()) if str(args.diagnostics_json).strip() else None
    if diagnostics_path is not None and diagnostics_path.resolve() == out_path.resolve():
        raise ValueError("--diagnostics-json deve essere diverso da --out")
    raw_checkpoint_games = str(args.checkpoint_games).strip()
    checkpoint_games: set[int] = set()
    if raw_checkpoint_games:
        for raw_item in raw_checkpoint_games.split(","):
            item = raw_item.strip()
            if not item:
                continue
            try:
                checkpoint_game = int(item)
            except ValueError as exc:
                raise ValueError(f"Checkpoint non valido in --checkpoint-games: {item!r}") from exc
            if checkpoint_game <= 0:
                raise ValueError("--checkpoint-games deve contenere solo valori > 0")
            if checkpoint_game > stop_after_games:
                raise ValueError("--checkpoint-games non può superare --stop-after-games del segmento")
            if resume_state is not None and checkpoint_game <= int(str(resume_state["games_completed"])):
                raise ValueError("--checkpoint-games deve contenere solo checkpoint successivi al resume")
            if checkpoint_game % int(args.update_every) != 0:
                raise ValueError("Ogni checkpoint deve essere multiplo di --update-every, per salvare dopo un update.")
            checkpoint_games.add(checkpoint_game)
    checkpoint_dir = Path(str(args.checkpoint_dir).strip()) if str(args.checkpoint_dir).strip() else out_path.parent
    checkpoint_prefix = str(args.checkpoint_prefix).strip() or out_path.stem
    encoder_version: EncoderVersion = str(args.encoder_version)
    rng_action = np.random.default_rng(args.seed)
    rng_game = np.random.default_rng(args.seed ^ 0x9E3779B9)
    rng_opponent_select = np.random.default_rng(args.seed ^ 0xA5A5A5A5)
    rng_opponent = random.Random(args.seed ^ 0xC0FFEE)
    suit_augmentation = str(args.suit_augmentation)
    rng_suit_augmentation = np.random.default_rng(args.seed ^ 0x51A17A9E) if suit_augmentation == "paired" else None
    suit_consistency_beta = float(args.suit_consistency_beta)
    rng_suit_consistency = np.random.default_rng(args.seed ^ 0x0C05157E) if suit_consistency_beta > 0.0 else None
    suit_margin_beta = float(args.suit_margin_beta)
    suit_margin_cap = float(args.suit_margin_cap)
    rng_suit_margin = np.random.default_rng(args.seed ^ 0x0A461A9E) if suit_margin_beta > 0.0 else None

    if resume_state is not None:
        raw_rng = resume_state.get("rng")
        if not isinstance(raw_rng, dict):
            raise ValueError("Checkpoint senza stato RNG")
        try:
            rng_action.bit_generator.state = raw_rng["action"]
            rng_game.bit_generator.state = raw_rng["game"]
            rng_opponent_select.bit_generator.state = raw_rng["opponent_select"]
            rng_opponent.setstate(tuple_tree(raw_rng["opponent_python"]))
            optional_rngs = {
                "suit_augmentation": rng_suit_augmentation,
                "suit_consistency": rng_suit_consistency,
                "suit_margin": rng_suit_margin,
            }
            for name, rng in optional_rngs.items():
                state = raw_rng.get(name)
                if (rng is None) != (state is None):
                    raise ValueError(f"Configurazione RNG {name} diversa dal checkpoint")
                if rng is not None:
                    if not isinstance(state, dict):
                        raise ValueError(f"Stato RNG {name} incompatibile")
                    rng.bit_generator.state = state
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stato RNG del checkpoint incompatibile") from exc

    opponent_pool: OpponentPool | None = None
    fast_numba_model_opponent: FastNumbaModelOpponent | None = None
    fast_numba_value_lookahead_opponent: FastNumbaValueLookaheadOpponent | None = None
    fast_numba_model_mix_name: str | None = None
    fast_numba_value_mix_name: str | None = None
    fast_numba_pimc_belief_name: str | None = None
    fast_numba_opponent_belief = None
    opponent_mix_raw = args.opponent_mix.strip()
    if opponent_mix_raw:
        items = parse_opponent_mix(opponent_mix_raw)
        if rollout_engine == "fast":
            model_mix_names = [item.name for item in items if item.name in {"best_a2c", "bc_model"}]
            value_mix_names = [item.name for item in items if item.name == "bc_model_value_lookahead_8x8"]
            pimc_mix_names = [item.name for item in items if item.name == "bc_model_pimc_belief"]
            unsupported = [
                item.name
                for item in items
                if item.name not in FAST_EVALUATION_AGENT_NAMES
                and not (
                    fast_rollout == "numba"
                    and item.name in {"best_a2c", "bc_model", "bc_model_value_lookahead_8x8", "bc_model_pimc_belief"}
                )
            ]
            if unsupported:
                supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
                raise ValueError(
                    f"`--rollout-engine fast` supporta opponent mix con: {supported}; "
                    "`best_a2c`/`bc_model`/`bc_model_value_lookahead_8x8` sono supportati solo "
                    "con `--fast-rollout numba`. "
                    f"Non supportati: {unsupported}"
                )
            if value_mix_names and "best_a2c" in model_mix_names:
                raise ValueError(
                    "`bc_model_value_lookahead_8x8` in opponent mix può essere combinato con `bc_model` "
                    "o baseline rule-based, ma non con `best_a2c`: il fast batch usa un solo set di pesi modello."
                )
            if len(set(model_mix_names)) > 1:
                raise ValueError(
                    "`--opponent-mix` fast Numba supporta al massimo un tipo di opponent modello "
                    "(`best_a2c` oppure `bc_model`) per batch."
                )
            if len(set(value_mix_names)) > 1:
                raise ValueError("`--opponent-mix` contiene più varianti value-lookahead non supportate.")
            if fast_rollout == "numba" and value_mix_names:
                fast_numba_value_mix_name = value_mix_names[0]
                fast_numba_value_lookahead_opponent = _load_fast_numba_value_lookahead_opponent(
                    opponent_model_path=str(args.opponent_model),
                    opponent_value_model_path=str(args.opponent_value_model),
                    max_unknown_cards=int(args.opponent_value_max_unknown_cards),
                )
            if fast_rollout == "numba" and pimc_mix_names:
                # Maestro PIMC belief nel mix: belief obbligatoria; il container condivide i pesi
                # opponent (se il VL non e' nel mix, value dummy — mode 3 non lo legge).
                belief_path = str(args.opponent_belief_model).strip()
                if not belief_path:
                    raise ValueError("`bc_model_pimc_belief` nel mix richiede `--opponent-belief-model <path.npz>`.")
                fast_numba_pimc_belief_name = pimc_mix_names[0]
                fast_numba_opponent_belief = load_belief_model_npz(belief_path)
                if fast_numba_value_lookahead_opponent is None:
                    fast_numba_value_lookahead_opponent = _load_fast_numba_pimc_belief_opponent(
                        opponent_model_path=str(args.opponent_model),
                        opponent_belief_model_path=belief_path,
                        max_unknown_cards=int(args.opponent_value_max_unknown_cards),
                    )
            if fast_rollout == "numba" and model_mix_names:
                fast_numba_model_mix_name = model_mix_names[0]
                if fast_numba_value_lookahead_opponent is not None and fast_numba_model_mix_name == "bc_model":
                    fast_numba_model_opponent = FastNumbaModelOpponent(
                        agent=fast_numba_value_lookahead_opponent.agent,
                        model=fast_numba_value_lookahead_opponent.model,
                    )
                else:
                    fast_numba_model_opponent = _load_fast_numba_model_opponent(
                        opponent_name=fast_numba_model_mix_name,
                        opponent_model_path=str(args.opponent_model),
                    )
        agents_by_name = {}
        for item in items:
            if item.name == "bc_model":
                if fast_numba_model_opponent is None or (
                    fast_numba_model_mix_name != "bc_model" and fast_numba_value_lookahead_opponent is None
                ):
                    raise ValueError("`bc_model` in `--opponent-mix` richiede fast Numba e `--opponent-model`.")
                agents_by_name[item.name] = NamedAgentProxy(item.name, fast_numba_model_opponent.agent)
            elif item.name == "bc_model_pimc_belief":
                if fast_numba_value_lookahead_opponent is None or fast_numba_opponent_belief is None:
                    raise ValueError(
                        "`bc_model_pimc_belief` in `--opponent-mix` richiede fast Numba, "
                        "`--opponent-model` e `--opponent-belief-model`."
                    )
                agents_by_name[item.name] = NamedAgentProxy(item.name, fast_numba_value_lookahead_opponent.agent)
            elif item.name == "bc_model_value_lookahead_8x8":
                if rollout_engine == "fast":
                    if fast_numba_value_lookahead_opponent is None:
                        raise ValueError(
                            "`bc_model_value_lookahead_8x8` in `--opponent-mix` richiede fast Numba, "
                            "`--opponent-model` e `--opponent-value-model`."
                        )
                    agents_by_name[item.name] = NamedAgentProxy(
                        item.name,
                        fast_numba_value_lookahead_opponent.agent,
                    )
                else:
                    if not str(args.opponent_model).strip():
                        raise ValueError(
                            "`bc_model_value_lookahead_8x8` in `--opponent-mix` richiede `--opponent-model <path.npz>`."
                        )
                    agents_by_name[item.name] = build_agent(
                        item.name,
                        model_path=Path(str(args.opponent_model).strip()),
                    )
            else:
                agents_by_name[item.name] = build_agent(item.name)
        opponent_pool = OpponentPool(items=items, agents_by_name=agents_by_name)
        opponent = agents_by_name[items[0].name]
    else:
        opponent_name = str(args.opponent)
        if rollout_engine == "fast" and opponent_name == "bc_model_pimc_belief":
            if fast_rollout != "numba":
                raise ValueError("Opponent PIMC belief nel fast path richiede `--fast-rollout numba`.")
            belief_path = str(args.opponent_belief_model).strip()
            if not belief_path:
                raise ValueError("`--opponent bc_model_pimc_belief` richiede `--opponent-belief-model <path.npz>`.")
            fast_numba_pimc_belief_name = opponent_name
            fast_numba_opponent_belief = load_belief_model_npz(belief_path)
            fast_numba_value_lookahead_opponent = _load_fast_numba_pimc_belief_opponent(
                opponent_model_path=str(args.opponent_model),
                opponent_belief_model_path=belief_path,
                max_unknown_cards=int(args.opponent_value_max_unknown_cards),
            )
            fast_numba_model_opponent = FastNumbaModelOpponent(
                agent=fast_numba_value_lookahead_opponent.agent,
                model=fast_numba_value_lookahead_opponent.model,
            )
            opponent = NamedAgentProxy(opponent_name, fast_numba_value_lookahead_opponent.agent)
        elif rollout_engine == "fast" and opponent_name == "bc_model_value_lookahead_8x8":
            if fast_rollout != "numba":
                raise ValueError("Opponent value-lookahead nel fast path richiede `--fast-rollout numba`.")
            fast_numba_value_mix_name = opponent_name
            fast_numba_value_lookahead_opponent = _load_fast_numba_value_lookahead_opponent(
                opponent_model_path=str(args.opponent_model),
                opponent_value_model_path=str(args.opponent_value_model),
                max_unknown_cards=int(args.opponent_value_max_unknown_cards),
            )
            fast_numba_model_opponent = FastNumbaModelOpponent(
                agent=fast_numba_value_lookahead_opponent.agent,
                model=fast_numba_value_lookahead_opponent.model,
            )
            opponent = NamedAgentProxy(opponent_name, fast_numba_value_lookahead_opponent.agent)
        elif rollout_engine == "fast" and opponent_name in {"best_a2c", "bc_model"}:
            if fast_rollout != "numba":
                raise ValueError("Opponent `.npz` nel fast path richiede `--fast-rollout numba`.")
            fast_numba_model_opponent = _load_fast_numba_model_opponent(
                opponent_name=opponent_name,
                opponent_model_path=str(args.opponent_model),
            )
            opponent = NamedAgentProxy(opponent_name, fast_numba_model_opponent.agent)
        elif rollout_engine == "fast" and opponent_name not in FAST_EVALUATION_AGENT_NAMES:
            supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
            raise ValueError(
                f"`--rollout-engine fast` supporta avversari fast-compatible ({supported}) "
                "oppure `best_a2c`/`bc_model`/`bc_model_value_lookahead_8x8` con `--fast-rollout numba`. "
                f"Ottenuto: {args.opponent!r}"
            )
        elif opponent_name == "bc_model_value_lookahead_8x8":
            if not str(args.opponent_model).strip():
                raise ValueError("`--opponent bc_model_value_lookahead_8x8` richiede `--opponent-model <path.npz>`.")
            opponent = build_agent(opponent_name, model_path=Path(str(args.opponent_model).strip()))
        elif opponent_name == "bc_model":
            if not str(args.opponent_model).strip():
                raise ValueError("`--opponent bc_model` richiede `--opponent-model <path.npz>`.")
            opponent = NamedAgentProxy(
                "bc_model", build_agent("bc_model", model_path=Path(str(args.opponent_model).strip()))
            )
        else:
            opponent = build_agent(opponent_name)

    resume_games = int(str(resume_state["games_completed"])) if resume_state is not None else 0
    raw_schedule_state = resume_state.get("schedule") if resume_state is not None else None
    if raw_schedule_state is not None and not isinstance(raw_schedule_state, dict):
        raise ValueError("Stato schedule del checkpoint incompatibile")
    schedule_digest = str(raw_schedule_state["sha256"]) if raw_schedule_state is not None else None
    schedule_consumed = int(raw_schedule_state["consumed_games"]) if raw_schedule_state is not None else 0
    if schedule_consumed != resume_games:
        raise ValueError("Cursore schedule e conteggio partite del checkpoint non coincidono")
    training_schedule = TrainingGameScheduleStream(
        mode=training_schedule_mode,
        seat_fair=bool(args.seat_fair),
        default_opponent_name=opponent.name,
        opponent_mix=opponent_pool.items if opponent_pool is not None else None,
        rng_game=rng_game,
        rng_opponent=rng_opponent_select,
        consumed_games=schedule_consumed,
        digest_hex=schedule_digest,
    )
    effective_seat_fair = bool(args.seat_fair) or training_schedule_mode == "paired"

    # Inizializzazione policy/critic.
    policy_belief = None
    policy_belief_path = str(args.policy_belief_model).strip()
    if (suit_augmentation == "paired" or suit_consistency_beta > 0.0 or suit_margin_beta > 0.0) and policy_belief_path:
        raise ValueError(
            "Le trasformazioni dei semi non supportano ancora --policy-belief-model: "
            "manca il contratto di permutazione delle 40 probabilità belief embedded."
        )
    if policy_belief_path:
        if encoder_version != "v4":
            raise ValueError("--policy-belief-model richiede --encoder-version v4 (la belief legge feature v4).")
        policy_belief = load_belief_model_npz(policy_belief_path)

    target_feature_dim = int(feature_dim_for_encoder_version(encoder_version))
    if policy_belief is not None:
        # Iterazione 1b: l'input della policy e' encoder v4 + 40 probabilita' belief.
        target_feature_dim += 40
    init_path = Path(args.init.strip()) if args.init.strip() else None
    init_contains_critic = False
    init_critic_shapes: dict[str, list[int]] | None = None
    original_init_artifact: dict[str, str | int] | None = None
    if resume_state is not None:
        raw_initialization = resume_state.get("initialization")
        if not isinstance(raw_initialization, dict):
            raise ValueError("Checkpoint senza provenienza dell'inizializzazione")
        init_contains_critic = bool(raw_initialization.get("init_contains_critic", False))
        raw_shapes = raw_initialization.get("init_critic_shapes")
        init_critic_shapes = raw_shapes if isinstance(raw_shapes, dict) else None
        raw_artifact = raw_initialization.get("init_artifact")
        original_init_artifact = raw_artifact if isinstance(raw_artifact, dict) else None
        w1 = resume_arrays["w1"].copy()
        b1 = resume_arrays["b1"].copy()
        w2 = resume_arrays["w2"].copy()
        b2 = resume_arrays["b2"].copy()
        wv = resume_arrays["wv"].copy()
        bv_values = resume_arrays["bv"].reshape(-1)
        if bv_values.size != 1:
            raise ValueError("Checkpoint con bias critic non scalare")
        bv = np.float32(bv_values[0])
        hdim = int(w1.shape[1])
        if int(w1.shape[0]) != target_feature_dim:
            raise ValueError(
                f"Feature dim del resume {int(w1.shape[0])}, attesa {target_feature_dim} per {encoder_version}"
            )
        if b1.shape != (hdim,) or w2.shape != (hdim, 40) or b2.shape != (40,) or wv.shape != (hdim,):
            raise ValueError("Shape policy/critic incompatibili nel checkpoint")
    elif init_path is not None:
        with np.load(init_path, allow_pickle=False) as init_archive:
            init_contains_critic = "wv" in init_archive and "bv" in init_archive
            if init_contains_critic:
                init_critic_shapes = {
                    "wv": list(np.asarray(init_archive["wv"]).shape),
                    "bv": list(np.asarray(init_archive["bv"]).shape),
                }
        loaded = load_bc_model_npz(init_path)
        if not isinstance(loaded, MLPBCModel):
            raise ValueError("--init deve puntare a un modello MLP (w1/b1/w2/b2).")
        w1 = loaded.w1.copy()
        b1 = loaded.b1.copy()
        w2 = loaded.w2.copy()
        b2 = loaded.b2.copy()
        hdim = int(w1.shape[1])
        init_dim = int(w1.shape[0])
        if init_dim != target_feature_dim:
            if (
                bool(args.upgrade_init_v1_to_v2)
                and init_dim == int(FEATURE_DIM_2P_V1)
                and target_feature_dim == int(FEATURE_DIM_2P_V2)
            ) or (
                # Pad automatico v4 -> v4+belief: le 40 feature belief partono a peso zero,
                # quindi la policy inizializzata e' ESATTAMENTE il modello di init.
                policy_belief is not None
                and init_dim == int(FEATURE_DIM_2P_V4)
                and target_feature_dim == int(FEATURE_DIM_2P_V4) + 40
            ):
                pad = np.zeros((target_feature_dim - init_dim, hdim), dtype=np.float32)
                w1 = np.vstack([w1, pad])
            else:
                raise ValueError(
                    "Feature dim mismatch tra `--init` e encoder scelto: "
                    f"init={init_dim} target={target_feature_dim} (encoder={encoder_version}). "
                    "Soluzioni: usa `--encoder-version` coerente, oppure abilita `--upgrade-init-v1-to-v2`."
                )
        original_init_artifact = _artifact(init_path)
        # Critic head: il warm-start storico usa sempre un critic vicino a zero.
        wv = np.zeros((hdim,), dtype=np.float32)
        bv = np.float32(0.0)
    else:
        hdim = int(args.hidden_dim)
        w1 = rng_action.normal(loc=0.0, scale=0.02, size=(target_feature_dim, hdim)).astype(np.float32)
        b1 = np.zeros((hdim,), dtype=np.float32)
        w2 = rng_action.normal(loc=0.0, scale=0.02, size=(hdim, 40)).astype(np.float32)
        b2 = np.zeros((40,), dtype=np.float32)
        wv = np.zeros((hdim,), dtype=np.float32)
        bv = np.float32(0.0)

    policy = A2CPolicy(w1=w1, b1=b1, w2=w2, b2=b2, wv=wv, bv=float(bv))

    def _policy_parameter_groups() -> A2CArrayGroups:
        """Vista corrente dei parametri, raggruppata per responsabilita' A2C."""
        return A2CArrayGroups(
            trunk=(policy.w1, policy.b1),
            actor_head=(policy.w2, policy.b2),
            critic_head=(policy.wv, np.asarray([policy.bv], dtype=np.float32)),
        )

    signal_diagnostics = A2CSignalAccumulator(policy.hidden_dim) if diagnostics_path is not None else None
    update_diagnostics: list[A2CUpdateDiagnostics] = []
    diagnostic_initial_parameter_l2: dict[str, float] | None = None
    if diagnostics_path is not None:
        if resume_state is None:
            initial_groups = _policy_parameter_groups()
            diagnostic_initial_parameter_l2 = {
                "trunk": array_group_l2(initial_groups.trunk),
                "actor_head": array_group_l2(initial_groups.actor_head),
                "critic_head": array_group_l2(initial_groups.critic_head),
            }
        else:
            raw_diagnostics = resume_state.get("diagnostics")
            if not isinstance(raw_diagnostics, dict):
                raise ValueError("Checkpoint senza storico diagnostico")
            raw_initial_l2 = raw_diagnostics.get("initial_parameter_l2")
            raw_updates = raw_diagnostics.get("updates")
            if not isinstance(raw_initial_l2, dict) or not isinstance(raw_updates, list):
                raise ValueError("Storico diagnostico del checkpoint incompatibile")
            diagnostic_initial_parameter_l2 = {
                "trunk": float(raw_initial_l2["trunk"]),
                "actor_head": float(raw_initial_l2["actor_head"]),
                "critic_head": float(raw_initial_l2["critic_head"]),
            }
            update_diagnostics = [A2CUpdateDiagnostics.from_json(row) for row in raw_updates]

    # Anchor BC (teacher) opzionale: deve avere stessa feature_dim dell'encoder corrente.
    bc_anchor: LoadedBCModel | None = None
    bc_anchor_path = str(args.bc_anchor).strip()
    if bc_anchor_path:
        loaded_anchor = load_bc_model_npz(Path(bc_anchor_path))
        if int(loaded_anchor.feature_dim) != int(policy.feature_dim):
            raise ValueError(
                "BC-anchor non compatibile con l'encoder corrente: "
                f"anchor.feature_dim={int(loaded_anchor.feature_dim)} policy.feature_dim={int(policy.feature_dim)}. "
                "Suggerimento: usa `--encoder-version` coerente con l'anchor (v1=248, v2=288)."
            )
        bc_anchor = loaded_anchor

    # Adam state: uno warm-start riparte da zero; un resume ripristina ogni momento.
    if resume_state is None:
        st_w1 = _adam_init(policy.w1)
        st_b1 = _adam_init(policy.b1)
        st_w2 = _adam_init(policy.w2)
        st_b2 = _adam_init(policy.b2)
        st_wv = _adam_init(policy.wv)
        st_bv = _adam_init(np.asarray([policy.bv], dtype=np.float32))
        t = 0
    else:
        st_w1 = AdamState(resume_arrays["resume_st_w1_m"], resume_arrays["resume_st_w1_v"])
        st_b1 = AdamState(resume_arrays["resume_st_b1_m"], resume_arrays["resume_st_b1_v"])
        st_w2 = AdamState(resume_arrays["resume_st_w2_m"], resume_arrays["resume_st_w2_v"])
        st_b2 = AdamState(resume_arrays["resume_st_b2_m"], resume_arrays["resume_st_b2_v"])
        st_wv = AdamState(resume_arrays["resume_st_wv_m"], resume_arrays["resume_st_wv_v"])
        st_bv = AdamState(resume_arrays["resume_st_bv_m"], resume_arrays["resume_st_bv_v"])
        expected_shapes = (
            (st_w1, policy.w1),
            (st_b1, policy.b1),
            (st_w2, policy.w2),
            (st_b2, policy.b2),
            (st_wv, policy.wv),
            (st_bv, np.asarray([policy.bv], dtype=np.float32)),
        )
        if any(state.m.shape != param.shape or state.v.shape != param.shape for state, param in expected_shapes):
            raise ValueError("Shape dello stato Adam incompatibili col checkpoint")
        t = int(str(resume_state.get("optimizer_updates", -1)))
        if t != resume_games // int(args.update_every):
            raise ValueError("Contatore Adam incoerente col numero di partite nel checkpoint")

    update_every = int(args.update_every)
    history_mode: HistoryMode = "full" if str(args.metrics_mode) == "full" else "summary"
    if resume_state is None:
        metrics = StreamingHistory[dict[str, object]](mode=history_mode)
        suit_consistency_metrics = StreamingHistory[dict[str, object]](mode=history_mode)
        suit_margin_metrics = StreamingHistory[dict[str, object]](mode=history_mode)
    else:
        raw_histories = resume_state.get("histories")
        if not isinstance(raw_histories, dict):
            raise ValueError("Checkpoint senza storici delle metriche")
        metrics = StreamingHistory[dict[str, object]].from_resume_state(
            raw_histories["metrics"], expected_mode=history_mode
        )
        suit_consistency_metrics = StreamingHistory[dict[str, object]].from_resume_state(
            raw_histories["suit_consistency"], expected_mode=history_mode
        )
        suit_margin_metrics = StreamingHistory[dict[str, object]].from_resume_state(
            raw_histories["suit_margin"], expected_mode=history_mode
        )

    # Accumulo grad (batch).
    gw1 = np.zeros_like(policy.w1)
    gb1 = np.zeros_like(policy.b1)
    gw2 = np.zeros_like(policy.w2)
    gb2 = np.zeros_like(policy.b2)
    gwv = np.zeros_like(policy.wv)
    gbv = 0.0

    # Logging accumulators.
    returns_buf: list[float] = []
    wins = 0
    draws = 0
    entropies: list[float] = []
    grad_step_count = 0
    value_loss_sum = 0.0
    anchor_ce_sum = 0.0
    anchor_ce_count = 0
    suit_consistency_kl_sum = 0.0
    suit_consistency_count = 0
    suit_margin_loss_sum = 0.0
    suit_margin_count = 0
    suit_margin_violation_count = 0
    suit_margin_teacher_sum = 0.0
    suit_margin_student_sum = 0.0

    use_numba_batch_rollout = rollout_engine == "fast" and fast_rollout == "numba"
    use_value_lookahead_numba_rollout = use_numba_batch_rollout and fast_numba_value_lookahead_opponent is not None
    numba_batch: NumbaA2CBatch | None = None
    numba_batch_offset = 0
    schedule_batch: tuple[ScheduledTrainingGame, ...] = ()
    schedule_batch_offset = 0

    def _configured_artifact(raw_path: str, *, enabled: bool) -> dict[str, str | int] | None:
        """Hasha soltanto gli asset che partecipano realmente al run."""
        value = raw_path.strip()
        return _artifact(Path(value)) if enabled and value else None

    config_payload: dict[str, object] = {
        "schema": "briscola.a2c_training_config.v1",
        "code": {
            "version": get_code_version(),
            "rules": get_rules_version(),
            "git_commit": _git_commit(),
            "trainer_sha256": _sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "seed": int(args.seed),
        "num_games": num_games,
        "architecture": {
            "encoder_version": encoder_version,
            "feature_dim": int(policy.feature_dim),
            "hidden_dim": int(policy.hidden_dim),
            "action_dim": 40,
            "policy_belief": policy_belief is not None,
        },
        "rollout": {
            "engine": rollout_engine,
            "fast_encoder": fast_encoder,
            "fast_rollout": fast_rollout,
        },
        "schedule": {
            "mode": training_schedule_mode,
            "seat_fair_requested": bool(args.seat_fair),
            "update_every": int(args.update_every),
        },
        "opponents": {
            "single": str(args.opponent) if opponent_pool is None else None,
            "mix": opponent_pool.to_metadata() if opponent_pool is not None else None,
            "pimc_determinizations": int(args.opponent_pimc_determinizations),
            "value_max_unknown_cards": int(args.opponent_value_max_unknown_cards),
        },
        "optimizer": {
            "name": "adam",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "entropy_beta": float(args.entropy_beta),
            "value_coef": float(args.value_coef),
            "gamma": float(args.gamma),
        },
        "regularization": {
            "bc_anchor_beta": float(args.bc_anchor_beta),
            "overkill_mode": str(args.overkill_penalty_mode),
            "overkill_beta": float(args.overkill_penalty_beta),
            "overkill_low_lead_points_max": int(args.overkill_low_lead_points_max),
            "suit_augmentation": suit_augmentation,
            "suit_consistency_beta": suit_consistency_beta,
            "suit_margin_beta": suit_margin_beta,
            "suit_margin_cap": suit_margin_cap,
        },
        "output_contract": {
            "metrics_mode": history_mode,
            "diagnostics_enabled": diagnostics_path is not None,
            "diagnostics_every": int(args.diagnostics_every),
            "inference_overkill_guard": bool(args.inference_overkill_guard),
        },
        "assets": {
            "initial_model": original_init_artifact,
            "bc_anchor": _configured_artifact(bc_anchor_path, enabled=bc_anchor is not None),
            "policy_belief": _configured_artifact(policy_belief_path, enabled=policy_belief is not None),
            "opponent_model": _configured_artifact(
                str(args.opponent_model),
                enabled=fast_numba_model_opponent is not None or fast_numba_value_lookahead_opponent is not None,
            ),
            "opponent_belief": _configured_artifact(
                str(args.opponent_belief_model), enabled=fast_numba_opponent_belief is not None
            ),
            "opponent_value": _configured_artifact(
                str(args.opponent_value_model), enabled=use_value_lookahead_numba_rollout
            ),
        },
    }
    training_config_fingerprint = config_fingerprint(config_payload)
    if resume_state is not None:
        expected_fingerprint = str(resume_state.get("config_fingerprint", ""))
        if training_config_fingerprint != expected_fingerprint:
            raise ValueError(
                "Configurazione del resume diversa dal checkpoint: ripetere gli stessi flag, asset e commit del run. "
                f"attuale={training_config_fingerprint} checkpoint={expected_fingerprint}"
            )

    # Metadati UI (opzionali ma utili per il dropdown dei modelli in frontend).
    def _format_num_games(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}k"
        return str(n)

    def _format_checkpoint_suffix(n: int) -> str:
        """Format a game count for compact checkpoint filenames."""
        return _format_num_games(n).replace(".0", "").lower()

    def _format_opponent_label() -> str:
        if opponent_pool is not None:
            parts = [f"{item.name} {float(item.prob):.2f}" for item in opponent_pool.items]
            return "mix(" + ", ".join(parts) + ")"
        return str(args.opponent).strip()

    def _metrics_metadata() -> dict[str, object]:
        """Return either full metric history or a compact summary for long runs."""
        return metrics.metadata(full_key="metrics", summary_key="metrics_summary")

    def _build_metadata(*, trained_games: int, is_checkpoint: bool) -> dict[str, object]:
        """Build metadata for a final model or an intermediate training checkpoint."""
        if training_schedule.consumed_games != int(trained_games):
            raise AssertionError(f"Schedule consumata={training_schedule.consumed_games}, modello={int(trained_games)}")
        observation_note = (
            "Osservazione anti-cheat: Fast2PState numerico con feature equivalenti a PlayerObservation."
            if rollout_engine == "fast"
            else "Osservazione anti-cheat: PlayerObservation (vista parziale lecita)."
        )
        payload: dict[str, object] = {
            "format": "mlp_a2c_shaped_v1",
            "label": f"A2C shaped {_format_num_games(int(trained_games))} game",
            "description_it": (
                "Policy addestrata con A2C (actor-critic) con reward shaping (delta punti per mano), "
                f"contro {_format_opponent_label()}. "
                f"{observation_note}"
            ),
            "feature_dim": int(policy.feature_dim),
            "hidden_dim": int(policy.hidden_dim),
            "action_dim": 40,
            "seed": int(args.seed),
            "rollout_engine": rollout_engine,
            "fast_encoder": fast_encoder if rollout_engine == "fast" else None,
            "fast_rollout": fast_rollout if rollout_engine == "fast" else None,
            "opponent": str(args.opponent) if not opponent_mix_raw else None,
            "opponent_model": str(args.opponent_model).strip() or None,
            "opponent_value_model": (
                str(args.opponent_value_model).strip() if fast_numba_value_lookahead_opponent is not None else None
            ),
            "opponent_value_max_unknown_cards": (
                int(args.opponent_value_max_unknown_cards) if fast_numba_value_lookahead_opponent is not None else None
            ),
            "opponent_value_lookahead_rollout": (
                "fast_numba_determinized" if use_value_lookahead_numba_rollout else None
            ),
            "opponent_mix": opponent_pool.to_metadata() if opponent_pool is not None else None,
            "init": original_init_artifact["path"] if original_init_artifact is not None else None,
            "resume_from": str(resume_path) if resume_path is not None else None,
            "encoder": f"encode_observation_2p:{encoder_version}",
            "encoder_version": encoder_version,
            "policy_belief_model": policy_belief_path or None,
            "policy_input": "v4+belief" if policy_belief is not None else encoder_version,
            "reward_shaping": "turn_based_trick_delta_points",
            "reward_shaping_overkill_penalty_mode": str(args.overkill_penalty_mode),
            "reward_shaping_overkill_penalty_beta": float(args.overkill_penalty_beta),
            "reward_shaping_overkill_low_lead_points_max": int(args.overkill_low_lead_points_max),
            "bc_anchor_path": bc_anchor_path or None,
            "bc_anchor_beta": float(args.bc_anchor_beta),
            "inference_overkill_guard": bool(args.inference_overkill_guard),
            "train": {
                "algorithm": "a2c",
                "optimizer": "adam",
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "entropy_beta": float(args.entropy_beta),
                "value_coef": float(args.value_coef),
                "gamma": float(args.gamma),
                "update_every": int(args.update_every),
                "seat_fair": effective_seat_fair,
                "seat_fair_requested": bool(args.seat_fair),
                "training_schedule": {
                    "mode": training_schedule_mode,
                    "pair_size": 2 if training_schedule_mode == "paired" else 1,
                    "scheduled_environment_draws": (
                        int(trained_games) // 2 if training_schedule_mode == "paired" else int(trained_games)
                    ),
                    "opponent_sampling_scope": "pair" if training_schedule_mode == "paired" else "game",
                    "seat_order": [0, 1] if training_schedule_mode == "paired" else None,
                    "sha256": training_schedule.sha256,
                    "digest_algorithm": "sha256_chain_v2",
                    "game_rng_seed": int(args.seed ^ 0x9E3779B9),
                    "opponent_rng_seed": int(args.seed ^ 0xA5A5A5A5),
                },
                "num_games": int(trained_games),
                "requested_num_games": num_games,
                "run_complete": int(trained_games) == num_games,
            },
            "training_config_fingerprint": training_config_fingerprint,
        }
        if diagnostics_path is not None:
            payload["a2c_diagnostics"] = {
                "schema": "briscola.a2c_training_diagnostics.v1",
                "report_path": str(diagnostics_path),
                "passive": True,
                "critic_initialization": "resume" if resume_state is not None else "reset_zero",
                "init_contains_critic": init_contains_critic,
            }
        if suit_augmentation == "paired":
            payload["suit_augmentation"] = {
                "mode": "paired",
                "copies_per_trajectory": 1,
                "permutation_scope": "whole_trajectory",
                "permutation_distribution": "uniform_nonidentity_23",
                "loss_normalization": "mean_over_original_and_copy",
                "rng_seed": int(args.seed ^ 0x51A17A9E),
            }
        if suit_consistency_beta > 0.0:
            payload["suit_consistency"] = {
                "mode": "forward_kl_stop_gradient",
                "beta": suit_consistency_beta,
                "target": "original_policy_distribution",
                "student": "nonidentity_suit_permutation",
                "permutation_scope": "whole_trajectory",
                "permutation_distribution": "uniform_nonidentity_23",
                "loss_normalization": "mean_over_original_on_policy_steps",
                "rng_seed": int(args.seed ^ 0x0C05157E),
            }
            payload.update(
                suit_consistency_metrics.metadata(
                    full_key="suit_consistency_metrics",
                    summary_key="suit_consistency_metrics_summary",
                )
            )
        if suit_margin_beta > 0.0:
            payload["suit_margin_consistency"] = {
                "mode": "teacher_argmax_hinge",
                "beta": suit_margin_beta,
                "margin_cap": suit_margin_cap,
                "teacher": "original_argmax_and_capped_logit_margin",
                "student": "nonidentity_suit_permutation",
                "forced_actions": "excluded",
                "permutation_scope": "whole_trajectory",
                "permutation_distribution": "uniform_nonidentity_23",
                "loss_normalization": "mean_over_original_on_policy_steps",
                "rng_seed": int(args.seed ^ 0x0A461A9E),
            }
            payload.update(
                suit_margin_metrics.metadata(
                    full_key="suit_margin_metrics",
                    summary_key="suit_margin_metrics_summary",
                )
            )
        if is_checkpoint:
            payload["checkpoint"] = {
                "games": int(trained_games),
                "final_num_games": num_games,
            }
        payload.update(_metrics_metadata())
        return payload

    def _build_resume_state(*, trained_games: int) -> dict[str, object]:
        """Cattura lo stato post-update sufficiente per una continuazione bit-identica."""
        if trained_games % update_every != 0:
            raise ValueError("Un resume A2C può essere salvato soltanto dopo un update completo")
        if training_schedule.consumed_games != trained_games:
            raise AssertionError("Cursore schedule incoerente durante il checkpoint")
        diagnostics_state = None
        if diagnostics_path is not None:
            assert diagnostic_initial_parameter_l2 is not None
            diagnostics_state = {
                "initial_parameter_l2": diagnostic_initial_parameter_l2,
                "updates": [row.to_json() for row in update_diagnostics],
            }
        return {
            "schema": A2C_RESUME_SCHEMA,
            "config_fingerprint": training_config_fingerprint,
            "config": config_payload,
            "games_completed": int(trained_games),
            "optimizer_updates": int(t),
            "schedule": {
                "consumed_games": training_schedule.consumed_games,
                "sha256": training_schedule.sha256,
                "digest_algorithm": "sha256_chain_v2",
            },
            "rng": {
                "action": json_compatible(rng_action.bit_generator.state),
                "game": json_compatible(rng_game.bit_generator.state),
                "opponent_select": json_compatible(rng_opponent_select.bit_generator.state),
                "opponent_python": json_compatible(rng_opponent.getstate()),
                "suit_augmentation": (
                    json_compatible(rng_suit_augmentation.bit_generator.state)
                    if rng_suit_augmentation is not None
                    else None
                ),
                "suit_consistency": (
                    json_compatible(rng_suit_consistency.bit_generator.state)
                    if rng_suit_consistency is not None
                    else None
                ),
                "suit_margin": (
                    json_compatible(rng_suit_margin.bit_generator.state) if rng_suit_margin is not None else None
                ),
            },
            "histories": {
                "metrics": metrics.resume_state(),
                "suit_consistency": suit_consistency_metrics.resume_state(),
                "suit_margin": suit_margin_metrics.resume_state(),
            },
            "diagnostics": diagnostics_state,
            "initialization": {
                "init_contains_critic": init_contains_critic,
                "init_critic_shapes": init_critic_shapes,
                "init_artifact": original_init_artifact,
            },
        }

    def _save_model(path: Path, *, trained_games: int, is_checkpoint: bool) -> None:
        """Salva modello e, quando utile, lo stato di resume con pubblicazione atomica."""
        payload = _build_metadata(trained_games=trained_games, is_checkpoint=is_checkpoint)
        # Nota compatibilità: `w1/b1/w2/b2` (actor) rende il file caricabile da `bc_model`.
        extra_arrays: dict[str, object] = {}
        if policy_belief is not None:
            # Artefatto SELF-CONTAINED: la belief congelata viaggia col modello, cosi'
            # bc_model puo' ricostruire l'input 409 a inference senza file esterni.
            extra_arrays = {
                "belief_w1": policy_belief.w1,
                "belief_b1": policy_belief.b1,
                "belief_w2": policy_belief.w2,
                "belief_b2": policy_belief.b2,
            }
        if is_checkpoint or trained_games < num_games:
            extra_arrays.update(
                {
                    "resume_st_w1_m": st_w1.m,
                    "resume_st_w1_v": st_w1.v,
                    "resume_st_b1_m": st_b1.m,
                    "resume_st_b1_v": st_b1.v,
                    "resume_st_w2_m": st_w2.m,
                    "resume_st_w2_v": st_w2.v,
                    "resume_st_b2_m": st_b2.m,
                    "resume_st_b2_v": st_b2.v,
                    "resume_st_wv_m": st_wv.m,
                    "resume_st_wv_v": st_wv.v,
                    "resume_st_bv_m": st_bv.m,
                    "resume_st_bv_v": st_bv.v,
                    "resume_state_json": canonical_json(_build_resume_state(trained_games=trained_games)),
                }
            )
        atomic_savez(
            path,
            w1=policy.w1,
            b1=policy.b1,
            w2=policy.w2,
            b2=policy.b2,
            wv=policy.wv,
            bv=np.asarray([policy.bv], dtype=np.float32),
            metadata_json=json.dumps(payload, ensure_ascii=False, indent=2),
            **extra_arrays,
        )
        kind = "checkpoint" if is_checkpoint else "model"
        print(f"Saved {kind}: {path}")

    def _write_diagnostics_report() -> None:
        """Salva la telemetria passiva dopo che il modello finale esiste."""
        if diagnostics_path is None:
            return
        assert diagnostic_initial_parameter_l2 is not None
        report = {
            "schema": "briscola.a2c_training_diagnostics.v1",
            "method": {
                "passive": True,
                "signal_scope": "original on-policy steps; suit copies excluded",
                "gradient_scope": "mean per policy step plus configured weight decay, immediately before Adam",
                "trunk_gradient": "combined actor and critic contribution in the shared trunk",
                "update_scope": "actual parameter delta produced by Adam",
                "anti_cheat": "diagnostics aggregate tensors derived from legal observations; no hidden cards stored",
                "sampling": (
                    "first, every N updates, checkpoints and last update of each resumed segment; "
                    "unsampled updates are not retained"
                ),
            },
            "config": {
                "seed": int(args.seed),
                "num_games": num_games,
                "games_completed": stop_after_games,
                "update_every": int(args.update_every),
                "rollout_engine": rollout_engine,
                "fast_rollout": fast_rollout if rollout_engine == "fast" else None,
                "encoder_version": encoder_version,
                "feature_dim": int(policy.feature_dim),
                "hidden_dim": int(policy.hidden_dim),
                "training_schedule": training_schedule_mode,
                "training_schedule_sha256": training_schedule.sha256,
                "diagnostics_every": int(args.diagnostics_every),
                "opponent": str(args.opponent) if not opponent_mix_raw else None,
                "opponent_mix": opponent_pool.to_metadata() if opponent_pool is not None else None,
                "lr": float(args.lr),
                "value_coef": float(args.value_coef),
                "entropy_beta": float(args.entropy_beta),
                "gamma": float(args.gamma),
                "weight_decay": float(args.weight_decay),
                "bc_anchor_beta": float(args.bc_anchor_beta),
                "overkill_penalty_mode": str(args.overkill_penalty_mode),
                "overkill_penalty_beta": float(args.overkill_penalty_beta),
            },
            "initialization": {
                "critic_mode": "reset_zero_then_resumed" if resume_state is not None else "reset_zero",
                "init_contains_critic": init_contains_critic,
                "init_critic_shapes": init_critic_shapes,
                "init_critic_used": False,
                "note_it": (
                    "Il warm-start iniziale reinizializza il critic; i segmenti successivi "
                    "ripristinano esattamente critic e optimizer dal checkpoint."
                ),
                "parameter_l2": diagnostic_initial_parameter_l2,
            },
            "artifacts": {
                "init": original_init_artifact,
                "resume_from": _artifact(resume_path) if resume_path is not None else None,
                "model_out": _artifact(out_path),
            },
            "updates": [row.to_json() for row in update_diagnostics],
            "summary": summarize_update_diagnostics(update_diagnostics),
            "versions": {
                "code": get_code_version(),
                "rules": get_rules_version(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "git_commit": _git_commit(),
            },
        }
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved diagnostics: {diagnostics_path}")

    def _checkpoint_path(trained_games: int) -> Path:
        suffix = _format_checkpoint_suffix(trained_games)
        return checkpoint_dir / f"{checkpoint_prefix}_{suffix}.npz"

    for game_idx in range(resume_games + 1, stop_after_games + 1):
        if schedule_batch_offset >= len(schedule_batch):
            games_until_update = update_every - ((game_idx - 1) % update_every)
            batch_size = min(games_until_update, stop_after_games - game_idx + 1)
            schedule_batch = training_schedule.take(batch_size)
            schedule_batch_offset = 0
        scheduled_game = schedule_batch[schedule_batch_offset]
        schedule_batch_offset += 1
        if scheduled_game.ordinal != game_idx:
            raise AssertionError(f"Ordinal schedule {scheduled_game.ordinal}, atteso {game_idx}")
        policy_seat = scheduled_game.policy_seat
        game_seed = scheduled_game.game_seed
        current_opponent = (
            opponent_pool.agents_by_name[scheduled_game.opponent_name] if opponent_pool is not None else opponent
        )
        numba_traj_for_backprop = None
        traj: list[StepRecord]

        if rollout_engine == "fast":
            if fast_rollout == "numba":
                if use_numba_batch_rollout:
                    if numba_batch is None or numba_batch_offset >= int(numba_batch.step_counts.shape[0]):
                        batch_schedule = schedule_batch
                        batch_size = len(batch_schedule)
                        game_seeds = np.asarray([game.game_seed for game in batch_schedule], dtype=np.int64)
                        policy_seats = np.asarray([game.policy_seat for game in batch_schedule], dtype=np.int64)
                        opponent_codes = None
                        opponent_model_enabled_flags = None
                        opponent_modes = None
                        if opponent_pool is not None:
                            sampled_names = [game.opponent_name for game in batch_schedule]
                            if use_value_lookahead_numba_rollout:
                                opponent_modes = np.asarray(
                                    [
                                        _fast_numba_opponent_mode_for_name(
                                            name,
                                            value_lookahead_name=fast_numba_value_mix_name,
                                            pimc_belief_name=fast_numba_pimc_belief_name,
                                        )
                                        for name in sampled_names
                                    ],
                                    dtype=np.int64,
                                )
                                opponent_codes = np.asarray(
                                    [
                                        numba_agent_code(name)
                                        if _fast_numba_opponent_mode_for_name(
                                            name,
                                            value_lookahead_name=fast_numba_value_mix_name,
                                            pimc_belief_name=fast_numba_pimc_belief_name,
                                        )
                                        == OPPONENT_MODE_RULE
                                        else 0
                                        for name in sampled_names
                                    ],
                                    dtype=np.int64,
                                )
                            else:
                                opponent_codes = np.asarray(
                                    [
                                        0 if name == fast_numba_model_mix_name else numba_agent_code(name)
                                        for name in sampled_names
                                    ],
                                    dtype=np.int64,
                                )
                                opponent_model_enabled_flags = np.asarray(
                                    [name == fast_numba_model_mix_name for name in sampled_names],
                                    dtype=np.bool_,
                                )
                        if use_value_lookahead_numba_rollout:
                            assert fast_numba_value_lookahead_opponent is not None
                            value_model = fast_numba_value_lookahead_opponent.value_model
                            if opponent_modes is None:
                                mode = _fast_numba_opponent_mode_for_name(
                                    current_opponent.name,
                                    value_lookahead_name=fast_numba_value_mix_name,
                                    pimc_belief_name=fast_numba_pimc_belief_name,
                                )
                                opponent_modes = np.full(batch_size, mode, dtype=np.int64)
                            if opponent_codes is None:
                                code = (
                                    numba_agent_code(current_opponent.name)
                                    if int(opponent_modes[0]) == OPPONENT_MODE_RULE
                                    else 0
                                )
                                opponent_codes = np.full(batch_size, code, dtype=np.int64)
                            numba_batch = collect_a2c_batch_numba_value_lookahead_2p(
                                opponent_belief_w1=fast_numba_opponent_belief.w1
                                if fast_numba_opponent_belief is not None
                                else None,
                                opponent_belief_b1=fast_numba_opponent_belief.b1
                                if fast_numba_opponent_belief is not None
                                else None,
                                opponent_belief_w2=fast_numba_opponent_belief.w2
                                if fast_numba_opponent_belief is not None
                                else None,
                                opponent_belief_b2=fast_numba_opponent_belief.b2
                                if fast_numba_opponent_belief is not None
                                else None,
                                opponent_pimc_determinizations=int(args.opponent_pimc_determinizations),
                                policy_belief_w1=policy_belief.w1 if policy_belief is not None else None,
                                policy_belief_b1=policy_belief.b1 if policy_belief is not None else None,
                                policy_belief_w2=policy_belief.w2 if policy_belief is not None else None,
                                policy_belief_b2=policy_belief.b2 if policy_belief is not None else None,
                                w1=policy.w1,
                                b1=policy.b1,
                                w2=policy.w2,
                                b2=policy.b2,
                                wv=policy.wv,
                                bv=float(policy.bv),
                                opponent_modes=opponent_modes,
                                opponent_codes=opponent_codes,
                                opponent_w1=fast_numba_value_lookahead_opponent.model.w1,
                                opponent_b1=fast_numba_value_lookahead_opponent.model.b1,
                                opponent_w2=fast_numba_value_lookahead_opponent.model.w2,
                                opponent_b2=fast_numba_value_lookahead_opponent.model.b2,
                                opponent_overkill_guard=bool(
                                    fast_numba_value_lookahead_opponent.agent.overkill_guard_enabled
                                ),
                                value_w1=value_model.w1,
                                value_b1=value_model.b1,
                                value_w2=value_model.w2,
                                value_b2=float(value_model.b2),
                                value_target_scale=float(value_model.metadata.get("target_scale", 120.0) or 120.0),
                                value_target_is_residual=value_model.metadata.get("target") == "residual",
                                value_max_unknown_cards=int(fast_numba_value_lookahead_opponent.max_unknown_cards),
                                game_seeds=game_seeds,
                                policy_seats=policy_seats,
                                overkill_penalty_beta=float(args.overkill_penalty_beta),
                                overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                                overkill_penalty_mode=str(args.overkill_penalty_mode),
                            )
                        else:
                            numba_batch = collect_a2c_batch_numba_2p(
                                policy_belief_w1=policy_belief.w1 if policy_belief is not None else None,
                                policy_belief_b1=policy_belief.b1 if policy_belief is not None else None,
                                policy_belief_w2=policy_belief.w2 if policy_belief is not None else None,
                                policy_belief_b2=policy_belief.b2 if policy_belief is not None else None,
                                w1=policy.w1,
                                b1=policy.b1,
                                w2=policy.w2,
                                b2=policy.b2,
                                wv=policy.wv,
                                bv=float(policy.bv),
                                opponent_name=current_opponent.name,
                                opponent_w1=(
                                    fast_numba_model_opponent.model.w1
                                    if fast_numba_model_opponent is not None
                                    else None
                                ),
                                opponent_b1=(
                                    fast_numba_model_opponent.model.b1
                                    if fast_numba_model_opponent is not None
                                    else None
                                ),
                                opponent_w2=(
                                    fast_numba_model_opponent.model.w2
                                    if fast_numba_model_opponent is not None
                                    else None
                                ),
                                opponent_b2=(
                                    fast_numba_model_opponent.model.b2
                                    if fast_numba_model_opponent is not None
                                    else None
                                ),
                                opponent_overkill_guard=(
                                    bool(fast_numba_model_opponent.agent.overkill_guard_enabled)
                                    if fast_numba_model_opponent is not None
                                    else False
                                ),
                                game_seeds=game_seeds,
                                policy_seats=policy_seats,
                                opponent_codes=opponent_codes,
                                opponent_model_enabled_flags=opponent_model_enabled_flags,
                                overkill_penalty_beta=float(args.overkill_penalty_beta),
                                overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                                overkill_penalty_mode=str(args.overkill_penalty_mode),
                            )
                        batch_grad_stats = _accumulate_numba_batch_grads(
                            policy=policy,
                            batch=numba_batch,
                            gamma=float(args.gamma),
                            entropy_beta=float(args.entropy_beta),
                            value_coef=float(args.value_coef),
                            bc_anchor=bc_anchor,
                            bc_anchor_beta=float(args.bc_anchor_beta),
                            gw1=gw1,
                            gb1=gb1,
                            gw2=gw2,
                            gb2=gb2,
                            gwv=gwv,
                            suit_augmentation_rng=rng_suit_augmentation,
                            suit_consistency_rng=rng_suit_consistency,
                            suit_consistency_beta=suit_consistency_beta,
                            suit_margin_rng=rng_suit_margin,
                            suit_margin_beta=suit_margin_beta,
                            suit_margin_cap=suit_margin_cap,
                            encoder_version=(
                                encoder_version
                                if (
                                    rng_suit_augmentation is not None
                                    or rng_suit_consistency is not None
                                    or rng_suit_margin is not None
                                )
                                else None
                            ),
                            signal_diagnostics=signal_diagnostics,
                        )
                        gbv += batch_grad_stats.gbv
                        grad_step_count += batch_grad_stats.steps
                        value_loss_sum += batch_grad_stats.value_loss_sum
                        anchor_ce_sum += batch_grad_stats.anchor_ce_sum
                        anchor_ce_count += batch_grad_stats.anchor_ce_count
                        suit_consistency_kl_sum += batch_grad_stats.suit_consistency_kl_sum
                        suit_consistency_count += batch_grad_stats.suit_consistency_count
                        suit_margin_loss_sum += batch_grad_stats.suit_margin_loss_sum
                        suit_margin_count += batch_grad_stats.suit_margin_count
                        suit_margin_violation_count += batch_grad_stats.suit_margin_violation_count
                        suit_margin_teacher_sum += batch_grad_stats.suit_margin_teacher_sum
                        suit_margin_student_sum += batch_grad_stats.suit_margin_student_sum
                        numba_batch_offset = 0
                    assert numba_batch is not None
                    numba_traj = _numba_batch_trajectory_at(numba_batch, numba_batch_offset)
                    numba_batch_offset += 1
                    if numba_batch_offset >= int(numba_batch.step_counts.shape[0]):
                        numba_batch = None
                else:
                    if use_value_lookahead_numba_rollout:
                        assert fast_numba_value_lookahead_opponent is not None
                        value_model = fast_numba_value_lookahead_opponent.value_model
                        mode = _fast_numba_opponent_mode_for_name(
                            current_opponent.name,
                            value_lookahead_name=fast_numba_value_mix_name,
                            pimc_belief_name=fast_numba_pimc_belief_name,
                        )
                        code = numba_agent_code(current_opponent.name) if mode == OPPONENT_MODE_RULE else 0
                        numba_traj = collect_a2c_trajectory_numba_value_lookahead_2p(
                            opponent_belief_w1=fast_numba_opponent_belief.w1
                            if fast_numba_opponent_belief is not None
                            else None,
                            opponent_belief_b1=fast_numba_opponent_belief.b1
                            if fast_numba_opponent_belief is not None
                            else None,
                            opponent_belief_w2=fast_numba_opponent_belief.w2
                            if fast_numba_opponent_belief is not None
                            else None,
                            opponent_belief_b2=fast_numba_opponent_belief.b2
                            if fast_numba_opponent_belief is not None
                            else None,
                            opponent_pimc_determinizations=int(args.opponent_pimc_determinizations),
                            policy_belief_w1=policy_belief.w1 if policy_belief is not None else None,
                            policy_belief_b1=policy_belief.b1 if policy_belief is not None else None,
                            policy_belief_w2=policy_belief.w2 if policy_belief is not None else None,
                            policy_belief_b2=policy_belief.b2 if policy_belief is not None else None,
                            w1=policy.w1,
                            b1=policy.b1,
                            w2=policy.w2,
                            b2=policy.b2,
                            wv=policy.wv,
                            bv=float(policy.bv),
                            opponent_mode=mode,
                            opponent_code=code,
                            opponent_w1=fast_numba_value_lookahead_opponent.model.w1,
                            opponent_b1=fast_numba_value_lookahead_opponent.model.b1,
                            opponent_w2=fast_numba_value_lookahead_opponent.model.w2,
                            opponent_b2=fast_numba_value_lookahead_opponent.model.b2,
                            opponent_overkill_guard=bool(
                                fast_numba_value_lookahead_opponent.agent.overkill_guard_enabled
                            ),
                            value_w1=value_model.w1,
                            value_b1=value_model.b1,
                            value_w2=value_model.w2,
                            value_b2=float(value_model.b2),
                            value_target_scale=float(value_model.metadata.get("target_scale", 120.0) or 120.0),
                            value_target_is_residual=value_model.metadata.get("target") == "residual",
                            value_max_unknown_cards=int(fast_numba_value_lookahead_opponent.max_unknown_cards),
                            game_seed=game_seed,
                            policy_seat=policy_seat,
                            overkill_penalty_beta=float(args.overkill_penalty_beta),
                            overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                            overkill_penalty_mode=str(args.overkill_penalty_mode),
                        )
                    else:
                        numba_traj = collect_a2c_trajectory_numba_2p(
                            policy_belief_w1=policy_belief.w1 if policy_belief is not None else None,
                            policy_belief_b1=policy_belief.b1 if policy_belief is not None else None,
                            policy_belief_w2=policy_belief.w2 if policy_belief is not None else None,
                            policy_belief_b2=policy_belief.b2 if policy_belief is not None else None,
                            w1=policy.w1,
                            b1=policy.b1,
                            w2=policy.w2,
                            b2=policy.b2,
                            wv=policy.wv,
                            bv=float(policy.bv),
                            opponent_name=current_opponent.name,
                            opponent_w1=(
                                fast_numba_model_opponent.model.w1 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_b1=(
                                fast_numba_model_opponent.model.b1 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_w2=(
                                fast_numba_model_opponent.model.w2 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_b2=(
                                fast_numba_model_opponent.model.b2 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_overkill_guard=(
                                bool(fast_numba_model_opponent.agent.overkill_guard_enabled)
                                if fast_numba_model_opponent is not None
                                else False
                            ),
                            game_seed=game_seed,
                            policy_seat=policy_seat,
                            overkill_penalty_beta=float(args.overkill_penalty_beta),
                            overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                            overkill_penalty_mode=str(args.overkill_penalty_mode),
                        )
                numba_traj_for_backprop = None if use_numba_batch_rollout else numba_traj
                traj = []
                avg_entropy = float(numba_traj.avg_entropy)
                policy_points = int(numba_traj.policy_points)
                opp_points = int(numba_traj.opponent_points)
                ep_return = float(policy_points - opp_points) / 120.0
            else:
                final_fast_state, traj, avg_entropy = _play_one_fast_game_2p_collect(
                    policy=policy,
                    opponent_name=current_opponent.name,
                    rng_opponent=rng_opponent,
                    rng_action=rng_action,
                    game_seed=game_seed,
                    policy_seat=policy_seat,
                    encoder_version=encoder_version,
                    fast_encoder=fast_encoder,
                    bc_anchor=bc_anchor,
                    bc_anchor_beta=float(args.bc_anchor_beta),
                )
                ep_return = float(_points_diff_fast(final_fast_state, policy_seat=policy_seat)) / 120.0
                policy_points = int(final_fast_state.points[policy_seat])
                opp_points = int(final_fast_state.points[1 - policy_seat])
        else:
            final_state, traj, avg_entropy = _play_one_game_2p_collect(
                policy=policy,
                opponent=current_opponent,
                rng_opponent=rng_opponent,
                rng_action=rng_action,
                game_seed=game_seed,
                policy_seat=policy_seat,
                entropy_beta=float(args.entropy_beta),
                encoder_version=encoder_version,
                overkill_penalty_beta=float(args.overkill_penalty_beta),
                overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                overkill_penalty_mode=str(args.overkill_penalty_mode),
                bc_anchor=bc_anchor,
                bc_anchor_beta=float(args.bc_anchor_beta),
            )
            ep_return = float(_points_diff(final_state, policy_seat=policy_seat)) / 120.0
            p0 = final_state.players[0].points
            p1 = final_state.players[1].points
            policy_points = p0 if policy_seat == 0 else p1
            opp_points = p1 if policy_seat == 0 else p0
        entropies.append(avg_entropy)

        # Episodic return (consistente con shaped reward): diff punti finale / 120.
        returns_buf.append(ep_return)

        # Win/draw tracking (in termini di punti).
        if policy_points > opp_points:
            wins += 1
        elif policy_points == opp_points:
            draws += 1

        if numba_traj_for_backprop is not None:
            returns_to_go_arr = _compute_returns_array(numba_traj_for_backprop.rewards, gamma=float(args.gamma))
            grad_stats = _accumulate_numba_trajectory_grads(
                policy=policy,
                xs=numba_traj_for_backprop.xs,
                z1s=numba_traj_for_backprop.z1s,
                hs=numba_traj_for_backprop.hs,
                action_masks=numba_traj_for_backprop.action_masks,
                probs=numba_traj_for_backprop.probs,
                action_ids=numba_traj_for_backprop.action_ids,
                value_preds=numba_traj_for_backprop.value_preds,
                returns_to_go=returns_to_go_arr,
                entropy_beta=float(args.entropy_beta),
                value_coef=float(args.value_coef),
                bc_anchor=bc_anchor,
                bc_anchor_beta=float(args.bc_anchor_beta),
                gw1=gw1,
                gb1=gb1,
                gw2=gw2,
                gb2=gb2,
                gwv=gwv,
                signal_diagnostics=signal_diagnostics,
            )
            if rng_suit_augmentation is not None:
                paired_stats = _accumulate_paired_suit_trajectory_grads(
                    policy=policy,
                    xs=numba_traj_for_backprop.xs,
                    action_masks=numba_traj_for_backprop.action_masks,
                    action_ids=numba_traj_for_backprop.action_ids,
                    returns_to_go=returns_to_go_arr,
                    encoder_version=encoder_version,
                    permutation=sample_nonidentity_suit_permutation(rng_suit_augmentation),
                    entropy_beta=float(args.entropy_beta),
                    value_coef=float(args.value_coef),
                    bc_anchor=bc_anchor,
                    bc_anchor_beta=float(args.bc_anchor_beta),
                    gw1=gw1,
                    gb1=gb1,
                    gw2=gw2,
                    gb2=gb2,
                    gwv=gwv,
                )
                grad_stats = _add_gradient_stats(grad_stats, paired_stats)
            if rng_suit_consistency is not None:
                consistency_stats = _accumulate_suit_consistency_grads(
                    policy=policy,
                    xs=numba_traj_for_backprop.xs,
                    action_masks=numba_traj_for_backprop.action_masks,
                    original_probs=numba_traj_for_backprop.probs,
                    encoder_version=encoder_version,
                    permutation=sample_nonidentity_suit_permutation(rng_suit_consistency),
                    beta=suit_consistency_beta,
                    gw1=gw1,
                    gb1=gb1,
                    gw2=gw2,
                    gb2=gb2,
                )
                grad_stats = _add_gradient_stats(grad_stats, consistency_stats)
            if rng_suit_margin is not None:
                margin_stats = _accumulate_suit_margin_grads(
                    policy=policy,
                    xs=numba_traj_for_backprop.xs,
                    action_masks=numba_traj_for_backprop.action_masks,
                    original_probs=numba_traj_for_backprop.probs,
                    encoder_version=encoder_version,
                    permutation=sample_nonidentity_suit_permutation(rng_suit_margin),
                    beta=suit_margin_beta,
                    margin_cap=suit_margin_cap,
                    gw1=gw1,
                    gb1=gb1,
                    gw2=gw2,
                    gb2=gb2,
                )
                grad_stats = _add_gradient_stats(grad_stats, margin_stats)
            gbv += grad_stats.gbv
            grad_step_count += grad_stats.steps
            value_loss_sum += grad_stats.value_loss_sum
            anchor_ce_sum += grad_stats.anchor_ce_sum
            anchor_ce_count += grad_stats.anchor_ce_count
            suit_consistency_kl_sum += grad_stats.suit_consistency_kl_sum
            suit_consistency_count += grad_stats.suit_consistency_count
            suit_margin_loss_sum += grad_stats.suit_margin_loss_sum
            suit_margin_count += grad_stats.suit_margin_count
            suit_margin_violation_count += grad_stats.suit_margin_violation_count
            suit_margin_teacher_sum += grad_stats.suit_margin_teacher_sum
            suit_margin_student_sum += grad_stats.suit_margin_student_sum
        else:
            rewards = [step_rec.reward for step_rec in traj]
            returns_to_go = _compute_returns(rewards, gamma=float(args.gamma))

            if signal_diagnostics is not None and traj:
                signal_diagnostics.observe(
                    returns_to_go=np.asarray(returns_to_go, dtype=np.float32),
                    value_preds=np.asarray([step_rec.value_pred for step_rec in traj], dtype=np.float32),
                    hidden=np.stack([step_rec.h for step_rec in traj]).astype(np.float32, copy=False),
                )

            # Backprop per ogni step della traiettoria (Monte Carlo A2C).
            for step_rec, g in zip(traj, returns_to_go, strict=True):
                grad_step_count += 1
                v = float(step_rec.value_pred)
                adv = float(g - v)

                # Policy gradient (loss = -adv * log pi(a|s)).
                dlogits = step_rec.probs.copy()
                dlogits[step_rec.action_id] -= 1.0
                dlogits *= float(adv)

                beta = float(args.entropy_beta)
                if beta > 0.0:
                    # Loss include `-beta * H(pi)` per incoraggiare esplorazione.
                    logp = np.log(step_rec.probs + 1e-12)
                    s = float(np.sum(step_rec.probs * (logp + 1.0)))
                    dent = step_rec.probs * (logp + 1.0 - s)
                    dlogits += beta * dent

                # Regularization: stay-close-to-BC anchor (se attivo).
                #
                # Questo termine NON è pesato dall'advantage: è un vincolo "stile" separato dal reward.
                anchor_beta = float(args.bc_anchor_beta)
                if anchor_beta > 0.0 and step_rec.anchor_probs is not None:
                    grad_anchor = grad_ce_wrt_logits_from_probs(
                        pred_probs=step_rec.probs,
                        target_probs=step_rec.anchor_probs,
                        action_mask=step_rec.action_mask,
                    )
                    dlogits += anchor_beta * grad_anchor
                    anchor_ce_sum += float(step_rec.anchor_ce)
                    anchor_ce_count += 1

                # Actor head grads.
                gw2 += np.outer(step_rec.h, dlogits).astype(np.float32)
                gb2 += dlogits.astype(np.float32)
                dh_policy = policy.w2 @ dlogits  # (H,)

                # Critic loss: 0.5 * value_coef * (V - G)^2
                dv = float(args.value_coef) * (v - float(g))
                value_loss_sum += 0.5 * float(args.value_coef) * (v - float(g)) ** 2

                gwv += (step_rec.h * dv).astype(np.float32)
                gbv += dv
                dh_value = policy.wv * dv  # (H,)

                dh = dh_policy + dh_value
                dz1 = dh * (step_rec.z1 > 0.0)
                gw1 += np.outer(step_rec.x, dz1).astype(np.float32)
                gb1 += dz1.astype(np.float32)

            if rng_suit_augmentation is not None and traj:
                paired_stats = _accumulate_paired_suit_trajectory_grads(
                    policy=policy,
                    xs=np.stack([step_rec.x for step_rec in traj]),
                    action_masks=np.stack([step_rec.action_mask for step_rec in traj]),
                    action_ids=np.asarray([step_rec.action_id for step_rec in traj], dtype=np.int64),
                    returns_to_go=np.asarray(returns_to_go, dtype=np.float32),
                    encoder_version=encoder_version,
                    permutation=sample_nonidentity_suit_permutation(rng_suit_augmentation),
                    entropy_beta=float(args.entropy_beta),
                    value_coef=float(args.value_coef),
                    bc_anchor=bc_anchor,
                    bc_anchor_beta=float(args.bc_anchor_beta),
                    gw1=gw1,
                    gb1=gb1,
                    gw2=gw2,
                    gb2=gb2,
                    gwv=gwv,
                )
                gbv += paired_stats.gbv
                grad_step_count += paired_stats.steps
                value_loss_sum += paired_stats.value_loss_sum
                anchor_ce_sum += paired_stats.anchor_ce_sum
                anchor_ce_count += paired_stats.anchor_ce_count
            if rng_suit_consistency is not None and traj:
                consistency_stats = _accumulate_suit_consistency_grads(
                    policy=policy,
                    xs=np.stack([step_rec.x for step_rec in traj]),
                    action_masks=np.stack([step_rec.action_mask for step_rec in traj]),
                    original_probs=np.stack([step_rec.probs for step_rec in traj]),
                    encoder_version=encoder_version,
                    permutation=sample_nonidentity_suit_permutation(rng_suit_consistency),
                    beta=suit_consistency_beta,
                    gw1=gw1,
                    gb1=gb1,
                    gw2=gw2,
                    gb2=gb2,
                )
                suit_consistency_kl_sum += consistency_stats.suit_consistency_kl_sum
                suit_consistency_count += consistency_stats.suit_consistency_count
            if rng_suit_margin is not None and traj:
                margin_stats = _accumulate_suit_margin_grads(
                    policy=policy,
                    xs=np.stack([step_rec.x for step_rec in traj]),
                    action_masks=np.stack([step_rec.action_mask for step_rec in traj]),
                    original_probs=np.stack([step_rec.probs for step_rec in traj]),
                    encoder_version=encoder_version,
                    permutation=sample_nonidentity_suit_permutation(rng_suit_margin),
                    beta=suit_margin_beta,
                    margin_cap=suit_margin_cap,
                    gw1=gw1,
                    gb1=gb1,
                    gw2=gw2,
                    gb2=gb2,
                )
                suit_margin_loss_sum += margin_stats.suit_margin_loss_sum
                suit_margin_count += margin_stats.suit_margin_count
                suit_margin_violation_count += margin_stats.suit_margin_violation_count
                suit_margin_teacher_sum += margin_stats.suit_margin_teacher_sum
                suit_margin_student_sum += margin_stats.suit_margin_student_sum

        # Update ogni `update_every` partite.
        if game_idx % update_every == 0:
            t += 1

            # Normalizziamo per numero di step policy osservati (più robusto di /update_every).
            # In 2-player i step per game sono ~20, ma può variare per seat-fair/fine partita.
            total_steps = max(1, grad_step_count)
            scale = 1.0 / float(total_steps)
            gw1 *= scale
            gb1 *= scale
            gw2 *= scale
            gb2 *= scale
            gwv *= scale
            gbv *= scale

            wd = float(args.weight_decay)
            if wd > 0.0:
                gw1 += wd * policy.w1
                gw2 += wd * policy.w2
                gwv += wd * policy.wv

            diagnostic_signal_snapshot = None
            diagnostic_gradient_groups = None
            diagnostic_parameters_before = None
            should_sample_diagnostics = signal_diagnostics is not None and (
                t == 1
                or t % int(args.diagnostics_every) == 0
                or game_idx in checkpoint_games
                or game_idx == stop_after_games
            )
            if should_sample_diagnostics:
                assert signal_diagnostics is not None
                diagnostic_signal_snapshot = signal_diagnostics.snapshot()
                diagnostic_gradient_groups = A2CArrayGroups(
                    trunk=(gw1, gb1),
                    actor_head=(gw2, gb2),
                    critic_head=(gwv, np.asarray([gbv], dtype=np.float32)),
                )
                diagnostic_parameters_before = _policy_parameter_groups().copied()

            _adam_update(policy.w1, gw1, state=st_w1, lr=float(args.lr), t=t)
            _adam_update(policy.b1, gb1, state=st_b1, lr=float(args.lr), t=t)
            _adam_update(policy.w2, gw2, state=st_w2, lr=float(args.lr), t=t)
            _adam_update(policy.b2, gb2, state=st_b2, lr=float(args.lr), t=t)
            _adam_update(policy.wv, gwv, state=st_wv, lr=float(args.lr), t=t)

            # `bv` lo aggiorniamo come un array 1D di lunghezza 1 per riusare Adam.
            bv_arr = np.asarray([policy.bv], dtype=np.float32)
            _adam_update(bv_arr, np.asarray([gbv], dtype=np.float32), state=st_bv, lr=float(args.lr), t=t)
            policy.bv = float(bv_arr[0])

            if should_sample_diagnostics:
                assert signal_diagnostics is not None
                assert diagnostic_signal_snapshot is not None
                assert diagnostic_gradient_groups is not None
                assert diagnostic_parameters_before is not None
                update_diagnostics.append(
                    build_update_diagnostics(
                        iteration=t,
                        games=game_idx,
                        signals=diagnostic_signal_snapshot,
                        gradients=diagnostic_gradient_groups,
                        parameters_before=diagnostic_parameters_before,
                        parameters_after=_policy_parameter_groups(),
                    )
                )
            if signal_diagnostics is not None:
                signal_diagnostics.reset()

            gw1.fill(0.0)
            gb1.fill(0.0)
            gw2.fill(0.0)
            gb2.fill(0.0)
            gwv.fill(0.0)
            gbv = 0.0

            avg_ret = float(np.mean(returns_buf)) if returns_buf else 0.0
            win_rate = float(wins) / float(update_every)
            draw_rate = float(draws) / float(update_every)
            avg_ent = float(np.mean(entropies)) if entropies else 0.0
            vloss = float(value_loss_sum) / float(grad_step_count) if grad_step_count > 0 else 0.0
            avg_anchor_ce = float(anchor_ce_sum) / float(anchor_ce_count) if anchor_ce_count > 0 else 0.0
            avg_suit_consistency_kl = (
                float(suit_consistency_kl_sum) / float(suit_consistency_count) if suit_consistency_count > 0 else 0.0
            )
            avg_suit_margin_loss = (
                float(suit_margin_loss_sum) / float(suit_margin_count) if suit_margin_count > 0 else 0.0
            )
            suit_margin_violation_rate = (
                float(suit_margin_violation_count) / float(suit_margin_count) if suit_margin_count > 0 else 0.0
            )
            avg_suit_teacher_margin = (
                float(suit_margin_teacher_sum) / float(suit_margin_count) if suit_margin_count > 0 else 0.0
            )
            avg_suit_student_margin = (
                float(suit_margin_student_sum) / float(suit_margin_count) if suit_margin_count > 0 else 0.0
            )

            row = TrainMetrics(
                iter=t,
                games=game_idx,
                avg_return=avg_ret,
                win_rate=win_rate,
                draw_rate=draw_rate,
                avg_entropy=avg_ent,
                value_loss=vloss,
                avg_anchor_ce=avg_anchor_ce,
            )
            metrics.append(asdict(row))
            if suit_consistency_beta > 0.0:
                suit_consistency_metrics.append(
                    {
                        "iter": t,
                        "games": game_idx,
                        "avg_kl": avg_suit_consistency_kl,
                    }
                )
            if suit_margin_beta > 0.0:
                suit_margin_metrics.append(
                    {
                        "iter": t,
                        "games": game_idx,
                        "avg_hinge": avg_suit_margin_loss,
                        "violation_rate": suit_margin_violation_rate,
                        "avg_teacher_margin": avg_suit_teacher_margin,
                        "avg_student_margin": avg_suit_student_margin,
                    }
                )

            if t % int(args.log_every) == 0 or game_idx == update_every:
                anchor_hint = "" if float(args.bc_anchor_beta) <= 0.0 else f" | anchor_ce {row.avg_anchor_ce:.3f}"
                consistency_hint = "" if suit_consistency_beta <= 0.0 else f" | suit_kl {avg_suit_consistency_kl:.4f}"
                margin_hint = (
                    ""
                    if suit_margin_beta <= 0.0
                    else f" | suit_hinge {avg_suit_margin_loss:.3f} viol {suit_margin_violation_rate:.3f}"
                )
                print(
                    f"iter {t:04d} | games {game_idx:06d} | "
                    f"avg_return {row.avg_return:+.3f} | win {row.win_rate:.3f} draw {row.draw_rate:.3f} | "
                    f"entropy {row.avg_entropy:.3f} | vloss {row.value_loss:.4f}"
                    f"{anchor_hint}{consistency_hint}{margin_hint}"
                )

            returns_buf.clear()
            wins = 0
            draws = 0
            entropies.clear()
            grad_step_count = 0
            value_loss_sum = 0.0
            anchor_ce_sum = 0.0
            anchor_ce_count = 0
            suit_consistency_kl_sum = 0.0
            suit_consistency_count = 0
            suit_margin_loss_sum = 0.0
            suit_margin_count = 0
            suit_margin_violation_count = 0
            suit_margin_teacher_sum = 0.0
            suit_margin_student_sum = 0.0

            if game_idx in checkpoint_games:
                _save_model(_checkpoint_path(game_idx), trained_games=game_idx, is_checkpoint=True)

    _save_model(out_path, trained_games=stop_after_games, is_checkpoint=False)
    _write_diagnostics_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
