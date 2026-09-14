"""
Agente V-lookahead: policy base + solver finale + value model sulle foglie.

Questo modulo implementa lo Stage 1 dell'ipotesi V-lookahead. L'agente non prova a
distillare la decisione PIMC in una policy reattiva: mantiene una lookahead corta a runtime
e usa una rete di valore scalare per valutare le foglie.

Anti-cheat
----------
L'agente riceve solo `PlayerObservation`. Quando serve ragionare su carte ignote, campiona
stati completi con `determinize_observation`, che usa solo informazione pubblica e mano del
giocatore. Il value model viene sempre interrogato su una nuova `PlayerObservation` lecita,
mai su `GameState` completo.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from ...domain.engine import PlayCardAction, step
from ...domain.observation import PlayerObservation, make_player_observation
from ...domain.state import GameState
from ..encoding.observation_encoder import encode_player_observation_2p
from ..endgame.fast_solver import solve_endgame_fast
from ..models.bc_model import apply_overkill_guard_second_hand
from ..models.belief_model import MLPBeliefModel
from ..models.value_model import MLPValueModel, infer_value_encoder_version
from .base import Agent
from .hybrid_endgame import reconstruct_endgame_state
from .pimc import belief_card_weights, determinize_observation, rollout_to_terminal, unknown_live_card_count


@dataclass(slots=True)
class ValueLookaheadStats:
    """Contatori runtime dell'agente V-lookahead."""

    total_decisions: int = 0
    lookahead_decisions: int = 0
    fallback_decisions: int = 0
    endgame_solver_decisions: int = 0
    failed_determinizations: int = 0
    successful_determinizations: int = 0
    failed_leaf_evaluations: int = 0
    completed_leaf_evaluations: int = 0
    overkill_guard_adjustments: int = 0
    search_elapsed_seconds: float = 0.0

    @property
    def seconds_per_lookahead_decision(self) -> float:
        """Tempo medio per decisione in cui la lookahead è stata davvero usata."""
        if self.lookahead_decisions <= 0:
            return 0.0
        return self.search_elapsed_seconds / float(self.lookahead_decisions)


def _score_delta_for_player(state: GameState, player_index: int) -> int:
    """Delta punti dal punto di vista di `player_index` in 2-player."""
    opponent = 1 - int(player_index)
    return int(state.players[player_index].points) - int(state.players[opponent].points)


def _safe_card_index(agent: Agent, observation: PlayerObservation, *, rng: random.Random) -> int:
    """Chiede una mossa a un agente e rifiuta indici fuori mano."""
    card_index = int(agent.choose_card_index(observation, rng=rng))
    if not 0 <= card_index < len(observation.hand):
        raise ValueError(f"Agente {agent.name!r} ha prodotto card_index={card_index}, mano={len(observation.hand)}")
    return card_index


def _resolve_current_trick(
    state: GameState,
    *,
    continuation_agent: Agent,
    rng: random.Random,
) -> GameState:
    """
    Porta lo stato al prossimo decision boundary dopo la carta candidata.

    Se la carta candidata ha aperto una presa, facciamo rispondere l'avversario con la policy
    di continuazione. Se invece ha chiuso la presa, `domain.step` ha già risolto il trick.
    """
    if state.game_over or len(state.table_cards) != 1:
        return state

    current = int(state.current_turn)
    observation = make_player_observation(state, current)
    card_index = _safe_card_index(continuation_agent, observation, rng=rng)
    next_state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
    if result.error:
        raise RuntimeError(f"Errore dominio durante risposta di continuazione: {result.error}")
    return next_state


@dataclass
class ValueLookaheadAgent:
    """
    Agente depth-1 basato su value model.

    Concorrenza (assunzione di contratto): UNA istanza per partita, mai condivisa tra
    richieste concorrenti — `metrics` è stato mutabile non thread-safe (vedi la nota
    analoga in `PIMCAgent`).

    Scelta:
    - mazzo vuoto: solver esatto ricostruito da `PlayerObservation`;
    - troppe carte ignote o errore: fallback `v6 + solver`;
    - finestra validata: campiona determinizzazioni compatibili, prova ogni carta, risolve la presa corrente
      con `continuation_agent`, valuta la foglia con `V`, media e sceglie il valore più alto.
    """

    value_model: MLPValueModel
    fallback: Agent
    continuation_agent: Agent
    num_determinizations: int = 8
    max_unknown_cards: int = 8
    overkill_guard_enabled: bool = True
    # Fase 2 (belief): come in PIMCAgent, pesa le determinizzazioni con la belief network.
    belief_model: MLPBeliefModel | None = None
    belief_uniform_mix: float = 0.10
    name: str = "value_lookahead"
    metrics: ValueLookaheadStats = field(default_factory=ValueLookaheadStats)

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        if not observation.hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        self.metrics.total_decisions += 1

        if observation.deck_size == 0:
            try:
                card_index = solve_endgame_fast(reconstruct_endgame_state(observation)).best_card_index
                if 0 <= card_index < len(observation.hand):
                    self.metrics.endgame_solver_decisions += 1
                    return card_index
            except ValueError:
                pass
            self.metrics.fallback_decisions += 1
            return _safe_card_index(self.fallback, observation, rng=rng)

        try:
            if unknown_live_card_count(observation) > int(self.max_unknown_cards):
                self.metrics.fallback_decisions += 1
                return _safe_card_index(self.fallback, observation, rng=rng)
        except ValueError:
            self.metrics.fallback_decisions += 1
            return _safe_card_index(self.fallback, observation, rng=rng)

        try:
            choice = self._choose_with_lookahead(observation, rng=rng)
        except RuntimeError, ValueError:
            self.metrics.fallback_decisions += 1
            return _safe_card_index(self.fallback, observation, rng=rng)
        if choice is None:
            self.metrics.fallback_decisions += 1
            return _safe_card_index(self.fallback, observation, rng=rng)
        if self.overkill_guard_enabled:
            guarded = apply_overkill_guard_second_hand(observation, chosen_card_index=int(choice))
            if int(guarded) != int(choice):
                self.metrics.overkill_guard_adjustments += 1
            return int(guarded)
        return int(choice)

    def _choose_with_lookahead(self, observation: PlayerObservation, *, rng: random.Random) -> int | None:
        """Esegue depth-1 lookahead su determinizzazioni compatibili."""
        legal_indices = list(range(len(observation.hand)))
        totals = [0.0 for _ in legal_indices]
        counts = [0 for _ in legal_indices]
        encoder_version = infer_value_encoder_version(self.value_model)

        self.metrics.lookahead_decisions += 1
        started = time.perf_counter()
        card_weights: dict[int, float] | None = None
        if self.belief_model is not None:
            card_weights = belief_card_weights(self.belief_model, observation, uniform_mix=self.belief_uniform_mix)
        try:
            for sample_idx in range(max(1, int(self.num_determinizations))):
                sample_rng = random.Random(rng.randrange(0, 2**32) ^ (sample_idx * 0x9E3779B9))
                try:
                    sampled_state = determinize_observation(observation, rng=sample_rng, card_weights=card_weights)
                except ValueError:
                    self.metrics.failed_determinizations += 1
                    continue
                self.metrics.successful_determinizations += 1
                if sampled_state.current_turn != observation.player_index:
                    self.metrics.failed_determinizations += 1
                    continue

                for pos, card_index in enumerate(legal_indices):
                    next_state, result = step(
                        sampled_state,
                        PlayCardAction(player_index=sampled_state.current_turn, card_index=card_index),
                    )
                    if result.error:
                        self.metrics.failed_leaf_evaluations += 1
                        continue
                    leaf_rng = random.Random(sample_rng.randrange(0, 2**32) ^ (card_index * 0x85EBCA6B))
                    try:
                        leaf_state = _resolve_current_trick(
                            next_state,
                            continuation_agent=self.continuation_agent,
                            rng=leaf_rng,
                        )
                        score = self._value_leaf_score_for_root(
                            leaf_state,
                            root_player=observation.player_index,
                            encoder_version=encoder_version,
                            rng=leaf_rng,
                        )
                    except RuntimeError, ValueError:
                        self.metrics.failed_leaf_evaluations += 1
                        continue
                    totals[pos] += float(score)
                    counts[pos] += 1
                    self.metrics.completed_leaf_evaluations += 1
        finally:
            self.metrics.search_elapsed_seconds += time.perf_counter() - started

        valid_positions = [pos for pos, count in enumerate(counts) if count > 0]
        if not valid_positions:
            return None
        best_pos = max(valid_positions, key=lambda pos: (totals[pos] / counts[pos], -legal_indices[pos]))
        return legal_indices[best_pos]

    def _value_leaf_score_for_root(
        self,
        leaf_state: GameState,
        *,
        root_player: int,
        encoder_version,
        rng: random.Random,
    ) -> float:
        """Valuta una foglia in punti dal punto di vista del root player."""
        if leaf_state.game_over:
            return float(_score_delta_for_player(leaf_state, root_player))

        if len(leaf_state.deck) == 0:
            terminal = rollout_to_terminal(
                leaf_state,
                rollout_agent=self.continuation_agent,
                rng=rng,
                use_endgame_solver=True,
            )
            return float(_score_delta_for_player(terminal, root_player))

        leaf_player = int(leaf_state.current_turn)
        observation = make_player_observation(leaf_state, leaf_player)
        encoded = encode_player_observation_2p(observation, version=encoder_version)
        current_delta = float(_score_delta_for_player(leaf_state, leaf_player))
        pred_for_leaf_player = float(
            self.value_model.predict_points(
                np.asarray(encoded.features, dtype=np.float32), current_score_delta=current_delta
            )
        )
        return pred_for_leaf_player if leaf_player == root_player else -pred_for_leaf_player
