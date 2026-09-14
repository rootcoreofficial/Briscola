"""
Stima controfattuale degli errori decisionali di una policy 2-player.

La forza media di un modello non spiega *dove* perda punti. Questo modulo parte da una
sola :class:`PlayerObservation`, prova ogni carta legale su mondi nascosti compatibili e
stima il vantaggio della migliore alternativa rispetto alla carta scelta dalla policy.

Il confine anti-cheat e' intenzionalmente strutturale: l'API pubblica accetta una
``PlayerObservation`` e non un ``GameState`` reale. A mazzo vuoto la mano avversaria e'
deducibile dall'informazione pubblica e il solver fornisce un risultato esatto. Prima
dell'endgame usiamo determinizzazioni PIMC e rollout, quindi parliamo sempre di *regret
stimato*, non di errore matematicamente provato.

Per ridurre il winner's curse, le determinizzazioni hanno due ruoli separati:

1. la prima meta' sceglie una sola alternativa candidata;
2. la seconda meta' stima, con differenze paired, il vantaggio candidato-mossa policy.

Una variante non viene quindi scelta e certificata sugli stessi campioni.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from statistics import fmean

from ...domain.card_id import card_to_id
from ...domain.engine import PlayCardAction, step
from ...domain.models import Card
from ...domain.observation import PlayerObservation
from ...domain.rules import who_wins_trick
from ..agents.base import Agent
from ..agents.hybrid_endgame import reconstruct_endgame_state
from ..agents.pimc import belief_card_weights, determinize_observation, rollout_to_terminal, unknown_live_card_count
from ..endgame.fast_solver import solve_endgame_fast
from ..models.belief_model import MLPBeliefModel
from ..training.reward_shaping import card_conservation_cost

type ActionScoreRow = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PolicyRegretConfig:
    """Parametri statistici e di rollout della stima controfattuale."""

    determinizations: int = 64
    min_regret_points: float = 1.0
    confidence_z: float = 2.576
    belief_uniform_mix: float = 0.10
    max_unknown_cards_for_phase: int = 8
    use_endgame_solver: bool = True

    def validate(self) -> None:
        """Rifiuta configurazioni che non consentono selection/evaluation indipendenti."""
        if self.determinizations < 4 or self.determinizations % 2 != 0:
            raise ValueError("determinizations deve essere pari e >= 4")
        if self.min_regret_points < 0.0:
            raise ValueError("min_regret_points deve essere >= 0")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z deve essere > 0")
        if not 0.0 <= self.belief_uniform_mix <= 1.0:
            raise ValueError("belief_uniform_mix deve essere tra 0 e 1")
        if self.max_unknown_cards_for_phase < 0:
            raise ValueError("max_unknown_cards_for_phase deve essere >= 0")


@dataclass(frozen=True, slots=True)
class PolicyRegretActionValue:
    """Valore medio di una carta nei due split e nell'insieme completo dei campioni."""

    card_index: int
    card_id: int
    selection_mean_score: float
    evaluation_mean_score: float
    overall_mean_score: float


@dataclass(frozen=True, slots=True)
class PolicyRegretEstimate:
    """Risultato completo per una singola decisione non forzata."""

    method: str
    phase: str
    position: str
    unknown_live_cards: int
    chosen_card_index: int
    chosen_card_id: int
    alternative_card_index: int
    alternative_card_id: int
    selection_best_card_index: int
    evaluation_best_card_index: int
    candidate_confirmed_as_evaluation_best: bool
    selection_sample_count: int
    evaluation_sample_count: int
    successful_determinizations: int
    failed_determinizations: int
    regret_mean: float
    regret_standard_error: float
    regret_confidence_low: float
    regret_confidence_high: float
    reliable_error: bool
    tags: tuple[str, ...]
    action_values: tuple[PolicyRegretActionValue, ...]


def observation_phase(observation: PlayerObservation, *, max_unknown_cards: int = 8) -> str:
    """Bucket di fase coerente con il collector value del progetto."""
    unknown = unknown_live_card_count(observation)
    if observation.deck_size == 0:
        return "endgame"
    if unknown <= int(max_unknown_cards):
        return "pimc_window"
    if observation.deck_size <= 10:
        return "mid"
    return "early"


def _validate_observation(observation: PlayerObservation, chosen_card_index: int) -> None:
    """Valida lo scope senza accettare accidentalmente stato completo o partite 4-player."""
    if observation.num_players != 2 or observation.is_team_game:
        raise ValueError("La sonda policy-regret supporta solo osservazioni 2-player")
    if observation.game_over:
        raise ValueError("Partita gia' terminata")
    if observation.current_turn != observation.player_index:
        raise ValueError("L'osservazione non appartiene al giocatore di turno")
    if len(observation.hand) < 2:
        raise ValueError("La decisione e' forzata: servono almeno due carte in mano")
    if chosen_card_index < 0 or chosen_card_index >= len(observation.hand):
        raise ValueError(f"chosen_card_index fuori range: {chosen_card_index}")


def _standard_error(values: tuple[float, ...]) -> float:
    """Errore standard campionario; con un solo valore non c'e' informazione di varianza."""
    if len(values) <= 1:
        return 0.0
    mean = fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance / len(values))


def _best_action_index(rows: tuple[ActionScoreRow, ...]) -> int:
    """Argmax dei valori medi, con tie-break stabile sull'indice mano piu' basso."""
    if not rows or not rows[0]:
        raise ValueError("Matrice score vuota")
    action_count = len(rows[0])
    if any(len(row) != action_count for row in rows):
        raise ValueError("Righe score con numero di azioni incoerente")
    means = [fmean(row[action_index] for row in rows) for action_index in range(action_count)]
    return max(range(action_count), key=lambda action_index: (means[action_index], -action_index))


def _card_wins_current_trick(observation: PlayerObservation, card: Card) -> bool | None:
    """Esito pubblico della presa corrente, oppure ``None`` quando il player apre."""
    if len(observation.table_cards) != 1 or observation.trump_card is None:
        return None
    lead_card, lead_player = observation.table_cards[0]
    trick = ((lead_card, lead_player), (card, observation.player_index))
    return who_wins_trick(trick, observation.trump_card.suit) == observation.player_index


def classify_policy_regret(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    alternative_card_index: int,
) -> tuple[str, ...]:
    """
    Assegna etichette meccaniche basate soltanto su mano, tavolo e briscola pubblica.

    Le etichette descrivono il cambiamento tra mossa policy e alternativa; non pretendono
    di essere una spiegazione causale del modello. Possono sovrapporsi.
    """
    chosen = observation.hand[chosen_card_index]
    alternative = observation.hand[alternative_card_index]
    trump_suit = observation.trump_card.suit if observation.trump_card is not None else None
    chosen_is_trump = trump_suit is not None and chosen.suit == trump_suit
    alternative_is_trump = trump_suit is not None and alternative.suit == trump_suit
    tags: list[str] = []

    if not observation.table_cards:
        if chosen_is_trump and not alternative_is_trump:
            tags.append("premature_trump_lead")
        if chosen.rank.points >= 10 and alternative.rank.points < 10:
            tags.append("high_card_exposure_lead")
    else:
        chosen_wins = bool(_card_wins_current_trick(observation, chosen))
        alternative_wins = bool(_card_wins_current_trick(observation, alternative))
        lead_card, _lead_player = observation.table_cards[0]

        if alternative_wins and not chosen_wins:
            tags.append("missed_winning_reply")
            if lead_card.rank.points >= 4:
                tags.append("missed_rich_trick")
            if lead_card.rank.points >= 10:
                tags.append("missed_load_capture")
        if chosen_is_trump and not alternative_is_trump:
            tags.append("trump_to_non_trump")
        if chosen_wins and not alternative_wins:
            tags.append("give_up_current_trick")
            if lead_card.rank.points + chosen.rank.points <= 2:
                tags.append("give_up_low_value_trick")
        if (
            chosen_is_trump
            and alternative_is_trump
            and chosen_wins
            and alternative_wins
            and card_conservation_cost(chosen) > card_conservation_cost(alternative)
        ):
            tags.append("trump_overkill")
        if chosen.rank.points >= 10 and not chosen_wins and alternative.rank.points < chosen.rank.points:
            tags.append("high_card_discard")

    return tuple(tags or ("other",))


def _crossfit_estimate(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    score_rows: tuple[ActionScoreRow, ...],
    failed_determinizations: int,
    config: PolicyRegretConfig,
) -> PolicyRegretEstimate:
    """Sceglie l'alternativa sul primo split e ne stima il regret sul secondo."""
    if len(score_rows) < 4:
        raise ValueError("Servono almeno quattro determinizzazioni riuscite")
    split = len(score_rows) // 2
    selection_rows = score_rows[:split]
    evaluation_rows = score_rows[split:]
    if len(selection_rows) < 2 or len(evaluation_rows) < 2:
        raise ValueError("Gli split selection/evaluation devono contenere almeno due campioni")

    alternative_index = _best_action_index(selection_rows)
    evaluation_best_index = _best_action_index(evaluation_rows)
    regret_samples = tuple(row[alternative_index] - row[chosen_card_index] for row in evaluation_rows)
    regret_mean = fmean(regret_samples)
    regret_se = _standard_error(regret_samples)
    ci_low = regret_mean - config.confidence_z * regret_se
    ci_high = regret_mean + config.confidence_z * regret_se
    is_reliable = alternative_index != chosen_card_index and regret_mean >= config.min_regret_points and ci_low > 0.0

    action_values = tuple(
        PolicyRegretActionValue(
            card_index=card_index,
            card_id=card_to_id(card),
            selection_mean_score=fmean(row[card_index] for row in selection_rows),
            evaluation_mean_score=fmean(row[card_index] for row in evaluation_rows),
            overall_mean_score=fmean(row[card_index] for row in score_rows),
        )
        for card_index, card in enumerate(observation.hand)
    )
    return PolicyRegretEstimate(
        method="sampled_crossfit",
        phase=observation_phase(observation, max_unknown_cards=config.max_unknown_cards_for_phase),
        position="lead" if not observation.table_cards else "response",
        unknown_live_cards=unknown_live_card_count(observation),
        chosen_card_index=chosen_card_index,
        chosen_card_id=card_to_id(observation.hand[chosen_card_index]),
        alternative_card_index=alternative_index,
        alternative_card_id=card_to_id(observation.hand[alternative_index]),
        selection_best_card_index=alternative_index,
        evaluation_best_card_index=evaluation_best_index,
        candidate_confirmed_as_evaluation_best=alternative_index == evaluation_best_index,
        selection_sample_count=len(selection_rows),
        evaluation_sample_count=len(evaluation_rows),
        successful_determinizations=len(score_rows),
        failed_determinizations=failed_determinizations,
        regret_mean=regret_mean,
        regret_standard_error=regret_se,
        regret_confidence_low=ci_low,
        regret_confidence_high=ci_high,
        reliable_error=is_reliable,
        tags=classify_policy_regret(
            observation,
            chosen_card_index=chosen_card_index,
            alternative_card_index=alternative_index,
        ),
        action_values=action_values,
    )


def _score_from_player_view(*, final_delta_p0_p1: float, player_index: int) -> float:
    """Converte il delta canonico P0-P1 nel punto di vista del decisore."""
    return float(final_delta_p0_p1 if player_index == 0 else -final_delta_p0_p1)


def _estimate_exact_endgame(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    config: PolicyRegretConfig,
) -> PolicyRegretEstimate:
    """Valuta ogni carta con minimax esatto sullo stato deducibile a mazzo vuoto."""
    reconstructed = reconstruct_endgame_state(observation)
    scores: list[float] = []
    for card_index in range(len(observation.hand)):
        child, result = step(
            reconstructed,
            PlayCardAction(player_index=reconstructed.current_turn, card_index=card_index),
        )
        if result.error:
            raise RuntimeError(f"Errore dominio nel controfattuale endgame: {result.error}")
        if child.game_over:
            final_delta = child.players[0].points - child.players[1].points
        else:
            final_delta = solve_endgame_fast(child).final_delta_p0_p1
        scores.append(_score_from_player_view(final_delta_p0_p1=final_delta, player_index=observation.player_index))

    alternative_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
    regret = scores[alternative_index] - scores[chosen_card_index]
    action_values = tuple(
        PolicyRegretActionValue(
            card_index=card_index,
            card_id=card_to_id(card),
            selection_mean_score=scores[card_index],
            evaluation_mean_score=scores[card_index],
            overall_mean_score=scores[card_index],
        )
        for card_index, card in enumerate(observation.hand)
    )
    return PolicyRegretEstimate(
        method="exact_endgame",
        phase="endgame",
        position="lead" if not observation.table_cards else "response",
        unknown_live_cards=unknown_live_card_count(observation),
        chosen_card_index=chosen_card_index,
        chosen_card_id=card_to_id(observation.hand[chosen_card_index]),
        alternative_card_index=alternative_index,
        alternative_card_id=card_to_id(observation.hand[alternative_index]),
        selection_best_card_index=alternative_index,
        evaluation_best_card_index=alternative_index,
        candidate_confirmed_as_evaluation_best=True,
        selection_sample_count=1,
        evaluation_sample_count=1,
        successful_determinizations=1,
        failed_determinizations=0,
        regret_mean=regret,
        regret_standard_error=0.0,
        regret_confidence_low=regret,
        regret_confidence_high=regret,
        reliable_error=alternative_index != chosen_card_index and regret >= config.min_regret_points,
        tags=classify_policy_regret(
            observation,
            chosen_card_index=chosen_card_index,
            alternative_card_index=alternative_index,
        ),
        action_values=action_values,
    )


def estimate_policy_regret(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    rollout_agent: Agent,
    rng: random.Random,
    config: PolicyRegretConfig,
    belief_model: MLPBeliefModel | None = None,
) -> PolicyRegretEstimate:
    """
    Stima il costo della scelta policy rispetto a una sola alternativa cross-fitted.

    Ogni riga della matrice usa lo stesso mondo determinizzato e lo stesso seed di
    continuazione per tutte le carte. Cio' rende il confronto paired e riduce il rumore
    senza mostrare al modello la mano avversaria reale.
    """
    config.validate()
    _validate_observation(observation, chosen_card_index)
    if observation.deck_size == 0 and config.use_endgame_solver:
        return _estimate_exact_endgame(observation, chosen_card_index=chosen_card_index, config=config)

    card_weights: dict[int, float] | None = None
    if belief_model is not None:
        card_weights = belief_card_weights(
            belief_model,
            observation,
            uniform_mix=config.belief_uniform_mix,
        )

    score_rows: list[ActionScoreRow] = []
    failed = 0
    for _sample_index in range(config.determinizations):
        determinization_seed = rng.randrange(0, 2**63)
        rollout_seed = rng.randrange(0, 2**63)
        try:
            sampled_state = determinize_observation(
                observation,
                rng=random.Random(determinization_seed),
                card_weights=card_weights,
            )
        except ValueError:
            failed += 1
            continue

        row: list[float] = []
        row_failed = False
        for card_index in range(len(observation.hand)):
            child, result = step(
                sampled_state,
                PlayCardAction(player_index=sampled_state.current_turn, card_index=card_index),
            )
            if result.error:
                row_failed = True
                break
            try:
                final_state = rollout_to_terminal(
                    child,
                    rollout_agent=rollout_agent,
                    rng=random.Random(rollout_seed),
                    use_endgame_solver=config.use_endgame_solver,
                )
            except RuntimeError:
                row_failed = True
                break
            final_delta = final_state.players[0].points - final_state.players[1].points
            row.append(_score_from_player_view(final_delta_p0_p1=final_delta, player_index=observation.player_index))
        if row_failed:
            failed += 1
            continue
        score_rows.append(tuple(row))

    return _crossfit_estimate(
        observation,
        chosen_card_index=chosen_card_index,
        score_rows=tuple(score_rows),
        failed_determinizations=failed,
        config=config,
    )


__all__ = [
    "ActionScoreRow",
    "PolicyRegretActionValue",
    "PolicyRegretConfig",
    "PolicyRegretEstimate",
    "classify_policy_regret",
    "estimate_policy_regret",
    "observation_phase",
]
