"""
Diagnostica di equivarianza della policy alla rinomina dei semi.

La Briscola non attribuisce un significato assoluto ai nomi `clubs`, `cups`, `coins` e
`swords`: se rinominiamo coerentemente tutte le carte, inclusa la briscola e la storia
pubblica, la distribuzione sulle mosse dovrebbe rinominarsi nello stesso modo.

Questo modulo lavora esclusivamente su :class:`PlayerObservation`. Non permuta il vettore
di feature a mano: trasforma l'osservazione semanticamente e riesegue l'encoder, così anche
i blocchi aggregati v3/v4 seguono le stesse regole del runtime reale.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import cast

import numpy as np

from ...domain.card_id import SUIT_ORDER, SUIT_TO_INDEX, card_to_id, id_to_card
from ...domain.models import Card, Suit
from ...domain.observation import PlayerObservation
from ...domain.state import TrickRecord
from ..encoding.observation_encoder import encode_player_observation_2p
from ..models.bc_model import BCModelAgent, MLPBCModel

SuitPermutation = tuple[Suit, Suit, Suit, Suit]
"""Permutazione `source -> target`, indicizzata secondo ``SUIT_ORDER``."""

IDENTITY_SUIT_PERMUTATION: SuitPermutation = (Suit.CLUBS, Suit.CUPS, Suit.COINS, Suit.SWORDS)


@dataclass(frozen=True, slots=True)
class SuitSymmetryComparison:
    """
    Confronto della policy per una singola rinomina dei semi.

    ``near_tie`` è vero se il gap top-2 è sotto soglia nella baseline oppure nella
    distribuzione rinominata e rimappata. In questo modo un flip non viene dichiarato
    robusto quando uno dei due argmax dipende da un quasi-pareggio.
    """

    permutation: tuple[str, str, str, str]
    is_identity: bool
    agreement: bool
    remapped_action_id: int
    remapped_top2_gap: float
    js_divergence_bits: float
    max_abs_probability_delta: float
    near_tie: bool


@dataclass(frozen=True, slots=True)
class ObservationSuitSymmetry:
    """
    Risultato completo sulle 24 rinomine di una osservazione.

    ``remapped_probabilities`` segue lo stesso ordine di ``comparisons``; ogni riga è già
    riportata negli action id dell'osservazione originale, quindi le righe sono confrontabili
    direttamente anche a coppie.
    """

    baseline_action_id: int
    baseline_top2_gap: float
    comparisons: tuple[SuitSymmetryComparison, ...]
    remapped_probabilities: tuple[tuple[float, ...], ...]


def validate_suit_permutation(permutation: Sequence[Suit]) -> SuitPermutation:
    """Valida e normalizza una biiezione sui quattro semi canonici."""
    if len(permutation) != len(SUIT_ORDER):
        raise ValueError(f"Permutazione semi di lunghezza {len(permutation)} (atteso 4)")
    normalized = tuple(permutation)
    if any(not isinstance(suit, Suit) for suit in normalized):
        raise ValueError("La permutazione deve contenere soltanto valori Suit")
    if set(normalized) != set(SUIT_ORDER):
        raise ValueError("La permutazione deve essere una biiezione dei quattro semi")
    return normalized  # type: ignore[return-value]


def all_suit_permutations() -> tuple[SuitPermutation, ...]:
    """Ritorna le 24 permutazioni, con l'identità come primo elemento."""
    return tuple(validate_suit_permutation(item) for item in itertools.permutations(SUIT_ORDER))


def inverse_suit_permutation(permutation: Sequence[Suit]) -> SuitPermutation:
    """Costruisce l'inversa di una permutazione `source -> target`."""
    normalized = validate_suit_permutation(permutation)
    inverse: list[Suit | None] = [None] * len(SUIT_ORDER)
    for source_index, target_suit in enumerate(normalized):
        inverse[SUIT_TO_INDEX[target_suit]] = SUIT_ORDER[source_index]
    if any(suit is None for suit in inverse):  # difesa: la validazione garantisce la biiezione
        raise AssertionError("Inversa della permutazione incompleta")
    return validate_suit_permutation(inverse)  # type: ignore[arg-type]


@lru_cache(maxsize=24)
def _card_id_mapping(permutation: SuitPermutation) -> tuple[int, ...]:
    """Precalcola la mappa dei 40 id usando le conversioni canoniche del dominio."""
    mapping: list[int] = []
    for card_id in range(40):
        card = id_to_card(card_id)
        target_suit = permutation[SUIT_TO_INDEX[card.suit]]
        mapping.append(card_to_id(Card(suit=target_suit, rank=card.rank)))
    return tuple(mapping)


def permute_card_id(card_id: int, permutation: Sequence[Suit]) -> int:
    """Applica la rinomina al card/action id canonico in ``[0, 39]``."""
    normalized = validate_suit_permutation(permutation)
    if card_id < 0 or card_id >= 40:
        raise ValueError(f"card_id fuori range: {card_id} (atteso 0..39)")
    return _card_id_mapping(normalized)[card_id]


def permute_card(card: Card, permutation: Sequence[Suit]) -> Card:
    """Rinomina il seme di una carta preservandone il rango."""
    normalized = validate_suit_permutation(permutation)
    return Card(suit=normalized[SUIT_TO_INDEX[card.suit]], rank=card.rank)


def permute_action_vector[T](values: Sequence[T], permutation: Sequence[Suit]) -> tuple[T, ...]:
    """Push-forward di un vettore carta: ``out[permute(i)] = values[i]``."""
    if len(values) != 40:
        raise ValueError(f"Vettore azioni di lunghezza {len(values)} (atteso 40)")
    normalized = validate_suit_permutation(permutation)
    # La lunghezza validata rende sicuro l'accesso e permette anche vettori che
    # contengono legittimamente ``None``, senza usarlo come sentinella interna.
    out = [values[0] for _ in range(40)]
    for card_id, target_id in enumerate(_card_id_mapping(normalized)):
        out[target_id] = values[card_id]
    return tuple(out)


def _permute_trick_record(record: TrickRecord, permutation: Sequence[Suit]) -> TrickRecord:
    """Rinomina le carte pubbliche di una presa senza cambiare ordine o giocatori."""
    return replace(
        record,
        cards=tuple((permute_card(card, permutation), player_index) for card, player_index in record.cards),
    )


def permute_player_observation(
    observation: PlayerObservation,
    permutation: Sequence[Suit],
) -> PlayerObservation:
    """
    Rinomina ogni campo-carta di una osservazione lecita.

    Indici giocatore, punteggi, dimensioni, vincitore e ordine delle giocate restano
    invariati. One-hot e storia vengono trasformati insieme alla mano: ometterli darebbe
    una falsa diagnosi sull'encoder v4.
    """
    normalized = validate_suit_permutation(permutation)
    return replace(
        observation,
        hand=tuple(permute_card(card, normalized) for card in observation.hand),
        trump_card=permute_card(observation.trump_card, normalized) if observation.trump_card is not None else None,
        table_cards=tuple(
            (permute_card(card, normalized), player_index) for card, player_index in observation.table_cards
        ),
        seen_cards_onehot=permute_action_vector(observation.seen_cards_onehot, normalized),
        out_of_play_cards_onehot=permute_action_vector(observation.out_of_play_cards_onehot, normalized),
        trick_history=tuple(_permute_trick_record(record, normalized) for record in observation.trick_history),
    )


def masked_softmax(logits: Sequence[float], action_mask: Sequence[bool | int]) -> np.ndarray:
    """Softmax float64 stabile sulle sole azioni legali."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    mask_arr = np.asarray(action_mask, dtype=bool)
    if logits_arr.shape != (40,) or mask_arr.shape != (40,):
        raise ValueError(f"Shape logits/mask invalida: {logits_arr.shape}/{mask_arr.shape} (atteso (40,)/(40,))")
    if not bool(np.any(mask_arr)):
        raise ValueError("Action mask vuota")
    if not bool(np.all(np.isfinite(logits_arr[mask_arr]))):
        raise ValueError("Logits non finiti su azioni legali")

    legal_logits = logits_arr[mask_arr]
    shifted = legal_logits - float(np.max(legal_logits))
    legal_exp = np.exp(shifted)
    total = float(np.sum(legal_exp))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Softmax non normalizzabile")
    probabilities = np.zeros(40, dtype=np.float64)
    probabilities[mask_arr] = legal_exp / total
    return probabilities


def jensen_shannon_divergence_bits(left: Sequence[float], right: Sequence[float]) -> float:
    """Jensen-Shannon in base 2, numericamente stabile e limitata a ``[0, 1]``."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Distribuzioni con shape diverse: {p.shape} vs {q.shape}")
    if not bool(np.all(np.isfinite(p))) or not bool(np.all(np.isfinite(q))):
        raise ValueError("Distribuzioni non finite")
    if bool(np.any(p < 0.0)) or bool(np.any(q < 0.0)):
        raise ValueError("Distribuzioni con probabilità negative")
    p_total = float(np.sum(p))
    q_total = float(np.sum(q))
    if p_total <= 0.0 or q_total <= 0.0:
        raise ValueError("Distribuzioni a massa nulla")
    p = p / p_total
    q = q / q_total
    middle = 0.5 * (p + q)

    def _kl_bits(source: np.ndarray) -> float:
        positive = source > 0.0
        return float(np.sum(source[positive] * np.log2(source[positive] / middle[positive])))

    value = 0.5 * (_kl_bits(p) + _kl_bits(q))
    # Tolleriamo soltanto rumore floating point fuori dall'intervallo teorico.
    return float(np.clip(value, 0.0, 1.0))


def _policy_inputs(agent: BCModelAgent, observations: Sequence[PlayerObservation]) -> tuple[np.ndarray, np.ndarray]:
    """Codifica un batch di osservazioni e ritorna input policy + mask."""
    inputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    has_belief = bool(getattr(agent.model, "has_belief_input", False))
    expected_encoder_dim = agent.model.feature_dim - 40 if has_belief else agent.model.feature_dim
    for observation in observations:
        encoded = encode_player_observation_2p(observation, version=agent.encoder_version)
        encoder_x = np.asarray(encoded.features, dtype=np.float32)
        if encoder_x.shape != (expected_encoder_dim,):
            raise ValueError(f"Feature dim mismatch: encoder={encoder_x.shape} model={agent.model.feature_dim}")
        policy_x = (
            agent.model.policy_input(encoder_x) if has_belief and isinstance(agent.model, MLPBCModel) else encoder_x
        )
        inputs.append(np.asarray(policy_x, dtype=np.float32))
        masks.append(np.asarray(encoded.action_mask, dtype=bool))
    return np.stack(inputs), np.stack(masks)


def evaluate_observation_suit_symmetry(
    agent: BCModelAgent,
    observation: PlayerObservation,
    *,
    near_tie_threshold: float = 1e-8,
) -> ObservationSuitSymmetry:
    """
    Valuta la policy grezza sulle 24 rinomine e rimappa gli output all'orientamento originale.

    Il post-processing dell'agente non viene applicato. Per v13 coincide col runtime perché
    `overkill_guard` è disabilitato; l'output della CLI dichiara comunque questo confine.
    ``near_tie_threshold`` è applicato al gap di probabilità top-2 dopo il softmax
    mascherato a temperatura 1. Ogni confronto è fragile se il gap è sotto soglia nella
    baseline oppure nella distribuzione rinominata e rimappata.
    """
    if observation.num_players != 2:
        raise ValueError("La sonda policy supporta osservazioni 2-player")
    if len(observation.hand) < 2:
        raise ValueError("Servono almeno due azioni legali per una misura informativa")
    if near_tie_threshold < 0.0:
        raise ValueError("near_tie_threshold deve essere >= 0")

    permutations = all_suit_permutations()
    permuted_observations = [permute_player_observation(observation, permutation) for permutation in permutations]
    policy_inputs, masks = _policy_inputs(agent, permuted_observations)
    logits = np.asarray(agent.model.logits(policy_inputs), dtype=np.float64)
    if logits.shape != (len(permutations), 40):
        raise ValueError(f"Output modello batch invalido: {logits.shape}")
    probabilities = np.stack([masked_softmax(row, mask) for row, mask in zip(logits, masks, strict=True)])

    baseline = probabilities[0]
    baseline_mask = masks[0]
    baseline_action = int(np.argmax(baseline))
    legal_probabilities = np.sort(baseline[baseline_mask])
    top2_gap = float(legal_probabilities[-1] - legal_probabilities[-2])
    comparisons: list[SuitSymmetryComparison] = []
    remapped_distributions: list[tuple[float, ...]] = []

    for permutation, permuted_probabilities, permuted_mask in zip(
        permutations,
        probabilities,
        masks,
        strict=True,
    ):
        inverse = inverse_suit_permutation(permutation)
        remapped_probabilities = np.asarray(
            permute_action_vector(permuted_probabilities.tolist(), inverse),
            dtype=np.float64,
        )
        remapped_mask = np.asarray(permute_action_vector(permuted_mask.tolist(), inverse), dtype=bool)
        if not np.array_equal(remapped_mask, baseline_mask):
            raise AssertionError("La mask rimappata non coincide con la baseline")
        remapped_action = int(np.argmax(remapped_probabilities))
        remapped_legal_probabilities = np.sort(remapped_probabilities[remapped_mask])
        remapped_top2_gap = float(remapped_legal_probabilities[-1] - remapped_legal_probabilities[-2])
        remapped_distributions.append(tuple(float(value) for value in remapped_probabilities))
        comparisons.append(
            SuitSymmetryComparison(
                permutation=cast(tuple[str, str, str, str], tuple(suit.value for suit in permutation)),
                is_identity=permutation == IDENTITY_SUIT_PERMUTATION,
                agreement=remapped_action == baseline_action,
                remapped_action_id=remapped_action,
                remapped_top2_gap=remapped_top2_gap,
                js_divergence_bits=jensen_shannon_divergence_bits(baseline.tolist(), remapped_probabilities.tolist()),
                max_abs_probability_delta=float(np.max(np.abs(baseline - remapped_probabilities))),
                near_tie=min(top2_gap, remapped_top2_gap) <= near_tie_threshold,
            )
        )

    return ObservationSuitSymmetry(
        baseline_action_id=baseline_action,
        baseline_top2_gap=top2_gap,
        comparisons=tuple(comparisons),
        remapped_probabilities=tuple(remapped_distributions),
    )


__all__ = [
    "IDENTITY_SUIT_PERMUTATION",
    "ObservationSuitSymmetry",
    "SuitPermutation",
    "SuitSymmetryComparison",
    "all_suit_permutations",
    "evaluate_observation_suit_symmetry",
    "inverse_suit_permutation",
    "jensen_shannon_divergence_bits",
    "masked_softmax",
    "permute_action_vector",
    "permute_card",
    "permute_card_id",
    "permute_player_observation",
    "validate_suit_permutation",
]
