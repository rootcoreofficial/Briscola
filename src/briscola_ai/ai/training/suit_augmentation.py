"""
Augmentation paired dei semi sulle osservazioni già encodate.

Il trainer A2C veloce conserva feature numeriche, mask e action id, non l'intera
``PlayerObservation``. Per duplicare una traiettoria sotto una rinomina dei semi serve
quindi l'azione del gruppo anche sul layout v1-v4 dell'encoder.

Questa trasformazione è una pura permutazione di coordinate:

* i blocchi da 40 carte seguono la mappa card id ``source -> target``;
* le one-hot e i contatori indicizzati per seme seguono la stessa mappa;
* feature relative alla briscola, punteggi, fase e relazioni lead/response restano
  invariate.

La definizione è tenuta esplicita e coperta da test contro il riferimento semantico
``PlayerObservation -> permute_player_observation -> encoder``. Non va estesa a un nuovo
encoder senza aggiungere prima la relativa parità.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ...domain.card_id import SUIT_ORDER, SUIT_TO_INDEX
from ..encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    FEATURE_DIM_2P_V3,
    EncoderVersion,
    feature_dim_for_encoder_version,
)
from ..evaluation.suit_symmetry import (
    SuitPermutation,
    all_suit_permutations,
    permute_card_id,
    validate_suit_permutation,
)

_NON_IDENTITY_PERMUTATIONS = all_suit_permutations()[1:]


@dataclass(frozen=True, slots=True)
class PermutedEncodedTrajectory:
    """Copia numerica di una traiettoria sotto una sola rinomina coerente dei semi."""

    xs: np.ndarray
    action_masks: np.ndarray
    action_ids: np.ndarray
    permutation: SuitPermutation


def _map_equal_blocks(
    mapping: list[int],
    *,
    start: int,
    block_size: int,
    source_to_target: tuple[int, ...],
) -> None:
    """Imposta ``mapping[start + source] = start + target`` per blocchi isomorfi."""
    if len(source_to_target) != block_size:
        raise ValueError(f"Mappa locale di lunghezza {len(source_to_target)} (atteso {block_size})")
    for source, target in enumerate(source_to_target):
        mapping[start + source] = start + target


@lru_cache(maxsize=4 * 24)
def encoded_feature_index_mapping(
    version: EncoderVersion,
    permutation: SuitPermutation,
) -> tuple[int, ...]:
    """
    Ritorna la mappa ``source_feature_index -> target_feature_index`` per un encoder.

    Gli offset sono il contratto documentato in ``observation_encoder.py``. La verifica
    finale impone che ogni coordinata compaia esattamente una volta.
    """
    normalized = validate_suit_permutation(permutation)
    feature_dim = int(feature_dim_for_encoder_version(version))
    mapping = list(range(feature_dim))
    action_mapping = tuple(permute_card_id(card_id, normalized) for card_id in range(40))
    suit_mapping = tuple(SUIT_TO_INDEX[normalized[source]] for source in range(len(SUIT_ORDER)))

    # v1: mano e tavolo, ciascuno one-hot / punti / forza.
    for start in (0, 40, 80, 120, 160, 200):
        _map_equal_blocks(mapping, start=start, block_size=40, source_to_target=action_mapping)
    _map_equal_blocks(mapping, start=240, block_size=4, source_to_target=suit_mapping)

    if version in ("v2", "v3", "v4"):
        # v2: carte pubblicamente viste.
        _map_equal_blocks(mapping, start=FEATURE_DIM_2P_V1, block_size=40, source_to_target=action_mapping)

    if version in ("v3", "v4"):
        # v3: dopo quattro feature relative alla briscola, 4 blocchi-seme da 3.
        per_suit_start = FEATURE_DIM_2P_V2 + 4
        for source_suit, target_suit in enumerate(suit_mapping):
            for local_offset in range(3):
                mapping[per_suit_start + source_suit * 3 + local_offset] = (
                    per_suit_start + target_suit * 3 + local_offset
                )

    if version == "v4":
        # v4 A: primi quattro contatori = carte avversarie giocate per seme.
        _map_equal_blocks(mapping, start=FEATURE_DIM_2P_V3, block_size=4, source_to_target=suit_mapping)

        # v4 B: 4 prese recenti x 12; offset 2..5 = one-hot del seme di lead.
        recent_start = FEATURE_DIM_2P_V3 + 11
        for slot in range(4):
            _map_equal_blocks(
                mapping,
                start=recent_start + slot * 12 + 2,
                block_size=4,
                source_to_target=suit_mapping,
            )

    if len(mapping) != feature_dim or sorted(mapping) != list(range(feature_dim)):
        raise AssertionError(f"Mappa feature {version} non biiettiva")
    return tuple(mapping)


def permute_encoded_features(
    features: np.ndarray,
    *,
    version: EncoderVersion,
    permutation: SuitPermutation,
) -> np.ndarray:
    """Permuta l'ultima dimensione di un vettore o batch di feature v1-v4."""
    values = np.asarray(features)
    expected_dim = int(feature_dim_for_encoder_version(version))
    if values.ndim < 1 or values.shape[-1] != expected_dim:
        raise ValueError(f"Feature shape {values.shape} incompatibile con {version} ({expected_dim})")
    mapping = np.asarray(encoded_feature_index_mapping(version, validate_suit_permutation(permutation)), dtype=np.intp)
    out = np.empty_like(values)
    out[..., mapping] = values
    return out


def permute_action_vectors(values: np.ndarray, *, permutation: SuitPermutation) -> np.ndarray:
    """Permuta l'ultima dimensione di un vettore carta o di un batch di vettori."""
    vectors = np.asarray(values)
    if vectors.ndim < 1 or vectors.shape[-1] != 40:
        raise ValueError(f"Action vector shape invalida: {vectors.shape}")
    normalized = validate_suit_permutation(permutation)
    mapping = np.asarray([permute_card_id(card_id, normalized) for card_id in range(40)], dtype=np.intp)
    out = np.empty_like(vectors)
    out[..., mapping] = vectors
    return out


def permute_action_masks(action_masks: np.ndarray, *, permutation: SuitPermutation) -> np.ndarray:
    """Permuta una action mask delegando alla trasformazione generica dei 40 card id."""
    masks = np.asarray(action_masks)
    if not np.issubdtype(masks.dtype, np.bool_):
        raise ValueError(f"action_masks deve avere dtype bool, ottenuto {masks.dtype}")
    return permute_action_vectors(masks, permutation=permutation)


def permute_action_ids(action_ids: np.ndarray, *, permutation: SuitPermutation) -> np.ndarray:
    """Rinomina un array di action id preservandone shape e dtype intero."""
    ids = np.asarray(action_ids)
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"action_ids deve avere dtype intero, ottenuto {ids.dtype}")
    if bool(np.any(ids < 0)) or bool(np.any(ids >= 40)):
        raise ValueError("action_ids contiene valori fuori da [0, 39]")
    normalized = validate_suit_permutation(permutation)
    mapping = np.asarray([permute_card_id(card_id, normalized) for card_id in range(40)], dtype=ids.dtype)
    return mapping[ids]


def permute_encoded_trajectory(
    *,
    xs: np.ndarray,
    action_masks: np.ndarray,
    action_ids: np.ndarray,
    version: EncoderVersion,
    permutation: SuitPermutation,
) -> PermutedEncodedTrajectory:
    """Crea la copia paired usando la stessa rinomina per ogni step della traiettoria."""
    if xs.ndim != 2 or action_masks.ndim != 2 or action_ids.ndim != 1:
        raise ValueError(
            f"Shape traiettoria invalide: xs={xs.shape}, masks={action_masks.shape}, ids={action_ids.shape}"
        )
    if xs.shape[0] != action_masks.shape[0] or xs.shape[0] != action_ids.shape[0]:
        raise ValueError("Numero step incoerente tra feature, mask e action id")
    normalized = validate_suit_permutation(permutation)
    return PermutedEncodedTrajectory(
        xs=permute_encoded_features(xs, version=version, permutation=normalized),
        action_masks=permute_action_masks(action_masks, permutation=normalized),
        action_ids=permute_action_ids(action_ids, permutation=normalized),
        permutation=normalized,
    )


def sample_nonidentity_suit_permutation(rng: np.random.Generator) -> SuitPermutation:
    """Campiona uniformemente una delle 23 rinomine non identità."""
    index = int(rng.integers(0, len(_NON_IDENTITY_PERMUTATIONS)))
    return _NON_IDENTITY_PERMUTATIONS[index]


__all__ = [
    "PermutedEncodedTrajectory",
    "encoded_feature_index_mapping",
    "permute_action_ids",
    "permute_action_masks",
    "permute_action_vectors",
    "permute_encoded_features",
    "permute_encoded_trajectory",
    "sample_nonidentity_suit_permutation",
]
