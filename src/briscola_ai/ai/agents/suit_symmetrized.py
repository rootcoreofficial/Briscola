"""
Policy diagnostica resa equivariant per media sull'intero gruppo dei semi.

La Briscola non attribuisce significato ai nomi assoluti dei quattro semi. Questo
wrapper valuta la stessa policy sulle 24 rinomine possibili, riporta ogni vettore
di logits negli action id originali e ne calcola la media prima dell'argmax.

Il costo non equivale a 24 chiamate Python complete: le feature rinominate formano
un unico batch NumPy, elaborato dalla MLP con una sola chiamata. Il wrapper resta
deliberatamente fuori dal catalogo UI finche' il test causale non ne misura forza
e latenza.

Anti-cheat
----------
L'unico input e' :class:`PlayerObservation`. Le trasformazioni operano sulle feature
lecite prodotte dall'encoder e non accedono a mazzo o mano avversaria.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...domain.card_id import card_to_id
from ...domain.observation import PlayerObservation
from ..encoding.observation_encoder import EncoderVersion, encode_player_observation_2p
from ..evaluation.suit_symmetry import all_suit_permutations, permute_card_id
from ..models.bc_model import BCModelAgent
from ..training.suit_augmentation import encoded_feature_index_mapping


@lru_cache(maxsize=4)
def _orbit_gather_indices(version: EncoderVersion) -> tuple[np.ndarray, np.ndarray]:
    """
    Precalcola le gather per creare e riallineare il batch delle 24 rinomine.

    ``feature_gather[p, target]`` indica quale feature originale finisce nella
    coordinata ``target`` della rinomina ``p``. ``action_gather[p, original]``
    indica invece da quale output rinominato leggere il logit dell'azione originale.
    """
    feature_gathers: list[np.ndarray] = []
    action_gathers: list[np.ndarray] = []
    for permutation in all_suit_permutations():
        source_to_target = np.asarray(encoded_feature_index_mapping(version, permutation), dtype=np.intp)
        feature_gathers.append(np.argsort(source_to_target))
        action_gathers.append(
            np.asarray([permute_card_id(action_id, permutation) for action_id in range(40)], dtype=np.intp)
        )
    return np.stack(feature_gathers), np.stack(action_gathers)


@dataclass(frozen=True, slots=True)
class SuitSymmetrizedBCModelAgent:
    """
    Wrapper offline che rende una policy BC equivariant alla rinomina dei semi.

    La media e' eseguita sui logits riallineati, non sulle probabilita': preserva
    la scala relativa appresa dalla policy ed evita una softmax per ciascuna riga.
    Il risultato teorico e' esattamente equivariant; in float32/float64 possono
    restare differenze dell'ordine dell'arrotondamento numerico.
    """

    base_agent: BCModelAgent

    def __post_init__(self) -> None:
        """Rifiuta formati il cui input non e' soltanto l'encoder permutabile."""
        if bool(getattr(self.base_agent.model, "has_belief_input", False)):
            raise ValueError("La policy symmetrized non supporta modelli con belief embedded")
        if self.base_agent.overkill_guard_enabled:
            raise ValueError("La policy symmetrized richiede overkill_guard disabilitato per isolare la policy")

    @property
    def name(self) -> str:
        """Nome leggibile usato nei report di valutazione."""
        return f"bc_model_suit_symmetrized({self.base_agent.model_path.name},24x-batch)"

    @classmethod
    def from_npz(cls, path: str | Path) -> SuitSymmetrizedBCModelAgent:
        """Carica un modello BC e lo avvolge nella media sulle 24 rinomine."""
        return cls(base_agent=BCModelAgent.from_npz(path))

    def orbit_logits(self, observation: PlayerObservation) -> np.ndarray:
        """
        Ritorna i 24 vettori di logits gia' riallineati agli action id originali.

        La prima riga corrisponde all'identita' ed e' quindi il forward standard
        della policy. Esporre l'orbita permette test e benchmark senza duplicare
        la logica interna.
        """
        encoded = encode_player_observation_2p(observation, version=self.base_agent.encoder_version)
        features = np.asarray(encoded.features, dtype=np.float32)
        return self._orbit_logits_from_features(features)

    def _orbit_logits_from_features(self, features: np.ndarray) -> np.ndarray:
        """Esegue il batch partendo da feature già encodate, evitando lavoro doppio nell'argmax."""
        if features.shape != (self.base_agent.model.feature_dim,):
            raise ValueError(
                "Feature dim mismatch: "
                f"encoder={features.shape} model={self.base_agent.model.feature_dim} "
                f"({self.base_agent.model_path})"
            )

        feature_gather, action_gather = _orbit_gather_indices(self.base_agent.encoder_version)
        orbit_features = features[feature_gather]
        permuted_logits = np.asarray(self.base_agent.model.logits(orbit_features))
        if permuted_logits.shape != (24, 40):
            raise ValueError(f"Logits orbit shape invalida: {permuted_logits.shape} (attesa (24, 40))")

        row_indices = np.arange(24, dtype=np.intp)[:, None]
        return permuted_logits[row_indices, action_gather]

    def symmetrized_logits(self, observation: PlayerObservation) -> np.ndarray:
        """Media float64 dei 24 logits riallineati negli action id originali."""
        return np.asarray(np.mean(self.orbit_logits(observation), axis=0, dtype=np.float64), dtype=np.float64)

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        """Sceglie l'azione legale col logit simmetrizzato massimo."""
        del rng  # La policy e' deterministica; il parametro soddisfa il protocollo Agent.
        if not observation.hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        encoded = encode_player_observation_2p(observation, version=self.base_agent.encoder_version)
        mask = np.asarray(encoded.action_mask, dtype=bool)
        if mask.shape != (40,) or not bool(np.any(mask)):
            raise ValueError(f"Action mask invalida o vuota: {mask.shape}")

        features = np.asarray(encoded.features, dtype=np.float32)
        logits = np.mean(self._orbit_logits_from_features(features), axis=0, dtype=np.float64)
        action_id = int(np.argmax(np.where(mask, logits, -np.inf)))
        for card_index, card in enumerate(observation.hand):
            if card_to_id(card) == action_id:
                return card_index
        raise AssertionError(f"Action id legale {action_id} assente dalla mano")


__all__ = ["SuitSymmetrizedBCModelAgent"]
