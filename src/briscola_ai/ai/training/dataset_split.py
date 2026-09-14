"""Split riproducibile dei dataset supervisionati per partita intera.

Una partita produce molte osservazioni fortemente correlate. Se le singole righe vengono
mescolate prima dello split, il modello puo' vedere nel train quasi la stessa situazione
che ritrovera' in validation o test. Questo modulo assegna invece ogni ``game_id`` a un
solo insieme e salva una provenance compatta dell'assegnazione.

Il digest nei metadati non contiene gli identificativi in chiaro: serve a dimostrare che
due run hanno usato la stessa assegnazione senza gonfiare il modello salvato.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class GroupedDatasetSplit:
    """Indici train/validation/test e provenance dello split per gruppo."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    provenance: dict[str, Any]

    def indices(self, split: SplitName) -> np.ndarray:
        """Restituisce gli indici dello split richiesto."""
        if split == "train":
            return self.train_indices
        if split == "validation":
            return self.validation_indices
        return self.test_indices


def _canonical_group_id(raw: object) -> str:
    """Normalizza stringhe/interi senza confondere, per esempio, ``1`` con ``"1"``."""
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, bool):
        raise ValueError("game_id booleano non valido")
    if isinstance(raw, int):
        return f"int:{raw}"
    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError("game_id vuoto non valido")
        return f"str:{raw}"
    raise ValueError(f"game_id deve essere stringa o intero, ottenuto {type(raw).__name__}")


def _holdout_group_counts(
    num_groups: int,
    *,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    """Converte le frazioni in conteggi, lasciando sempre almeno una partita al train."""
    fractions = {
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
    }
    for name, fraction in fractions.items():
        if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError(f"{name} deve essere finita e in [0, 1)")
    if float(validation_fraction) + float(test_fraction) >= 1.0:
        raise ValueError("validation_fraction + test_fraction deve essere < 1")

    required_groups = 1 + int(validation_fraction > 0.0) + int(test_fraction > 0.0)
    if num_groups < required_groups:
        raise ValueError(
            "Partite distinte insufficienti per split non vuoti: "
            f"trovate {num_groups}, richieste almeno {required_groups}."
        )

    validation_count = 0
    if validation_fraction > 0.0:
        validation_count = max(1, int(round(num_groups * float(validation_fraction))))
    test_count = 0
    if test_fraction > 0.0:
        test_count = max(1, int(round(num_groups * float(test_fraction))))

    # Con dataset piccoli l'arrotondamento indipendente puo' esaurire tutti i gruppi.
    # Riduciamo il bucket piu' sovra-rappresentato rispetto alla frazione richiesta.
    while validation_count + test_count > num_groups - 1:
        validation_excess = validation_count - num_groups * float(validation_fraction)
        test_excess = test_count - num_groups * float(test_fraction)
        can_reduce_validation = validation_count > int(validation_fraction > 0.0)
        can_reduce_test = test_count > int(test_fraction > 0.0)
        if can_reduce_validation and (not can_reduce_test or validation_excess >= test_excess):
            validation_count -= 1
        elif can_reduce_test:
            test_count -= 1
        else:  # Guardia difensiva: ``required_groups`` dovrebbe rendere il ramo irraggiungibile.
            raise ValueError("Impossibile lasciare almeno una partita nel train")
    return validation_count, test_count


def make_grouped_dataset_split(
    group_ids: Sequence[object] | np.ndarray,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    group_key: str = "game_id",
) -> GroupedDatasetSplit:
    """Assegna ogni gruppo interamente a train, validation o test.

    L'assegnazione dipende solo da identificativi, frazioni e seed: cambiare l'ordine delle
    righe non cambia il gruppo assegnato a ciascuna partita. Le frazioni si riferiscono al
    numero di partite, non al numero di osservazioni.
    """
    raw_ids = np.asarray(group_ids, dtype=object)
    if raw_ids.ndim != 1 or raw_ids.size == 0:
        raise ValueError(f"group_ids deve essere un array 1D non vuoto, ottenuto {raw_ids.shape}")
    canonical_ids = np.asarray([_canonical_group_id(raw) for raw in raw_ids.tolist()], dtype=np.str_)
    unique_groups = np.asarray(sorted(set(canonical_ids.tolist())), dtype=np.str_)
    validation_count, test_count = _holdout_group_counts(
        int(unique_groups.size),
        validation_fraction=float(validation_fraction),
        test_fraction=float(test_fraction),
    )

    shuffled_groups = unique_groups.copy()
    np.random.default_rng(int(seed)).shuffle(shuffled_groups)
    validation_groups = set(shuffled_groups[:validation_count].tolist())
    test_start = validation_count
    test_groups = set(shuffled_groups[test_start : test_start + test_count].tolist())

    validation_mask = np.asarray([group in validation_groups for group in canonical_ids], dtype=bool)
    test_mask = np.asarray([group in test_groups for group in canonical_ids], dtype=bool)
    train_mask = ~(validation_mask | test_mask)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)

    assignments: list[tuple[str, SplitName]] = []
    for group in unique_groups.tolist():
        if group in validation_groups:
            split: SplitName = "validation"
        elif group in test_groups:
            split = "test"
        else:
            split = "train"
        assignments.append((group, split))
    assignment_payload = json.dumps(assignments, ensure_ascii=True, separators=(",", ":"))
    assignment_sha256 = hashlib.sha256(assignment_payload.encode("utf-8")).hexdigest()

    provenance: dict[str, Any] = {
        "format": "grouped_dataset_split_v1",
        "unit": "game",
        "group_key": str(group_key),
        "algorithm": "shuffle_sorted_unique_groups_v1",
        "seed": int(seed),
        "fractions_requested": {
            "train": 1.0 - float(validation_fraction) - float(test_fraction),
            "validation": float(validation_fraction),
            "test": float(test_fraction),
        },
        "group_counts": {
            "total": int(unique_groups.size),
            "train": int(unique_groups.size - validation_count - test_count),
            "validation": int(validation_count),
            "test": int(test_count),
        },
        "record_counts": {
            "total": int(raw_ids.size),
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
        },
        "assignment_sha256": assignment_sha256,
    }
    return GroupedDatasetSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        provenance=provenance,
    )
