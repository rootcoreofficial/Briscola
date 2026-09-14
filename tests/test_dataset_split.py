"""Test dello split supervisionato per partita intera."""

from __future__ import annotations

import numpy as np
import pytest

from briscola_ai.ai.training.dataset_split import GroupedDatasetSplit, make_grouped_dataset_split


def _groups_by_split(group_ids: np.ndarray, split: GroupedDatasetSplit) -> dict[str, set[object]]:
    """Ricostruisce gli insiemi di game_id assegnati dal risultato."""
    return {
        "train": set(group_ids[split.train_indices].tolist()),
        "validation": set(group_ids[split.validation_indices].tolist()),
        "test": set(group_ids[split.test_indices].tolist()),
    }


def test_grouped_split_keeps_each_game_in_one_split() -> None:
    """Tutte le osservazioni della stessa partita devono restare nello stesso insieme."""
    group_ids = np.asarray([game_id for game_id in range(10) for _ in range(game_id % 4 + 1)])

    split = make_grouped_dataset_split(
        group_ids,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=17,
    )

    groups = _groups_by_split(group_ids, split)
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert set.union(*groups.values()) == set(group_ids.tolist())
    assert split.provenance["group_counts"] == {"total": 10, "train": 6, "validation": 2, "test": 2}
    assert sum(split.provenance["record_counts"].values()) - split.provenance["record_counts"]["total"] == len(
        group_ids
    )


def test_grouped_split_is_independent_from_record_order() -> None:
    """Riordinare le righe non deve cambiare l'assegnazione delle partite."""
    group_ids = np.asarray(["game-c", "game-a", "game-c", "game-b", "game-d", "game-a"])
    first = make_grouped_dataset_split(
        group_ids,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=99,
    )
    reversed_ids = group_ids[::-1]
    second = make_grouped_dataset_split(
        reversed_ids,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=99,
    )

    assert first.provenance["assignment_sha256"] == second.provenance["assignment_sha256"]
    assert _groups_by_split(group_ids, first) == _groups_by_split(reversed_ids, second)


def test_grouped_split_requires_enough_valid_game_ids() -> None:
    """Con validation e test attivi servono almeno tre partite identificabili."""
    with pytest.raises(ValueError, match="Partite distinte insufficienti"):
        make_grouped_dataset_split(
            np.asarray(["a", "a", "b"]),
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=1,
        )
    with pytest.raises(ValueError, match="game_id vuoto"):
        make_grouped_dataset_split(
            np.asarray(["a", "", "c"]),
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=1,
        )
