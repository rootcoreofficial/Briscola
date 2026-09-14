"""
Test per il curriculum di training (stage easy→standard→hard).

Obiettivo didattico:
- garantire che lo split di `num_games` sia deterministico e sommi al totale;
- garantire che i preset ritornino stage non vuoti e coerenti.
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.training.curriculum import build_curriculum_stages


def test_curriculum_easy_standard_hard_sums_to_total() -> None:
    """Il preset deve produrre gli stage [easy, standard, hard] non vuoti e con `num_games`
    che somma esattamente al totale richiesto."""
    stages = build_curriculum_stages(preset="easy_standard_hard", total_games=200_000)
    assert [s.name for s in stages] == ["easy", "standard", "hard"]
    assert sum(s.num_games for s in stages) == 200_000
    assert all(s.num_games > 0 for s in stages)


def test_curriculum_split_is_deterministic_with_remainder() -> None:
    """Lo split deve essere deterministico anche con resto: il remainder della divisione
    va assegnato al primo stage (11 game -> 3/5/3)."""
    # 11 games con split 20/50/30 -> base 2/5/3 = 10, remainder=1 va al primo stage => 3/5/3
    stages = build_curriculum_stages(preset="easy_standard_hard", total_games=11)
    assert [s.num_games for s in stages] == [3, 5, 3]


def test_curriculum_rejects_non_positive_total() -> None:
    """Un `total_games` non positivo deve sollevare ValueError invece di generare stage degeneri."""
    with pytest.raises(ValueError):
        build_curriculum_stages(preset="easy_standard_hard", total_games=0)
