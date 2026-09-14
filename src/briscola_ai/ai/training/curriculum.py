"""
Curriculum di training (didattico) per Briscola AI.

Motivazione
-----------
Quando un modello RL diventa forte, allenarlo sempre contro la *stessa* distribuzione di avversari
può essere sub-ottimale:
- all'inizio può essere troppo difficile (learning lento/instabile);
- più avanti può essere troppo facile (il modello smette di migliorare);
- se aggiorni l'avversario “in tempo reale” (self-play simmetrico) rischi chasing e regressioni.

Un approccio semplice e spesso efficace è un **curriculum a stage**:
1) easy   -> avversari deboli (impari i fondamentali)
2) standard -> mix “bilanciato” (robustezza)
3) hard   -> include un best congelato (league)

Questo modulo definisce preset *minimali* per costruire in modo deterministico una lista di stage,
usata dalla pipeline `scripts/run_experiment.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CurriculumPreset = Literal["easy_standard_hard"]


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """Uno stage di training: mix avversari + numero partite."""

    name: str
    opponent_mix: str
    num_games: int


def _split_total_games(total: int, *, fractions: tuple[float, ...]) -> list[int]:
    """
    Divide `total` in N stage secondo `fractions`.

    Proprietà:
    - rounding deterministico (floor + distribuzione resto)
    - somma esatta = total
    """
    if total <= 0:
        raise ValueError("total deve essere > 0")
    if not fractions:
        raise ValueError("fractions vuota")
    if any(f <= 0.0 for f in fractions):
        raise ValueError("fractions deve contenere solo valori > 0")

    s = float(sum(fractions))
    weights = [float(f) / s for f in fractions]

    raw = [total * w for w in weights]
    base = [int(x) for x in raw]  # floor
    remainder = int(total - sum(base))
    if remainder < 0:
        raise ValueError("split interno invalido (remainder<0)")

    # Distribuiamo il resto ai primi stage (deterministico).
    for i in range(remainder):
        base[i % len(base)] += 1

    if sum(base) != total:
        raise ValueError("split interno invalido (somma != total)")
    if any(x <= 0 for x in base):
        raise ValueError("split interno invalido (stage <= 0)")
    return base


def build_curriculum_stages(*, preset: CurriculumPreset, total_games: int) -> list[CurriculumStage]:
    """
    Costruisce una lista di stage per un preset.

    Nota:
    i mix sono stringhe in formato compatibile con `--opponent-mix`.
    """
    total = int(total_games)
    if preset == "easy_standard_hard":
        # 20% easy, 50% standard, 30% hard.
        n_easy, n_std, n_hard = _split_total_games(total, fractions=(0.2, 0.5, 0.3))
        return [
            CurriculumStage(
                name="easy",
                opponent_mix="random:0.7,greedy_points:0.3",
                num_games=n_easy,
            ),
            CurriculumStage(
                name="standard",
                opponent_mix="heuristic_v1:0.7,random:0.2,greedy_points:0.1",
                num_games=n_std,
            ),
            CurriculumStage(
                name="hard",
                opponent_mix="best_a2c:0.6,heuristic_v1:0.3,random:0.1",
                num_games=n_hard,
            ),
        ]

    raise ValueError(f"Preset curriculum non supportato: {preset!r}")
