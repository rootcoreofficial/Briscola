"""
Dominio "puro" (Phase 2B): stato + transizioni.

Questa cartella contiene il motore canonico della Briscola, pensato per:
- test unitari deterministici
- replay riproducibili (seed controllato)
- futuri use-case ML (simulazioni, self-play, dataset)

Componenti principali:
- uno stato esplicito (`GameState`)
- una transizione pura (`step`) che ritorna un nuovo stato e un risultato
"""

from .engine import PlayCardAction, StepResult, step
from .models import Card, Rank, Suit
from .rules import trick_points, who_wins_trick
from .state import GameState, new_game_state
from .version import RULES_VERSION

__all__ = [
    "Card",
    "GameState",
    "PlayCardAction",
    "Rank",
    "RULES_VERSION",
    "StepResult",
    "Suit",
    "new_game_state",
    "step",
    "trick_points",
    "who_wins_trick",
]
