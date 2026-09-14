"""Solver e agenti di supporto per il finale a informazione perfetta."""

from typing import Any

from .fast_solver import solve_endgame_fast
from .solver import EndgameSolution, solve_endgame

__all__ = [
    "EndgameSolution",
    "choose_endgame_card_numba",
    "solve_endgame",
    "solve_endgame_fast",
    "warm_up_numba_endgame_solver",
]


def __getattr__(name: str) -> Any:
    """
    Import PIGRO dei simboli Numba (PEP 562).

    Dal 2026-07-07 il runtime web è tutto-python: non deve pagare `import numba`
    (llvmlite pesa ~100-200 MB di RSS e secondi di import sul container piccolo).
    Training e benchmark che chiedono i simboli JIT li ottengono al primo accesso,
    con la stessa sintassi di prima (`from briscola_ai.ai.endgame import ...`).
    """
    if name in ("choose_endgame_card_numba", "warm_up_numba_endgame_solver"):
        from . import numba_solver

        return getattr(numba_solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
