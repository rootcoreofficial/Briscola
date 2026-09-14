"""
Kernel JIT condiviso: vincitore di una presa 2-player su card id.

Perché è un modulo FOGLIA separato:
sia il core Numba (`ai/numba/core.py`) sia il solver endgame (`ai/endgame/numba_solver.py`)
hanno bisogno di questa funzione, ma importare l'uno dall'altro crea un ciclo
(`endgame.numba_solver -> ai.numba.__init__ -> value_lookahead -> endgame.numba_solver`).
Tenere il kernel qui, con dipendenze solo verso `card_tables`, spezza il ciclo e lascia
una sola implementazione (storicamente ne esistevano due copie byte-identiche).

La semantica è identica a `domain.rules.who_wins_trick`; la parità è protetta dai
test-àncora (`test_card_tables_parity`).
"""

from __future__ import annotations

from numba import njit

from .card_tables import CARD_STRENGTH_BY_ID_NP, CARD_SUIT_BY_ID_NP


@njit(cache=True)
def who_wins_trick_numba_2p(
    first_card: int, first_player: int, second_card: int, second_player: int, trump_card: int
) -> int:
    """
    Determina il vincitore di una presa 2-player usando solo card id.

    Nota sul tie-break `>=`: due carte vengono confrontate sulla forza solo quando sono
    dello stesso seme (entrambe briscola o entrambe seme di uscita), e dentro un seme le
    forze sono tutte distinte — quindi il caso di parità non può verificarsi e il `>=`
    coincide col comportamento di `max(...)` del dominio.
    """
    trump_suit = CARD_SUIT_BY_ID_NP[trump_card]
    first_suit = CARD_SUIT_BY_ID_NP[first_card]
    second_suit = CARD_SUIT_BY_ID_NP[second_card]

    first_is_trump = first_suit == trump_suit
    second_is_trump = second_suit == trump_suit
    if first_is_trump or second_is_trump:
        if first_is_trump and not second_is_trump:
            return first_player
        if second_is_trump and not first_is_trump:
            return second_player
        if CARD_STRENGTH_BY_ID_NP[first_card] >= CARD_STRENGTH_BY_ID_NP[second_card]:
            return first_player
        return second_player

    if second_suit != first_suit:
        return first_player
    if CARD_STRENGTH_BY_ID_NP[first_card] >= CARD_STRENGTH_BY_ID_NP[second_card]:
        return first_player
    return second_player
