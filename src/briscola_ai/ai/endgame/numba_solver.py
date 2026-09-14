"""
Scelta endgame JIT con Numba.

`fast_solver.solve_endgame_fast` mantiene il contratto completo `EndgameSolution`
(best move, delta finale e principal variation). I loop di training/runtime usano spesso
solo `best_card_index`: calcolare e materializzare la PV a ogni mossa finale è lavoro
inutile nel loop caldo.

Questo modulo espone quindi una API choose-only compilata con Numba. Il kernel usa un DFS
minimax iterativo con stack fisso (al massimo 6 plie residue) invece di ricorsione, così è
stabile per chiamate ripetute nei training lunghi.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ...domain.card_id import card_to_id
from ...domain.state import GameState
from ..card_tables import (
    CARD_NUMBER_BY_ID_NP,
    CARD_POINTS_BY_ID_NP,
    CARD_STRENGTH_BY_ID_NP,
    CARD_SUIT_BY_ID_NP,
)
from ..trick_kernel import who_wins_trick_numba_2p as _who_wins_trick_numba
from .solver import _validate

_HAND_CAPACITY = 3
_MAX_STACK_DEPTH = 8
_ACTION_DIM = 40
# Tabelle numeriche derivate dal dominio canonico: vedi `ai/card_tables.py` (fonte unica).
_CARD_SUIT_NUMBA = CARD_SUIT_BY_ID_NP
_CARD_NUMBER_NUMBA = CARD_NUMBER_BY_ID_NP
_CARD_POINTS_NUMBA = CARD_POINTS_BY_ID_NP
_CARD_STRENGTH_NUMBA = CARD_STRENGTH_BY_ID_NP


def _arrays_from_state(
    state: GameState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Converte uno stato endgame valido in array compatibili col kernel JIT."""
    _validate(state)
    if state.trump_card is None:
        raise ValueError("Briscola assente: impossibile risolvere l'endgame")

    hands = np.full((2, _HAND_CAPACITY), -1, dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)
    for player_index in range(2):
        hand = state.players[player_index].hand
        hand_sizes[player_index] = len(hand)
        for card_index, card in enumerate(hand):
            hands[player_index, card_index] = card_to_id(card)

    points = np.asarray([state.players[0].points, state.players[1].points], dtype=np.int64)
    table_cards = np.full(1, -1, dtype=np.int64)
    table_players = np.full(1, -1, dtype=np.int64)
    if state.table_cards:
        table_cards[0] = card_to_id(state.table_cards[0][0])
        table_players[0] = int(state.table_cards[0][1])

    return (
        hands,
        hand_sizes,
        points,
        table_cards,
        table_players,
        len(state.table_cards),
        int(state.current_turn),
        card_to_id(state.trump_card),
    )


# Nota de-duplicazione: il vincitore della presa è calcolato da `_who_wins_trick_numba`
# importata da `ai/numba/core.py` — le funzioni `@njit` sono importabili cross-module
# come normali riferimenti globali, quindi non serve (né conviene) una copia locale.


@njit(cache=True)
def _card_at3(c0: int, c1: int, c2: int, card_index: int) -> int:
    """Legge una carta da una mano rappresentata da tre slot."""
    if card_index == 0:
        return c0
    if card_index == 1:
        return c1
    return c2


@njit(cache=True)
def _remove_at3(c0: int, c1: int, c2: int, size: int, card_index: int) -> tuple[int, int, int, int, int]:
    """Rimuove una carta da tre slot preservando l'ordine residuo."""
    played = _card_at3(c0, c1, c2, card_index)
    if card_index == 0:
        return played, c1, c2, -1, size - 1
    if card_index == 1:
        return played, c0, c2, -1, size - 1
    return played, c0, c1, -1, size - 1


@njit(cache=True)
def _solve_endgame_stack_numba(
    root_h00: int,
    root_h01: int,
    root_h02: int,
    root_h0n: int,
    root_h10: int,
    root_h11: int,
    root_h12: int,
    root_h1n: int,
    root_p0: int,
    root_p1: int,
    root_table_card: int,
    root_table_player: int,
    root_table_size: int,
    root_current_turn: int,
    trump_card: int,
) -> tuple[int, int]:
    """
    Risolve l'endgame con DFS minimax iterativo.

    Ritorna `(best_card_index, final_delta_p0_p1)`, con `best_card_index` nell'ordine mano
    dello stato radice.
    """
    h00 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    h01 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    h02 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    h0n = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    h10 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    h11 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    h12 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    h1n = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    p0 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    p1 = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    table_card = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    table_player = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    table_size = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    current_turn = np.empty(_MAX_STACK_DEPTH, dtype=np.int64)
    next_child = np.zeros(_MAX_STACK_DEPTH, dtype=np.int64)
    best_value = np.zeros(_MAX_STACK_DEPTH, dtype=np.int64)
    best_index = np.full(_MAX_STACK_DEPTH, -1, dtype=np.int64)
    has_best = np.zeros(_MAX_STACK_DEPTH, dtype=np.int64)
    move_from_parent = np.full(_MAX_STACK_DEPTH, -1, dtype=np.int64)

    h00[0] = root_h00
    h01[0] = root_h01
    h02[0] = root_h02
    h0n[0] = root_h0n
    h10[0] = root_h10
    h11[0] = root_h11
    h12[0] = root_h12
    h1n[0] = root_h1n
    p0[0] = root_p0
    p1[0] = root_p1
    table_card[0] = root_table_card
    table_player[0] = root_table_player
    table_size[0] = root_table_size
    current_turn[0] = root_current_turn

    depth = 0
    while depth >= 0:
        mover = int(current_turn[depth])
        hand_size = int(h0n[depth] if mover == 0 else h1n[depth])
        terminal = h0n[depth] == 0 and h1n[depth] == 0 and table_size[depth] == 0

        if terminal or hand_size <= 0:
            node_value = int(p0[depth] - p1[depth])
            node_best = -1
            node_done = True
        elif next_child[depth] < hand_size:
            card_index = int(next_child[depth])
            next_child[depth] += 1

            if mover == 0:
                played, nh00, nh01, nh02, nh0n = _remove_at3(
                    int(h00[depth]),
                    int(h01[depth]),
                    int(h02[depth]),
                    int(h0n[depth]),
                    card_index,
                )
                nh10 = int(h10[depth])
                nh11 = int(h11[depth])
                nh12 = int(h12[depth])
                nh1n = int(h1n[depth])
            else:
                played, nh10, nh11, nh12, nh1n = _remove_at3(
                    int(h10[depth]),
                    int(h11[depth]),
                    int(h12[depth]),
                    int(h1n[depth]),
                    card_index,
                )
                nh00 = int(h00[depth])
                nh01 = int(h01[depth])
                nh02 = int(h02[depth])
                nh0n = int(h0n[depth])

            child = depth + 1
            h00[child] = nh00
            h01[child] = nh01
            h02[child] = nh02
            h0n[child] = nh0n
            h10[child] = nh10
            h11[child] = nh11
            h12[child] = nh12
            h1n[child] = nh1n
            p0[child] = p0[depth]
            p1[child] = p1[depth]

            if table_size[depth] == 0:
                table_card[child] = played
                table_player[child] = mover
                table_size[child] = 1
                current_turn[child] = 1 - mover
            else:
                winner = _who_wins_trick_numba(
                    int(table_card[depth]),
                    int(table_player[depth]),
                    int(played),
                    mover,
                    trump_card,
                )
                gained = int(_CARD_POINTS_NUMBA[int(table_card[depth])]) + int(_CARD_POINTS_NUMBA[int(played)])
                if winner == 0:
                    p0[child] += gained
                else:
                    p1[child] += gained
                table_card[child] = -1
                table_player[child] = -1
                table_size[child] = 0
                current_turn[child] = winner

            next_child[child] = 0
            best_value[child] = 0
            best_index[child] = -1
            has_best[child] = 0
            move_from_parent[child] = card_index
            depth = child
            continue
        else:
            node_value = int(best_value[depth])
            node_best = int(best_index[depth])
            node_done = True

        if node_done:
            if depth == 0:
                return int(node_best), int(node_value)

            child_value = int(node_value)
            child_move = int(move_from_parent[depth])
            depth -= 1
            maximize = current_turn[depth] == 0
            if has_best[depth] == 0:
                best_value[depth] = child_value
                best_index[depth] = child_move
                has_best[depth] = 1
            elif maximize:
                if child_value > best_value[depth]:
                    best_value[depth] = child_value
                    best_index[depth] = child_move
            elif child_value < best_value[depth]:
                best_value[depth] = child_value
                best_index[depth] = child_move

    return -1, 0


@njit(cache=True)
def choose_endgame_card_numba_arrays(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    current_turn: int,
    trump_card: int,
) -> tuple[int, int]:
    """
    Entry point JIT da array compatibili col fast/numba path.

    Ritorna `(best_card_index, final_delta_p0_p1)`. È pensato per i futuri loop di training
    che già mantengono `hands`, `hand_sizes`, `points` e tavolo in array NumPy.
    """
    return _solve_endgame_stack_numba(
        int(hands[0, 0]),
        int(hands[0, 1]),
        int(hands[0, 2]),
        int(hand_sizes[0]),
        int(hands[1, 0]),
        int(hands[1, 1]),
        int(hands[1, 2]),
        int(hand_sizes[1]),
        int(points[0]),
        int(points[1]),
        int(table_cards[0]) if table_size > 0 else -1,
        int(table_players[0]) if table_size > 0 else -1,
        int(table_size),
        int(current_turn),
        int(trump_card),
    )


def choose_endgame_card_numba(state: GameState) -> int:
    """
    Ritorna la mossa ottima endgame usando il kernel Numba choose-only.

    Il wrapper valida lo stato con le stesse precondizioni del solver canonico. Il valore
    ritornato è un indice nella mano corrente, con lo stesso tie-break del solver completo.
    """
    (
        hands,
        hand_sizes,
        points,
        table_cards,
        table_players,
        table_size,
        current_turn,
        trump_card,
    ) = _arrays_from_state(state)
    best_card_index, _final_delta = choose_endgame_card_numba_arrays(
        hands,
        hand_sizes,
        points,
        table_cards,
        table_players,
        table_size,
        current_turn,
        trump_card,
    )
    if int(best_card_index) < 0:
        raise ValueError("Stato endgame senza mossa ottima")
    return int(best_card_index)


def warm_up_numba_endgame_solver() -> None:
    """Compila il kernel endgame Numba con uno stato minimale artificiale."""
    hands = np.asarray([[0, -1, -1], [1, -1, -1]], dtype=np.int64)
    hand_sizes = np.asarray([1, 1], dtype=np.int64)
    points = np.asarray([0, 0], dtype=np.int64)
    table_cards = np.asarray([-1], dtype=np.int64)
    table_players = np.asarray([-1], dtype=np.int64)
    choose_endgame_card_numba_arrays(hands, hand_sizes, points, table_cards, table_players, 0, 0, 10)
