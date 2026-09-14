"""
Solver endgame 2-player su stato numerico compatto — ORACOLO PER I TEST, non runtime.

Ruolo attuale (importante):
nessun agente o loop di training chiama questo modulo a runtime. Il runtime usa
`numba_solver.choose_endgame_card_numba` (choose-only, JIT); questo modulo resta come
PONTE DI PARITÀ tra il solver canonico (`solver.solve_endgame`, su `GameState` +
`domain.step`, con principal variation) e il kernel numerico: implementa lo stesso
minimax sul formato `card_id` 0..39 mantenendo il contratto completo `EndgameSolution`.

Perché tenerlo:
- verifica il minimax numerico con la PV completa (il kernel Numba è choose-only e non
  può essere confrontato sulla PV);
- se il kernel Numba e il canonico divergessero, questo intermedio localizza il bug
  (formato numerico vs implementazione JIT).

Invariante: `solve_endgame_fast(state)` deve restituire la stessa mossa ottima e lo stesso
delta finale di `solve_endgame(state)` per ogni stato valido. I test di parita' proteggono
questa proprieta'.
"""

from __future__ import annotations

from ...domain.card_id import card_to_id
from ...domain.state import GameState
from ..fast.state_2p import CARD_POINTS, fast_who_wins_trick_2p
from .solver import EndgameSolution, _validate

type _FastEndgameState = tuple[
    tuple[int, ...],  # mano P0, ordine identico al dominio
    tuple[int, ...],  # mano P1, ordine identico al dominio
    int,  # punti P0 gia' acquisiti
    int,  # punti P1 gia' acquisiti
    tuple[int, ...],  # carte sul tavolo, lunghezza 0/1
    tuple[int, ...],  # player delle carte sul tavolo, lunghezza 0/1
    int,  # current_turn
]


def _signature_from_state(state: GameState) -> tuple[_FastEndgameState, int]:
    """Converte uno stato endgame canonico in firma numerica compatta."""
    _validate(state)
    if state.trump_card is None:
        raise ValueError("Briscola assente: impossibile risolvere l'endgame")

    table_cards = tuple(card_to_id(card) for card, _player_index in state.table_cards)
    table_players = tuple(int(player_index) for _card, player_index in state.table_cards)
    signature: _FastEndgameState = (
        tuple(card_to_id(card) for card in state.players[0].hand),
        tuple(card_to_id(card) for card in state.players[1].hand),
        int(state.players[0].points),
        int(state.players[1].points),
        table_cards,
        table_players,
        int(state.current_turn),
    )
    return signature, card_to_id(state.trump_card)


def _is_terminal(signature: _FastEndgameState) -> bool:
    """Ritorna True quando non restano mani ne' carta pendente sul tavolo."""
    hand0, hand1, _points0, _points1, table_cards, _table_players, _current_turn = signature
    return not hand0 and not hand1 and not table_cards


def _play_card(signature: _FastEndgameState, *, card_index: int, trump_card: int) -> _FastEndgameState:
    """Applica una carta alla firma numerica, replicando il sottoinsieme endgame di `domain.step`."""
    hand0, hand1, points0, points1, table_cards, table_players, current_turn = signature
    hands = [hand0, hand1]
    hand = hands[current_turn]
    played_card = hand[card_index]
    hands[current_turn] = hand[:card_index] + hand[card_index + 1 :]

    if not table_cards:
        return (
            hands[0],
            hands[1],
            points0,
            points1,
            (played_card,),
            (current_turn,),
            1 - current_turn,
        )

    first_card = table_cards[0]
    first_player = table_players[0]
    second_card = played_card
    second_player = current_turn
    winner = fast_who_wins_trick_2p(
        first_card=first_card,
        first_player=first_player,
        second_card=second_card,
        second_player=second_player,
        trump_card=trump_card,
    )
    trick_points = int(CARD_POINTS[first_card]) + int(CARD_POINTS[second_card])
    if winner == 0:
        points0 += trick_points
    else:
        points1 += trick_points

    return (
        hands[0],
        hands[1],
        points0,
        points1,
        tuple(),
        tuple(),
        winner,
    )


def _minimax(
    signature: _FastEndgameState,
    *,
    trump_card: int,
    memo: dict[_FastEndgameState, tuple[int, int | None]],
) -> tuple[int, int | None]:
    """Minimax esatto su firma numerica. Ritorna `(delta_p0_p1, best_card_index)`."""
    if _is_terminal(signature):
        return int(signature[2] - signature[3]), None

    cached = memo.get(signature)
    if cached is not None:
        return cached

    mover = int(signature[6])
    hand = signature[0] if mover == 0 else signature[1]
    if not hand:
        raise ValueError("Stato endgame incoerente: il player corrente non ha carte")

    maximize = mover == 0
    best_value: int | None = None
    best_index: int | None = None

    for card_index in range(len(hand)):
        child = _play_card(signature, card_index=card_index, trump_card=trump_card)
        child_value, _child_move = _minimax(child, trump_card=trump_card, memo=memo)
        if best_value is None:
            best_value = child_value
            best_index = card_index
        elif maximize:
            if child_value > best_value:
                best_value = child_value
                best_index = card_index
        elif child_value < best_value:
            best_value = child_value
            best_index = card_index

    assert best_value is not None and best_index is not None
    result = (int(best_value), int(best_index))
    memo[signature] = result
    return result


def solve_endgame_fast(state: GameState) -> EndgameSolution:
    """
    Risolve l'endgame come `solve_endgame`, ma usando un minimax numerico compatto.

    Argomenti:
        state: `GameState` 2-player valido, mazzo vuoto, tavolo vuoto o con una carta.

    Ritorna:
        `EndgameSolution` con la stessa convenzione del solver canonico: mossa ottima
        per `state.current_turn`, delta finale dal punto di vista di P0 e principal variation
        in indici mano locali.
    """
    signature, trump_card = _signature_from_state(state)
    memo: dict[_FastEndgameState, tuple[int, int | None]] = {}
    final_delta, best_index = _minimax(signature, trump_card=trump_card, memo=memo)
    assert best_index is not None

    principal_variation: list[tuple[int, int]] = []
    cursor = signature
    while not _is_terminal(cursor):
        _value, move_index = _minimax(cursor, trump_card=trump_card, memo=memo)
        if move_index is None:
            break
        mover = int(cursor[6])
        principal_variation.append((mover, int(move_index)))
        cursor = _play_card(cursor, card_index=int(move_index), trump_card=trump_card)

    return EndgameSolution(
        best_card_index=int(best_index),
        final_delta_p0_p1=int(final_delta),
        principal_variation=tuple(principal_variation),
    )
