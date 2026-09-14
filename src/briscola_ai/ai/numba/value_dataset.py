"""
Collector Numba per dataset di valore `V(observation)`.

Il generatore canonico `scripts/generate_value_dataset.py` passa dal dominio Python e salva
JSONL leggibile. Questo modulo è il corrispettivo ad alto throughput per run lunghe:
gioca partite 2-player su array numerici, encoda direttamente le osservazioni e produce
target di continuazione deterministica `policy + solver`.

Anti-cheat:
le feature salvate sono sempre osservazioni dal punto di vista del giocatore di turno,
costruite con `encode_fast_observation_arrays_numba`. La continuazione usa lo stato completo
solo per generare il target supervisionato offline, non per l'osservazione data alla rete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ..encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4
from ..endgame.numba_solver import choose_endgame_card_numba_arrays
from .core import ACTION_DIM, _shuffle_deck_numba
from .observation import (
    _action_id_to_hand_index_numba,
    _apply_numba_card_index,
    _apply_overkill_guard_numba,
    _argmax_mlp_policy_action_numba,
    encode_fast_observation_arrays_numba,
    encode_fast_observation_arrays_v4_numba,
)

_MAX_DECK_SIZE_2P = 34
_HAND_CAPACITY_2P = 3
_TABLE_CAPACITY_2P = 2
_MAX_STATES_PER_GAME_2P = 40

PHASE_EARLY = 0
PHASE_MID = 1
PHASE_PIMC_WINDOW = 2
PHASE_ENDGAME = 3

COLLECT_ALL = 0
COLLECT_WINDOW = 1


@dataclass(frozen=True, slots=True)
class NumbaValueDatasetBatch:
    """Batch fisso `(games, max_states, ...)` prodotto dal collector value Numba."""

    xs: np.ndarray
    current_delta: np.ndarray
    final_delta: np.ndarray
    phase: np.ndarray
    unknown_live_cards: np.ndarray
    deck_size: np.ndarray
    exploratory_action: np.ndarray
    valid: np.ndarray
    games_completed: int


def phase_name_from_code(code: int) -> str:
    """Converte il codice fase compatto in etichetta stabile usata dai report."""
    if int(code) == PHASE_EARLY:
        return "early"
    if int(code) == PHASE_MID:
        return "mid"
    if int(code) == PHASE_PIMC_WINDOW:
        return "pimc_window"
    if int(code) == PHASE_ENDGAME:
        return "endgame"
    return "unknown"


@njit(cache=True)
def _validate_mlp_weights_numba(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
    """Guardie minime sui pesi MLP usati dal collector."""
    feature_dim = int(w1.shape[0])
    hidden_dim = int(w1.shape[1])
    if (
        feature_dim != int(FEATURE_DIM_2P_V1)
        and feature_dim != int(FEATURE_DIM_2P_V2)
        and feature_dim != int(FEATURE_DIM_2P_V3)
        and feature_dim != int(FEATURE_DIM_2P_V4)
    ):
        raise ValueError("feature_dim non supportata")
    if int(b1.shape[0]) != hidden_dim:
        raise ValueError("b1 shape non compatibile")
    if int(w2.shape[0]) != hidden_dim or int(w2.shape[1]) != ACTION_DIM:
        raise ValueError("w2 shape non compatibile")
    if int(b2.shape[0]) != ACTION_DIM:
        raise ValueError("b2 shape non compatibile")


@njit(cache=True)
def _copy_compact_state_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    deck: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copia lo stato numerico compatto per etichettare una continuazione senza mutare il self-play."""
    hands_copy = np.empty((2, _HAND_CAPACITY_2P), dtype=np.int64)
    for player_index in range(2):
        for card_index in range(_HAND_CAPACITY_2P):
            hands_copy[player_index, card_index] = hands[player_index, card_index]

    hand_sizes_copy = np.empty(2, dtype=np.int64)
    points_copy = np.empty(2, dtype=np.int64)
    for player_index in range(2):
        hand_sizes_copy[player_index] = hand_sizes[player_index]
        points_copy[player_index] = points[player_index]

    deck_copy = np.empty(_MAX_DECK_SIZE_2P, dtype=np.int64)
    for index in range(_MAX_DECK_SIZE_2P):
        deck_copy[index] = deck[index]

    table_cards_copy = np.empty(_TABLE_CAPACITY_2P, dtype=np.int64)
    table_players_copy = np.empty(_TABLE_CAPACITY_2P, dtype=np.int64)
    for index in range(_TABLE_CAPACITY_2P):
        table_cards_copy[index] = table_cards[index]
        table_players_copy[index] = table_players[index]

    seen_copy = np.empty(ACTION_DIM, dtype=np.int64)
    out_copy = np.empty(ACTION_DIM, dtype=np.int64)
    for card_id in range(ACTION_DIM):
        seen_copy[card_id] = seen_cards[card_id]
        out_copy[card_id] = out_of_play_cards[card_id]

    return (
        hands_copy,
        hand_sizes_copy,
        points_copy,
        deck_copy,
        table_cards_copy,
        table_players_copy,
        seen_copy,
        out_copy,
    )


@njit(cache=True)
def _choose_hybrid_mlp_card_index_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    overkill_guard_enabled: bool,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    deck: np.ndarray,
    deck_size: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    current_turn: int,
    trump_card: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    trick_hist: np.ndarray,
    num_tricks: int,
) -> int:
    """
    Sceglie una carta con `MLP + solver endgame`.

    È la policy di continuazione che vogliamo etichettare: MLP deterministico finché il
    mazzo non è vuoto, solver esatto quando l'endgame è a informazione perfetta.
    """
    if deck_size == 0:
        best_card_index, _delta = choose_endgame_card_numba_arrays(
            hands,
            hand_sizes,
            points,
            table_cards,
            table_players,
            table_size,
            current_turn,
            trump_card,
        )
        if 0 <= best_card_index < hand_sizes[current_turn]:
            return int(best_card_index)

    action_id = _argmax_mlp_policy_action_numba(
        w1,
        b1,
        w2,
        b2,
        hands,
        hand_sizes,
        points,
        table_cards,
        table_size,
        deck_size,
        current_turn,
        trump_card,
        current_turn,
        seen_cards,
        out_of_play_cards,
        trick_hist,
        num_tricks,
    )
    action_id = _apply_overkill_guard_numba(
        action_id,
        hands,
        hand_sizes,
        current_turn,
        table_cards,
        table_players,
        table_size,
        trump_card,
        overkill_guard_enabled,
    )
    return _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)


@njit(cache=True)
def _play_hybrid_mlp_to_terminal_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    overkill_guard_enabled: bool,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    deck: np.ndarray,
    deck_size: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    current_turn: int,
    trump_card: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    trick_hist: np.ndarray,
    trick_count: np.ndarray,
) -> tuple[int, int]:
    """
    Completa una partita deterministica con `MLP + solver`, ritornando i punti finali.

    `trick_hist`/`trick_count` sono la storia delle prese (COPIA, mutata dal rollout):
    con una policy v4 la continuazione deve giocare con la memoria REALE, altrimenti
    le etichette del value dataset rifletterebbero un giocatore "smemorato".
    """
    safety = 256
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0 and table_size == 0:
            break
        card_index = _choose_hybrid_mlp_card_index_numba(
            w1,
            b1,
            w2,
            b2,
            overkill_guard_enabled,
            hands,
            hand_sizes,
            points,
            deck,
            deck_size,
            table_cards,
            table_players,
            table_size,
            current_turn,
            trump_card,
            seen_cards,
            out_of_play_cards,
            trick_hist,
            trick_count[0],
        )
        deck_size, table_size, current_turn = _apply_numba_card_index(
            hands,
            hand_sizes,
            points,
            deck,
            deck_size,
            table_cards,
            table_players,
            table_size,
            current_turn,
            trump_card,
            card_index,
            seen_cards,
            out_of_play_cards,
            trick_hist,
            trick_count,
        )
    return int(points[0]), int(points[1])


@njit(cache=True)
def _unknown_live_cards_numba(hand_sizes: np.ndarray, deck_size: int, player_index: int) -> int:
    """Carte vive non note al player: mano avversaria + mazzo."""
    return int(hand_sizes[1 - player_index]) + int(deck_size)


@njit(cache=True)
def _phase_code_numba(deck_size: int, unknown_live_cards: int, max_unknown_cards: int) -> int:
    """Bucket fase compatto, coerente con il generatore Python."""
    if deck_size == 0:
        return PHASE_ENDGAME
    if unknown_live_cards <= max_unknown_cards:
        return PHASE_PIMC_WINDOW
    if deck_size <= 10:
        return PHASE_MID
    return PHASE_EARLY


@njit(cache=True)
def _should_collect_numba(
    collect_mode: int,
    include_endgame: bool,
    deck_size: int,
    unknown_live_cards: int,
    max_unknown_cards: int,
) -> bool:
    """Decide se salvare lo stato corrente nel dataset value."""
    if collect_mode == COLLECT_ALL:
        return not (deck_size == 0 and not include_endgame)
    if collect_mode == COLLECT_WINDOW:
        if deck_size == 0:
            return bool(include_endgame)
        return unknown_live_cards <= max_unknown_cards
    return False


@njit(cache=True)
def _collect_value_game_into_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    overkill_guard_enabled: bool,
    epsilon: float,
    collect_mode: int,
    max_unknown_cards: int,
    include_endgame: bool,
    seed: int,
    xs: np.ndarray,
    current_delta_out: np.ndarray,
    final_delta_out: np.ndarray,
    phase_out: np.ndarray,
    unknown_out: np.ndarray,
    deck_size_out: np.ndarray,
    exploratory_out: np.ndarray,
    valid_out: np.ndarray,
) -> int:
    """
    Gioca una partita epsilon-greedy e salva target di continuazione deterministica.

    Ritorna il numero di stati validi scritti per questa partita.
    """
    _validate_mlp_weights_numba(w1, b1, w2, b2)
    shuffled = _shuffle_deck_numba(seed)

    deck = np.empty(_MAX_DECK_SIZE_2P, dtype=np.int64)
    hands = np.empty((2, _HAND_CAPACITY_2P), dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)

    deck_size_source = ACTION_DIM
    for _ in range(3):
        for player_index in range(2):
            deck_size_source -= 1
            hands[player_index, hand_sizes[player_index]] = shuffled[deck_size_source]
            hand_sizes[player_index] += 1

    deck_size_source -= 1
    trump_card = shuffled[deck_size_source]
    deck[0] = trump_card
    for i in range(deck_size_source):
        deck[i + 1] = shuffled[i]
    deck_size = deck_size_source + 1

    points = np.zeros(2, dtype=np.int64)
    table_cards = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    table_players = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    table_size = 0
    current_turn = 0
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[trump_card] = 1
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    trick_hist = np.zeros((20, 5), dtype=np.int64)
    trick_count = np.zeros(1, dtype=np.int64)

    record_count = 0
    ply_count = 0
    safety = 256
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0 and table_size == 0:
            break
        if ply_count >= _MAX_STATES_PER_GAME_2P:
            break

        player_index = current_turn
        unknown = _unknown_live_cards_numba(hand_sizes, deck_size, player_index)
        phase = _phase_code_numba(deck_size, unknown, max_unknown_cards)
        should_collect = _should_collect_numba(collect_mode, include_endgame, deck_size, unknown, max_unknown_cards)

        exploratory = False
        if epsilon > 0.0 and np.random.random() < epsilon:
            card_index = np.random.randint(0, hand_sizes[player_index])
            exploratory = True
        else:
            card_index = _choose_hybrid_mlp_card_index_numba(
                w1,
                b1,
                w2,
                b2,
                overkill_guard_enabled,
                hands,
                hand_sizes,
                points,
                deck,
                deck_size,
                table_cards,
                table_players,
                table_size,
                current_turn,
                trump_card,
                seen_cards,
                out_of_play_cards,
                trick_hist,
                trick_count[0],
            )

        if should_collect and record_count < _MAX_STATES_PER_GAME_2P:
            # Le feature registrate seguono l'encoder della policy di self-play:
            # con una policy v4 il dataset diventa v4 (storia REALE del loop, non dummy).
            if int(w1.shape[0]) == int(FEATURE_DIM_2P_V4):
                features, _mask = encode_fast_observation_arrays_v4_numba(
                    hands,
                    hand_sizes,
                    points,
                    table_cards,
                    table_size,
                    deck_size,
                    current_turn,
                    trump_card,
                    player_index,
                    seen_cards,
                    out_of_play_cards,
                    trick_hist,
                    trick_count[0],
                )
            else:
                features, _mask = encode_fast_observation_arrays_numba(
                    hands,
                    hand_sizes,
                    points,
                    table_cards,
                    table_size,
                    deck_size,
                    current_turn,
                    trump_card,
                    player_index,
                    seen_cards,
                    out_of_play_cards,
                    int(w1.shape[0]),
                )
            for feature_index in range(int(w1.shape[0])):
                xs[record_count, feature_index] = features[feature_index]

            (
                label_hands,
                label_hand_sizes,
                label_points,
                label_deck,
                label_table_cards,
                label_table_players,
                label_seen,
                label_out,
            ) = _copy_compact_state_numba(
                hands,
                hand_sizes,
                points,
                deck,
                table_cards,
                table_players,
                seen_cards,
                out_of_play_cards,
            )
            label_trick_hist = trick_hist.copy()
            label_trick_count = trick_count.copy()
            final0, final1 = _play_hybrid_mlp_to_terminal_numba(
                w1,
                b1,
                w2,
                b2,
                overkill_guard_enabled,
                label_hands,
                label_hand_sizes,
                label_points,
                label_deck,
                deck_size,
                label_table_cards,
                label_table_players,
                table_size,
                current_turn,
                trump_card,
                label_seen,
                label_out,
                label_trick_hist,
                label_trick_count,
            )

            current_delta = points[player_index] - points[1 - player_index]
            final_delta = final0 - final1
            if player_index == 1:
                final_delta = -final_delta

            current_delta_out[record_count] = float(current_delta)
            final_delta_out[record_count] = float(final_delta)
            phase_out[record_count] = int(phase)
            unknown_out[record_count] = int(unknown)
            deck_size_out[record_count] = int(deck_size)
            exploratory_out[record_count] = bool(exploratory)
            valid_out[record_count] = True
            record_count += 1

        deck_size, table_size, current_turn = _apply_numba_card_index(
            hands,
            hand_sizes,
            points,
            deck,
            deck_size,
            table_cards,
            table_players,
            table_size,
            current_turn,
            trump_card,
            card_index,
            seen_cards,
            out_of_play_cards,
            trick_hist,
            trick_count,
        )
        ply_count += 1

    return int(record_count)


@njit(cache=True, parallel=True)
def _collect_value_batch_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    overkill_guard_enabled: bool,
    epsilon: float,
    collect_mode: int,
    max_unknown_cards: int,
    include_endgame: bool,
    game_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collector batch parallelo: ogni partita scrive al massimo 40 plie nel proprio slot."""
    batch_size = int(game_seeds.shape[0])
    feature_dim = int(w1.shape[0])
    xs = np.zeros((batch_size, _MAX_STATES_PER_GAME_2P, feature_dim), dtype=np.float32)
    current_delta = np.zeros((batch_size, _MAX_STATES_PER_GAME_2P), dtype=np.float32)
    final_delta = np.zeros((batch_size, _MAX_STATES_PER_GAME_2P), dtype=np.float32)
    phase = np.full((batch_size, _MAX_STATES_PER_GAME_2P), -1, dtype=np.int64)
    unknown_live_cards = np.zeros((batch_size, _MAX_STATES_PER_GAME_2P), dtype=np.int64)
    deck_size = np.zeros((batch_size, _MAX_STATES_PER_GAME_2P), dtype=np.int64)
    exploratory_action = np.zeros((batch_size, _MAX_STATES_PER_GAME_2P), dtype=np.bool_)
    valid = np.zeros((batch_size, _MAX_STATES_PER_GAME_2P), dtype=np.bool_)
    record_counts = np.zeros(batch_size, dtype=np.int64)

    for game_index in prange(batch_size):
        count = _collect_value_game_into_numba(
            w1,
            b1,
            w2,
            b2,
            overkill_guard_enabled,
            epsilon,
            collect_mode,
            max_unknown_cards,
            include_endgame,
            int(game_seeds[game_index]),
            xs[game_index],
            current_delta[game_index],
            final_delta[game_index],
            phase[game_index],
            unknown_live_cards[game_index],
            deck_size[game_index],
            exploratory_action[game_index],
            valid[game_index],
        )
        record_counts[game_index] = int(count)

    return (
        xs,
        current_delta,
        final_delta,
        phase,
        unknown_live_cards,
        deck_size,
        exploratory_action,
        valid,
        record_counts,
    )


def collect_value_dataset_batch_numba(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    overkill_guard_enabled: bool,
    epsilon: float,
    collect_mode: int,
    max_unknown_cards: int,
    include_endgame: bool,
    game_seeds: np.ndarray,
) -> NumbaValueDatasetBatch:
    """Wrapper Python validato per il collector batch Numba."""
    seeds = np.asarray(game_seeds, dtype=np.int64)
    if seeds.ndim != 1:
        raise ValueError(f"game_seeds deve essere 1D, ottenuto shape={seeds.shape}")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon deve essere in [0,1]")
    if int(collect_mode) not in {COLLECT_ALL, COLLECT_WINDOW}:
        raise ValueError(f"collect_mode non supportato: {collect_mode}")
    if int(max_unknown_cards) < 0:
        raise ValueError("max_unknown_cards deve essere >= 0")

    w1_arr = np.ascontiguousarray(np.asarray(w1, dtype=np.float32))
    b1_arr = np.ascontiguousarray(np.asarray(b1, dtype=np.float32))
    w2_arr = np.ascontiguousarray(np.asarray(w2, dtype=np.float32))
    b2_arr = np.ascontiguousarray(np.asarray(b2, dtype=np.float32))
    if w1_arr.ndim != 2 or b1_arr.ndim != 1 or w2_arr.ndim != 2 or b2_arr.ndim != 1:
        raise ValueError("Pesi MLP non validi: attesi w1/w2 2D e b1/b2 1D")
    if w1_arr.shape[1] != b1_arr.shape[0] or w2_arr.shape != (w1_arr.shape[1], ACTION_DIM):
        raise ValueError("Shape MLP non compatibili")
    if b2_arr.shape != (ACTION_DIM,):
        raise ValueError("b2 deve avere shape (40,)")

    (
        xs,
        current_delta,
        final_delta,
        phase,
        unknown_live_cards,
        deck_size,
        exploratory_action,
        valid,
        _record_counts,
    ) = _collect_value_batch_numba(
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        bool(overkill_guard_enabled),
        float(epsilon),
        int(collect_mode),
        int(max_unknown_cards),
        bool(include_endgame),
        seeds,
    )
    return NumbaValueDatasetBatch(
        xs=xs,
        current_delta=current_delta,
        final_delta=final_delta,
        phase=phase,
        unknown_live_cards=unknown_live_cards,
        deck_size=deck_size,
        exploratory_action=exploratory_action,
        valid=valid,
        games_completed=int(seeds.shape[0]),
    )


def warm_up_numba_value_dataset() -> None:
    """Compila il collector value con pesi sintetici minimali."""
    feature_dim = int(FEATURE_DIM_2P_V3)
    w1 = np.zeros((feature_dim, 1), dtype=np.float32)
    b1 = np.zeros(1, dtype=np.float32)
    w2 = np.zeros((1, ACTION_DIM), dtype=np.float32)
    b2 = np.zeros(ACTION_DIM, dtype=np.float32)
    collect_value_dataset_batch_numba(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        overkill_guard_enabled=True,
        epsilon=0.0,
        collect_mode=COLLECT_WINDOW,
        max_unknown_cards=8,
        include_endgame=False,
        game_seeds=np.asarray([1], dtype=np.int64),
    )
