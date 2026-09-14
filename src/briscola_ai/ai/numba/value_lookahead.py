"""
Kernel Numba per V-lookahead su stati numerici determinizzati.

Il runtime web usa `ValueLookaheadAgent`, che parte da `PlayerObservation` e campiona
determinizzazioni anti-cheat. Nei training futuri, invece, spesso abbiamo gia' uno stato
2-player completo o determinizzato in array NumPy: in quel caso riconvertire a `GameState`,
`PlayerObservation` e oggetti Python a ogni mossa spreca throughput.

Questo modulo espone quindi un entrypoint JIT che:
- lavora direttamente su `hands`, `deck`, `points`, `table_cards`;
- prova le carte in mano al giocatore corrente;
- usa la policy MLP come continuazione quando la carta candidata apre una presa;
- usa il solver endgame Numba quando la foglia e' a mazzo vuoto;
- valuta le altre foglie con il value model MLP.

Il kernel non fa determinizzazione: il chiamante deve passargli uno stato numerico coerente
con il proprio protocollo di training/evaluation.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ...domain.card_id import card_to_id
from ...domain.state import GameState
from ..encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4
from ..endgame.numba_solver import choose_endgame_card_numba_arrays
from .core import ACTION_DIM, _choose_policy_card_index_numba, _shuffle_deck_numba
from .observation import (
    _action_id_to_hand_index_numba,
    _apply_numba_card_index,
    _apply_overkill_guard_numba,
    _argmax_mlp_policy_action_numba,
    _record_mlp_policy_decision_numba,
    _trump_overkill_penalty_numba,
    encode_fast_observation_arrays_numba,
    encode_fast_observation_arrays_v4_numba,
)
from .pimc import _belief_card_weights_numba, choose_pimc_card_numba_arrays
from .types import NumbaA2CBatch, NumbaA2CTrajectory

_MAX_DECK_SIZE_2P = 34
_HAND_CAPACITY_2P = 3
_TABLE_CAPACITY_2P = 2
OPPONENT_MODE_RULE = 0
OPPONENT_MODE_MODEL = 1
OPPONENT_MODE_VALUE_LOOKAHEAD = 2
OPPONENT_MODE_PIMC_BELIEF = 3


def arrays_from_game_state_for_value_lookahead(
    state: GameState,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Converte un `GameState` 2-player in array compatibili con il kernel V-lookahead.

    Il formato del mazzo segue il fast/numba path del progetto: la prossima pescata e'
    `deck[deck_size - 1]`, come `domain.engine.step` pesca con `deck.pop()`.
    """
    if state.num_players != 2 or state.is_team_game:
        raise ValueError("Il V-lookahead Numba supporta solo partite 2-player")
    if state.trump_card is None:
        raise ValueError("Briscola assente")

    hands = np.full((2, _HAND_CAPACITY_2P), -1, dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)
    for player_index in range(2):
        hand = state.players[player_index].hand
        if len(hand) > _HAND_CAPACITY_2P:
            raise ValueError(f"Mano troppo grande per 2-player: player={player_index} size={len(hand)}")
        hand_sizes[player_index] = len(hand)
        for card_index, card in enumerate(hand):
            hands[player_index, card_index] = card_to_id(card)

    deck = np.full(_MAX_DECK_SIZE_2P, -1, dtype=np.int64)
    if len(state.deck) > _MAX_DECK_SIZE_2P:
        raise ValueError(f"Mazzo troppo grande per 2-player: {len(state.deck)}")
    for index, card in enumerate(state.deck):
        deck[index] = card_to_id(card)

    points = np.asarray([state.players[0].points, state.players[1].points], dtype=np.int64)
    table_cards = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    table_players = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    if len(state.table_cards) > _TABLE_CAPACITY_2P:
        raise ValueError(f"Tavolo troppo grande per 2-player: {len(state.table_cards)}")
    for index, (card, player_index) in enumerate(state.table_cards):
        table_cards[index] = card_to_id(card)
        table_players[index] = int(player_index)

    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[card_to_id(state.trump_card)] = 1
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    for card, _player_index in state.table_cards:
        card_id = card_to_id(card)
        seen_cards[card_id] = 1
        out_of_play_cards[card_id] = 1
    for player in state.players:
        for card in player.captured_cards:
            card_id = card_to_id(card)
            seen_cards[card_id] = 1
            out_of_play_cards[card_id] = 1

    trick_hist = np.zeros((20, 5), dtype=np.int64)
    num_tricks = min(len(state.trick_history), 20)
    for index in range(num_tricks):
        record = state.trick_history[index]
        (lead_card, lead_player), (resp_card, _resp_player) = record.cards
        trick_hist[index, 0] = card_to_id(lead_card)
        trick_hist[index, 1] = int(lead_player)
        trick_hist[index, 2] = card_to_id(resp_card)
        trick_hist[index, 3] = int(record.winner_index)
        trick_hist[index, 4] = int(record.points)

    return (
        hands,
        hand_sizes,
        points,
        deck,
        len(state.deck),
        table_cards,
        table_players,
        len(state.table_cards),
        int(state.current_turn),
        card_to_id(state.trump_card),
        seen_cards,
        out_of_play_cards,
        trick_hist,
        num_tricks,
    )


@njit(cache=True)
def _copy_state_arrays(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    deck: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copia lo stato compatto per simulare una carta candidata senza mutare la radice."""
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
def _terminal_or_endgame_score_for_root(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    current_turn: int,
    trump_card: int,
    root_player: int,
) -> tuple[bool, float]:
    """
    Valuta terminali ed endgame con solver.

    Ritorna `(handled, score_root)`. `handled=False` significa che la foglia va valutata da V.
    """
    if hand_sizes[0] == 0 and hand_sizes[1] == 0 and table_size == 0:
        if root_player == 0:
            return True, float(points[0] - points[1])
        return True, float(points[1] - points[0])

    best_card_index, final_delta_p0_p1 = choose_endgame_card_numba_arrays(
        hands,
        hand_sizes,
        points,
        table_cards,
        table_players,
        table_size,
        current_turn,
        trump_card,
    )
    if best_card_index >= 0 and hand_sizes[0] + hand_sizes[1] > 0:
        if root_player == 0:
            return True, float(final_delta_p0_p1)
        return True, float(-final_delta_p0_p1)
    return False, 0.0


@njit(cache=True)
def _predict_value_points_numba(
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_size: int,
    deck_size: int,
    current_turn: int,
    trump_card: int,
    player_index: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    trick_hist: np.ndarray,
    num_tricks: int,
) -> float:
    """Forward del value model scalare in punti dal punto di vista di `player_index`."""
    feature_dim = value_w1.shape[0]
    hidden_dim = value_w1.shape[1]
    if feature_dim == int(FEATURE_DIM_2P_V4):
        features, _action_mask = encode_fast_observation_arrays_v4_numba(
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
            num_tricks,
        )
    else:
        features, _action_mask = encode_fast_observation_arrays_numba(
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
            feature_dim,
        )

    raw = float(value_b2)
    for hidden_index in range(hidden_dim):
        hidden_value = value_b1[hidden_index]
        for feature_index in range(feature_dim):
            hidden_value += features[feature_index] * value_w1[feature_index, hidden_index]
        if hidden_value < 0.0:
            hidden_value = 0.0
        raw += hidden_value * float(value_w2[hidden_index])

    pred = raw * float(value_target_scale)
    if value_target_is_residual:
        opponent = 1 - player_index
        pred += float(points[player_index] - points[opponent])
    return float(pred)


@njit(cache=True)
def choose_value_lookahead_card_numba_arrays(
    policy_w1: np.ndarray,
    policy_b1: np.ndarray,
    policy_w2: np.ndarray,
    policy_b2: np.ndarray,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
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
    overkill_guard_enabled: bool,
) -> tuple[int, float]:
    """
    Sceglie una carta con depth-1 V-lookahead su uno stato determinizzato.

    Ritorna `(card_index, expected_score_for_root)`, dove `card_index` e' un indice nella
    mano del `current_turn` radice.
    """
    root_player = int(current_turn)
    hand_size = int(hand_sizes[root_player])
    if hand_size <= 0:
        return -1, 0.0

    if int(deck_size) == 0:
        best_card_index, final_delta_p0_p1 = choose_endgame_card_numba_arrays(
            hands,
            hand_sizes,
            points,
            table_cards,
            table_players,
            table_size,
            current_turn,
            trump_card,
        )
        if root_player == 0:
            return int(best_card_index), float(final_delta_p0_p1)
        return int(best_card_index), float(-final_delta_p0_p1)

    best_card_index = 0
    best_score = -1.0e30

    for candidate_index in range(hand_size):
        (
            candidate_hands,
            candidate_hand_sizes,
            candidate_points,
            candidate_deck,
            candidate_table_cards,
            candidate_table_players,
            candidate_seen,
            candidate_out_of_play,
        ) = _copy_state_arrays(
            hands, hand_sizes, points, deck, table_cards, table_players, seen_cards, out_of_play_cards
        )
        # Storia REALE copiata per-candidato: la simulazione la estende con le prese
        # simulate, cosi' policy base e value model v4 giocano/valutano CON la memoria
        # (le copie isolano i candidati tra loro e dallo stato radice).
        sim_trick_hist = trick_hist.copy()
        sim_trick_count = np.zeros(1, dtype=np.int64)
        sim_trick_count[0] = num_tricks

        candidate_deck_size, candidate_table_size, candidate_turn = _apply_numba_card_index(
            candidate_hands,
            candidate_hand_sizes,
            candidate_points,
            candidate_deck,
            int(deck_size),
            candidate_table_cards,
            candidate_table_players,
            int(table_size),
            int(current_turn),
            int(trump_card),
            candidate_index,
            candidate_seen,
            candidate_out_of_play,
            sim_trick_hist,
            sim_trick_count,
        )

        if candidate_table_size == 1:
            if candidate_deck_size == 0:
                response_index, _response_delta = choose_endgame_card_numba_arrays(
                    candidate_hands,
                    candidate_hand_sizes,
                    candidate_points,
                    candidate_table_cards,
                    candidate_table_players,
                    candidate_table_size,
                    candidate_turn,
                    trump_card,
                )
            else:
                response_action = _argmax_mlp_policy_action_numba(
                    policy_w1,
                    policy_b1,
                    policy_w2,
                    policy_b2,
                    candidate_hands,
                    candidate_hand_sizes,
                    candidate_points,
                    candidate_table_cards,
                    candidate_table_size,
                    candidate_deck_size,
                    candidate_turn,
                    trump_card,
                    candidate_turn,
                    candidate_seen,
                    candidate_out_of_play,
                    sim_trick_hist,
                    sim_trick_count[0],
                )
                response_action = _apply_overkill_guard_numba(
                    response_action,
                    candidate_hands,
                    candidate_hand_sizes,
                    candidate_turn,
                    candidate_table_cards,
                    candidate_table_players,
                    candidate_table_size,
                    trump_card,
                    overkill_guard_enabled,
                )
                response_index = _action_id_to_hand_index_numba(
                    candidate_hands, candidate_hand_sizes, candidate_turn, response_action
                )

            candidate_deck_size, candidate_table_size, candidate_turn = _apply_numba_card_index(
                candidate_hands,
                candidate_hand_sizes,
                candidate_points,
                candidate_deck,
                candidate_deck_size,
                candidate_table_cards,
                candidate_table_players,
                candidate_table_size,
                candidate_turn,
                trump_card,
                response_index,
                candidate_seen,
                candidate_out_of_play,
                sim_trick_hist,
                sim_trick_count,
            )

        if candidate_deck_size == 0:
            handled, score = _terminal_or_endgame_score_for_root(
                candidate_hands,
                candidate_hand_sizes,
                candidate_points,
                candidate_table_cards,
                candidate_table_players,
                candidate_table_size,
                candidate_turn,
                trump_card,
                root_player,
            )
            if not handled:
                score = 0.0
        elif candidate_hand_sizes[0] == 0 and candidate_hand_sizes[1] == 0 and candidate_table_size == 0:
            if root_player == 0:
                score = float(candidate_points[0] - candidate_points[1])
            else:
                score = float(candidate_points[1] - candidate_points[0])
        else:
            leaf_player = int(candidate_turn)
            leaf_score = _predict_value_points_numba(
                value_w1,
                value_b1,
                value_w2,
                value_b2,
                value_target_scale,
                value_target_is_residual,
                candidate_hands,
                candidate_hand_sizes,
                candidate_points,
                candidate_table_cards,
                candidate_table_size,
                candidate_deck_size,
                candidate_turn,
                trump_card,
                leaf_player,
                candidate_seen,
                candidate_out_of_play,
                sim_trick_hist,
                sim_trick_count[0],
            )
            score = leaf_score if leaf_player == root_player else -leaf_score

        if score > best_score:
            best_score = score
            best_card_index = candidate_index

    chosen_action = hands[root_player, best_card_index]
    guarded_action = _apply_overkill_guard_numba(
        chosen_action,
        hands,
        hand_sizes,
        root_player,
        table_cards,
        table_players,
        table_size,
        trump_card,
        overkill_guard_enabled,
    )
    if int(guarded_action) != int(chosen_action):
        best_card_index = _action_id_to_hand_index_numba(hands, hand_sizes, root_player, guarded_action)

    return int(best_card_index), float(best_score)


def choose_value_lookahead_card_numba(
    *,
    policy_w1: np.ndarray,
    policy_b1: np.ndarray,
    policy_w2: np.ndarray,
    policy_b2: np.ndarray,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    state: GameState,
    overkill_guard_enabled: bool = True,
) -> tuple[int, float]:
    """Wrapper Python comodo per test/benchmark da `GameState` canonico 2-player."""
    arrays = arrays_from_game_state_for_value_lookahead(state)
    return choose_value_lookahead_card_numba_arrays(
        policy_w1,
        policy_b1,
        policy_w2,
        policy_b2,
        value_w1,
        value_b1,
        value_w2,
        float(value_b2),
        float(value_target_scale),
        bool(value_target_is_residual),
        *arrays,
        bool(overkill_guard_enabled),
    )


def warm_up_numba_value_lookahead() -> None:
    """Compila il kernel V-lookahead con pesi minimali sintetici."""
    policy_w1 = np.zeros((310, 1), dtype=np.float32)
    policy_b1 = np.zeros(1, dtype=np.float32)
    policy_w2 = np.zeros((1, ACTION_DIM), dtype=np.float32)
    policy_b2 = np.zeros(ACTION_DIM, dtype=np.float32)
    value_w1 = np.zeros((310, 1), dtype=np.float32)
    value_b1 = np.zeros(1, dtype=np.float32)
    value_w2 = np.zeros(1, dtype=np.float32)

    hands = np.asarray([[0, 1, 2], [10, 11, 12]], dtype=np.int64)
    hand_sizes = np.asarray([3, 3], dtype=np.int64)
    points = np.asarray([0, 0], dtype=np.int64)
    deck = np.full(_MAX_DECK_SIZE_2P, -1, dtype=np.int64)
    deck[:4] = np.asarray([20, 21, 22, 23], dtype=np.int64)
    table_cards = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    table_players = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[20] = 1

    choose_value_lookahead_card_numba_arrays(
        policy_w1,
        policy_b1,
        policy_w2,
        policy_b2,
        value_w1,
        value_b1,
        value_w2,
        0.0,
        120.0,
        True,
        hands,
        hand_sizes,
        points,
        deck,
        4,
        table_cards,
        table_players,
        0,
        0,
        20,
        seen_cards,
        out_of_play_cards,
        np.zeros((20, 5), dtype=np.int64),
        0,
        True,
    )


def _as_float32_matrix(name: str, value: np.ndarray) -> np.ndarray:
    """Normalizza un peso 2D a float32 contiguo."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} deve essere una matrice 2D, ottenuto shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _as_float32_vector(name: str, value: np.ndarray) -> np.ndarray:
    """Normalizza un vettore 1D a float32 contiguo."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} deve essere un vettore 1D, ottenuto shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _overkill_penalty_mode_code(mode: str) -> int:
    """Codifica la modalità reward shaping anti-overkill per il core JIT."""
    normalized = str(mode).strip().lower()
    if normalized == "flat":
        return 0
    if normalized == "gap":
        return 1
    raise ValueError(f"overkill_penalty_mode non supportato: {mode!r}")


def _validate_a2c_value_lookahead_inputs(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Valida e normalizza i tensori necessari al collector value-lookahead."""
    w1_arr = _as_float32_matrix("w1", w1)
    b1_arr = _as_float32_vector("b1", b1)
    w2_arr = _as_float32_matrix("w2", w2)
    b2_arr = _as_float32_vector("b2", b2)
    wv_arr = _as_float32_vector("wv", wv)
    opponent_w1_arr = _as_float32_matrix("opponent_w1", opponent_w1)
    opponent_b1_arr = _as_float32_vector("opponent_b1", opponent_b1)
    opponent_w2_arr = _as_float32_matrix("opponent_w2", opponent_w2)
    opponent_b2_arr = _as_float32_vector("opponent_b2", opponent_b2)
    value_w1_arr = _as_float32_matrix("value_w1", value_w1)
    value_b1_arr = _as_float32_vector("value_b1", value_b1)
    value_w2_arr = _as_float32_vector("value_w2", value_w2)

    feature_dim = int(w1_arr.shape[0])
    hidden_dim = int(w1_arr.shape[1])
    if feature_dim not in (
        int(FEATURE_DIM_2P_V1),
        int(FEATURE_DIM_2P_V2),
        int(FEATURE_DIM_2P_V3),
        int(FEATURE_DIM_2P_V4),
        int(FEATURE_DIM_2P_V4) + 40,  # iterazione 1b: input v4 + belief
    ):
        raise ValueError(
            f"w1 feature_dim={feature_dim}; "
            f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
            f"{int(FEATURE_DIM_2P_V3)}, {int(FEATURE_DIM_2P_V4)} o {int(FEATURE_DIM_2P_V4) + 40}"
        )
    if b1_arr.shape != (hidden_dim,):
        raise ValueError(f"b1 shape={b1_arr.shape}; atteso {(hidden_dim,)}")
    if w2_arr.shape != (hidden_dim, ACTION_DIM):
        raise ValueError(f"w2 shape={w2_arr.shape}; atteso {(hidden_dim, ACTION_DIM)}")
    if b2_arr.shape != (ACTION_DIM,):
        raise ValueError(f"b2 shape={b2_arr.shape}; atteso {(ACTION_DIM,)}")
    if wv_arr.shape != (hidden_dim,):
        raise ValueError(f"wv shape={wv_arr.shape}; atteso {(hidden_dim,)}")

    opponent_feature_dim = int(opponent_w1_arr.shape[0])
    opponent_hidden_dim = int(opponent_w1_arr.shape[1])
    if opponent_feature_dim not in (
        int(FEATURE_DIM_2P_V1),
        int(FEATURE_DIM_2P_V2),
        int(FEATURE_DIM_2P_V3),
        int(FEATURE_DIM_2P_V4),
    ):
        raise ValueError(
            f"opponent_w1 feature_dim={opponent_feature_dim}; "
            f"atteso {int(FEATURE_DIM_2P_V1)}, {int(FEATURE_DIM_2P_V2)}, "
            f"{int(FEATURE_DIM_2P_V3)} o {int(FEATURE_DIM_2P_V4)}"
        )
    if opponent_b1_arr.shape != (opponent_hidden_dim,):
        raise ValueError(f"opponent_b1 shape={opponent_b1_arr.shape}; atteso {(opponent_hidden_dim,)}")
    if opponent_w2_arr.shape != (opponent_hidden_dim, ACTION_DIM):
        raise ValueError(f"opponent_w2 shape={opponent_w2_arr.shape}; atteso {(opponent_hidden_dim, ACTION_DIM)}")
    if opponent_b2_arr.shape != (ACTION_DIM,):
        raise ValueError(f"opponent_b2 shape={opponent_b2_arr.shape}; atteso {(ACTION_DIM,)}")

    value_hidden_dim = int(value_w1_arr.shape[1])
    if int(value_w1_arr.shape[0]) != int(opponent_w1_arr.shape[0]):
        raise ValueError(
            f"value_w1 feature_dim={int(value_w1_arr.shape[0])}; "
            f"atteso feature_dim opponent={int(opponent_w1_arr.shape[0])}"
        )
    if value_b1_arr.shape != (value_hidden_dim,):
        raise ValueError(f"value_b1 shape={value_b1_arr.shape}; atteso {(value_hidden_dim,)}")
    if value_w2_arr.shape != (value_hidden_dim,):
        raise ValueError(f"value_w2 shape={value_w2_arr.shape}; atteso {(value_hidden_dim,)}")

    return (
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        wv_arr,
        opponent_w1_arr,
        opponent_b1_arr,
        opponent_w2_arr,
        opponent_b2_arr,
        value_w1_arr,
        value_b1_arr,
        value_w2_arr,
    )


@njit(cache=True)
def _choose_training_opponent_card_index_numba(
    opponent_mode: int,
    opponent_code: int,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    value_max_unknown_cards: int,
    opponent_belief_w1: np.ndarray,
    opponent_belief_b1: np.ndarray,
    opponent_belief_w2: np.ndarray,
    opponent_belief_b2: np.ndarray,
    opponent_pimc_determinizations: int,
    opponent_belief_uniform_mix: float,
    opponent_overkill_guard: bool,
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
    """Dispatch dell'avversario nel collector A2C: rule / model / V-lookahead / PIMC belief."""
    if opponent_mode == OPPONENT_MODE_PIMC_BELIEF:
        # Maestro PIMC belief: la search vede SOLO l'informazione lecita del suo seat
        # (la propria mano + pubblico); la finestra riusa `value_max_unknown_cards`.
        if deck_size == 0:
            # Finale a informazione completa deducibile: solver esatto, come nel runtime.
            solver_index, _delta = choose_endgame_card_numba_arrays(
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                current_turn,
                trump_card,
            )
            if 0 <= solver_index < hand_sizes[current_turn]:
                return int(solver_index)
        unknown_live_cards = int(hand_sizes[1 - current_turn]) + int(deck_size)
        if deck_size > 0 and unknown_live_cards <= int(value_max_unknown_cards):
            my_hand = np.full(3, -1, dtype=np.int64)
            my_hand_size = int(hand_sizes[current_turn])
            for i in range(my_hand_size):
                my_hand[i] = hands[current_turn, i]
            in_my_hand = np.zeros(ACTION_DIM, dtype=np.int64)
            for i in range(my_hand_size):
                in_my_hand[my_hand[i]] = 1

            v4_features, _mask = encode_fast_observation_arrays_v4_numba(
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
            weights = np.empty(ACTION_DIM, dtype=np.float64)
            _belief_card_weights_numba(
                v4_features,
                opponent_belief_w1,
                opponent_belief_b1,
                opponent_belief_w2,
                opponent_belief_b2,
                in_my_hand,
                out_of_play_cards,
                opponent_belief_uniform_mix,
                weights,
            )
            card_index = choose_pimc_card_numba_arrays(
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                weights,
                my_hand,
                my_hand_size,
                int(hand_sizes[1 - current_turn]),
                int(deck_size),
                table_cards,
                table_players,
                table_size,
                current_turn,
                trump_card,
                points,
                out_of_play_cards,
                seen_cards,
                trick_hist,
                num_tricks,
                opponent_pimc_determinizations,
                -1,
            )
            if 0 <= card_index < my_hand_size:
                return int(card_index)
        # Fuori finestra o kernel non applicabile: stesso fallback MLP del runtime.
        opponent_mode = OPPONENT_MODE_MODEL

    if opponent_mode == OPPONENT_MODE_VALUE_LOOKAHEAD:
        unknown_live_cards = int(hand_sizes[1 - current_turn]) + int(deck_size)
        if deck_size == 0 or unknown_live_cards <= int(value_max_unknown_cards):
            card_index, _score = choose_value_lookahead_card_numba_arrays(
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                value_w1,
                value_b1,
                value_w2,
                value_b2,
                value_target_scale,
                value_target_is_residual,
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
                num_tricks,
                opponent_overkill_guard,
            )
            if 0 <= card_index < hand_sizes[current_turn]:
                return int(card_index)

        # Fuori finestra: stesso fallback MLP dell'agente runtime (solver gestito sopra a mazzo vuoto).
        opponent_mode = OPPONENT_MODE_MODEL

    if opponent_mode == OPPONENT_MODE_MODEL:
        action_id = _argmax_mlp_policy_action_numba(
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
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
            opponent_overkill_guard,
        )
        return _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)

    return _choose_policy_card_index_numba(
        opponent_code,
        hands,
        hand_sizes,
        current_turn,
        table_cards,
        table_players,
        table_size,
        deck_size,
        trump_card,
        seen_cards,
    )


@njit(cache=True)
def _collect_mlp_policy_game_value_lookahead_opponent_into_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_mode: int,
    opponent_code: int,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    value_max_unknown_cards: int,
    opponent_belief_w1: np.ndarray,
    opponent_belief_b1: np.ndarray,
    opponent_belief_w2: np.ndarray,
    opponent_belief_b2: np.ndarray,
    opponent_pimc_determinizations: int,
    opponent_belief_uniform_mix: float,
    belief_enabled: bool,
    belief_w1: np.ndarray,
    belief_b1: np.ndarray,
    belief_w2: np.ndarray,
    belief_b2: np.ndarray,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int,
    overkill_penalty_mode_code: int,
    seed: int,
    policy_seat: int,
    xs: np.ndarray,
    z1s: np.ndarray,
    hs: np.ndarray,
    masks: np.ndarray,
    probs: np.ndarray,
    action_ids: np.ndarray,
    value_preds: np.ndarray,
    rewards: np.ndarray,
) -> tuple[int, int, int, int, float]:
    """
    Raccoglie una partita A2C full-JIT contro opponent value-lookahead determinized.

    La policy trainata resta anti-cheat: le sue osservazioni sono sempre encode da vista pubblica.
    L'opponent avanzato, invece, usa lo stato determinizzato del rollout come una singola
    determinizzazione: e' un avversario di training forte, non una replica bit-a-bit della UI.
    """
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
    table_cards = np.empty(_TABLE_CAPACITY_2P, dtype=np.int64)
    table_players = np.empty(_TABLE_CAPACITY_2P, dtype=np.int64)
    table_size = 0
    current_turn = 0
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[trump_card] = 1
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    # Storia REALE delle prese: alimenta l'encoder v4 della policy in training.
    trick_hist = np.zeros((20, 5), dtype=np.int64)
    trick_count = np.zeros(1, dtype=np.int64)

    step_count = 0
    entropy_sum = 0.0
    safety = 5000
    while safety > 0:
        safety -= 1

        while not (hand_sizes[0] == 0 and hand_sizes[1] == 0) and current_turn != policy_seat:
            opp_card_index = _choose_training_opponent_card_index_numba(
                opponent_mode,
                opponent_code,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                value_w1,
                value_b1,
                value_w2,
                value_b2,
                value_target_scale,
                value_target_is_residual,
                value_max_unknown_cards,
                opponent_belief_w1,
                opponent_belief_b1,
                opponent_belief_w2,
                opponent_belief_b2,
                opponent_pimc_determinizations,
                opponent_belief_uniform_mix,
                opponent_overkill_guard,
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
                opp_card_index,
                seen_cards,
                out_of_play_cards,
                trick_hist,
                trick_count,
            )

        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        diff_before = points[policy_seat] - points[1 - policy_seat]
        action_id, value_pred, entropy = _record_mlp_policy_decision_numba(
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
            hands,
            hand_sizes,
            points,
            table_cards,
            table_size,
            deck_size,
            current_turn,
            trump_card,
            policy_seat,
            seen_cards,
            out_of_play_cards,
            trick_hist,
            trick_count[0],
            belief_enabled,
            belief_w1,
            belief_b1,
            belief_w2,
            belief_b2,
            xs,
            z1s,
            hs,
            masks,
            probs,
            step_count,
        )
        policy_card_index = _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)
        action_ids[step_count] = action_id
        value_preds[step_count] = value_pred
        entropy_sum += entropy
        extra_penalty = _trump_overkill_penalty_numba(
            hands,
            hand_sizes,
            table_cards,
            table_players,
            table_size,
            trump_card,
            policy_seat,
            policy_card_index,
            overkill_penalty_beta,
            overkill_low_lead_points_max,
            overkill_penalty_mode_code,
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
            policy_card_index,
            seen_cards,
            out_of_play_cards,
            trick_hist,
            trick_count,
        )

        while not (hand_sizes[0] == 0 and hand_sizes[1] == 0) and current_turn != policy_seat:
            opp_card_index = _choose_training_opponent_card_index_numba(
                opponent_mode,
                opponent_code,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                value_w1,
                value_b1,
                value_w2,
                value_b2,
                value_target_scale,
                value_target_is_residual,
                value_max_unknown_cards,
                opponent_belief_w1,
                opponent_belief_b1,
                opponent_belief_w2,
                opponent_belief_b2,
                opponent_pimc_determinizations,
                opponent_belief_uniform_mix,
                opponent_overkill_guard,
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
                opp_card_index,
                seen_cards,
                out_of_play_cards,
                trick_hist,
                trick_count,
            )

        diff_after = points[policy_seat] - points[1 - policy_seat]
        rewards[step_count] = float(diff_after - diff_before) / 120.0 + extra_penalty
        step_count += 1

    policy_points = points[policy_seat]
    opponent_points = points[1 - policy_seat]
    if policy_points > opponent_points:
        winner_out = 0
    elif opponent_points > policy_points:
        winner_out = 1
    else:
        winner_out = -1
    avg_entropy = entropy_sum / float(step_count) if step_count > 0 else 0.0
    return (
        int(policy_points),
        int(opponent_points),
        int(winner_out),
        int(step_count),
        float(avg_entropy),
    )


@njit(cache=True)
def _collect_mlp_policy_value_lookahead_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_mode: int,
    opponent_code: int,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    value_max_unknown_cards: int,
    opponent_belief_w1: np.ndarray,
    opponent_belief_b1: np.ndarray,
    opponent_belief_w2: np.ndarray,
    opponent_belief_b2: np.ndarray,
    opponent_pimc_determinizations: int,
    opponent_belief_uniform_mix: float,
    belief_enabled: bool,
    belief_w1: np.ndarray,
    belief_b1: np.ndarray,
    belief_w2: np.ndarray,
    belief_b2: np.ndarray,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int,
    overkill_penalty_mode_code: int,
    seed: int,
    policy_seat: int,
) -> tuple[
    int,
    int,
    int,
    int,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Wrapper JIT: alloca i buffer per una singola partita contro opponent value-aware."""
    max_steps = 20
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]
    xs = np.zeros((max_steps, feature_dim), dtype=np.float32)
    z1s = np.zeros((max_steps, hidden_dim), dtype=np.float32)
    hs = np.zeros((max_steps, hidden_dim), dtype=np.float32)
    masks = np.zeros((max_steps, ACTION_DIM), dtype=np.bool_)
    probs = np.zeros((max_steps, ACTION_DIM), dtype=np.float32)
    action_ids = np.full(max_steps, -1, dtype=np.int64)
    value_preds = np.zeros(max_steps, dtype=np.float32)
    rewards = np.zeros(max_steps, dtype=np.float32)

    policy_points, opponent_points, winner, step_count, avg_entropy = (
        _collect_mlp_policy_game_value_lookahead_opponent_into_numba(
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
            opponent_mode,
            opponent_code,
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            opponent_overkill_guard,
            value_w1,
            value_b1,
            value_w2,
            value_b2,
            value_target_scale,
            value_target_is_residual,
            value_max_unknown_cards,
            opponent_belief_w1,
            opponent_belief_b1,
            opponent_belief_w2,
            opponent_belief_b2,
            opponent_pimc_determinizations,
            opponent_belief_uniform_mix,
            belief_enabled,
            belief_w1,
            belief_b1,
            belief_w2,
            belief_b2,
            overkill_penalty_beta,
            overkill_low_lead_points_max,
            overkill_penalty_mode_code,
            seed,
            policy_seat,
            xs,
            z1s,
            hs,
            masks,
            probs,
            action_ids,
            value_preds,
            rewards,
        )
    )
    return (
        int(policy_points),
        int(opponent_points),
        int(winner),
        int(step_count),
        float(avg_entropy),
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    )


@njit(cache=True, parallel=True)
def _collect_mlp_policy_value_lookahead_batch_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_modes: np.ndarray,
    opponent_codes: np.ndarray,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    value_max_unknown_cards: int,
    opponent_belief_w1: np.ndarray,
    opponent_belief_b1: np.ndarray,
    opponent_belief_w2: np.ndarray,
    opponent_belief_b2: np.ndarray,
    opponent_pimc_determinizations: int,
    opponent_belief_uniform_mix: float,
    belief_enabled: bool,
    belief_w1: np.ndarray,
    belief_b1: np.ndarray,
    belief_w2: np.ndarray,
    belief_b2: np.ndarray,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int,
    overkill_penalty_mode_code: int,
    game_seeds: np.ndarray,
    policy_seats: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Batch full-JIT per training A2C contro opponent rule/model/value-lookahead."""
    batch_size = game_seeds.shape[0]
    max_steps = 20
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]

    policy_points = np.zeros(batch_size, dtype=np.int64)
    opponent_points = np.zeros(batch_size, dtype=np.int64)
    winners = np.zeros(batch_size, dtype=np.int64)
    step_counts = np.zeros(batch_size, dtype=np.int64)
    avg_entropies = np.zeros(batch_size, dtype=np.float32)
    xs = np.zeros((batch_size, max_steps, feature_dim), dtype=np.float32)
    z1s = np.zeros((batch_size, max_steps, hidden_dim), dtype=np.float32)
    hs = np.zeros((batch_size, max_steps, hidden_dim), dtype=np.float32)
    masks = np.zeros((batch_size, max_steps, ACTION_DIM), dtype=np.bool_)
    probs = np.zeros((batch_size, max_steps, ACTION_DIM), dtype=np.float32)
    action_ids = np.full((batch_size, max_steps), -1, dtype=np.int64)
    value_preds = np.zeros((batch_size, max_steps), dtype=np.float32)
    rewards = np.zeros((batch_size, max_steps), dtype=np.float32)

    for game_idx in prange(batch_size):
        p_points, o_points, winner, steps, avg_entropy = _collect_mlp_policy_game_value_lookahead_opponent_into_numba(
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
            int(opponent_modes[game_idx]),
            int(opponent_codes[game_idx]),
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            opponent_overkill_guard,
            value_w1,
            value_b1,
            value_w2,
            value_b2,
            value_target_scale,
            value_target_is_residual,
            value_max_unknown_cards,
            opponent_belief_w1,
            opponent_belief_b1,
            opponent_belief_w2,
            opponent_belief_b2,
            opponent_pimc_determinizations,
            opponent_belief_uniform_mix,
            belief_enabled,
            belief_w1,
            belief_b1,
            belief_w2,
            belief_b2,
            overkill_penalty_beta,
            overkill_low_lead_points_max,
            overkill_penalty_mode_code,
            int(game_seeds[game_idx]),
            int(policy_seats[game_idx]),
            xs[game_idx],
            z1s[game_idx],
            hs[game_idx],
            masks[game_idx],
            probs[game_idx],
            action_ids[game_idx],
            value_preds[game_idx],
            rewards[game_idx],
        )
        policy_points[game_idx] = p_points
        opponent_points[game_idx] = o_points
        winners[game_idx] = winner
        step_counts[game_idx] = steps
        avg_entropies[game_idx] = avg_entropy

    return (
        policy_points,
        opponent_points,
        winners,
        step_counts,
        avg_entropies,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    )


def _prepare_policy_belief_arrays(
    w1: np.ndarray,
    belief_w1: np.ndarray | None,
    belief_b1: np.ndarray | None,
    belief_w2: np.ndarray | None,
    belief_b2: np.ndarray | None,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizza i pesi della belief per i kernel (iterazione 1b).

    Se la belief è data, la policy DEVE avere input 409 (= encoder v4 369 + 40 probabilità);
    senza belief passiamo dummy minimi (i kernel non li leggono con belief_enabled=False).
    """
    given = [belief_w1, belief_b1, belief_w2, belief_b2]
    if all(v is None for v in given):
        dummy1 = np.zeros((1, 1), dtype=np.float32)
        dummy_v = np.zeros(1, dtype=np.float32)
        dummy2 = np.zeros((1, ACTION_DIM), dtype=np.float32)
        dummy_b = np.zeros(ACTION_DIM, dtype=np.float32)
        return False, dummy1, dummy_v, dummy2, dummy_b
    if any(v is None for v in given):
        raise ValueError("Belief policy incompleta: servono belief_w1/b1/w2/b2.")
    bw1 = np.ascontiguousarray(np.asarray(belief_w1, dtype=np.float32))
    bb1 = np.ascontiguousarray(np.asarray(belief_b1, dtype=np.float32))
    bw2 = np.ascontiguousarray(np.asarray(belief_w2, dtype=np.float32))
    bb2 = np.ascontiguousarray(np.asarray(belief_b2, dtype=np.float32))
    if int(w1.shape[0]) != int(bw1.shape[0]) + ACTION_DIM:
        raise ValueError(
            "Policy con belief input: attesa feature_dim = encoder_belief + 40 "
            f"(policy={int(w1.shape[0])}, belief={int(bw1.shape[0])})"
        )
    return True, bw1, bb1, bw2, bb2


def collect_a2c_trajectory_numba_value_lookahead_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_mode: int,
    opponent_code: int,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    value_max_unknown_cards: int,
    game_seed: int,
    policy_seat: int,
    policy_belief_w1: np.ndarray | None = None,
    policy_belief_b1: np.ndarray | None = None,
    policy_belief_w2: np.ndarray | None = None,
    policy_belief_b2: np.ndarray | None = None,
    opponent_belief_w1: np.ndarray | None = None,
    opponent_belief_b1: np.ndarray | None = None,
    opponent_belief_w2: np.ndarray | None = None,
    opponent_belief_b2: np.ndarray | None = None,
    opponent_pimc_determinizations: int = 32,
    opponent_belief_uniform_mix: float = 0.10,
    overkill_penalty_beta: float = 0.0,
    overkill_low_lead_points_max: int = 2,
    overkill_penalty_mode: str = "flat",
) -> NumbaA2CTrajectory:
    """Raccoglie una traiettoria A2C contro opponent rule/model/value-lookahead determinized."""
    if policy_seat not in (0, 1):
        raise ValueError(f"policy_seat fuori range: {policy_seat}")
    if int(value_max_unknown_cards) < 0:
        raise ValueError("value_max_unknown_cards deve essere >= 0")
    (
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
    ) = _prepare_policy_belief_arrays(w1, policy_belief_w1, policy_belief_b1, policy_belief_w2, policy_belief_b2)
    # Belief del MAESTRO PIMC (opponent mode 3): array reali se dati, dummy altrimenti
    # (il kernel non li legge negli altri mode). Nessun vincolo sulla dim della policy.
    if opponent_belief_w1 is None:
        opp_belief_w1_arr = np.zeros((1, 1), dtype=np.float32)
        opp_belief_b1_arr = np.zeros(1, dtype=np.float32)
        opp_belief_w2_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)
        opp_belief_b2_arr = np.zeros(ACTION_DIM, dtype=np.float32)
    else:
        opp_belief_w1_arr = np.ascontiguousarray(np.asarray(opponent_belief_w1, dtype=np.float32))
        opp_belief_b1_arr = np.ascontiguousarray(np.asarray(opponent_belief_b1, dtype=np.float32))
        opp_belief_w2_arr = np.ascontiguousarray(np.asarray(opponent_belief_w2, dtype=np.float32))
        opp_belief_b2_arr = np.ascontiguousarray(np.asarray(opponent_belief_b2, dtype=np.float32))
    if float(overkill_penalty_beta) < 0.0:
        raise ValueError("overkill_penalty_beta deve essere >= 0")
    if int(overkill_low_lead_points_max) < 0:
        raise ValueError("overkill_low_lead_points_max deve essere >= 0")
    if int(opponent_mode) not in {OPPONENT_MODE_RULE, OPPONENT_MODE_MODEL, OPPONENT_MODE_VALUE_LOOKAHEAD}:
        raise ValueError(f"opponent_mode non supportato: {opponent_mode}")
    mode_code = _overkill_penalty_mode_code(overkill_penalty_mode)
    (
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        wv_arr,
        opponent_w1_arr,
        opponent_b1_arr,
        opponent_w2_arr,
        opponent_b2_arr,
        value_w1_arr,
        value_b1_arr,
        value_w2_arr,
    ) = _validate_a2c_value_lookahead_inputs(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
        value_w1=value_w1,
        value_b1=value_b1,
        value_w2=value_w2,
    )

    (
        policy_points,
        opponent_points,
        winner,
        step_count,
        avg_entropy,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    ) = _collect_mlp_policy_value_lookahead_game_numba(
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        wv_arr,
        float(bv),
        int(opponent_mode),
        int(opponent_code),
        opponent_w1_arr,
        opponent_b1_arr,
        opponent_w2_arr,
        opponent_b2_arr,
        bool(opponent_overkill_guard),
        value_w1_arr,
        value_b1_arr,
        value_w2_arr,
        float(value_b2),
        float(value_target_scale),
        bool(value_target_is_residual),
        int(value_max_unknown_cards),
        opp_belief_w1_arr,
        opp_belief_b1_arr,
        opp_belief_w2_arr,
        opp_belief_b2_arr,
        int(opponent_pimc_determinizations),
        float(opponent_belief_uniform_mix),
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
        float(overkill_penalty_beta),
        int(overkill_low_lead_points_max),
        int(mode_code),
        int(game_seed),
        int(policy_seat),
    )
    count = int(step_count)
    return NumbaA2CTrajectory(
        policy_points=int(policy_points),
        opponent_points=int(opponent_points),
        winner=int(winner),
        avg_entropy=float(avg_entropy),
        xs=xs[:count].copy(),
        z1s=z1s[:count].copy(),
        hs=hs[:count].copy(),
        action_masks=masks[:count].copy(),
        probs=probs[:count].copy(),
        action_ids=action_ids[:count].copy(),
        value_preds=value_preds[:count].copy(),
        rewards=rewards[:count].copy(),
    )


def collect_a2c_batch_numba_value_lookahead_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_modes: np.ndarray,
    opponent_codes: np.ndarray,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    value_max_unknown_cards: int,
    game_seeds: np.ndarray,
    policy_seats: np.ndarray,
    policy_belief_w1: np.ndarray | None = None,
    policy_belief_b1: np.ndarray | None = None,
    policy_belief_w2: np.ndarray | None = None,
    policy_belief_b2: np.ndarray | None = None,
    opponent_belief_w1: np.ndarray | None = None,
    opponent_belief_b1: np.ndarray | None = None,
    opponent_belief_w2: np.ndarray | None = None,
    opponent_belief_b2: np.ndarray | None = None,
    opponent_pimc_determinizations: int = 32,
    opponent_belief_uniform_mix: float = 0.10,
    overkill_penalty_beta: float = 0.0,
    overkill_low_lead_points_max: int = 2,
    overkill_penalty_mode: str = "flat",
) -> NumbaA2CBatch:
    """Raccoglie un batch A2C contro opponent rule/model/value-lookahead determinized."""
    seeds_arr = np.asarray(game_seeds, dtype=np.int64)
    seats_arr = np.asarray(policy_seats, dtype=np.int64)
    modes_arr = np.asarray(opponent_modes, dtype=np.int64)
    codes_arr = np.asarray(opponent_codes, dtype=np.int64)
    if seeds_arr.ndim != 1:
        raise ValueError(f"game_seeds deve essere 1D, ottenuto shape={seeds_arr.shape}")
    if seats_arr.shape != seeds_arr.shape:
        raise ValueError(f"Shape mismatch: policy_seats={seats_arr.shape} game_seeds={seeds_arr.shape}")
    if modes_arr.shape != seeds_arr.shape:
        raise ValueError(f"Shape mismatch: opponent_modes={modes_arr.shape} game_seeds={seeds_arr.shape}")
    if codes_arr.shape != seeds_arr.shape:
        raise ValueError(f"Shape mismatch: opponent_codes={codes_arr.shape} game_seeds={seeds_arr.shape}")
    if not np.all((seats_arr == 0) | (seats_arr == 1)):
        raise ValueError("policy_seats deve contenere solo 0/1")
    if not np.all(
        (modes_arr == OPPONENT_MODE_RULE)
        | (modes_arr == OPPONENT_MODE_MODEL)
        | (modes_arr == OPPONENT_MODE_VALUE_LOOKAHEAD)
        | (modes_arr == OPPONENT_MODE_PIMC_BELIEF)
    ):
        raise ValueError("opponent_modes contiene modalità non supportate")
    if int(value_max_unknown_cards) < 0:
        raise ValueError("value_max_unknown_cards deve essere >= 0")
    (
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
    ) = _prepare_policy_belief_arrays(w1, policy_belief_w1, policy_belief_b1, policy_belief_w2, policy_belief_b2)
    # Belief del MAESTRO PIMC (opponent mode 3): array reali se dati, dummy altrimenti
    # (il kernel non li legge negli altri mode). Nessun vincolo sulla dim della policy.
    if opponent_belief_w1 is None:
        opp_belief_w1_arr = np.zeros((1, 1), dtype=np.float32)
        opp_belief_b1_arr = np.zeros(1, dtype=np.float32)
        opp_belief_w2_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)
        opp_belief_b2_arr = np.zeros(ACTION_DIM, dtype=np.float32)
    else:
        opp_belief_w1_arr = np.ascontiguousarray(np.asarray(opponent_belief_w1, dtype=np.float32))
        opp_belief_b1_arr = np.ascontiguousarray(np.asarray(opponent_belief_b1, dtype=np.float32))
        opp_belief_w2_arr = np.ascontiguousarray(np.asarray(opponent_belief_w2, dtype=np.float32))
        opp_belief_b2_arr = np.ascontiguousarray(np.asarray(opponent_belief_b2, dtype=np.float32))
    if float(overkill_penalty_beta) < 0.0:
        raise ValueError("overkill_penalty_beta deve essere >= 0")
    if int(overkill_low_lead_points_max) < 0:
        raise ValueError("overkill_low_lead_points_max deve essere >= 0")
    mode_code = _overkill_penalty_mode_code(overkill_penalty_mode)
    (
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        wv_arr,
        opponent_w1_arr,
        opponent_b1_arr,
        opponent_w2_arr,
        opponent_b2_arr,
        value_w1_arr,
        value_b1_arr,
        value_w2_arr,
    ) = _validate_a2c_value_lookahead_inputs(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
        value_w1=value_w1,
        value_b1=value_b1,
        value_w2=value_w2,
    )

    (
        policy_points,
        opponent_points,
        winners,
        step_counts,
        avg_entropies,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    ) = _collect_mlp_policy_value_lookahead_batch_numba(
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        wv_arr,
        float(bv),
        np.ascontiguousarray(modes_arr),
        np.ascontiguousarray(codes_arr),
        opponent_w1_arr,
        opponent_b1_arr,
        opponent_w2_arr,
        opponent_b2_arr,
        bool(opponent_overkill_guard),
        value_w1_arr,
        value_b1_arr,
        value_w2_arr,
        float(value_b2),
        float(value_target_scale),
        bool(value_target_is_residual),
        int(value_max_unknown_cards),
        opp_belief_w1_arr,
        opp_belief_b1_arr,
        opp_belief_w2_arr,
        opp_belief_b2_arr,
        int(opponent_pimc_determinizations),
        float(opponent_belief_uniform_mix),
        belief_enabled,
        belief_w1_arr,
        belief_b1_arr,
        belief_w2_arr,
        belief_b2_arr,
        float(overkill_penalty_beta),
        int(overkill_low_lead_points_max),
        int(mode_code),
        np.ascontiguousarray(seeds_arr),
        np.ascontiguousarray(seats_arr),
    )
    return NumbaA2CBatch(
        policy_points=policy_points,
        opponent_points=opponent_points,
        winners=winners,
        step_counts=step_counts,
        avg_entropies=avg_entropies,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=masks,
        probs=probs,
        action_ids=action_ids,
        value_preds=value_preds,
        rewards=rewards,
    )
