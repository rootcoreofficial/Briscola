"""
Encoder Numba per osservazioni del fast path 2-player.

Questo modulo prepara l'integrazione del rollout A2C su stato numerico/JIT:
mantiene lo stesso layout di `encode_fast_observation_2p`, ma lo costruisce da array
compatti (`hands`, `hand_sizes`, `points`, `table_cards`, ...). Il wrapper Python
serve per i test di equivalenza; il target finale è chiamare la funzione JIT da un
rollout Numba senza riconvertire da liste Python a ogni step.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    FEATURE_DIM_2P_V3,
    FEATURE_DIM_2P_V4,
    EncodedObservation,
    EncoderVersion,
)
from ..fast.state_2p import Fast2PState
from .core import (
    ACTION_DIM,
    CARD_POINTS_NUMBA,
    CARD_STRENGTH_NUMBA,
    CARD_SUIT_NUMBA,
    _choose_policy_card_index_numba,
    _shuffle_deck_numba,
    _who_wins_trick_numba,
)


@njit(cache=True)
def encode_fast_observation_arrays_numba(
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
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encoda una osservazione fast partendo da array numerici.

    Ritorna `(features, action_mask)`:
    - `features`: float32 con layout v1/v2/v3 canonico;
    - `action_mask`: bool[40], True per le carte presenti nella mano del player.

    `out_of_play_cards` è usato solo da v3 (carte fuori gioco = prese + tavolo); per v1/v2 è ignorato.
    Il blocco v3 replica `_compute_v3_extra_features` del path domain (parità verificata dai test).
    """
    features = np.zeros(feature_dim, dtype=np.float32)
    action_mask = np.zeros(ACTION_DIM, dtype=np.bool_)

    for i in range(hand_sizes[player_index]):
        card_id = hands[player_index, i]
        action_mask[card_id] = True
        features[card_id] = 1.0
        features[ACTION_DIM + card_id] = float(CARD_POINTS_NUMBA[card_id])
        features[2 * ACTION_DIM + card_id] = float(CARD_STRENGTH_NUMBA[card_id])

    table_offset = 3 * ACTION_DIM
    for i in range(table_size):
        card_id = table_cards[i]
        features[table_offset + card_id] = 1.0
        features[table_offset + ACTION_DIM + card_id] = float(CARD_POINTS_NUMBA[card_id])
        features[table_offset + 2 * ACTION_DIM + card_id] = float(CARD_STRENGTH_NUMBA[card_id])

    scalar_offset = 6 * ACTION_DIM
    trump_suit = CARD_SUIT_NUMBA[trump_card]
    features[scalar_offset + trump_suit] = 1.0

    opp_index = 1 - player_index
    is_second_in_trick = 1.0 if current_turn == player_index and table_size == 1 else 0.0
    features[scalar_offset + 4] = float(deck_size) / 40.0
    features[scalar_offset + 5] = float(points[player_index]) / 120.0
    features[scalar_offset + 6] = float(points[opp_index]) / 120.0
    features[scalar_offset + 7] = is_second_in_trick

    if (
        feature_dim == int(FEATURE_DIM_2P_V2)
        or feature_dim == int(FEATURE_DIM_2P_V3)
        or feature_dim == int(FEATURE_DIM_2P_V4)
    ):
        seen_offset = int(FEATURE_DIM_2P_V1)
        for card_id in range(ACTION_DIM):
            features[seen_offset + card_id] = float(seen_cards[card_id])

    if feature_dim == int(FEATURE_DIM_2P_V3) or feature_dim == int(FEATURE_DIM_2P_V4):
        # Blocco v3 (22 feature) — stesso layout/normalizzazioni di `_compute_v3_extra_features`.
        # "ignota" = non vista e non in mano (action_mask True = carta in mano).
        v3 = int(FEATURE_DIM_2P_V2)
        base = trump_suit * 10
        unknown_trumps = 0
        for off in range(10):
            cid = base + off
            if seen_cards[cid] == 0 and not action_mask[cid]:
                unknown_trumps += 1
        features[v3 + 0] = float(unknown_trumps) / 10.0
        ace_t = base + 0
        three_t = base + 2
        king_t = base + 9
        features[v3 + 1] = 1.0 if (seen_cards[ace_t] == 0 and not action_mask[ace_t]) else 0.0
        features[v3 + 2] = 1.0 if (seen_cards[three_t] == 0 and not action_mask[three_t]) else 0.0
        features[v3 + 3] = 1.0 if (seen_cards[king_t] == 0 and not action_mask[king_t]) else 0.0

        for suit_index in range(4):
            ace_id = suit_index * 10 + 0
            three_id = suit_index * 10 + 2
            features[v3 + 4 + suit_index * 3 + 0] = 1.0 if out_of_play_cards[ace_id] == 1 else 0.0
            features[v3 + 4 + suit_index * 3 + 1] = 1.0 if out_of_play_cards[three_id] == 1 else 0.0
            load_unknown = 0
            if seen_cards[ace_id] == 0 and not action_mask[ace_id]:
                load_unknown += 1
            if seen_cards[three_id] == 0 and not action_mask[three_id]:
                load_unknown += 1
            features[v3 + 4 + suit_index * 3 + 2] = float(load_unknown) / 2.0

        features[v3 + 16] = float(deck_size) / 40.0
        features[v3 + 17] = float(hand_sizes[player_index]) / 3.0
        features[v3 + 18] = 1.0 if deck_size == 0 else 0.0

        if table_size > 0:
            lead_card = table_cards[0]
            trick_points = 0.0
            for i in range(table_size):
                trick_points += float(CARD_POINTS_NUMBA[table_cards[i]])
            features[v3 + 19] = trick_points / 11.0
            features[v3 + 20] = float(CARD_STRENGTH_NUMBA[lead_card]) / 10.0
            features[v3 + 21] = 1.0 if CARD_SUIT_NUMBA[lead_card] == trump_suit else 0.0

    return features, action_mask


@njit(cache=True)
def _v4_block_numba(
    features: np.ndarray,
    trick_hist: np.ndarray,
    num_tricks: int,
    player_index: int,
    trump_suit: int,
) -> None:
    """
    Scrive il blocco v4 (59 feature) in `features` a partire da offset FEATURE_DIM_2P_V3.

    `trick_hist` è la storia numerica delle prese, una riga per presa completata:
    `[lead_card, lead_player, resp_card, winner, points]`. Replica ESATTAMENTE
    `_compute_v4_extra_features` del path domain (parità protetta dai test):
    blocco A = 11 aggregati sul comportamento avversario; blocco B = ultime 4 prese x 12.
    """
    base = int(FEATURE_DIM_2P_V3)
    opp_index = 1 - player_index

    opp_suit0 = 0
    opp_suit1 = 0
    opp_suit2 = 0
    opp_suit3 = 0
    opp_lead = 0
    opp_trump_resp = 0
    opp_follow = 0
    opp_discard = 0
    opp_lead_trump = 0
    opp_lead_load = 0

    for t_idx in range(num_tricks):
        lead_card = trick_hist[t_idx, 0]
        lead_player = trick_hist[t_idx, 1]
        resp_card = trick_hist[t_idx, 2]
        lead_suit = lead_card // 10
        resp_suit = resp_card // 10

        # Conteggio per seme delle carte giocate dall'avversario (lead o risposta).
        opp_card_suit = lead_suit if lead_player == opp_index else resp_suit
        if opp_card_suit == 0:
            opp_suit0 += 1
        elif opp_card_suit == 1:
            opp_suit1 += 1
        elif opp_card_suit == 2:
            opp_suit2 += 1
        else:
            opp_suit3 += 1

        if lead_player == opp_index:
            opp_lead += 1
            if lead_suit == trump_suit:
                opp_lead_trump += 1
            if CARD_POINTS_NUMBA[lead_card] >= 10:  # asso (11) o tre (10)
                opp_lead_load += 1
        else:
            lead_is_trump = lead_suit == trump_suit
            resp_is_trump = resp_suit == trump_suit
            if resp_suit == lead_suit:
                opp_follow += 1
            elif resp_is_trump and not lead_is_trump:
                opp_trump_resp += 1
            else:
                opp_discard += 1

    features[base + 0] = opp_suit0 / 10.0
    features[base + 1] = opp_suit1 / 10.0
    features[base + 2] = opp_suit2 / 10.0
    features[base + 3] = opp_suit3 / 10.0
    features[base + 4] = opp_lead / 10.0
    features[base + 5] = opp_trump_resp / 10.0
    features[base + 6] = opp_follow / 10.0
    features[base + 7] = opp_discard / 10.0
    features[base + 8] = opp_lead_trump / 10.0
    features[base + 9] = opp_lead_load / 10.0
    features[base + 10] = num_tricks / 20.0

    for slot in range(4):
        t_idx = num_tricks - 1 - slot
        offset = base + 11 + slot * 12
        if t_idx < 0:
            continue  # gli slot mancanti restano a zero (padding)
        lead_card = trick_hist[t_idx, 0]
        lead_player = trick_hist[t_idx, 1]
        resp_card = trick_hist[t_idx, 2]
        winner = trick_hist[t_idx, 3]
        trick_points = trick_hist[t_idx, 4]
        lead_suit = lead_card // 10
        resp_suit = resp_card // 10

        features[offset + 0] = 1.0
        features[offset + 1] = 1.0 if lead_player == player_index else 0.0
        features[offset + 2 + lead_suit] = 1.0
        features[offset + 6] = float(CARD_STRENGTH_NUMBA[lead_card]) / 10.0
        features[offset + 7] = float(CARD_POINTS_NUMBA[lead_card]) / 11.0
        features[offset + 8] = 1.0 if resp_suit == lead_suit else 0.0
        features[offset + 9] = 1.0 if resp_suit == trump_suit else 0.0
        features[offset + 10] = 1.0 if winner == player_index else 0.0
        features[offset + 11] = float(trick_points) / 21.0


@njit(cache=True)
def encode_fast_observation_arrays_v4_numba(
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encoder v4: base (v1+v2+v3) + blocco storia prese (59 feature).

    Funzione SEPARATA dal base per non toccare i chiamanti che restano <= v3
    (value_lookahead/value_dataset lavorano su stati determinizzati senza storia).
    """
    features, action_mask = encode_fast_observation_arrays_numba(
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
        int(FEATURE_DIM_2P_V4),
    )
    _v4_block_numba(features, trick_hist, num_tricks, player_index, CARD_SUIT_NUMBA[trump_card])
    return features, action_mask


@njit(cache=True)
def _policy_input_v4_belief_numba(
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
    belief_w1: np.ndarray,
    belief_b1: np.ndarray,
    belief_w2: np.ndarray,
    belief_b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Input policy iterazione-1b: encoder v4 (369) + 40 probabilità della belief CONGELATA.

    La belief legge le stesse feature v4 e produce P(carta in mano avversaria) via sigmoid:
    l'inferenza comportamentale diventa parte dell'osservazione della policy. Anti-cheat
    invariato: la belief è a sua volta funzione della sola osservazione lecita.
    """
    v4_features, action_mask = encode_fast_observation_arrays_v4_numba(
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
    encoder_dim = int(FEATURE_DIM_2P_V4)
    belief_hidden = belief_w1.shape[1]

    features = np.zeros(encoder_dim + ACTION_DIM, dtype=np.float32)
    for f_idx in range(encoder_dim):
        features[f_idx] = v4_features[f_idx]

    hidden = np.empty(belief_hidden, dtype=np.float32)
    for h_idx in range(belief_hidden):
        value = belief_b1[h_idx]
        for f_idx in range(encoder_dim):
            value += v4_features[f_idx] * belief_w1[f_idx, h_idx]
        hidden[h_idx] = value if value > 0.0 else 0.0

    for card_id in range(ACTION_DIM):
        logit = belief_b2[card_id]
        for h_idx in range(belief_hidden):
            logit += hidden[h_idx] * belief_w2[h_idx, card_id]
        features[encoder_dim + card_id] = 1.0 / (1.0 + np.exp(-logit))

    return features, action_mask


@njit(cache=True)
def _action_id_to_hand_index_numba(hands: np.ndarray, hand_sizes: np.ndarray, player_index: int, action_id: int) -> int:
    """Converte action_id/card_id in indice nella mano numerica."""
    for i in range(hand_sizes[player_index]):
        if hands[player_index, i] == action_id:
            return i
    raise ValueError("action_id non presente nella mano")


@njit(cache=True)
def _trump_overkill_penalty_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
    player_index: int,
    chosen_card_index: int,
    beta: float,
    low_lead_points_max: int,
    mode_code: int,
) -> float:
    """Penalità anti-overkill equivalente al reward shaping canonico, ma su card id."""
    if beta <= 0.0:
        return 0.0
    if table_size != 1:
        return 0.0
    if chosen_card_index < 0 or chosen_card_index >= hand_sizes[player_index]:
        return 0.0

    lead_card = table_cards[0]
    lead_player = table_players[0]
    if low_lead_points_max >= 0 and CARD_POINTS_NUMBA[lead_card] > low_lead_points_max:
        return 0.0

    trump_suit = CARD_SUIT_NUMBA[trump_card]
    chosen = hands[player_index, chosen_card_index]
    if CARD_SUIT_NUMBA[chosen] != trump_suit:
        return 0.0
    if _who_wins_trick_numba(lead_card, lead_player, chosen, player_index, trump_card) != player_index:
        return 0.0

    min_points = 999
    min_strength = 999
    winning_trump_exists = False
    for hand_idx in range(hand_sizes[player_index]):
        card = hands[player_index, hand_idx]
        if CARD_SUIT_NUMBA[card] != trump_suit:
            continue
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if not winning_trump_exists or points < min_points or (points == min_points and strength < min_strength):
            winning_trump_exists = True
            min_points = points
            min_strength = strength

    if not winning_trump_exists:
        return 0.0

    chosen_points = CARD_POINTS_NUMBA[chosen]
    chosen_strength = CARD_STRENGTH_NUMBA[chosen]
    is_overkill = chosen_points > min_points or (chosen_points == min_points and chosen_strength > min_strength)
    if not is_overkill:
        return 0.0
    if mode_code == 0:
        return -float(beta)

    points_gap = chosen_points - min_points
    if points_gap < 0:
        points_gap = 0
    strength_gap = chosen_strength - min_strength
    if strength_gap < 0:
        strength_gap = 0
    gap = float(points_gap) / 11.0 + float(strength_gap) / 10.0
    return -float(beta) * gap


@njit(cache=True)
def _sample_mlp_policy_action_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_size: int,
    deck_size: int,
    current_turn: int,
    trump_card: int,
    policy_seat: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    trick_hist: np.ndarray,
    num_tricks: int,
) -> int:
    """Esegue encoder + forward MLP + sampling mascherato dentro Numba."""
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]
    if feature_dim == int(FEATURE_DIM_2P_V4):
        features, action_mask = encode_fast_observation_arrays_v4_numba(
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
            num_tricks,
        )
    else:
        features, action_mask = encode_fast_observation_arrays_numba(
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
            feature_dim,
        )

    hidden = np.empty(hidden_dim, dtype=np.float32)
    for h_idx in range(hidden_dim):
        value = b1[h_idx]
        for f_idx in range(feature_dim):
            value += features[f_idx] * w1[f_idx, h_idx]
        hidden[h_idx] = value if value > 0.0 else 0.0

    logits = np.empty(ACTION_DIM, dtype=np.float32)
    max_logit = -1.0e30
    last_valid = 0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            logits[action_id] = -1.0e30
            continue
        value = b2[action_id]
        for h_idx in range(hidden_dim):
            value += hidden[h_idx] * w2[h_idx, action_id]
        logits[action_id] = value
        last_valid = action_id
        if value > max_logit:
            max_logit = value

    total = 0.0
    for action_id in range(ACTION_DIM):
        if action_mask[action_id]:
            total += np.exp(float(logits[action_id] - max_logit))

    threshold = np.random.random() * total
    cumulative = 0.0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            continue
        cumulative += np.exp(float(logits[action_id] - max_logit))
        if cumulative >= threshold:
            return action_id
    return last_valid


@njit(cache=True)
def _record_mlp_policy_decision_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_size: int,
    deck_size: int,
    current_turn: int,
    trump_card: int,
    policy_seat: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    trick_hist: np.ndarray,
    num_tricks: int,
    belief_enabled: bool,
    belief_w1: np.ndarray,
    belief_b1: np.ndarray,
    belief_w2: np.ndarray,
    belief_b2: np.ndarray,
    xs: np.ndarray,
    z1s: np.ndarray,
    hs: np.ndarray,
    masks: np.ndarray,
    probs_out: np.ndarray,
    step_index: int,
) -> tuple[int, float, float]:
    """
    Registra uno step A2C completo e ritorna `(action_id, value_pred, entropy)`.

    Le matrici di output sono mutate in-place alla riga `step_index`.
    """
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]
    if belief_enabled:
        features, action_mask = _policy_input_v4_belief_numba(
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
            num_tricks,
            belief_w1,
            belief_b1,
            belief_w2,
            belief_b2,
        )
    elif feature_dim == int(FEATURE_DIM_2P_V4):
        features, action_mask = encode_fast_observation_arrays_v4_numba(
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
            num_tricks,
        )
    else:
        features, action_mask = encode_fast_observation_arrays_numba(
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
            feature_dim,
        )

    for f_idx in range(feature_dim):
        xs[step_index, f_idx] = features[f_idx]
    for action_id in range(ACTION_DIM):
        masks[step_index, action_id] = action_mask[action_id]

    hidden = np.empty(hidden_dim, dtype=np.float32)
    for h_idx in range(hidden_dim):
        z_value = b1[h_idx]
        for f_idx in range(feature_dim):
            z_value += features[f_idx] * w1[f_idx, h_idx]
        z1s[step_index, h_idx] = z_value
        h_value = z_value if z_value > 0.0 else 0.0
        hidden[h_idx] = h_value
        hs[step_index, h_idx] = h_value

    value_pred = float(bv)
    for h_idx in range(hidden_dim):
        value_pred += float(hidden[h_idx] * wv[h_idx])

    logits = np.empty(ACTION_DIM, dtype=np.float32)
    max_logit = -1.0e30
    last_valid = 0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            logits[action_id] = -1.0e30
            continue
        logit = b2[action_id]
        for h_idx in range(hidden_dim):
            logit += hidden[h_idx] * w2[h_idx, action_id]
        logits[action_id] = logit
        last_valid = action_id
        if logit > max_logit:
            max_logit = logit

    total = 0.0
    for action_id in range(ACTION_DIM):
        if action_mask[action_id]:
            total += np.exp(float(logits[action_id] - max_logit))

    threshold = np.random.random() * total
    cumulative = 0.0
    selected_action = last_valid
    entropy = 0.0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            probs_out[step_index, action_id] = 0.0
            continue
        prob = float(np.exp(float(logits[action_id] - max_logit)) / total)
        probs_out[step_index, action_id] = prob
        entropy -= prob * np.log(prob + 1.0e-12)
        cumulative += np.exp(float(logits[action_id] - max_logit))
        if selected_action == last_valid and cumulative >= threshold:
            selected_action = action_id

    return int(selected_action), float(value_pred), float(entropy)


@njit(cache=True)
def _argmax_mlp_policy_action_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
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
) -> int:
    """Forward MLP deterministico: ritorna l'action_id valido con logit massimo."""
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]
    if feature_dim == int(FEATURE_DIM_2P_V4):
        features, action_mask = encode_fast_observation_arrays_v4_numba(
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
        features, action_mask = encode_fast_observation_arrays_numba(
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

    hidden = np.empty(hidden_dim, dtype=np.float32)
    for h_idx in range(hidden_dim):
        value = b1[h_idx]
        for f_idx in range(feature_dim):
            value += features[f_idx] * w1[f_idx, h_idx]
        hidden[h_idx] = value if value > 0.0 else 0.0

    best_action = 0
    best_logit = -1.0e30
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            continue
        logit = b2[action_id]
        for h_idx in range(hidden_dim):
            logit += hidden[h_idx] * w2[h_idx, action_id]
        if logit > best_logit:
            best_logit = logit
            best_action = action_id

    return int(best_action)


@njit(cache=True)
def _apply_overkill_guard_numba(
    action_id: int,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
    enabled: bool,
) -> int:
    """Post-processing anti-overkill numerico equivalente al guard del `BCModelAgent`."""
    if not enabled:
        return action_id
    if table_size != 1:
        return action_id

    trump_suit = CARD_SUIT_NUMBA[trump_card]
    if CARD_SUIT_NUMBA[action_id] != trump_suit:
        return action_id

    lead_card = table_cards[0]
    lead_player = table_players[0]
    if _who_wins_trick_numba(lead_card, lead_player, action_id, player_index, trump_card) != player_index:
        return action_id

    best_action = action_id
    best_points = CARD_POINTS_NUMBA[action_id]
    best_strength = CARD_STRENGTH_NUMBA[action_id]
    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        if CARD_SUIT_NUMBA[card] != trump_suit:
            continue
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if points < best_points or (points == best_points and strength < best_strength):
            best_action = card
            best_points = points
            best_strength = strength

    return int(best_action)


@njit(cache=True)
def _trump_waste_metric_numba(
    action_id: int,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
) -> int:
    """
    Metric `trump_waste` numerica.

    Ritorna:
    - `-1` se la metrica non è applicabile (nessuna risposta vincente);
    - `0` se applicabile ma non waste;
    - `1` se la policy usa briscola pur avendo una risposta vincente non-briscola.
    """
    if table_size != 1:
        return -1

    lead_card = table_cards[0]
    lead_player = table_players[0]
    trump_suit = CARD_SUIT_NUMBA[trump_card]
    winning_any_exists = False
    winning_non_trump_exists = False
    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        winning_any_exists = True
        if CARD_SUIT_NUMBA[card] != trump_suit:
            winning_non_trump_exists = True

    if not winning_any_exists:
        return -1

    chosen_is_trump = CARD_SUIT_NUMBA[action_id] == trump_suit
    if chosen_is_trump and winning_non_trump_exists:
        return 1
    return 0


@njit(cache=True)
def _trump_overkill_metric_numba(
    action_id: int,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
) -> int:
    """
    Metric `trump_overkill` numerica.

    Ritorna:
    - `-1` se la scelta non è una briscola vincente;
    - `0` se è la briscola vincente minima;
    - `1` se esiste una briscola vincente più economica.
    """
    if table_size != 1:
        return -1

    trump_suit = CARD_SUIT_NUMBA[trump_card]
    if CARD_SUIT_NUMBA[action_id] != trump_suit:
        return -1

    lead_card = table_cards[0]
    lead_player = table_players[0]
    if _who_wins_trick_numba(lead_card, lead_player, action_id, player_index, trump_card) != player_index:
        return -1

    best_points = CARD_POINTS_NUMBA[action_id]
    best_strength = CARD_STRENGTH_NUMBA[action_id]
    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        if CARD_SUIT_NUMBA[card] != trump_suit:
            continue
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if points < best_points or (points == best_points and strength < best_strength):
            return 1

    return 0


@njit(cache=True)
def _choose_opponent_card_index_numba(
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    deck_size: int,
    current_turn: int,
    trump_card: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    trick_hist: np.ndarray,
    num_tricks: int,
) -> int:
    """Sceglie l'indice carta per opponent rule-based o MLP `.npz`."""
    if opponent_model_enabled:
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
def _play_mlp_policy_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    seed: int,
    policy_seat: int,
    policy_argmax: bool,
    policy_overkill_guard: bool,
) -> tuple[int, int, int]:
    """
    Gioca una partita full-JIT: MLP policy vs opponent fast-compatible.

    Ritorna `(policy_points, opponent_points, winner)` con `winner=0` per policy,
    `winner=1` per opponent e `winner=-1` in caso di pareggio.
    """
    shuffled = _shuffle_deck_numba(seed)

    deck = np.empty(34, dtype=np.int64)
    hands = np.empty((2, 3), dtype=np.int64)
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
    table_cards = np.empty(2, dtype=np.int64)
    table_players = np.empty(2, dtype=np.int64)
    table_size = 0
    current_turn = 0
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[trump_card] = 1
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    # Storia delle prese per l'encoder v4: [lead, lead_player, resp, winner, points] per riga.
    trick_hist = np.zeros((20, 5), dtype=np.int64)
    trick_count = np.zeros(1, dtype=np.int64)

    safety = 5000
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        if current_turn == policy_seat:
            if policy_argmax:
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
                    policy_seat,
                    seen_cards,
                    out_of_play_cards,
                    trick_hist,
                    trick_count[0],
                )
            else:
                action_id = _sample_mlp_policy_action_numba(
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
                    policy_seat,
                    seen_cards,
                    out_of_play_cards,
                    trick_hist,
                    trick_count[0],
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
                policy_overkill_guard,
            )
            card_index = _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)
        else:
            card_index = _choose_opponent_card_index_numba(
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                deck_size,
                current_turn,
                trump_card,
                seen_cards,
                out_of_play_cards,
                trick_hist,
                trick_count[0],
            )

        played_card = hands[current_turn, card_index]
        hand_size = hand_sizes[current_turn]
        for i in range(card_index, hand_size - 1):
            hands[current_turn, i] = hands[current_turn, i + 1]
        hand_sizes[current_turn] -= 1

        table_cards[table_size] = played_card
        table_players[table_size] = current_turn
        table_size += 1
        seen_cards[played_card] = 1
        out_of_play_cards[played_card] = 1

        if table_size == 1:
            current_turn = 1 - current_turn
            continue

        first_card = table_cards[0]
        second_card = table_cards[1]
        first_player = table_players[0]
        second_player = table_players[1]
        winner = _who_wins_trick_numba(first_card, first_player, second_card, second_player, trump_card)
        gained = CARD_POINTS_NUMBA[first_card] + CARD_POINTS_NUMBA[second_card]
        points[winner] += gained
        trick_hist[trick_count[0], 0] = first_card
        trick_hist[trick_count[0], 1] = first_player
        trick_hist[trick_count[0], 2] = second_card
        trick_hist[trick_count[0], 3] = winner
        trick_hist[trick_count[0], 4] = gained
        trick_count[0] += 1
        table_size = 0

        if deck_size > 0:
            for i in range(2):
                player_to_deal = (winner + i) % 2
                if deck_size <= 0:
                    break
                deck_size -= 1
                hands[player_to_deal, hand_sizes[player_to_deal]] = deck[deck_size]
                hand_sizes[player_to_deal] += 1

        current_turn = winner

    policy_points = points[policy_seat]
    opponent_points = points[1 - policy_seat]
    if policy_points > opponent_points:
        winner_out = 0
    elif opponent_points > policy_points:
        winner_out = 1
    else:
        winner_out = -1
    return int(policy_points), int(opponent_points), int(winner_out)


@njit(cache=True)
def _evaluate_mlp_policy_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    game_seeds: np.ndarray,
    seat_fair: bool,
    policy_argmax: bool,
    policy_overkill_guard: bool,
) -> tuple[int, int, int, int, int]:
    """Valuta una MLP policy contro un opponent JIT e aggrega i risultati."""
    wins_policy = 0
    wins_opponent = 0
    draws = 0
    sum_policy = 0
    sum_opponent = 0

    if seat_fair:
        for seed_index in range(game_seeds.shape[0]):
            game_seed = int(game_seeds[seed_index])
            for policy_seat in range(2):
                policy_points, opponent_points, winner = _play_mlp_policy_game_numba(
                    w1,
                    b1,
                    w2,
                    b2,
                    opponent_code,
                    opponent_model_enabled,
                    opponent_w1,
                    opponent_b1,
                    opponent_w2,
                    opponent_b2,
                    opponent_overkill_guard,
                    game_seed,
                    policy_seat,
                    policy_argmax,
                    policy_overkill_guard,
                )
                sum_policy += policy_points
                sum_opponent += opponent_points
                if winner == 0:
                    wins_policy += 1
                elif winner == 1:
                    wins_opponent += 1
                else:
                    draws += 1
    else:
        for seed_index in range(game_seeds.shape[0]):
            policy_points, opponent_points, winner = _play_mlp_policy_game_numba(
                w1,
                b1,
                w2,
                b2,
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                int(game_seeds[seed_index]),
                0,
                policy_argmax,
                policy_overkill_guard,
            )
            sum_policy += policy_points
            sum_opponent += opponent_points
            if winner == 0:
                wins_policy += 1
            elif winner == 1:
                wins_opponent += 1
            else:
                draws += 1

    return wins_policy, wins_opponent, draws, sum_policy, sum_opponent


@njit(cache=True, parallel=True)
def _evaluate_mlp_policy_numba_parallel_plain(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    game_seeds: np.ndarray,
    policy_argmax: bool,
    policy_overkill_guard: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Valuta una MLP policy in parallelo su partite indipendenti con seat fisso.

    Ritorna array per-partita: punti policy, punti opponent, winner (`0/1/-1`).
    L'aggregazione resta nel wrapper Python per evitare riduzioni complesse dentro `prange`.
    """
    total_games = game_seeds.shape[0]
    policy_points = np.empty(total_games, dtype=np.int64)
    opponent_points = np.empty(total_games, dtype=np.int64)
    winners = np.empty(total_games, dtype=np.int64)

    for game_index in prange(total_games):
        p_points, o_points, winner = _play_mlp_policy_game_numba(
            w1,
            b1,
            w2,
            b2,
            opponent_code,
            opponent_model_enabled,
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            opponent_overkill_guard,
            int(game_seeds[game_index]),
            0,
            policy_argmax,
            policy_overkill_guard,
        )
        policy_points[game_index] = p_points
        opponent_points[game_index] = o_points
        winners[game_index] = winner

    return policy_points, opponent_points, winners


@njit(cache=True, parallel=True)
def _evaluate_mlp_policy_numba_parallel_seat_fair(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    game_seeds: np.ndarray,
    policy_argmax: bool,
    policy_overkill_guard: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Valuta una MLP policy in parallelo con coppie seat-fair per ogni seed.

    Ogni seed genera due partite indipendenti a seat invertiti. Tenere questo kernel
    separato da quello a seat fisso evita inferenze ambigue degli indici dentro `prange`.
    """
    total_games = game_seeds.shape[0] * 2
    policy_points = np.empty(total_games, dtype=np.int64)
    opponent_points = np.empty(total_games, dtype=np.int64)
    winners = np.empty(total_games, dtype=np.int64)

    for game_index in prange(total_games):
        seed_index = game_index // 2
        policy_seat = game_index - seed_index * 2
        p_points, o_points, winner = _play_mlp_policy_game_numba(
            w1,
            b1,
            w2,
            b2,
            opponent_code,
            opponent_model_enabled,
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            opponent_overkill_guard,
            int(game_seeds[seed_index]),
            int(policy_seat),
            policy_argmax,
            policy_overkill_guard,
        )
        policy_points[game_index] = p_points
        opponent_points[game_index] = o_points
        winners[game_index] = winner

    return policy_points, opponent_points, winners


@njit(cache=True)
def _play_mlp_policy_quality_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    seed: int,
    policy_seat: int,
    policy_overkill_guard: bool,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    """
    Gioca una partita MLP-vs-baseline e raccoglie metriche qualità per la policy.

    Ritorna match result + contatori quality, tutti dal punto di vista della policy.
    """
    shuffled = _shuffle_deck_numba(seed)

    deck = np.empty(34, dtype=np.int64)
    hands = np.empty((2, 3), dtype=np.int64)
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
    table_cards = np.empty(2, dtype=np.int64)
    table_players = np.empty(2, dtype=np.int64)
    table_size = 0
    current_turn = 0
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[trump_card] = 1
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    # Storia delle prese per l'encoder v4: [lead, lead_player, resp, winner, points] per riga.
    trick_hist = np.zeros((20, 5), dtype=np.int64)
    trick_count = np.zeros(1, dtype=np.int64)

    q_second = 0
    q_second_with_win = 0
    q_waste = 0
    q_trump_wins = 0
    q_trump_overkill = 0
    q_trump_wins_low = 0
    q_trump_overkill_low = 0

    safety = 5000
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        if current_turn == policy_seat:
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
                policy_seat,
                seen_cards,
                out_of_play_cards,
                trick_hist,
                trick_count[0],
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
                policy_overkill_guard,
            )

            if table_size == 1:
                q_second += 1
                waste = _trump_waste_metric_numba(
                    action_id,
                    hands,
                    hand_sizes,
                    current_turn,
                    table_cards,
                    table_players,
                    table_size,
                    trump_card,
                )
                if waste >= 0:
                    q_second_with_win += 1
                    if waste == 1:
                        q_waste += 1

                overkill = _trump_overkill_metric_numba(
                    action_id,
                    hands,
                    hand_sizes,
                    current_turn,
                    table_cards,
                    table_players,
                    table_size,
                    trump_card,
                )
                if overkill >= 0:
                    q_trump_wins += 1
                    low_lead = CARD_POINTS_NUMBA[table_cards[0]] <= 2
                    if low_lead:
                        q_trump_wins_low += 1
                    if overkill == 1:
                        q_trump_overkill += 1
                        if low_lead:
                            q_trump_overkill_low += 1

            card_index = _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)
        else:
            card_index = _choose_opponent_card_index_numba(
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                deck_size,
                current_turn,
                trump_card,
                seen_cards,
                out_of_play_cards,
                trick_hist,
                trick_count[0],
            )

        played_card = hands[current_turn, card_index]
        hand_size = hand_sizes[current_turn]
        for i in range(card_index, hand_size - 1):
            hands[current_turn, i] = hands[current_turn, i + 1]
        hand_sizes[current_turn] -= 1

        table_cards[table_size] = played_card
        table_players[table_size] = current_turn
        table_size += 1
        seen_cards[played_card] = 1
        out_of_play_cards[played_card] = 1

        if table_size == 1:
            current_turn = 1 - current_turn
            continue

        first_card = table_cards[0]
        second_card = table_cards[1]
        first_player = table_players[0]
        second_player = table_players[1]
        winner = _who_wins_trick_numba(first_card, first_player, second_card, second_player, trump_card)
        gained = CARD_POINTS_NUMBA[first_card] + CARD_POINTS_NUMBA[second_card]
        points[winner] += gained
        trick_hist[trick_count[0], 0] = first_card
        trick_hist[trick_count[0], 1] = first_player
        trick_hist[trick_count[0], 2] = second_card
        trick_hist[trick_count[0], 3] = winner
        trick_hist[trick_count[0], 4] = gained
        trick_count[0] += 1
        table_size = 0

        if deck_size > 0:
            for i in range(2):
                player_to_deal = (winner + i) % 2
                if deck_size <= 0:
                    break
                deck_size -= 1
                hands[player_to_deal, hand_sizes[player_to_deal]] = deck[deck_size]
                hand_sizes[player_to_deal] += 1

        current_turn = winner

    policy_points = points[policy_seat]
    opponent_points = points[1 - policy_seat]
    if policy_points > opponent_points:
        winner_out = 0
    elif opponent_points > policy_points:
        winner_out = 1
    else:
        winner_out = -1
    return (
        int(policy_points),
        int(opponent_points),
        int(winner_out),
        q_second,
        q_second_with_win,
        q_waste,
        q_trump_wins,
        q_trump_overkill,
        q_trump_wins_low,
        q_trump_overkill_low,
    )


@njit(cache=True)
def _evaluate_mlp_policy_quality_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    game_seeds: np.ndarray,
    policy_overkill_guard: bool,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
    """Valuta seat-fair MLP-vs-baseline e aggrega match stats + quality stats."""
    wins_policy = 0
    wins_opponent = 0
    draws = 0
    sum_policy = 0
    sum_opponent = 0
    q_second = 0
    q_second_with_win = 0
    q_waste = 0
    q_trump_wins = 0
    q_trump_overkill = 0
    q_trump_wins_low = 0
    q_trump_overkill_low = 0

    for seed_index in range(game_seeds.shape[0]):
        game_seed = int(game_seeds[seed_index])
        for policy_seat in range(2):
            (
                policy_points,
                opponent_points,
                winner,
                second,
                second_with_win,
                waste,
                trump_wins,
                trump_overkill,
                trump_wins_low,
                trump_overkill_low,
            ) = _play_mlp_policy_quality_game_numba(
                w1,
                b1,
                w2,
                b2,
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                game_seed,
                policy_seat,
                policy_overkill_guard,
            )
            sum_policy += policy_points
            sum_opponent += opponent_points
            if winner == 0:
                wins_policy += 1
            elif winner == 1:
                wins_opponent += 1
            else:
                draws += 1

            q_second += second
            q_second_with_win += second_with_win
            q_waste += waste
            q_trump_wins += trump_wins
            q_trump_overkill += trump_overkill
            q_trump_wins_low += trump_wins_low
            q_trump_overkill_low += trump_overkill_low

    return (
        wins_policy,
        wins_opponent,
        draws,
        sum_policy,
        sum_opponent,
        q_second,
        q_second_with_win,
        q_waste,
        q_trump_wins,
        q_trump_overkill,
        q_trump_wins_low,
        q_trump_overkill_low,
    )


@njit(cache=True, parallel=True)
def _evaluate_mlp_policy_quality_numba_parallel(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    game_seeds: np.ndarray,
    policy_overkill_guard: bool,
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
]:
    """
    Valuta decision-quality MLP in parallelo con due partite seat-fair per seed.

    Il kernel produce array per-partita; l'aggregazione resta nel wrapper Python per
    mantenere il loop `prange` semplice e allineato al path evaluation MLP parallelo.
    """
    total_games = game_seeds.shape[0] * 2
    policy_points = np.empty(total_games, dtype=np.int64)
    opponent_points = np.empty(total_games, dtype=np.int64)
    winners = np.empty(total_games, dtype=np.int64)
    q_second = np.empty(total_games, dtype=np.int64)
    q_second_with_win = np.empty(total_games, dtype=np.int64)
    q_waste = np.empty(total_games, dtype=np.int64)
    q_trump_wins = np.empty(total_games, dtype=np.int64)
    q_trump_overkill = np.empty(total_games, dtype=np.int64)
    q_trump_wins_low = np.empty(total_games, dtype=np.int64)
    q_trump_overkill_low = np.empty(total_games, dtype=np.int64)

    for game_index in prange(total_games):
        seed_index = game_index // 2
        policy_seat = game_index - seed_index * 2
        (
            p_points,
            o_points,
            winner,
            second,
            second_with_win,
            waste,
            trump_wins,
            trump_overkill,
            trump_wins_low,
            trump_overkill_low,
        ) = _play_mlp_policy_quality_game_numba(
            w1,
            b1,
            w2,
            b2,
            opponent_code,
            opponent_model_enabled,
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            opponent_overkill_guard,
            int(game_seeds[seed_index]),
            int(policy_seat),
            policy_overkill_guard,
        )
        policy_points[game_index] = p_points
        opponent_points[game_index] = o_points
        winners[game_index] = winner
        q_second[game_index] = second
        q_second_with_win[game_index] = second_with_win
        q_waste[game_index] = waste
        q_trump_wins[game_index] = trump_wins
        q_trump_overkill[game_index] = trump_overkill
        q_trump_wins_low[game_index] = trump_wins_low
        q_trump_overkill_low[game_index] = trump_overkill_low

    return (
        policy_points,
        opponent_points,
        winners,
        q_second,
        q_second_with_win,
        q_waste,
        q_trump_wins,
        q_trump_overkill,
        q_trump_wins_low,
        q_trump_overkill_low,
    )


@njit(cache=True)
def _apply_numba_card_index(
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
    card_index: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    trick_hist: np.ndarray,
    trick_count: np.ndarray,
) -> tuple[int, int, int]:
    """
    Applica una giocata mutando gli array e ritorna `(deck_size, table_size, current_turn)`.

    `trick_hist` (20,5) e `trick_count` (1,) registrano la storia delle prese per
    l'encoder v4: una riga `[lead, lead_player, resp, winner, points]` per presa completata.
    """
    played_card = hands[current_turn, card_index]
    hand_size = hand_sizes[current_turn]
    for i in range(card_index, hand_size - 1):
        hands[current_turn, i] = hands[current_turn, i + 1]
    hand_sizes[current_turn] -= 1

    table_cards[table_size] = played_card
    table_players[table_size] = current_turn
    table_size += 1
    seen_cards[played_card] = 1
    out_of_play_cards[played_card] = 1

    if table_size == 1:
        return deck_size, table_size, 1 - current_turn

    first_card = table_cards[0]
    second_card = table_cards[1]
    first_player = table_players[0]
    second_player = table_players[1]
    winner = _who_wins_trick_numba(first_card, first_player, second_card, second_player, trump_card)
    gained = CARD_POINTS_NUMBA[first_card] + CARD_POINTS_NUMBA[second_card]
    points[winner] += gained
    trick_hist[trick_count[0], 0] = first_card
    trick_hist[trick_count[0], 1] = first_player
    trick_hist[trick_count[0], 2] = second_card
    trick_hist[trick_count[0], 3] = winner
    trick_hist[trick_count[0], 4] = gained
    trick_count[0] += 1
    table_size = 0

    if deck_size > 0:
        for i in range(2):
            player_to_deal = (winner + i) % 2
            if deck_size <= 0:
                break
            deck_size -= 1
            hands[player_to_deal, hand_sizes[player_to_deal]] = deck[deck_size]
            hand_sizes[player_to_deal] += 1

    return deck_size, table_size, winner


@njit(cache=True)
def _collect_mlp_policy_game_into_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
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
    Raccoglie una traiettoria A2C full-JIT per una singola partita.

    Ritorna punti, vincitore, step_count, entropia media e buffer di traiettoria.
    """
    shuffled = _shuffle_deck_numba(seed)

    deck = np.empty(34, dtype=np.int64)
    hands = np.empty((2, 3), dtype=np.int64)
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
    table_cards = np.empty(2, dtype=np.int64)
    table_players = np.empty(2, dtype=np.int64)
    table_size = 0
    current_turn = 0
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[trump_card] = 1
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    # Storia delle prese per l'encoder v4: [lead, lead_player, resp, winner, points] per riga.
    trick_hist = np.zeros((20, 5), dtype=np.int64)
    trick_count = np.zeros(1, dtype=np.int64)

    step_count = 0
    entropy_sum = 0.0
    safety = 5000
    while safety > 0:
        safety -= 1

        while not (hand_sizes[0] == 0 and hand_sizes[1] == 0) and current_turn != policy_seat:
            opp_card_index = _choose_opponent_card_index_numba(
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                deck_size,
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
            opp_card_index = _choose_opponent_card_index_numba(
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                deck_size,
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
def _collect_mlp_policy_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
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
    """Wrapper JIT compatibile: alloca i buffer e raccoglie una singola partita."""
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

    policy_points, opponent_points, winner, step_count, avg_entropy = _collect_mlp_policy_game_into_numba(
        w1,
        b1,
        w2,
        b2,
        wv,
        bv,
        opponent_code,
        opponent_model_enabled,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        opponent_overkill_guard,
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
def _collect_mlp_policy_batch_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    belief_enabled: bool,
    belief_w1: np.ndarray,
    belief_b1: np.ndarray,
    belief_w2: np.ndarray,
    belief_b2: np.ndarray,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int,
    overkill_penalty_mode_code: int,
    opponent_model_enabled_flags: np.ndarray,
    opponent_codes: np.ndarray,
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
    """Raccoglie un batch di partite A2C dentro una sola chiamata JIT."""
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
        p_points, o_points, winner, steps, avg_entropy = _collect_mlp_policy_game_into_numba(
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
            int(opponent_codes[game_idx]),
            bool(opponent_model_enabled_flags[game_idx]),
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            opponent_overkill_guard,
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


def _state_to_numba_arrays(state: Fast2PState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converte `Fast2PState` in array compatti per il wrapper/test Python."""
    hands = np.full((2, 3), -1, dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)
    for player_index in range(2):
        hand_sizes[player_index] = len(state.hands[player_index])
        for i, card_id in enumerate(state.hands[player_index]):
            hands[player_index, i] = int(card_id)

    points = np.asarray(state.points, dtype=np.int64)
    table_cards = np.full(2, -1, dtype=np.int64)
    for i, card_id in enumerate(state.table_cards):
        table_cards[i] = int(card_id)

    return hands, hand_sizes, points, table_cards


def encode_fast_observation_numba_2p(
    state: Fast2PState,
    *,
    player_index: int,
    seen_cards_onehot: tuple[int, ...],
    out_of_play_cards_onehot: tuple[int, ...] | None = None,
    version: EncoderVersion = "v1",
) -> EncodedObservation:
    """
    Wrapper Python dell'encoder JIT, con lo stesso contratto di `encode_fast_observation_2p`.

    Il wrapper valida input e converte liste Python in array; nel rollout JIT finale useremo
    direttamente `encode_fast_observation_arrays_numba`.

    `out_of_play_cards_onehot` serve solo per v3 (carte fuori gioco); per v1/v2 è ignorato.
    """
    if player_index not in (0, 1):
        raise ValueError(f"player_index fuori range: {player_index}")
    if len(seen_cards_onehot) != ACTION_DIM:
        raise ValueError(f"seen_cards_onehot len={len(seen_cards_onehot)} (atteso {ACTION_DIM})")
    if version == "v1":
        feature_dim = int(FEATURE_DIM_2P_V1)
    elif version == "v2":
        feature_dim = int(FEATURE_DIM_2P_V2)
    elif version in ("v3", "v4"):
        feature_dim = int(FEATURE_DIM_2P_V3) if version == "v3" else int(FEATURE_DIM_2P_V4)
        if out_of_play_cards_onehot is None:
            raise ValueError("Encoder v3/v4 (numba) richiede `out_of_play_cards_onehot`.")
        if len(out_of_play_cards_onehot) != ACTION_DIM:
            raise ValueError(f"out_of_play_cards_onehot len={len(out_of_play_cards_onehot)} (atteso {ACTION_DIM})")
    else:
        raise ValueError(f"Encoder version non supportata: {version!r}")

    seen_cards = np.asarray(seen_cards_onehot, dtype=np.int64)
    if not np.all((seen_cards == 0) | (seen_cards == 1)):
        raise ValueError("seen_cards_onehot deve contenere solo 0/1")

    if out_of_play_cards_onehot is None:
        out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    else:
        out_of_play_cards = np.asarray(out_of_play_cards_onehot, dtype=np.int64)
        if not np.all((out_of_play_cards == 0) | (out_of_play_cards == 1)):
            raise ValueError("out_of_play_cards_onehot deve contenere solo 0/1")

    hands, hand_sizes, points, table_cards = _state_to_numba_arrays(state)
    if version == "v4":
        # Storia numerica dal Fast2PState -> array (20,5) per il kernel.
        trick_hist = np.zeros((20, 5), dtype=np.int64)
        for i, (lead, lead_player, resp, winner, trick_points) in enumerate(state.trick_history):
            trick_hist[i, 0] = lead
            trick_hist[i, 1] = lead_player
            trick_hist[i, 2] = resp
            trick_hist[i, 3] = winner
            trick_hist[i, 4] = trick_points
        features, action_mask = encode_fast_observation_arrays_v4_numba(
            hands,
            hand_sizes,
            points,
            table_cards,
            len(state.table_cards),
            len(state.deck),
            int(state.current_turn),
            int(state.trump_card),
            int(player_index),
            seen_cards,
            out_of_play_cards,
            trick_hist,
            len(state.trick_history),
        )
    else:
        features, action_mask = encode_fast_observation_arrays_numba(
            hands,
            hand_sizes,
            points,
            table_cards,
            len(state.table_cards),
            len(state.deck),
            int(state.current_turn),
            int(state.trump_card),
            int(player_index),
            seen_cards,
            out_of_play_cards,
            feature_dim,
        )
    return EncodedObservation(features=features.astype(float).tolist(), action_mask=action_mask.astype(bool).tolist())


def warm_up_numba_observation() -> None:
    """Compila l'encoder osservazione Numba con un input minimo."""
    hands = np.full((2, 3), -1, dtype=np.int64)
    hands[0, 0] = 0
    hands[1, 0] = 10
    hand_sizes = np.asarray([1, 1], dtype=np.int64)
    points = np.asarray([0, 0], dtype=np.int64)
    table_cards = np.full(2, -1, dtype=np.int64)
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    # Warm-up esteso a v3 per compilare anche il blocco aggregato.
    encode_fast_observation_arrays_numba(
        hands,
        hand_sizes,
        points,
        table_cards,
        0,
        34,
        0,
        20,
        0,
        seen_cards,
        out_of_play_cards,
        int(FEATURE_DIM_2P_V3),
    )
    # Warm-up v4 (storia prese).
    encode_fast_observation_arrays_v4_numba(
        hands,
        hand_sizes,
        points,
        table_cards,
        0,
        34,
        0,
        20,
        0,
        seen_cards,
        out_of_play_cards,
        np.zeros((20, 5), dtype=np.int64),
        0,
    )
