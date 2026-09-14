"""
Kernel Numba per la search PIMC (belief-weighted) da osservazione numerica.

Porta nel JIT il ciclo caldo di `ai/agents/pimc.py`: determinizzazione anti-cheat
(campiona mani avversarie compatibili con la SOLA osservazione), rollout a terminale
con policy MLP + solver esatto a mazzo vuoto, media dei delta per carta candidata.

Semantica replicata da `determinize_observation` (parita' protetta dai test):
- pool ignoto = 40 carte − mano mia − out_of_play (la briscola scoperta e' nel pool);
- se la briscola e' ancora viva e il mazzo non e' vuoto, e' FORZATA in fondo al mazzo
  (indice 0 nel layout fast: viene pescata per ultima);
- mano avversaria campionata con "successive sampling" pesato (belief) o uniforme;
- il resto, mescolato, e' il mazzo.

La belief NON e' calcolata qui: il chiamante passa `card_weights[40]` gia' pronti
(vedi `belief_card_weights` nel modulo python) — pesi 0 per le carte non ignote.

Anti-cheat invariato: il kernel riceve solo cio' che l'osservazione lecita contiene.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .core import ACTION_DIM
from .observation import _apply_numba_card_index
from .value_dataset import _play_hybrid_mlp_to_terminal_numba

# Capienza massime del 2-player (come nel resto del path numba).
_HAND_CAPACITY_2P = 3
_MAX_DECK_SIZE_2P = 34
_TABLE_CAPACITY_2P = 2


@njit(cache=True)
def _weighted_sample_without_replacement_numba(
    pool: np.ndarray,
    pool_size: int,
    k: int,
    weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Campiona `k` id da `pool[:pool_size]` senza rimpiazzo, proporzionalmente a `weights[id]`.

    Stesso schema del python (`_weighted_sample_without_replacement`): a ogni estrazione
    la probabilita' e' proporzionale al peso residuo; con totale residuo 0 degrada
    all'uniforme sul resto (mai un crash per una belief mal calibrata).
    Muta `out[:k]` e riordina `pool` (gli estratti finiscono fuori da [:pool_size]).
    """
    n = pool_size
    for j in range(k):
        total = 0.0
        for i in range(n):
            w = weights[pool[i]]
            if w > 0.0:
                total += w
        if total <= 0.0:
            idx = np.random.randint(0, n)
        else:
            r = np.random.random() * total
            acc = 0.0
            idx = n - 1
            for i in range(n):
                w = weights[pool[i]]
                if w > 0.0:
                    acc += w
                if r <= acc:
                    idx = i
                    break
        out[j] = pool[idx]
        # swap-remove: l'estratto va in coda, il pool attivo si accorcia.
        pool[idx], pool[n - 1] = pool[n - 1], pool[idx]
        n -= 1


@njit(cache=True)
def _belief_card_weights_numba(
    v4_features: np.ndarray,
    belief_w1: np.ndarray,
    belief_b1: np.ndarray,
    belief_w2: np.ndarray,
    belief_b2: np.ndarray,
    in_my_hand: np.ndarray,
    out_of_play_cards: np.ndarray,
    uniform_mix: float,
    out_weights: np.ndarray,
) -> None:
    """
    Pesi di campionamento per-carta dalla belief network (replica JIT di
    `belief_card_weights`): sigmoid sulle sole carte IGNOTE, normalizzate a somma 1,
    mixate con l'uniforme (pavimento anti punti-ciechi). `out_weights[40]` mutato,
    0 per le carte non ignote; belief degenere (somma 0) -> uniforme.
    """
    hidden_dim = belief_w1.shape[1]
    feature_dim = belief_w1.shape[0]
    hidden = np.empty(hidden_dim, dtype=np.float64)
    for h_idx in range(hidden_dim):
        value = float(belief_b1[h_idx])
        for f_idx in range(feature_dim):
            value += float(v4_features[f_idx]) * float(belief_w1[f_idx, h_idx])
        hidden[h_idx] = value if value > 0.0 else 0.0

    n_unknown = 0
    total = 0.0
    for card_id in range(ACTION_DIM):
        out_weights[card_id] = 0.0
        if in_my_hand[card_id] == 0 and out_of_play_cards[card_id] == 0:
            logit = float(belief_b2[card_id])
            for h_idx in range(hidden_dim):
                logit += hidden[h_idx] * float(belief_w2[h_idx, card_id])
            prob = 1.0 / (1.0 + np.exp(-logit))
            out_weights[card_id] = prob
            total += prob
            n_unknown += 1

    if n_unknown == 0:
        return
    if total <= 0.0:
        for card_id in range(ACTION_DIM):
            if in_my_hand[card_id] == 0 and out_of_play_cards[card_id] == 0:
                out_weights[card_id] = 1.0
        return
    uniform = 1.0 / n_unknown
    for card_id in range(ACTION_DIM):
        if out_weights[card_id] > 0.0 or (in_my_hand[card_id] == 0 and out_of_play_cards[card_id] == 0):
            out_weights[card_id] = (1.0 - uniform_mix) * (out_weights[card_id] / total) + uniform_mix * uniform


@njit(cache=True)
def choose_pimc_card_numba_arrays(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    overkill_guard_enabled: bool,
    card_weights: np.ndarray,
    my_hand: np.ndarray,
    my_hand_size: int,
    opponent_hand_size: int,
    deck_size: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    my_index: int,
    trump_card: int,
    points: np.ndarray,
    out_of_play_cards: np.ndarray,
    seen_cards: np.ndarray,
    trick_hist: np.ndarray,
    num_tricks: int,
    num_determinizations: int,
    seed: int,
) -> int:
    """
    Sceglie l'indice carta (nella mano osservata) con la search PIMC su array numerici.

    Per ogni determinizzazione campiona una mano avversaria compatibile (pesata da
    `card_weights`), poi per ogni carta candidata gioca la continuazione fino a fine
    partita con `policy MLP (argmax + guard) + solver esatto a mazzo vuoto` e accumula
    il delta punti finale dal punto di vista di `my_index`. Vince la media piu' alta.

    Ritorna -1 se lo stato non e' determinizzabile (il chiamante usa il fallback).

    `seed >= 0` risemina lo stream numpy (riproducibilita' runtime); `seed < 0` continua
    lo stream corrente (uso nel collector di training, che ha gia' il suo seeding).
    """
    if seed >= 0:
        np.random.seed(seed)

    # ---- pool ignoto: non in mano mia, non fuori gioco ----
    in_my_hand = np.zeros(ACTION_DIM, dtype=np.int64)
    for i in range(my_hand_size):
        in_my_hand[my_hand[i]] = 1

    pool = np.empty(ACTION_DIM, dtype=np.int64)
    pool_size = 0
    for card_id in range(ACTION_DIM):
        if in_my_hand[card_id] == 0 and out_of_play_cards[card_id] == 0:
            pool[pool_size] = card_id
            pool_size += 1

    if pool_size != opponent_hand_size + deck_size:
        return -1  # osservazione incoerente: delega al fallback python
    if deck_size == 0:
        return -1  # endgame: il chiamante usa il solver esatto, non la search

    # La briscola scoperta, se ancora viva, esce dal pool campionabile: e' pubblica
    # e per regola viene pescata per ultima (fondo del mazzo).
    trump_alive = 0
    for i in range(pool_size):
        if pool[i] == trump_card:
            pool[i], pool[pool_size - 1] = pool[pool_size - 1], pool[i]
            pool_size -= 1
            trump_alive = 1
            break

    opp_sample = np.empty(_HAND_CAPACITY_2P, dtype=np.int64)
    scores = np.zeros(_HAND_CAPACITY_2P, dtype=np.float64)
    counts = np.zeros(_HAND_CAPACITY_2P, dtype=np.int64)

    sim_hands = np.full((2, _HAND_CAPACITY_2P), -1, dtype=np.int64)
    sim_hand_sizes = np.zeros(2, dtype=np.int64)
    sim_points = np.zeros(2, dtype=np.int64)
    sim_deck = np.full(_MAX_DECK_SIZE_2P, -1, dtype=np.int64)
    sim_table_cards = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    sim_table_players = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    sim_seen = np.zeros(ACTION_DIM, dtype=np.int64)
    sim_out = np.zeros(ACTION_DIM, dtype=np.int64)
    deck_rest = np.empty(_MAX_DECK_SIZE_2P, dtype=np.int64)

    opp_index = 1 - my_index

    for _det in range(num_determinizations):
        # ---- campiona la mano avversaria (pesata) ----
        _weighted_sample_without_replacement_numba(pool, pool_size, opponent_hand_size, card_weights, opp_sample)

        # Il resto del pool attivo (dopo gli swap del sampling, gli estratti sono in coda).
        rest_size = 0
        for i in range(pool_size - opponent_hand_size):
            deck_rest[rest_size] = pool[i]
            rest_size += 1
        # Fisher-Yates sul resto.
        for i in range(rest_size - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            deck_rest[i], deck_rest[j] = deck_rest[j], deck_rest[i]

        # ---- per ogni candidata: costruisci lo stato, gioca, rollout a terminale ----
        for cand in range(my_hand_size):
            # stato determinizzato fresco
            for p in range(2):
                for s in range(_HAND_CAPACITY_2P):
                    sim_hands[p, s] = -1
            for i in range(my_hand_size):
                sim_hands[my_index, i] = my_hand[i]
            for i in range(opponent_hand_size):
                sim_hands[opp_index, i] = opp_sample[i]
            sim_hand_sizes[my_index] = my_hand_size
            sim_hand_sizes[opp_index] = opponent_hand_size
            sim_points[0] = points[0]
            sim_points[1] = points[1]

            # mazzo: briscola (se viva) all'indice 0 = pescata per ultima nel layout fast.
            sim_deck_size = 0
            if trump_alive == 1:
                sim_deck[0] = trump_card
                sim_deck_size = 1
            for i in range(rest_size):
                sim_deck[sim_deck_size] = deck_rest[i]
                sim_deck_size += 1

            for i in range(_TABLE_CAPACITY_2P):
                sim_table_cards[i] = table_cards[i] if i < table_size else -1
                sim_table_players[i] = table_players[i] if i < table_size else -1
            for card_id in range(ACTION_DIM):
                sim_seen[card_id] = seen_cards[card_id]
                sim_out[card_id] = out_of_play_cards[card_id]
            sim_trick_hist = trick_hist.copy()
            sim_trick_count = np.zeros(1, dtype=np.int64)
            sim_trick_count[0] = num_tricks

            # gioca la candidata
            sim_deck_size2, sim_table_size2, sim_turn2 = _apply_numba_card_index(
                sim_hands,
                sim_hand_sizes,
                sim_points,
                sim_deck,
                sim_deck_size,
                sim_table_cards,
                sim_table_players,
                table_size,
                my_index,
                trump_card,
                cand,
                sim_seen,
                sim_out,
                sim_trick_hist,
                sim_trick_count,
            )

            # continua fino a fine partita: MLP argmax (+guard) per entrambi + solver nel finale.
            final0, final1 = _play_hybrid_mlp_to_terminal_numba(
                w1,
                b1,
                w2,
                b2,
                overkill_guard_enabled,
                sim_hands,
                sim_hand_sizes,
                sim_points,
                sim_deck,
                sim_deck_size2,
                sim_table_cards,
                sim_table_players,
                sim_table_size2,
                sim_turn2,
                trump_card,
                sim_seen,
                sim_out,
                sim_trick_hist,
                sim_trick_count,
            )
            delta = float(final0 - final1) if my_index == 0 else float(final1 - final0)
            scores[cand] += delta
            counts[cand] += 1

    best_cand = 0
    best_score = -1.0e30
    for cand in range(my_hand_size):
        if counts[cand] == 0:
            continue
        avg = scores[cand] / counts[cand]
        if avg > best_score:
            best_score = avg
            best_cand = cand
    return best_cand


def warm_up_numba_pimc() -> None:
    """Compila il kernel PIMC con un input minimo coerente (2 carte ignote vive)."""
    w1 = np.zeros((248, 4), dtype=np.float32)
    b1 = np.zeros(4, dtype=np.float32)
    w2 = np.zeros((4, ACTION_DIM), dtype=np.float32)
    b2 = np.zeros(ACTION_DIM, dtype=np.float32)
    # Stato LEGALE minimo: mani 3+3, mazzo 2 (briscola + 1) -> 5 carte ignote vive.
    # (Nel 2-player il mazzo ha sempre dimensione pari: stati con mazzo dispari sono
    # impossibili e mandano il rollout in configurazioni degeneri.)
    my_hand = np.asarray([0, 1, 2], dtype=np.int64)
    table_cards = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    table_players = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    out_of_play = np.ones(ACTION_DIM, dtype=np.int64)
    for live in (0, 1, 2, 10, 11, 12, 13, 20):
        out_of_play[live] = 0
    seen = np.zeros(ACTION_DIM, dtype=np.int64)
    seen[20] = 1
    choose_pimc_card_numba_arrays(
        w1,
        b1,
        w2,
        b2,
        True,
        np.ones(ACTION_DIM, dtype=np.float64),
        my_hand,
        3,
        3,
        2,
        table_cards,
        table_players,
        0,
        0,
        20,
        np.zeros(2, dtype=np.int64),
        out_of_play,
        seen,
        np.zeros((20, 5), dtype=np.int64),
        0,
        2,
        0,
    )
