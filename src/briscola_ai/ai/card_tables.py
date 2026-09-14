"""
Tabelle numeriche per-card_id derivate dal dominio canonico (fonte UNICA).

Perché esiste questo modulo
---------------------------
Il fast path (puro Python), i kernel Numba e l'encoding lavorano su card id interi
`[0,39]` e hanno bisogno di tabelle "carta -> seme/numero/punti/forza". Storicamente
ogni modulo manteneva la propria copia scritta a mano: un typo in una copia avrebbe
corrotto training e dataset in modo silenzioso (i test aggregati tipo "somma punti=120"
non lo avrebbero intercettato).

Qui le tabelle sono DERIVATE da `domain.models.Rank` e `domain.card_id`, quindi non
possono divergere dal dominio per costruzione. I test-àncora (`test_card_tables_parity`)
restano come rete di sicurezza contro regressioni di questo stesso modulo.

Convenzione carta: `card_id = suit_index * 10 + (number - 1)` (vedi `domain/card_id.py`).
"""

from __future__ import annotations

import numpy as np

from ..domain.card_id import SUIT_TO_INDEX, id_to_card

ACTION_DIM = 40

_CARDS = tuple(id_to_card(card_id) for card_id in range(ACTION_DIM))

# Tuple immutabili per il fast path puro-Python (lookup O(1) senza overhead NumPy).
CARD_SUIT_BY_ID: tuple[int, ...] = tuple(SUIT_TO_INDEX[card.suit] for card in _CARDS)
CARD_NUMBER_BY_ID: tuple[int, ...] = tuple(card.rank.number for card in _CARDS)
CARD_POINTS_BY_ID: tuple[int, ...] = tuple(card.rank.points for card in _CARDS)
CARD_STRENGTH_BY_ID: tuple[int, ...] = tuple(card.rank.trick_strength for card in _CARDS)

# Array NumPy int64 per i kernel `@njit` (Numba richiede array, non tuple, nei loop caldi).
CARD_SUIT_BY_ID_NP = np.asarray(CARD_SUIT_BY_ID, dtype=np.int64)
CARD_NUMBER_BY_ID_NP = np.asarray(CARD_NUMBER_BY_ID, dtype=np.int64)
CARD_POINTS_BY_ID_NP = np.asarray(CARD_POINTS_BY_ID, dtype=np.int64)
CARD_STRENGTH_BY_ID_NP = np.asarray(CARD_STRENGTH_BY_ID, dtype=np.int64)

# Tabelle indicizzate per `Rank.number` (1..10), usate dall'encoding.
POINTS_BY_NUMBER: dict[int, int] = {card.rank.number: card.rank.points for card in _CARDS[:10]}
TRICK_STRENGTH_BY_NUMBER: dict[int, int] = {card.rank.number: card.rank.trick_strength for card in _CARDS[:10]}
