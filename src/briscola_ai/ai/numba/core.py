"""
Core Numba sperimentale per simulazioni 2-player.

Questo modulo è il primo passo JIT: tiene tutto il loop partita dentro funzioni `njit`,
usando solo interi e array NumPy. Per ora copre random-vs-random, così possiamo misurare
il limite superiore del motore compilato senza introdurre policy neurali o oggetti dominio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from ..card_tables import (
    CARD_NUMBER_BY_ID_NP,
    CARD_POINTS_BY_ID_NP,
    CARD_STRENGTH_BY_ID_NP,
    CARD_SUIT_BY_ID_NP,
)
from ..evaluation import MatchStats, SeatFairStats

# Kernel condiviso "chi vince la presa": vive in `ai/trick_kernel.py` (modulo foglia) per
# essere importabile anche dal solver endgame senza cicli. L'alias mantiene il nome storico.
from ..trick_kernel import who_wins_trick_numba_2p as _who_wins_trick_numba

ACTION_DIM = 40
NUMBA_AGENT_RANDOM = 0
NUMBA_AGENT_GREEDY_POINTS = 1
NUMBA_AGENT_HEURISTIC_V1 = 2
NUMBA_AGENT_HEURISTIC_V2 = 3
NUMBA_AGENT_HEURISTIC_TRUMP_SAVER = 4

NUMBA_EVALUATION_AGENT_NAMES: frozenset[str] = frozenset(
    {"random", "greedy_points", "heuristic_v1", "heuristic_v2", "heuristic_trump_saver"}
)
_NUMBA_AGENT_CODES: dict[str, int] = {
    "random": NUMBA_AGENT_RANDOM,
    "greedy_points": NUMBA_AGENT_GREEDY_POINTS,
    "heuristic_v1": NUMBA_AGENT_HEURISTIC_V1,
    "heuristic_v2": NUMBA_AGENT_HEURISTIC_V2,
    "heuristic_trump_saver": NUMBA_AGENT_HEURISTIC_TRUMP_SAVER,
}
# Massimo codice agente valido: derivato dal registry dei codici, così i range check
# esterni (es. `collect_a2c_batch_numba_2p`) non possono divergere quando si aggiunge un agente.
NUMBA_AGENT_CODE_MAX: int = max(_NUMBA_AGENT_CODES.values())

# Soglia "carico" di `heuristic_trump_saver`: asso (11) e tre (10) — vedi `HeuristicTrumpSaverAgent`.
_TRUMP_SAVER_CARICO_POINTS = 10
# Tabelle numeriche derivate dal dominio canonico: vedi `ai/card_tables.py` (fonte unica).
# Gli alias locali mantengono i nomi storici usati dai kernel JIT.
CARD_SUIT_NUMBA = CARD_SUIT_BY_ID_NP
CARD_NUMBER_NUMBA = CARD_NUMBER_BY_ID_NP
CARD_POINTS_NUMBA = CARD_POINTS_BY_ID_NP
CARD_STRENGTH_NUMBA = CARD_STRENGTH_BY_ID_NP


@dataclass(frozen=True, slots=True)
class NumbaRandomSummary:
    """Risultato aggregato del benchmark random-vs-random compilato."""

    num_games: int
    wins0: int
    wins1: int
    draws: int
    sum0: int
    sum1: int

    def to_match_stats(self) -> MatchStats:
        """Converte il summary nel DTO statistico usato dal resto della evaluation."""
        return MatchStats(
            num_games=self.num_games,
            agent0_name="random_numba",
            agent1_name="random_numba",
            wins_agent0=self.wins0,
            wins_agent1=self.wins1,
            draws=self.draws,
            avg_points_agent0=self.sum0 / self.num_games if self.num_games else 0.0,
            avg_points_agent1=self.sum1 / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent0_minus_agent1=(self.sum0 - self.sum1) / self.num_games if self.num_games else 0.0,
        )


def numba_agent_code(agent_name: str) -> int:
    """Ritorna il codice numerico usato dal core JIT per un agente fast-compatible."""
    try:
        return _NUMBA_AGENT_CODES[agent_name]
    except KeyError as exc:
        supported = ", ".join(sorted(NUMBA_EVALUATION_AGENT_NAMES))
        raise ValueError(f"Numba supporta solo agenti fast-compatible: {supported}. Ottenuto: {agent_name!r}") from exc


@njit(cache=True)
def _shuffle_deck_numba(seed: int) -> np.ndarray:
    """Crea e mescola un mazzo numerico con RNG Numba locale alla funzione."""
    np.random.seed(seed)
    deck = np.arange(ACTION_DIM, dtype=np.int64)
    for i in range(ACTION_DIM - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp
    return deck


@njit(cache=True)
def _choose_discard_card_index_numba(
    hands: np.ndarray, hand_sizes: np.ndarray, player_index: int, trump_suit: int
) -> int:
    """Sceglie lo scarto piu' economico: pochi punti, evita briscole, forza bassa."""
    best_idx = 0
    best_card = hands[player_index, 0]
    best_points = CARD_POINTS_NUMBA[best_card]
    best_is_trump = 1 if CARD_SUIT_NUMBA[best_card] == trump_suit else 0
    best_strength = CARD_STRENGTH_NUMBA[best_card]

    for i in range(1, hand_sizes[player_index]):
        card = hands[player_index, i]
        points = CARD_POINTS_NUMBA[card]
        is_trump = 1 if CARD_SUIT_NUMBA[card] == trump_suit else 0
        strength = CARD_STRENGTH_NUMBA[card]
        if (
            points < best_points
            or (points == best_points and is_trump < best_is_trump)
            or (points == best_points and is_trump == best_is_trump and strength < best_strength)
        ):
            best_idx = i
            best_points = points
            best_is_trump = is_trump
            best_strength = strength
    return best_idx


@njit(cache=True)
def _choose_greedy_points_card_index_numba(hands: np.ndarray, hand_sizes: np.ndarray, player_index: int) -> int:
    """Sceglie una carta tra quelle con piu' punti, con tie-break casuale."""
    best_points = -1
    candidates = 0
    for i in range(hand_sizes[player_index]):
        points = CARD_POINTS_NUMBA[hands[player_index, i]]
        if points > best_points:
            best_points = points
            candidates = 1
        elif points == best_points:
            candidates += 1

    selected = np.random.randint(0, candidates)
    seen = 0
    for i in range(hand_sizes[player_index]):
        if CARD_POINTS_NUMBA[hands[player_index, i]] != best_points:
            continue
        if seen == selected:
            return i
        seen += 1
    return 0


@njit(cache=True)
def _choose_heuristic_lead_card_index_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    deck_size: int,
    trump_card: int,
) -> int:
    """Versione JIT della scelta lead di `heuristic_v1/v2`."""
    trump_suit = CARD_SUIT_NUMBA[trump_card]

    if deck_size <= 0:
        best_idx = 0
        best_card = hands[player_index, 0]
        best_strength = CARD_STRENGTH_NUMBA[best_card]
        best_points = CARD_POINTS_NUMBA[best_card]
        for i in range(1, hand_sizes[player_index]):
            card = hands[player_index, i]
            strength = CARD_STRENGTH_NUMBA[card]
            points = CARD_POINTS_NUMBA[card]
            if strength > best_strength or (strength == best_strength and points > best_points):
                best_idx = i
                best_strength = strength
                best_points = points
        return best_idx

    return _choose_discard_card_index_numba(hands, hand_sizes, player_index, trump_suit)


@njit(cache=True)
def _choose_heuristic_v1_response_card_index_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    deck_size: int,
    trump_card: int,
) -> int:
    """Versione JIT della risposta di `heuristic_v1`."""
    lead_card = table_cards[0]
    lead_player = table_players[0]
    trump_suit = CARD_SUIT_NUMBA[trump_card]
    best_win_idx = -1
    best_win_is_trump = 0
    best_win_points = 0
    best_win_strength = 0

    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        winner = _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card)
        if winner != player_index:
            continue
        is_trump = 1 if CARD_SUIT_NUMBA[card] == trump_suit else 0
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if (
            best_win_idx < 0
            or is_trump < best_win_is_trump
            or (is_trump == best_win_is_trump and points < best_win_points)
            or (is_trump == best_win_is_trump and points == best_win_points and strength < best_win_strength)
        ):
            best_win_idx = i
            best_win_is_trump = is_trump
            best_win_points = points
            best_win_strength = strength

    if best_win_idx >= 0:
        best_win_card = hands[player_index, best_win_idx]
        total_trick_points = CARD_POINTS_NUMBA[lead_card] + CARD_POINTS_NUMBA[best_win_card]
        best_is_trump = CARD_SUIT_NUMBA[best_win_card] == trump_suit
        best_is_free = CARD_POINTS_NUMBA[best_win_card] == 0 and not best_is_trump
        if total_trick_points >= 10 or deck_size <= 0 or best_is_free:
            return best_win_idx

    return _choose_discard_card_index_numba(hands, hand_sizes, player_index, trump_suit)


@njit(cache=True)
def _count_remaining_trumps_public_numba(seen_cards: np.ndarray, trump_suit: int) -> int:
    """Conta quante briscole non sono ancora viste pubblicamente."""
    remaining = 0
    for card_id in range(ACTION_DIM):
        if seen_cards[card_id] == 0 and CARD_SUIT_NUMBA[card_id] == trump_suit:
            remaining += 1
    return remaining


@njit(cache=True)
def _hand_contains_card_numba(hands: np.ndarray, hand_size: int, player_index: int, card_id: int) -> bool:
    """Ritorna True se `card_id` e' nella mano numerica del player."""
    # SIM110 suggerirebbe `any(...)`, ma le generator expression non compilano in @njit:
    # il loop esplicito e' l'unica forma supportata da Numba.
    for i in range(hand_size):  # noqa: SIM110
        if hands[player_index, i] == card_id:
            return True
    return False


@njit(cache=True)
def _count_unknown_high_trumps_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    seen_cards: np.ndarray,
    trump_suit: int,
    strength_threshold: int,
) -> int:
    """Stima quante briscole alte non viste e non in mano al player restano ignote."""
    unknown = 0
    hand_size = hand_sizes[player_index]
    for card_id in range(ACTION_DIM):
        if seen_cards[card_id] != 0:
            continue
        if _hand_contains_card_numba(hands, hand_size, player_index, card_id):
            continue
        if CARD_SUIT_NUMBA[card_id] == trump_suit and CARD_STRENGTH_NUMBA[card_id] >= strength_threshold:
            unknown += 1
    return unknown


@njit(cache=True)
def _should_take_with_trump_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    lead_card: int,
    response_card: int,
    trump_suit: int,
    deck_size: int,
    seen_cards: np.ndarray,
) -> bool:
    """Versione JIT della regola di conservazione briscole di `heuristic_v2`."""
    total_points = CARD_POINTS_NUMBA[lead_card] + CARD_POINTS_NUMBA[response_card]
    if deck_size <= 0:
        return True
    if total_points >= 10:
        return True

    remaining_trumps_public = _count_remaining_trumps_public_numba(seen_cards, trump_suit)
    unknown_high_trumps = _count_unknown_high_trumps_numba(hands, hand_sizes, player_index, seen_cards, trump_suit, 9)
    is_high_trump = CARD_STRENGTH_NUMBA[response_card] >= 8
    is_late = deck_size <= 4 or remaining_trumps_public <= 2

    if is_late:
        if CARD_POINTS_NUMBA[response_card] == 0 and not is_high_trump:
            return True
        if unknown_high_trumps == 0 and total_points >= 3:
            return True

    return False


@njit(cache=True)
def _choose_heuristic_v2_response_card_index_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    deck_size: int,
    trump_card: int,
    seen_cards: np.ndarray,
) -> int:
    """Versione JIT della risposta di `heuristic_v2`."""
    lead_card = table_cards[0]
    lead_player = table_players[0]
    trump_suit = CARD_SUIT_NUMBA[trump_card]

    best_non_trump_idx = -1
    best_non_trump_points = 0
    best_non_trump_strength = 0
    best_trump_idx = -1
    best_trump_points = 0
    best_trump_strength = 0

    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        winner = _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card)
        if winner != player_index:
            continue

        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if CARD_SUIT_NUMBA[card] == trump_suit:
            if (
                best_trump_idx < 0
                or points < best_trump_points
                or (points == best_trump_points and strength < best_trump_strength)
            ):
                best_trump_idx = i
                best_trump_points = points
                best_trump_strength = strength
        elif (
            best_non_trump_idx < 0
            or points < best_non_trump_points
            or (points == best_non_trump_points and strength < best_non_trump_strength)
        ):
            best_non_trump_idx = i
            best_non_trump_points = points
            best_non_trump_strength = strength

    if best_non_trump_idx >= 0:
        return best_non_trump_idx

    if best_trump_idx >= 0:
        best_trump = hands[player_index, best_trump_idx]
        if _should_take_with_trump_numba(
            hands, hand_sizes, player_index, lead_card, best_trump, trump_suit, deck_size, seen_cards
        ):
            return best_trump_idx

    return _choose_discard_card_index_numba(hands, hand_sizes, player_index, trump_suit)


@njit(cache=True)
def _is_master_numba(
    card_id: int,
    hands: np.ndarray,
    hand_size: int,
    player_index: int,
    seen_cards: np.ndarray,
    trump_card: int,
) -> bool:
    """
    Versione JIT di `HeuristicTrumpSaverAgent._is_master`.

    True se `card_id`, guidata da primi, vince contro OGNI carta "viva e ignota"
    (non vista pubblicamente e non in mano al player). Riusa il kernel condiviso
    `who_wins_trick_numba_2p` (ancorato al dominio dai test-àncora): niente regole duplicate.
    """
    for other in range(ACTION_DIM):
        if seen_cards[other] != 0:
            continue
        if _hand_contains_card_numba(hands, hand_size, player_index, other):
            continue
        if _who_wins_trick_numba(card_id, 0, other, 1, trump_card) != 0:
            return False
    return True


@njit(cache=True)
def _choose_trump_saver_lead_card_index_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    deck_size: int,
    trump_card: int,
    seen_cards: np.ndarray,
) -> int:
    """
    Versione JIT della scelta lead di `heuristic_trump_saver`.

    Stesse priorità del dominio:
    1. incassare il master più ricco se conveniente (a mazzo vuoto sempre; durante le pescate
       solo se è un carico NON di briscola);
    2. altrimenti guidare "liscio": minimizza (carico, punti, briscola, forza).
    I tie-break replicano `max`/`min` di Python: a parità di chiave vince il PRIMO indice.
    """
    trump_suit = CARD_SUIT_NUMBA[trump_card]
    hand_size = hand_sizes[player_index]

    # Master più "ricco": max su (punti, forza); si sostituisce solo con chiave strettamente maggiore.
    richest_idx = -1
    richest_points = -1
    richest_strength = -1
    for i in range(hand_size):
        card = hands[player_index, i]
        if not _is_master_numba(card, hands, hand_size, player_index, seen_cards, trump_card):
            continue
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if points > richest_points or (points == richest_points and strength > richest_strength):
            richest_idx = i
            richest_points = points
            richest_strength = strength

    if richest_idx >= 0:
        richest_card = hands[player_index, richest_idx]
        cashable_now = deck_size <= 0 or (
            CARD_SUIT_NUMBA[richest_card] != trump_suit
            and CARD_POINTS_NUMBA[richest_card] >= _TRUMP_SAVER_CARICO_POINTS
        )
        if cashable_now:
            return richest_idx

    # Guida "liscia": min lessicografico su (is_carico, punti, is_trump, forza).
    best_idx = 0
    best_card = hands[player_index, 0]
    best_is_carico = 1 if CARD_POINTS_NUMBA[best_card] >= _TRUMP_SAVER_CARICO_POINTS else 0
    best_points = CARD_POINTS_NUMBA[best_card]
    best_is_trump = 1 if CARD_SUIT_NUMBA[best_card] == trump_suit else 0
    best_strength = CARD_STRENGTH_NUMBA[best_card]
    for i in range(1, hand_size):
        card = hands[player_index, i]
        is_carico = 1 if CARD_POINTS_NUMBA[card] >= _TRUMP_SAVER_CARICO_POINTS else 0
        points = CARD_POINTS_NUMBA[card]
        is_trump = 1 if CARD_SUIT_NUMBA[card] == trump_suit else 0
        strength = CARD_STRENGTH_NUMBA[card]
        if (
            is_carico < best_is_carico
            or (is_carico == best_is_carico and points < best_points)
            or (is_carico == best_is_carico and points == best_points and is_trump < best_is_trump)
            or (
                is_carico == best_is_carico
                and points == best_points
                and is_trump == best_is_trump
                and strength < best_strength
            )
        ):
            best_idx = i
            best_is_carico = is_carico
            best_points = points
            best_is_trump = is_trump
            best_strength = strength
    return best_idx


@njit(cache=True)
def _choose_trump_saver_response_card_index_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    deck_size: int,
    trump_card: int,
) -> int:
    """
    Versione JIT della risposta di `heuristic_trump_saver`.

    Stesse priorità del dominio:
    1. vincere senza briscola -> incassa il massimo dei punti (a parità la forza MINIMA,
       per conservare la carta più forte);
    2. con briscola: taglia SEMPRE il carico, piatti medi (>=3 punti) solo con briscolina
       a 0 punti, mai piatti poveri durante le pescate; a mazzo vuoto prende comunque;
    3. altrimenti scarto economico.
    """
    lead_card = table_cards[0]
    lead_player = table_players[0]
    trump_suit = CARD_SUIT_NUMBA[trump_card]

    # Non-briscola vincente con più punti: max su (punti, -forza); si sostituisce solo
    # con chiave strettamente maggiore (a parità di punti vince la forza più BASSA).
    best_cash_idx = -1
    best_cash_points = -1
    best_cash_strength = 0
    # Briscola vincente minima: min su (punti, forza).
    best_trump_idx = -1
    best_trump_points = 0
    best_trump_strength = 0

    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        winner = _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card)
        if winner != player_index:
            continue

        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if CARD_SUIT_NUMBA[card] == trump_suit:
            if (
                best_trump_idx < 0
                or points < best_trump_points
                or (points == best_trump_points and strength < best_trump_strength)
            ):
                best_trump_idx = i
                best_trump_points = points
                best_trump_strength = strength
        elif points > best_cash_points or (points == best_cash_points and strength < best_cash_strength):
            best_cash_idx = i
            best_cash_points = points
            best_cash_strength = strength

    if best_cash_idx >= 0:
        return best_cash_idx

    if best_trump_idx >= 0:
        lead_points = CARD_POINTS_NUMBA[lead_card]
        if deck_size <= 0:
            # Endgame: senza pescate prendere (e guidare la prossima) vale quasi sempre.
            return best_trump_idx
        if lead_points >= _TRUMP_SAVER_CARICO_POINTS:
            # Il carico avversario si taglia SEMPRE (l'exploit-chiave, anche con l'asso di briscola).
            return best_trump_idx
        if lead_points >= 3 and best_trump_points == 0:
            # Piatto medio (re/cavallo): solo se il taglio è gratis (briscolina a 0 punti).
            return best_trump_idx
        # Piatti poveri durante le pescate: la briscola si conserva, si scarta.

    return _choose_discard_card_index_numba(hands, hand_sizes, player_index, trump_suit)


@njit(cache=True)
def _choose_policy_card_index_numba(
    agent_code: int,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    deck_size: int,
    trump_card: int,
    seen_cards: np.ndarray,
) -> int:
    """Dispatch JIT per agenti fast-compatible."""
    hand_size = hand_sizes[player_index]
    if hand_size <= 0:
        raise ValueError("Mano vuota: nessuna azione possibile")

    if agent_code == NUMBA_AGENT_RANDOM:
        return np.random.randint(0, hand_size)

    if agent_code == NUMBA_AGENT_GREEDY_POINTS:
        return _choose_greedy_points_card_index_numba(hands, hand_sizes, player_index)

    if agent_code == NUMBA_AGENT_HEURISTIC_V1:
        if table_size == 0:
            return _choose_heuristic_lead_card_index_numba(hands, hand_sizes, player_index, deck_size, trump_card)
        return _choose_heuristic_v1_response_card_index_numba(
            hands, hand_sizes, player_index, table_cards, table_players, deck_size, trump_card
        )

    if agent_code == NUMBA_AGENT_HEURISTIC_V2:
        if table_size == 0:
            return _choose_heuristic_lead_card_index_numba(hands, hand_sizes, player_index, deck_size, trump_card)
        return _choose_heuristic_v2_response_card_index_numba(
            hands, hand_sizes, player_index, table_cards, table_players, deck_size, trump_card, seen_cards
        )

    if agent_code == NUMBA_AGENT_HEURISTIC_TRUMP_SAVER:
        if table_size == 0:
            return _choose_trump_saver_lead_card_index_numba(
                hands, hand_sizes, player_index, deck_size, trump_card, seen_cards
            )
        return _choose_trump_saver_response_card_index_numba(
            hands, hand_sizes, player_index, table_cards, table_players, deck_size, trump_card
        )

    raise ValueError("Codice agente Numba non supportato")


@njit(cache=True)
def _play_random_game_numba(seed: int) -> tuple[int, int, int]:
    """
    Gioca una partita random-vs-random interamente in Numba.

    Ritorna `(points0, points1, winner)` con `winner=-1` in caso di pareggio.
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

    safety = 5000
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        hand_size = hand_sizes[current_turn]
        card_index = np.random.randint(0, hand_size)
        played_card = hands[current_turn, card_index]

        for i in range(card_index, hand_size - 1):
            hands[current_turn, i] = hands[current_turn, i + 1]
        hand_sizes[current_turn] -= 1

        table_cards[table_size] = played_card
        table_players[table_size] = current_turn
        table_size += 1

        if table_size == 1:
            current_turn = 1 - current_turn
            continue

        first_card = table_cards[0]
        second_card = table_cards[1]
        first_player = table_players[0]
        second_player = table_players[1]
        winner = _who_wins_trick_numba(first_card, first_player, second_card, second_player, trump_card)
        points[winner] += CARD_POINTS_NUMBA[first_card] + CARD_POINTS_NUMBA[second_card]
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

    if points[0] > points[1]:
        winner_out = 0
    elif points[1] > points[0]:
        winner_out = 1
    else:
        winner_out = -1
    return int(points[0]), int(points[1]), int(winner_out)


@njit(cache=True)
def _play_random_games_numba(num_games: int, seed: int) -> tuple[int, int, int, int, int]:
    """Gioca `num_games` partite random-vs-random e aggrega i risultati."""
    wins0 = 0
    wins1 = 0
    draws = 0
    sum0 = 0
    sum1 = 0

    for game_index in range(num_games):
        p0, p1, winner = _play_random_game_numba(seed + game_index)
        sum0 += p0
        sum1 += p1
        if winner == 0:
            wins0 += 1
        elif winner == 1:
            wins1 += 1
        else:
            draws += 1

    return wins0, wins1, draws, sum0, sum1


@njit(cache=True)
def _play_policy_game_numba(agent0_code: int, agent1_code: int, seed: int) -> tuple[int, int, int]:
    """
    Gioca una partita tra due policy fast-compatible interamente in Numba.

    Ritorna `(points0, points1, winner)` con `winner=-1` in caso di pareggio.
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

    safety = 5000
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        agent_code = agent0_code if current_turn == 0 else agent1_code
        card_index = _choose_policy_card_index_numba(
            agent_code,
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
        played_card = hands[current_turn, card_index]

        hand_size = hand_sizes[current_turn]
        for i in range(card_index, hand_size - 1):
            hands[current_turn, i] = hands[current_turn, i + 1]
        hand_sizes[current_turn] -= 1

        table_cards[table_size] = played_card
        table_players[table_size] = current_turn
        table_size += 1
        seen_cards[played_card] = 1

        if table_size == 1:
            current_turn = 1 - current_turn
            continue

        first_card = table_cards[0]
        second_card = table_cards[1]
        first_player = table_players[0]
        second_player = table_players[1]
        winner = _who_wins_trick_numba(first_card, first_player, second_card, second_player, trump_card)
        points[winner] += CARD_POINTS_NUMBA[first_card] + CARD_POINTS_NUMBA[second_card]
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

    if points[0] > points[1]:
        winner_out = 0
    elif points[1] > points[0]:
        winner_out = 1
    else:
        winner_out = -1
    return int(points[0]), int(points[1]), int(winner_out)


@njit(cache=True)
def _evaluate_policy_match_numba(
    agent0_code: int, agent1_code: int, num_games: int, seed: int
) -> tuple[int, int, int, int, int]:
    """Valuta due policy Numba senza seat-fair e aggrega i risultati."""
    wins0 = 0
    wins1 = 0
    draws = 0
    sum0 = 0
    sum1 = 0

    for game_index in range(num_games):
        p0, p1, winner = _play_policy_game_numba(agent0_code, agent1_code, seed + game_index)
        sum0 += p0
        sum1 += p1
        if winner == 0:
            wins0 += 1
        elif winner == 1:
            wins1 += 1
        else:
            draws += 1

    return wins0, wins1, draws, sum0, sum1


@njit(cache=True)
def _evaluate_policy_seat_fair_numba(
    agent_a_code: int, agent_b_code: int, num_games: int, seed: int
) -> tuple[int, int, int, int, int, int, int, int, float]:
    """
    Valuta due policy Numba in modalità seat-fair e aggrega i risultati.

    Oltre agli aggregati per partita, accumula anche le somme per COPPIA (stesso mazzo,
    seat scambiati): sono l'unità statistica indipendente su cui calcolare CI corrette.
    """
    wins_a = 0
    wins_b = 0
    draws = 0
    sum_a = 0
    sum_b = 0
    sum_diff = 0
    sum_sq_diff = 0
    sum_sq_pair_diff = 0
    sum_sq_pair_score = 0.0
    num_pairs = num_games // 2

    for pair_index in range(num_pairs):
        game_seed = seed + pair_index

        p0, p1, winner = _play_policy_game_numba(agent_a_code, agent_b_code, game_seed)
        sum_a += p0
        sum_b += p1
        diff1 = p0 - p1
        sum_diff += diff1
        sum_sq_diff += diff1 * diff1
        if winner == 0:
            wins_a += 1
            score1 = 1.0
        elif winner == 1:
            wins_b += 1
            score1 = 0.0
        else:
            draws += 1
            score1 = 0.5

        p0, p1, winner = _play_policy_game_numba(agent_b_code, agent_a_code, game_seed)
        sum_a += p1
        sum_b += p0
        diff2 = p1 - p0
        sum_diff += diff2
        sum_sq_diff += diff2 * diff2
        if winner == 0:
            wins_b += 1
            score2 = 0.0
        elif winner == 1:
            wins_a += 1
            score2 = 1.0
        else:
            draws += 1
            score2 = 0.5

        pair_diff = diff1 + diff2
        sum_sq_pair_diff += pair_diff * pair_diff
        pair_score = (score1 + score2) / 2.0
        sum_sq_pair_score += pair_score * pair_score

    return wins_a, wins_b, draws, sum_a, sum_b, sum_diff, sum_sq_diff, sum_sq_pair_diff, sum_sq_pair_score


def warm_up_numba() -> None:
    """Compila le funzioni Numba principali con un input minimo."""
    _play_random_games_numba(1, 0)


def warm_up_numba_evaluation() -> None:
    """Compila anche il core JIT con policy euristiche."""
    _evaluate_policy_match_numba(NUMBA_AGENT_HEURISTIC_V2, NUMBA_AGENT_HEURISTIC_V1, 2, 0)
    _evaluate_policy_seat_fair_numba(NUMBA_AGENT_HEURISTIC_V2, NUMBA_AGENT_HEURISTIC_V1, 2, 0)
    _evaluate_policy_match_numba(NUMBA_AGENT_HEURISTIC_TRUMP_SAVER, NUMBA_AGENT_HEURISTIC_V1, 2, 0)


def play_random_game_numba(seed: int) -> tuple[int, int, int]:
    """Wrapper Python per una partita random-vs-random JIT."""
    return _play_random_game_numba(int(seed))


def play_policy_game_numba(agent0_name: str, agent1_name: str, *, seed: int) -> tuple[int, int, int]:
    """Wrapper Python per una partita JIT tra agenti fast-compatible."""
    return _play_policy_game_numba(numba_agent_code(agent0_name), numba_agent_code(agent1_name), int(seed))


def evaluate_random_numba_2p(*, num_games: int, seed: int) -> NumbaRandomSummary:
    """Esegue random-vs-random con core Numba e ritorna statistiche aggregate."""
    if int(num_games) < 0:
        raise ValueError("num_games deve essere >= 0")
    wins0, wins1, draws, sum0, sum1 = _play_random_games_numba(int(num_games), int(seed))
    return NumbaRandomSummary(
        num_games=int(num_games),
        wins0=int(wins0),
        wins1=int(wins1),
        draws=int(draws),
        sum0=int(sum0),
        sum1=int(sum1),
    )


def evaluate_numba_match_2p(agent0_name: str, agent1_name: str, *, num_games: int, seed: int) -> MatchStats:
    """Valuta due agenti fast-compatible con il core Numba senza seat-fair."""
    num_games = int(num_games)
    if num_games < 0:
        raise ValueError("num_games deve essere >= 0")
    wins0, wins1, draws, sum0, sum1 = _evaluate_policy_match_numba(
        numba_agent_code(agent0_name),
        numba_agent_code(agent1_name),
        num_games,
        int(seed),
    )
    return MatchStats(
        num_games=num_games,
        agent0_name=agent0_name,
        agent1_name=agent1_name,
        wins_agent0=int(wins0),
        wins_agent1=int(wins1),
        draws=int(draws),
        avg_points_agent0=sum0 / num_games if num_games else 0.0,
        avg_points_agent1=sum1 / num_games if num_games else 0.0,
        avg_point_diff_agent0_minus_agent1=(sum0 - sum1) / num_games if num_games else 0.0,
    )


def evaluate_numba_seat_fair_match_2p(
    agent_a_name: str,
    agent_b_name: str,
    *,
    num_games: int,
    seed: int,
) -> SeatFairStats:
    """Valuta due agenti fast-compatible con il core Numba in modalità seat-fair."""
    num_games = int(num_games)
    if num_games < 0:
        raise ValueError("num_games deve essere >= 0")
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari.")
    (
        wins_a,
        wins_b,
        draws,
        sum_a,
        sum_b,
        sum_diff,
        sum_sq_diff,
        sum_sq_pair_diff,
        sum_sq_pair_score,
    ) = _evaluate_policy_seat_fair_numba(
        numba_agent_code(agent_a_name),
        numba_agent_code(agent_b_name),
        num_games,
        int(seed),
    )
    return SeatFairStats(
        num_games=num_games,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        wins_agent_a=int(wins_a),
        wins_agent_b=int(wins_b),
        draws=int(draws),
        avg_points_agent_a=sum_a / num_games if num_games else 0.0,
        avg_points_agent_b=sum_b / num_games if num_games else 0.0,
        avg_point_diff_agent_a_minus_agent_b=sum_diff / num_games if num_games else 0.0,
        sum_sq_point_diff_agent_a_minus_agent_b=float(sum_sq_diff),
        sum_sq_pair_point_diff_agent_a_minus_agent_b=float(sum_sq_pair_diff),
        sum_sq_pair_score_agent_a=float(sum_sq_pair_score),
    )
