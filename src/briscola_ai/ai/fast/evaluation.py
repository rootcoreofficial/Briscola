"""
Valutazione sperimentale basata sul fast path 2-player.

Questo modulo è volutamente separato da `ai.evaluation`:
- `ai.evaluation` resta il path canonico, anti-cheat e compatibile con tutti gli agenti;
- questo path fast supporta agenti traducibili direttamente su card id numerici.

L'obiettivo è misurare il guadagno del motore mutabile 2-player prima di integrarlo in training/evaluation
più complessi. I test devono dimostrare equivalenza aggregata col dominio per gli agenti supportati.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..evaluation import MatchStats, SeatFairStats
from .state_2p import (
    CARD_POINTS,
    CARD_STRENGTH,
    CARD_SUIT,
    Fast2PState,
    fast_who_wins_trick_2p,
    new_fast_2p_state,
    step_fast_2p,
)

FAST_EVALUATION_AGENT_NAMES: frozenset[str] = frozenset(
    {"random", "greedy_points", "heuristic_v1", "heuristic_v2", "heuristic_trump_saver"}
)

# Soglia "carico" di `heuristic_trump_saver`: asso (11) e tre (10) — vedi `HeuristicTrumpSaverAgent`.
_TRUMP_SAVER_CARICO_POINTS = 10


def _validate_fast_agent_name(agent_name: str) -> None:
    """Fallisce presto se un agente non è ancora supportato dal path fast."""
    if agent_name not in FAST_EVALUATION_AGENT_NAMES:
        supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
        raise ValueError(f"`--engine fast` supporta solo agenti fast-compatible: {supported}. Ottenuto: {agent_name!r}")


def _trick_points_fast(cards: tuple[int, ...]) -> int:
    """Somma i punti delle carte numeriche."""
    return sum(CARD_POINTS[card_id] for card_id in cards)


def _fast_card_wins_trick(
    *,
    lead_card: int,
    lead_player: int,
    response_card: int,
    response_player: int,
    trump_card: int,
) -> bool:
    """Ritorna True se `response_card` vincerebbe la mano."""
    winner = fast_who_wins_trick_2p(
        first_card=lead_card,
        first_player=lead_player,
        second_card=response_card,
        second_player=response_player,
        trump_card=trump_card,
    )
    return winner == response_player


def _choose_fast_heuristic_lead_card_index(state: Fast2PState, *, player_index: int) -> int:
    """Versione numerica della scelta lead di `heuristic_v1/v2`."""
    hand = state.hands[player_index]
    if len(state.deck) <= 0:
        return max(range(len(hand)), key=lambda i: (CARD_STRENGTH[hand[i]], CARD_POINTS[hand[i]]))

    trump_suit = CARD_SUIT[state.trump_card]
    return min(
        range(len(hand)),
        key=lambda i: (
            CARD_POINTS[hand[i]],
            1 if CARD_SUIT[hand[i]] == trump_suit else 0,
            CARD_STRENGTH[hand[i]],
        ),
    )


def _choose_fast_heuristic_v1_response_card_index(state: Fast2PState, *, player_index: int) -> int:
    """Versione numerica della risposta di `heuristic_v1`."""
    hand = state.hands[player_index]
    lead_card = state.table_cards[0]
    lead_player = state.table_players[0]
    trump_suit = CARD_SUIT[state.trump_card]

    winning_candidates: list[int] = []
    for i, card_id in enumerate(hand):
        if _fast_card_wins_trick(
            lead_card=lead_card,
            lead_player=lead_player,
            response_card=card_id,
            response_player=player_index,
            trump_card=state.trump_card,
        ):
            winning_candidates.append(i)

    if winning_candidates:
        best_win = min(
            winning_candidates,
            key=lambda idx: (
                1 if CARD_SUIT[hand[idx]] == trump_suit else 0,
                CARD_POINTS[hand[idx]],
                CARD_STRENGTH[hand[idx]],
            ),
        )
        best_win_card = hand[best_win]
        total_trick_points = _trick_points_fast((lead_card, best_win_card))
        best_is_trump = CARD_SUIT[best_win_card] == trump_suit
        best_is_free = CARD_POINTS[best_win_card] == 0 and not best_is_trump

        if total_trick_points >= 10 or len(state.deck) <= 0 or best_is_free:
            return best_win

    return min(
        range(len(hand)),
        key=lambda idx: (
            CARD_POINTS[hand[idx]],
            1 if CARD_SUIT[hand[idx]] == trump_suit else 0,
            CARD_STRENGTH[hand[idx]],
        ),
    )


def _count_remaining_trumps_public_fast(*, seen_cards_onehot: tuple[int, ...], trump_suit: int) -> int:
    """Conta quante briscole non risultano ancora viste pubblicamente."""
    return sum(1 for card_id, seen in enumerate(seen_cards_onehot) if not seen and CARD_SUIT[card_id] == trump_suit)


def _count_unknown_high_trumps_fast(
    *,
    hand: list[int],
    seen_cards_onehot: tuple[int, ...],
    trump_suit: int,
    strength_threshold: int = 9,
) -> int:
    """Stima quante briscole alte non viste e non in mano al player potrebbero ancora essere in giro."""
    hand_ids = set(hand)
    unknown = 0
    for card_id, seen in enumerate(seen_cards_onehot):
        if seen:
            continue
        if card_id in hand_ids:
            continue
        if CARD_SUIT[card_id] == trump_suit and CARD_STRENGTH[card_id] >= strength_threshold:
            unknown += 1
    return unknown


def _should_take_with_trump_fast(
    *,
    hand: list[int],
    lead_card: int,
    response_card: int,
    trump_suit: int,
    deck_size: int,
    seen_cards_onehot: tuple[int, ...],
) -> bool:
    """Versione numerica della regola `_should_take_with_trump` di `heuristic_v2`."""
    total_points = _trick_points_fast((lead_card, response_card))
    if deck_size <= 0:
        return True
    if total_points >= 10:
        return True

    remaining_trumps_public = _count_remaining_trumps_public_fast(
        seen_cards_onehot=seen_cards_onehot,
        trump_suit=trump_suit,
    )
    unknown_high_trumps = _count_unknown_high_trumps_fast(
        hand=hand,
        seen_cards_onehot=seen_cards_onehot,
        trump_suit=trump_suit,
        strength_threshold=9,
    )
    is_high_trump = CARD_STRENGTH[response_card] >= 8
    is_late = deck_size <= 4 or remaining_trumps_public <= 2

    if is_late:
        if CARD_POINTS[response_card] == 0 and not is_high_trump:
            return True
        if unknown_high_trumps == 0 and total_points >= 3:
            return True

    return False


def _choose_fast_heuristic_v2_response_card_index(
    state: Fast2PState,
    *,
    player_index: int,
    seen_cards_onehot: tuple[int, ...],
) -> int:
    """Versione numerica della risposta di `heuristic_v2`."""
    hand = state.hands[player_index]
    lead_card = state.table_cards[0]
    lead_player = state.table_players[0]
    trump_suit = CARD_SUIT[state.trump_card]

    winning_non_trumps: list[int] = []
    winning_trumps: list[int] = []
    for i, card_id in enumerate(hand):
        if not _fast_card_wins_trick(
            lead_card=lead_card,
            lead_player=lead_player,
            response_card=card_id,
            response_player=player_index,
            trump_card=state.trump_card,
        ):
            continue
        if CARD_SUIT[card_id] == trump_suit:
            winning_trumps.append(i)
        else:
            winning_non_trumps.append(i)

    if winning_non_trumps:
        return min(winning_non_trumps, key=lambda idx: (CARD_POINTS[hand[idx]], CARD_STRENGTH[hand[idx]]))

    if winning_trumps:
        best_trump_idx = min(winning_trumps, key=lambda idx: (CARD_POINTS[hand[idx]], CARD_STRENGTH[hand[idx]]))
        best_trump = hand[best_trump_idx]
        if _should_take_with_trump_fast(
            hand=hand,
            lead_card=lead_card,
            response_card=best_trump,
            trump_suit=trump_suit,
            deck_size=len(state.deck),
            seen_cards_onehot=seen_cards_onehot,
        ):
            return best_trump_idx

    return min(
        range(len(hand)),
        key=lambda idx: (
            CARD_POINTS[hand[idx]],
            1 if CARD_SUIT[hand[idx]] == trump_suit else 0,
            CARD_STRENGTH[hand[idx]],
        ),
    )


def _is_master_fast(
    card_id: int,
    *,
    hand: list[int],
    trump_card: int,
    seen_cards_onehot: tuple[int, ...],
) -> bool:
    """
    Versione numerica di `HeuristicTrumpSaverAgent._is_master`.

    True se `card_id`, guidata da primi, vince contro OGNI carta "viva e ignota"
    (non vista pubblicamente e non in mano a noi). Riusa il kernel di presa fast
    (già ancorato a `domain.rules.who_wins_trick` dai test-àncora) invece di duplicare le regole.
    """
    hand_ids = set(hand)
    for other, seen in enumerate(seen_cards_onehot):
        if seen or other in hand_ids:
            continue
        if _fast_card_wins_trick(
            lead_card=card_id,
            lead_player=0,
            response_card=other,
            response_player=1,
            trump_card=trump_card,
        ):
            return False
    return True


def _choose_fast_trump_saver_lead_card_index(
    state: Fast2PState,
    *,
    player_index: int,
    seen_cards_onehot: tuple[int, ...],
) -> int:
    """
    Versione numerica della scelta lead di `heuristic_trump_saver`.

    Stesse priorità del dominio:
    1. incassare un master conveniente (a mazzo vuoto qualsiasi master; durante le pescate
       solo un carico NON di briscola divenuto imbattibile);
    2. altrimenti guidare "liscio": mai un carico; minimizza (carico, punti, briscola, forza).
    """
    hand = state.hands[player_index]
    trump_suit = CARD_SUIT[state.trump_card]

    masters = [
        i
        for i in range(len(hand))
        if _is_master_fast(hand[i], hand=hand, trump_card=state.trump_card, seen_cards_onehot=seen_cards_onehot)
    ]
    if masters:
        richest = max(masters, key=lambda i: (CARD_POINTS[hand[i]], CARD_STRENGTH[hand[i]]))
        card_id = hand[richest]
        cashable_now = len(state.deck) <= 0 or (
            CARD_SUIT[card_id] != trump_suit and CARD_POINTS[card_id] >= _TRUMP_SAVER_CARICO_POINTS
        )
        if cashable_now:
            return richest

    return min(
        range(len(hand)),
        key=lambda i: (
            1 if CARD_POINTS[hand[i]] >= _TRUMP_SAVER_CARICO_POINTS else 0,
            CARD_POINTS[hand[i]],
            1 if CARD_SUIT[hand[i]] == trump_suit else 0,
            CARD_STRENGTH[hand[i]],
        ),
    )


def _choose_fast_trump_saver_response_card_index(state: Fast2PState, *, player_index: int) -> int:
    """
    Versione numerica della risposta di `heuristic_trump_saver`.

    Stesse priorità del dominio:
    1. vincere senza briscola -> incassa il massimo dei punti (a parità conserva la più forte);
    2. con briscola: taglia SEMPRE il carico, taglia piatti medi (>=3 punti) solo con briscolina
       a 0 punti, mai piatti poveri durante le pescate; a mazzo vuoto prende comunque;
    3. altrimenti scarto economico.
    """
    hand = state.hands[player_index]
    lead_card = state.table_cards[0]
    lead_player = state.table_players[0]
    trump_suit = CARD_SUIT[state.trump_card]

    winning_non_trumps: list[int] = []
    winning_trumps: list[int] = []
    for i, card_id in enumerate(hand):
        if not _fast_card_wins_trick(
            lead_card=lead_card,
            lead_player=lead_player,
            response_card=card_id,
            response_player=player_index,
            trump_card=state.trump_card,
        ):
            continue
        if CARD_SUIT[card_id] == trump_suit:
            winning_trumps.append(i)
        else:
            winning_non_trumps.append(i)

    if winning_non_trumps:
        # Incassa: massimo punti; a parità conserva la carta più forte per dopo (strength minima).
        return max(winning_non_trumps, key=lambda idx: (CARD_POINTS[hand[idx]], -CARD_STRENGTH[hand[idx]]))

    if winning_trumps:
        best_trump_idx = min(winning_trumps, key=lambda idx: (CARD_POINTS[hand[idx]], CARD_STRENGTH[hand[idx]]))
        best_trump = hand[best_trump_idx]
        lead_points = CARD_POINTS[lead_card]

        if len(state.deck) <= 0:
            # Endgame: senza pescate prendere (e guidare la prossima) vale quasi sempre.
            return best_trump_idx
        if lead_points >= _TRUMP_SAVER_CARICO_POINTS:
            # Il carico avversario si taglia SEMPRE (l'exploit-chiave, anche con l'asso di briscola).
            return best_trump_idx
        if lead_points >= 3 and CARD_POINTS[best_trump] == 0:
            # Piatto medio (re/cavallo): solo se il taglio è gratis (briscolina a 0 punti).
            return best_trump_idx
        # Piatti poveri durante le pescate: la briscola si conserva, si scarta.

    return min(
        range(len(hand)),
        key=lambda idx: (
            CARD_POINTS[hand[idx]],
            1 if CARD_SUIT[hand[idx]] == trump_suit else 0,
            CARD_STRENGTH[hand[idx]],
        ),
    )


def choose_fast_card_index(
    agent_name: str,
    state: Fast2PState,
    player_index: int,
    *,
    rng: random.Random,
    seen_cards_onehot: tuple[int, ...] | None = None,
) -> int:
    """
    Sceglie una carta usando solo lo stato numerico fast.

    Gli agenti implementati qui devono consumare RNG nello stesso modo degli agenti canonici equivalenti,
    così i test possono confrontare esiti fast/domain con lo stesso seed.
    """
    hand = state.hands[player_index]
    if not hand:
        raise ValueError("Mano vuota: nessuna azione possibile")

    if agent_name == "random":
        return rng.randrange(len(hand))

    if agent_name == "greedy_points":
        best_points = max(CARD_POINTS[card_id] for card_id in hand)
        candidates = [i for i, card_id in enumerate(hand) if CARD_POINTS[card_id] == best_points]
        return candidates[rng.randrange(len(candidates))]

    if agent_name == "heuristic_v1":
        if not state.table_cards:
            return _choose_fast_heuristic_lead_card_index(state, player_index=player_index)
        return _choose_fast_heuristic_v1_response_card_index(state, player_index=player_index)

    if agent_name == "heuristic_v2":
        if not state.table_cards:
            return _choose_fast_heuristic_lead_card_index(state, player_index=player_index)
        if seen_cards_onehot is None:
            raise ValueError("heuristic_v2 fast richiede `seen_cards_onehot`")
        return _choose_fast_heuristic_v2_response_card_index(
            state,
            player_index=player_index,
            seen_cards_onehot=seen_cards_onehot,
        )

    if agent_name == "heuristic_trump_saver":
        if not state.table_cards:
            # Solo la scelta lead usa il card counting (check "master"); la risposta no.
            if seen_cards_onehot is None:
                raise ValueError("heuristic_trump_saver fast richiede `seen_cards_onehot`")
            return _choose_fast_trump_saver_lead_card_index(
                state,
                player_index=player_index,
                seen_cards_onehot=seen_cards_onehot,
            )
        return _choose_fast_trump_saver_response_card_index(state, player_index=player_index)

    _validate_fast_agent_name(agent_name)
    raise AssertionError("Agente validato ma non implementato nel path fast")


def _winner_index_fast_2p(state: Fast2PState) -> int | None:
    """Ritorna 0/1 se c'è un vincitore, altrimenti None (pareggio)."""
    if state.points[0] > state.points[1]:
        return 0
    if state.points[1] > state.points[0]:
        return 1
    return None


def play_one_fast_game_2p(
    agent0_name: str,
    agent1_name: str,
    *,
    rng: random.Random,
    game_seed: int,
) -> Fast2PState:
    """
    Simula una singola partita 2-player con il fast path.

    Supporta agenti numerici fast-compatible senza costruire `PlayerObservation`.
    """
    _validate_fast_agent_name(agent0_name)
    _validate_fast_agent_name(agent1_name)

    state = new_fast_2p_state(seed=game_seed)
    seen = [0] * 40
    seen[state.trump_card] = 1
    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        agent_name = agent0_name if current == 0 else agent1_name
        card_index = choose_fast_card_index(agent_name, state, current, rng=rng, seen_cards_onehot=tuple(seen))
        result = step_fast_2p(state, player_index=current, card_index=card_index)
        seen[result.played_card] = 1

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita fast non termina")

    return state


def evaluate_fast_match_2p(
    agent0_name: str,
    agent1_name: str,
    *,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
) -> MatchStats:
    """
    Valuta due agenti fast-compatible usando il fast path.

    La semantica di seed replica `evaluate_match_2p`: un RNG per gli shuffle e uno per le azioni.
    """
    _validate_fast_agent_name(agent0_name)
    _validate_fast_agent_name(agent1_name)

    rng_game = random.Random(seed)
    rng_action = random.Random(seed ^ 0x9E3779B9)

    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_games)]
    if len(seeds) < num_games:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_games}, ottenuti {len(seeds)}")

    wins0 = 0
    wins1 = 0
    draws = 0
    sum0 = 0
    sum1 = 0
    sum_diff = 0

    for i in range(num_games):
        final_state = play_one_fast_game_2p(agent0_name, agent1_name, rng=rng_action, game_seed=seeds[i])
        p0, p1 = final_state.points[0], final_state.points[1]
        sum0 += p0
        sum1 += p1
        sum_diff += p0 - p1

        winner = _winner_index_fast_2p(final_state)
        if winner is None:
            draws += 1
        elif winner == 0:
            wins0 += 1
        else:
            wins1 += 1

    return MatchStats(
        num_games=num_games,
        agent0_name=agent0_name,
        agent1_name=agent1_name,
        wins_agent0=wins0,
        wins_agent1=wins1,
        draws=draws,
        avg_points_agent0=sum0 / num_games if num_games else 0.0,
        avg_points_agent1=sum1 / num_games if num_games else 0.0,
        avg_point_diff_agent0_minus_agent1=sum_diff / num_games if num_games else 0.0,
    )


def evaluate_fast_seat_fair_match_2p(
    agent_a_name: str,
    agent_b_name: str,
    *,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
) -> SeatFairStats:
    """
    Valuta due agenti fast-compatible in modalità seat-fair usando il fast path.

    `num_games` deve essere pari, come nel path canonico.
    """
    _validate_fast_agent_name(agent_a_name)
    _validate_fast_agent_name(agent_b_name)
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari (giochiamo a coppie).")

    rng_game = random.Random(seed)
    rng_action = random.Random(seed ^ 0x9E3779B9)
    num_pairs = num_games // 2

    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_pairs)]
    if len(seeds) < num_pairs:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_pairs}, ottenuti {len(seeds)}")

    wins_a = 0
    wins_b = 0
    draws = 0
    sum_a = 0
    sum_b = 0
    sum_diff = 0
    sum_sq_diff = 0
    sum_sq_pair_diff = 0
    sum_sq_pair_score = 0.0

    for i in range(num_pairs):
        game_seed = seeds[i]

        s1 = play_one_fast_game_2p(agent_a_name, agent_b_name, rng=rng_action, game_seed=game_seed)
        p0, p1 = s1.points[0], s1.points[1]
        sum_a += p0
        sum_b += p1
        diff1 = p0 - p1
        sum_diff += diff1
        sum_sq_diff += diff1 * diff1
        w = _winner_index_fast_2p(s1)
        if w is None:
            draws += 1
            score1 = 0.5
        elif w == 0:
            wins_a += 1
            score1 = 1.0
        else:
            wins_b += 1
            score1 = 0.0

        s2 = play_one_fast_game_2p(agent_b_name, agent_a_name, rng=rng_action, game_seed=game_seed)
        p0, p1 = s2.points[0], s2.points[1]
        sum_a += p1
        sum_b += p0
        diff2 = p1 - p0
        sum_diff += diff2
        sum_sq_diff += diff2 * diff2
        w = _winner_index_fast_2p(s2)
        if w is None:
            draws += 1
            score2 = 0.5
        elif w == 0:
            wins_b += 1
            score2 = 0.0
        else:
            wins_a += 1
            score2 = 1.0

        # Accumulo per COPPIA (unità statistica indipendente), come nel path canonico.
        pair_diff = diff1 + diff2
        sum_sq_pair_diff += pair_diff * pair_diff
        pair_score = (score1 + score2) / 2.0
        sum_sq_pair_score += pair_score * pair_score

    return SeatFairStats(
        num_games=num_games,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        wins_agent_a=wins_a,
        wins_agent_b=wins_b,
        draws=draws,
        avg_points_agent_a=sum_a / num_games if num_games else 0.0,
        avg_points_agent_b=sum_b / num_games if num_games else 0.0,
        avg_point_diff_agent_a_minus_agent_b=sum_diff / num_games if num_games else 0.0,
        sum_sq_point_diff_agent_a_minus_agent_b=float(sum_sq_diff),
        sum_sq_pair_point_diff_agent_a_minus_agent_b=float(sum_sq_pair_diff),
        sum_sq_pair_score_agent_a=float(sum_sq_pair_score),
    )
