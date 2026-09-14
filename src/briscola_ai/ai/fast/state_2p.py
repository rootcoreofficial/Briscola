"""
Motore numerico sperimentale per simulazioni Briscola 2-player.

Questo modulo NON sostituisce il dominio canonico (`domain.engine.step`).
Serve come base per un futuro core veloce, più adatto a parallelizzazione spinta
e a un eventuale JIT con Numba:

- carte come interi `0..39`
- stato mutabile e compatto
- niente `Card`, `Enum`, dataclass frozen o `replace` nel loop caldo

La regola architetturale resta: prima test di equivalenza col dominio canonico,
poi eventuale integrazione in training/evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..card_tables import CARD_NUMBER_BY_ID, CARD_POINTS_BY_ID, CARD_STRENGTH_BY_ID, CARD_SUIT_BY_ID

ACTION_DIM = 40

# Tabelle numeriche derivate dal dominio canonico: vedi `ai/card_tables.py` (fonte unica).
# Gli alias locali mantengono i nomi storici usati in tutto il fast path.
CARD_SUIT: tuple[int, ...] = CARD_SUIT_BY_ID
CARD_NUMBER: tuple[int, ...] = CARD_NUMBER_BY_ID
CARD_POINTS: tuple[int, ...] = CARD_POINTS_BY_ID
CARD_STRENGTH: tuple[int, ...] = CARD_STRENGTH_BY_ID


@dataclass(slots=True)
class Fast2PState:
    """
    Stato mutabile minimale per una partita 2-player.

    Campi:
        deck: carte residue; come nel dominio canonico si pesca con `pop()` dalla fine.
        hands: due mani, ognuna lista di card_id.
        points: punti raccolti da player 0/1.
        trump_card: card_id della briscola scoperta.
        table_cards/table_players: carte sul tavolo in ordine di gioco.
        current_turn: player che deve giocare.
    """

    deck: list[int]
    hands: list[list[int]]
    points: list[int]
    trump_card: int
    table_cards: list[int]
    table_players: list[int]
    current_turn: int
    first_player: int
    game_over: bool
    winner_index: int | None
    # Storia delle prese completate, come nel dominio (`TrickRecord`) ma numerica:
    # (lead_card, lead_player, resp_card, winner, points). Alimenta l'encoder v4.
    trick_history: list[tuple[int, int, int, int, int]]


@dataclass(frozen=True, slots=True)
class Fast2PStepResult:
    """Esito compatto di uno step fast."""

    played_card: int
    player: int
    trick_completed: bool
    trick_winner: int | None
    cards_dealt: bool


def new_fast_2p_state(*, seed: int = 0) -> Fast2PState:
    """
    Crea uno stato iniziale 2-player equivalente a `domain.state.new_game_state`.

    Manteniamo lo stesso ordine canonico del mazzo e lo stesso `random.Random(seed).shuffle`.
    """
    deck = list(range(ACTION_DIM))
    rng = random.Random(seed)
    rng.shuffle(deck)

    hands: list[list[int]] = [[], []]
    for _ in range(3):
        for player_index in range(2):
            hands[player_index].append(deck.pop())

    trump_card = deck.pop()
    deck.insert(0, trump_card)

    return Fast2PState(
        deck=deck,
        hands=hands,
        points=[0, 0],
        trump_card=trump_card,
        table_cards=[],
        table_players=[],
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        trick_history=[],
    )


def fast_who_wins_trick_2p(
    *, first_card: int, first_player: int, second_card: int, second_player: int, trump_card: int
) -> int:
    """
    Determina il vincitore di una presa 2-player.

    Regole uguali al dominio:
    - se almeno una carta è briscola, vince la briscola più forte;
    - altrimenti vince la carta più forte del seme di uscita.
    """
    trump_suit = CARD_SUIT[trump_card]
    first_suit = CARD_SUIT[first_card]
    second_suit = CARD_SUIT[second_card]

    first_is_trump = first_suit == trump_suit
    second_is_trump = second_suit == trump_suit
    if first_is_trump or second_is_trump:
        if first_is_trump and not second_is_trump:
            return first_player
        if second_is_trump and not first_is_trump:
            return second_player
        return first_player if CARD_STRENGTH[first_card] >= CARD_STRENGTH[second_card] else second_player

    if second_suit != first_suit:
        return first_player
    return first_player if CARD_STRENGTH[first_card] >= CARD_STRENGTH[second_card] else second_player


def step_fast_2p(state: Fast2PState, *, player_index: int, card_index: int) -> Fast2PStepResult:
    """
    Applica una giocata mutando `state`.

    Questo path assume input valido, come accade nei loop di training/evaluation dove gli agenti
    scelgono sempre un indice in mano. Le validazioni pesanti restano nel dominio canonico.
    """
    if state.game_over:
        raise ValueError("Partita già terminata")
    if player_index != state.current_turn:
        raise ValueError("Non è il turno del giocatore richiesto")
    hand = state.hands[player_index]
    if card_index < 0 or card_index >= len(hand):
        raise ValueError(f"Azione non valida: {card_index}")

    played_card = hand.pop(card_index)
    state.table_cards.append(played_card)
    state.table_players.append(player_index)

    if len(state.table_cards) == 1:
        state.current_turn = 1 - state.current_turn
        return Fast2PStepResult(
            played_card=played_card,
            player=player_index,
            trick_completed=False,
            trick_winner=None,
            cards_dealt=False,
        )

    first_card, second_card = state.table_cards
    first_player, second_player = state.table_players
    winner = fast_who_wins_trick_2p(
        first_card=first_card,
        first_player=first_player,
        second_card=second_card,
        second_player=second_player,
        trump_card=state.trump_card,
    )
    trick_points = CARD_POINTS[first_card] + CARD_POINTS[second_card]
    state.points[winner] += trick_points
    state.trick_history.append((first_card, first_player, second_card, winner, trick_points))

    state.table_cards.clear()
    state.table_players.clear()

    cards_dealt = False
    if state.deck:
        cards_dealt = True
        for i in range(2):
            player_to_deal = (winner + i) % 2
            if not state.deck:
                break
            state.hands[player_to_deal].append(state.deck.pop())

    state.game_over = not state.hands[0] and not state.hands[1]
    if state.game_over:
        if state.points[0] > state.points[1]:
            state.winner_index = 0
        elif state.points[1] > state.points[0]:
            state.winner_index = 1
        else:
            state.winner_index = None

    state.first_player = winner
    state.current_turn = winner
    return Fast2PStepResult(
        played_card=played_card,
        player=player_index,
        trick_completed=True,
        trick_winner=winner,
        cards_dealt=cards_dealt,
    )


def play_random_fast_2p(*, seed: int, action_seed: int = 0) -> Fast2PState:
    """
    Smoke helper: gioca una partita fast con scelte casuali valide.

    Utile per benchmark e test non-policy-specific.
    """
    state = new_fast_2p_state(seed=seed)
    rng = random.Random(action_seed)
    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        hand_size = len(state.hands[state.current_turn])
        card_index = rng.randrange(hand_size)
        step_fast_2p(state, player_index=state.current_turn, card_index=card_index)
    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita fast non termina")
    return state
