"""
Transizione pura `step(state, action)` (Phase 2B).

Questa implementazione codifica le regole della Briscola lavorando su `GameState` immutabile.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import Card
from .rules import who_wins_trick
from .state import GameState, PlayerState, TrickRecord


@dataclass(frozen=True, slots=True)
class PlayCardAction:
    """Azione: il giocatore `player_index` gioca la carta `card_index` dalla propria mano."""

    player_index: int
    card_index: int


@dataclass(frozen=True, slots=True)
class StepResult:
    """
    Output della transizione.

    È pensato per uso didattico e per mantenere parità con i payload API:
    - `played_card`: carta giocata in questo step
    - `trick_completed`: True quando la mano si completa
    - `trick_cards`: snapshot delle carte giocate nella mano (ordine di gioco)
    - `trick_winner`: vincitore della mano (indice player)
    - `cards_dealt`: True se (2-player) sono state pescate nuove carte dopo la mano
    - `error`: stringa se l'azione è invalida
    """

    played_card: Card | None
    player: int | None
    trick_completed: bool
    trick_winner: int | None
    trick_cards: tuple[tuple[Card, int], ...]
    cards_dealt: bool
    error: str | None = None


def _valid_actions(state: GameState) -> list[int]:
    """In Briscola qualunque carta in mano è giocabile."""
    if state.game_over:
        return []
    return list(range(len(state.players[state.current_turn].hand)))


def step(state: GameState, action: PlayCardAction) -> tuple[GameState, StepResult]:
    """
    Applica un'azione e ritorna (nuovo_stato, risultato).
    """
    if state.game_over:
        return state, StepResult(
            played_card=None,
            player=None,
            trick_completed=False,
            trick_winner=None,
            trick_cards=tuple(),
            cards_dealt=False,
            error="Partita già terminata",
        )

    if action.player_index != state.current_turn:
        return state, StepResult(
            played_card=None,
            player=None,
            trick_completed=False,
            trick_winner=None,
            trick_cards=tuple(),
            cards_dealt=False,
            error="Non è il turno del giocatore richiesto",
        )

    valid = _valid_actions(state)
    if action.card_index not in valid:
        return state, StepResult(
            played_card=None,
            player=None,
            trick_completed=False,
            trick_winner=None,
            trick_cards=tuple(),
            cards_dealt=False,
            error=f"Azione non valida: {action.card_index}",
        )

    players = list(state.players)
    current_player = players[state.current_turn]
    hand = list(current_player.hand)
    played_card = hand.pop(action.card_index)
    players[state.current_turn] = PlayerState(
        name=current_player.name,
        hand=tuple(hand),
        captured_cards=current_player.captured_cards,
        points=current_player.points,
    )

    table_cards = list(state.table_cards)
    table_cards.append((played_card, state.current_turn))

    # Mano non completa: passa al prossimo player
    if len(table_cards) != state.num_players:
        new_state = replace(
            state,
            players=tuple(players),
            table_cards=tuple(table_cards),
            current_turn=(state.current_turn + 1) % state.num_players,
            winner_index=None,
            winning_team=None,
        )
        return new_state, StepResult(
            played_card=played_card,
            player=state.current_turn,
            trick_completed=False,
            trick_winner=None,
            trick_cards=tuple(table_cards),
            cards_dealt=False,
        )

    # Mano completa: determina vincitore e aggiorna punti/carte raccolte
    trick_cards = tuple(table_cards)
    trump_suit = state.trump_card.suit if state.trump_card else None
    winner = who_wins_trick(trick_cards, trump_suit)

    # Registra la presa nella storia PUBBLICA (ordine di gioco + attribuzione + vincitore):
    # è ciò che ogni giocatore ha visto al tavolo, e alimenta l'encoder v4 (opponent modeling).
    trick_record = TrickRecord(
        cards=trick_cards,
        winner_index=winner,
        points=sum(card.rank.points for card, _ in trick_cards),
    )

    captured_cards = [card for card, _ in trick_cards]
    winner_player = players[winner]
    new_captured = list(winner_player.captured_cards) + captured_cards
    new_points = sum(card.rank.points for card in new_captured)
    players[winner] = PlayerState(
        name=winner_player.name,
        hand=winner_player.hand,
        captured_cards=tuple(new_captured),
        points=new_points,
    )

    # Pescata post-mano (solo 2-player): dal vincitore, pescando dalla fine del deck
    deck = list(state.deck)
    cards_dealt = False
    if not state.is_team_game and len(deck) > 0:
        cards_dealt = True
        for i in range(state.num_players):
            player_idx = (winner + i) % state.num_players
            if not deck:
                break
            drawn = deck.pop()
            p = players[player_idx]
            players[player_idx] = PlayerState(
                name=p.name,
                hand=tuple(list(p.hand) + [drawn]),
                captured_cards=p.captured_cards,
                points=p.points,
            )

    # Fine partita: tutte le mani vuote
    game_over = all(len(p.hand) == 0 for p in players)
    winner_index: int | None = None
    winning_team: int | None = None
    if game_over:
        if state.is_team_game and state.teams:
            team0 = sum(players[i].points for i in state.teams[0])
            team1 = sum(players[i].points for i in state.teams[1])
            if team0 > team1:
                winning_team = 0
            elif team1 > team0:
                winning_team = 1
        else:
            if players[0].points > players[1].points:
                winner_index = 0
            elif players[1].points > players[0].points:
                winner_index = 1

    new_state = replace(
        state,
        players=tuple(players),
        deck=tuple(deck),
        table_cards=tuple(),  # tavolo ripulito
        first_player=winner,
        current_turn=winner,
        game_over=game_over,
        winner_index=winner_index,
        winning_team=winning_team,
        trick_history=state.trick_history + (trick_record,),
    )

    return new_state, StepResult(
        played_card=played_card,
        player=action.player_index,
        trick_completed=True,
        trick_winner=winner,
        trick_cards=trick_cards,
        cards_dealt=cards_dealt,
    )
