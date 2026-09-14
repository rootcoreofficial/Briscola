"""
Builder di DTO “view” a partire dallo stato di dominio.

Perché esiste questo modulo?
----------------------------
La conversione `DomainGameState -> DTO (HTTP/WS)` è necessaria in più contesti:
- backend FastAPI (snapshot via HTTP e WS),
- pipeline dati (self-play / replay) che vogliono loggare observation coerenti
  con il contratto DTO.

Per evitare duplicazione e drift, centralizziamo qui i builder.
"""

from __future__ import annotations

from ..domain.card_id import card_to_id
from ..domain.state import GameState as DomainGameState
from .dto import CardDTO, GameStateDTO, ObservationDTO, PlayerInfoDTO, PlayerStateDTO, TableCardDTO, TrickRecordDTO


def build_observation_dto(state: DomainGameState, player_index: int, server_version: int) -> ObservationDTO:
    """
    Costruisce un ObservationDTO dal dominio (stato puro).

    Questa funzione centralizza la conversione da stato di gioco a payload WS/HTTP,
    garantendo che il formato sia coerente con il contratto DTO.
    Supporta sia modalità 2-player che 4-player.
    """
    if player_index < 0 or player_index >= state.num_players:
        raise ValueError(f"L'indice giocatore deve essere compreso tra 0 e {state.num_players - 1}")

    me = state.players[player_index]
    my_turn = state.current_turn == player_index

    # Converti carte in mano
    my_hand = [CardDTO.from_domain(card) for card in me.hand]

    # Carta di briscola:
    # - In 2-player la briscola scoperta è "sotto il mazzo" e viene pescata per ultima.
    # - Quando il mazzo è vuoto, mostrare anche la carta qui può risultare confusivo perché
    #   la stessa carta può essere già finita nella mano di un giocatore.
    #   In quel caso inviamo solo `trump_suit` (sempre) e lasciamo `trump_card=None`.
    trump_card = CardDTO.from_domain(state.trump_card) if state.trump_card and len(state.deck) > 0 else None
    trump_suit = state.trump_card.suit.value if state.trump_card else None

    # Converti carte sul tavolo
    table_cards = [TableCardDTO.from_domain(card, idx) for card, idx in state.table_cards]

    # Costruisci lista players (sostituisce player_{n}_* dinamici)
    players: list[PlayerInfoDTO] = []
    for i, player in enumerate(state.players):
        players.append(
            PlayerInfoDTO(
                index=i,
                name=player.name,
                points=player.points,
                hand_size=len(player.hand),
            )
        )

    # Storia pubblica (40 carte): tavolo + prese + briscola scoperta.
    seen = [0] * 40
    if state.trump_card is not None:
        seen[card_to_id(state.trump_card)] = 1
    for card, _ in state.table_cards:
        seen[card_to_id(card)] = 1
    for p in state.players:
        for card in p.captured_cards:
            seen[card_to_id(card)] = 1

    # Carte fuori gioco (40): SOLO tavolo + prese (no briscola scoperta finché pescabile/in mano).
    out_of_play = [0] * 40
    for card, _ in state.table_cards:
        out_of_play[card_to_id(card)] = 1
    for p in state.players:
        for card in p.captured_cards:
            out_of_play[card_to_id(card)] = 1

    # Campi 4-player (None se 2-player)
    my_team = None
    teammate_index = None
    teammate_points = None
    my_team_points = None
    opponent_team_points = None
    if state.is_team_game and state.teams is not None:
        if player_index in state.teams[0]:
            my_team = 0
            teammate_index = state.teams[0][0] if state.teams[0][1] == player_index else state.teams[0][1]
        else:
            my_team = 1
            teammate_index = state.teams[1][0] if state.teams[1][1] == player_index else state.teams[1][1]

        teammate_points = state.players[teammate_index].points if teammate_index is not None else 0
        my_team_points = sum(state.players[i].points for i in state.teams[my_team]) if my_team is not None else 0
        opponent_team_points = (
            sum(state.players[i].points for i in state.teams[1 - my_team]) if my_team is not None else 0
        )

    # Storia ordinata/attribuita delle prese (pubblica): base dell'encoder v4.
    trick_history = [TrickRecordDTO.from_domain(record) for record in state.trick_history]

    return ObservationDTO(
        server_version=server_version,
        my_index=player_index,
        my_hand=my_hand,
        my_points=me.points,
        my_turn=my_turn,
        first_player=state.first_player,
        trump_card=trump_card,
        trump_suit=trump_suit,
        table_cards=table_cards,
        cards_remaining_in_deck=len(state.deck),
        valid_actions=list(range(len(me.hand))) if my_turn and not state.game_over else [],
        game_over=state.game_over,
        num_players=state.num_players,
        is_team_game=state.is_team_game,
        players=players,
        seen_cards_onehot=seen,
        out_of_play_cards_onehot=out_of_play,
        trick_history=trick_history,
        my_team=my_team,
        teammate_index=teammate_index,
        teammate_points=teammate_points,
        my_team_points=my_team_points,
        opponent_team_points=opponent_team_points,
    )


def build_game_state_dto(state: DomainGameState, server_version: int) -> GameStateDTO:
    """
    Costruisce un GameStateDTO (stato completo) dal dominio.

    Uso previsto:
    - endpoint HTTP `GET /games/{id}` senza `player_index` (debug/spectator)

    Nota sicurezza/fair-play:
    Questo payload contiene tutte le mani e quindi NON deve essere usato da un client
    che rappresenta un singolo giocatore umano.
    """
    # Stesso criterio di `ObservationDTO`: se il mazzo è vuoto, non ripetiamo la carta di briscola.
    trump_card = CardDTO.from_domain(state.trump_card) if state.trump_card and len(state.deck) > 0 else None
    trump_suit = state.trump_card.suit.value if state.trump_card else None
    next_deck_card = CardDTO.from_domain(state.deck[-1]) if state.deck else None
    table_cards = [TableCardDTO.from_domain(card, idx) for card, idx in state.table_cards]

    players: list[PlayerStateDTO] = []
    for i, player in enumerate(state.players):
        players.append(
            PlayerStateDTO(
                index=i,
                name=player.name,
                points=player.points,
                hand=[CardDTO.from_domain(card) for card in player.hand],
                hand_size=len(player.hand),
                captured_cards=[CardDTO.from_domain(card) for card in player.captured_cards],
            )
        )

    teams = list(state.teams) if state.teams is not None else None
    team_0_points = sum(state.players[i].points for i in state.teams[0]) if state.teams is not None else None
    team_1_points = sum(state.players[i].points for i in state.teams[1]) if state.teams is not None else None

    return GameStateDTO(
        server_version=server_version,
        num_players=state.num_players,
        is_team_game=state.is_team_game,
        trump_card=trump_card,
        trump_suit=trump_suit,
        next_deck_card=next_deck_card,
        table_cards=table_cards,
        current_turn=state.current_turn,
        first_player=state.first_player,
        cards_remaining_in_deck=len(state.deck),
        valid_actions=list(range(len(state.players[state.current_turn].hand))) if not state.game_over else [],
        game_over=state.game_over,
        trick_in_progress=len(state.table_cards) > 0,
        trick_size=len(state.table_cards),
        expected_trick_size=state.num_players,
        players=players,
        teams=teams,
        team_0_points=team_0_points,
        team_1_points=team_1_points,
    )
