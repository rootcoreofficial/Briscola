"""
Serializzazione JSON-safe del `GameState` (per game store condiviso, es. Redis).

Le carte sono codificate come id canonici 0..39 (`card_to_id`/`id_to_card`), così il dump è
compatto e privo di enum. Funzioni pure e testabili: il dominio non dipende da Redis/JSON, qui
forniamo solo la conversione `GameState <-> dict`.
"""

from __future__ import annotations

from typing import Any

from .card_id import card_to_id, id_to_card
from .state import GameState, PlayerState, TrickRecord

# Versione dello schema di serializzazione (per migrazioni future).
# Storia: 1 = baseline; 2 = aggiunge `trick_history` (encoder v4 / opponent modeling).
SERIALIZATION_SCHEMA = 2


def _player_to_dict(player: PlayerState) -> dict[str, Any]:
    return {
        "name": player.name,
        "hand": [card_to_id(c) for c in player.hand],
        "captured_cards": [card_to_id(c) for c in player.captured_cards],
        "points": int(player.points),
    }


def _player_from_dict(data: dict[str, Any]) -> PlayerState:
    return PlayerState(
        name=str(data["name"]),
        hand=tuple(id_to_card(int(i)) for i in data["hand"]),
        captured_cards=tuple(id_to_card(int(i)) for i in data["captured_cards"]),
        points=int(data["points"]),
    )


def game_state_to_dict(state: GameState) -> dict[str, Any]:
    """Converte un `GameState` in un dict JSON-serializzabile."""
    return {
        "schema": SERIALIZATION_SCHEMA,
        "num_players": int(state.num_players),
        "is_team_game": bool(state.is_team_game),
        "teams": [list(t) for t in state.teams] if state.teams is not None else None,
        "players": [_player_to_dict(p) for p in state.players],
        "deck": [card_to_id(c) for c in state.deck],
        "trump_card": (card_to_id(state.trump_card) if state.trump_card is not None else None),
        "table_cards": [[card_to_id(card), int(player_idx)] for card, player_idx in state.table_cards],
        "current_turn": int(state.current_turn),
        "first_player": int(state.first_player),
        "game_over": bool(state.game_over),
        "winner_index": state.winner_index,
        "winning_team": state.winning_team,
        "trick_history": [
            {
                "cards": [[card_to_id(card), int(player_idx)] for card, player_idx in record.cards],
                "winner_index": int(record.winner_index),
                "points": int(record.points),
            }
            for record in state.trick_history
        ],
    }


def game_state_from_dict(data: dict[str, Any]) -> GameState:
    """
    Ricostruisce un `GameState` da un dict prodotto da `game_state_to_dict`.

    Nota sulla fiducia nell'input: il dict arriva dallo store interno (Redis/in-memory),
    quindi non validiamo qui tutti gli invarianti di dominio (40 carte uniche, ecc.).
    Il campo `schema` però viene VERIFICATO: se in futuro lo schema evolve, un processo
    vecchio che legge un dump nuovo deve fallire in modo esplicito, non ricostruire
    silenziosamente uno stato sbagliato.
    """
    schema = data.get("schema")
    if schema not in (1, SERIALIZATION_SCHEMA):
        raise ValueError(f"Schema di serializzazione non supportato: {schema!r} (atteso 1..{SERIALIZATION_SCHEMA})")

    # Migrazione schema 1 -> 2: i dump legacy non hanno `trick_history`. Ricostruirla non è
    # possibile (l'ordine intra-presa è perso), quindi la storia riparte vuota: le feature v4
    # per quelle sessioni in corso saranno parziali, ma lo stato di gioco resta corretto.
    trick_history_raw = data.get("trick_history", [])
    trick_history = tuple(
        TrickRecord(
            cards=tuple((id_to_card(int(cid)), int(player_idx)) for cid, player_idx in record["cards"]),
            winner_index=int(record["winner_index"]),
            points=int(record["points"]),
        )
        for record in trick_history_raw
    )

    teams_raw = data.get("teams")
    teams: Any = None
    if teams_raw is not None:
        teams = tuple(tuple(int(x) for x in pair) for pair in teams_raw)

    trump_raw = data.get("trump_card")
    winner_index = data.get("winner_index")
    winning_team = data.get("winning_team")

    return GameState(
        num_players=int(data["num_players"]),
        is_team_game=bool(data["is_team_game"]),
        teams=teams,
        players=tuple(_player_from_dict(p) for p in data["players"]),
        deck=tuple(id_to_card(int(i)) for i in data["deck"]),
        trump_card=(id_to_card(int(trump_raw)) if trump_raw is not None else None),
        table_cards=tuple((id_to_card(int(cid)), int(player_idx)) for cid, player_idx in data["table_cards"]),
        current_turn=int(data["current_turn"]),
        first_player=int(data["first_player"]),
        game_over=bool(data["game_over"]),
        winner_index=int(winner_index) if winner_index is not None else None,
        winning_team=int(winning_team) if winning_team is not None else None,
        trick_history=trick_history,
    )
