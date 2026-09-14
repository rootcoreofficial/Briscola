"""
Osservazioni (information set) per agenti/ML.

Problema
--------
Nel dominio, `GameState` contiene informazione *completa* (mazzo e mani di tutti).
Se passiamo `GameState` direttamente a un agente, una IA potrebbe "barare" leggendo:
- l'ordine del mazzo (`state.deck`)
- le carte in mano agli avversari (`state.players[i].hand`)

Soluzione
---------
Questo modulo definisce una vista *parziale* e “lecita” dello stato:
`PlayerObservation` contiene solo ciò che un giocatore potrebbe conoscere durante una partita.

Obiettivo didattico:
- rendere esplicito il confine tra “stato completo” (debug/replay) e “osservazione”
  (policy/ML), così da evitare leak informativi accidentali.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Card, Suit
from .state import GameState, TrickRecord


def _empty_seen_cards_onehot() -> tuple[int, ...]:
    """Default (backward-compatible): nessuna carta “vista” (lunghezza 40)."""
    return (0,) * 40


def _empty_out_of_play_cards_onehot() -> tuple[int, ...]:
    """Default (backward-compatible): nessuna carta “fuori gioco” (lunghezza 40)."""
    return (0,) * 40


def _card_to_id_fast(card: Card) -> int:
    """
    Converte una carta in id canonico nel path caldo di osservazione.

    Evitiamo il lookup su dict/Enum usato dall'helper pubblico `card_to_id`, perche'
    `make_player_observation` viene chiamata decine di migliaia di volte durante training
    ed evaluation. La convenzione resta identica: `suit_index * 10 + (number - 1)`.
    """
    if card.suit is Suit.CLUBS:
        suit_index = 0
    elif card.suit is Suit.CUPS:
        suit_index = 1
    elif card.suit is Suit.COINS:
        suit_index = 2
    elif card.suit is Suit.SWORDS:
        suit_index = 3
    else:
        raise ValueError(f"Seme non supportato: {card.suit!r}")
    return suit_index * 10 + (int(card.rank.number) - 1)


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """
    Osservazione del gioco dal punto di vista di un singolo giocatore.

    Invarianti (anti-cheat):
    - NON contiene `deck` né informazioni su carte specifiche in mano agli avversari.
    - Contiene solo:
      - mano del giocatore osservante
      - briscola scoperta (se presente)
      - carte sul tavolo (pubbliche) e chi le ha giocate
      - dimensione del mazzo (pubblica)
      - punteggi e dimensioni delle mani (informazione pubblica, dato che le prese sono visibili)

    Nota:
    I campi sono volutamente ridondanti/espliciti per rendere semplice costruire feature per ML.
    """

    num_players: int
    is_team_game: bool
    teams: tuple[tuple[int, int], tuple[int, int]] | None

    player_index: int
    player_name: str

    hand: tuple[Card, ...]

    trump_card: Card | None
    deck_size: int

    table_cards: tuple[tuple[Card, int], ...]
    current_turn: int
    first_player: int

    game_over: bool
    winner_index: int | None  # 2-player
    winning_team: int | None  # 4-player

    players_points: tuple[int, ...]
    players_hand_sizes: tuple[int, ...]

    # Storia pubblica (card counting lecito): one-hot sulle 40 carte.
    #
    # Contiene solo informazione pubblica, quindi è anti-cheat:
    # - carte già giocate e finite nelle prese (captured_cards di tutti)
    # - carte attualmente sul tavolo
    # - briscola scoperta (se presente)
    #
    # Non include mai carte specifiche in mano agli avversari.
    seen_cards_onehot: tuple[int, ...] = field(default_factory=_empty_seen_cards_onehot)

    # Carte "fuori gioco" (one-hot sulle 40 carte): definitivamente non più disponibili.
    #
    # Differenza con `seen_cards_onehot` (volutamente distinte):
    # - `seen_cards_onehot` è "informazione pubblica vista", e include sempre la briscola scoperta;
    # - `out_of_play_cards_onehot` è "carta non più in gioco", quindi SOLO prese + tavolo.
    #   La briscola scoperta NON è qui finché resta pescabile (nel mazzo) o è stata pescata in mano:
    #   compare solo quando finisce in una presa o sul tavolo.
    #
    # È anti-cheat: dipende solo da prese (pubbliche) e tavolo. Default a 40 zeri per backward
    # compatibility con dataset/osservazioni vecchie.
    out_of_play_cards_onehot: tuple[int, ...] = field(default_factory=_empty_out_of_play_cards_onehot)

    # Storia ORDINATA e ATTRIBUITA delle prese completate (vedi `TrickRecord`).
    #
    # È anti-cheat per costruzione: ogni carta di ogni presa è stata giocata a tavolo scoperto
    # davanti a tutti; qui conserviamo solo ciò che un giocatore con memoria perfetta ricorda.
    # È la base dell'encoder v4 (opponent modeling comportamentale): senza ordine e attribuzione,
    # inferenze come "l'avversario ha tagliato a coppe" erano irrappresentabili.
    # Default vuoto per backward compatibility con osservazioni/dataset precedenti.
    trick_history: tuple[TrickRecord, ...] = ()


def make_player_observation(state: GameState, player_index: int) -> PlayerObservation:
    """
    Costruisce l'osservazione lecita per `player_index`.

    Argomenti:
        state: stato completo del dominio (include informazione nascosta)
        player_index: indice del giocatore osservante (0..num_players-1)

    Ritorna:
        Un `PlayerObservation` che può essere passato a policy/ML senza leak informativi.
    """
    if player_index < 0 or player_index >= state.num_players:
        raise ValueError(f"player_index fuori range: {player_index} (num_players={state.num_players})")

    # Costruiamo la storia pubblica (40 carte) senza leak:
    # - prese di tutti i player (pubbliche)
    # - tavolo (pubblico)
    # - briscola scoperta (pubblica)
    seen = [0] * 40
    if state.trump_card is not None:
        seen[_card_to_id_fast(state.trump_card)] = 1
    for card, _ in state.table_cards:
        seen[_card_to_id_fast(card)] = 1
    for p in state.players:
        for card in p.captured_cards:
            seen[_card_to_id_fast(card)] = 1

    # Carte fuori gioco (40): SOLO prese + tavolo, senza la briscola scoperta (che resta in gioco
    # finché è pescabile o in mano). Quando la briscola viene catturata o giocata, finisce qui
    # automaticamente perché appartiene a `captured_cards`/`table_cards`.
    out_of_play = [0] * 40
    for card, _ in state.table_cards:
        out_of_play[_card_to_id_fast(card)] = 1
    for p in state.players:
        for card in p.captured_cards:
            out_of_play[_card_to_id_fast(card)] = 1

    return PlayerObservation(
        num_players=state.num_players,
        is_team_game=state.is_team_game,
        teams=state.teams,
        player_index=player_index,
        player_name=state.players[player_index].name,
        hand=state.players[player_index].hand,
        trump_card=state.trump_card,
        deck_size=len(state.deck),
        table_cards=state.table_cards,
        current_turn=state.current_turn,
        first_player=state.first_player,
        game_over=state.game_over,
        winner_index=state.winner_index,
        winning_team=state.winning_team,
        players_points=tuple(p.points for p in state.players),
        players_hand_sizes=tuple(len(p.hand) for p in state.players),
        seen_cards_onehot=tuple(seen),
        out_of_play_cards_onehot=tuple(out_of_play),
        trick_history=state.trick_history,
    )
