"""
DTO (Data Transfer Objects) per i messaggi WebSocket e HTTP.

Questo modulo definisce i modelli Pydantic v2 per tutti i messaggi scambiati
tra backend e frontend. I DTO garantiscono:
- Validazione automatica dei payload
- Documentazione implicita del contratto API
- Serializzazione coerente (no encoder custom)

Note architetturali:
- **Supporto 2-player e 4-player**: ObservationDTO include campi opzionali per
  la modalità a squadre (my_team, teammate_index, ecc.). In 2-player sono None.
- **HTTP e WS allineati**: l'endpoint `GET /games/{id}?player_index=X` restituisce
  lo stesso formato di ObservationDTO usato nei messaggi WS.
  `GET /games/{id}` (senza player_index) restituisce invece un `GameStateDTO`
  (stato completo per debugging/spettatori, include tutte le mani).

Nota: i DTO sono separati dai modelli di dominio (`game/models.py`) per mantenere
la separazione tra logica di gioco e serializzazione API.
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..domain.models import Card
from ..domain.state import TrickRecord as DomainTrickRecord


class CardDTO(BaseModel):
    """
    Rappresentazione JSON di una carta.

    Campi:
    - suit: seme (clubs, cups, coins, swords)
    - rank: nome del rango (ACE, TWO, THREE, ...)
    - number: valore numerico stampato (1-10)
    - points: punti Briscola (11, 10, 4, 3, 2, 0)
    """

    suit: str
    rank: str
    number: int
    points: int

    @classmethod
    def from_domain(cls, card: Card) -> CardDTO:
        """Converte una Card di dominio in CardDTO."""
        return cls(
            suit=card.suit.value,
            rank=card.rank.name,
            number=card.rank.number,
            points=card.rank.points,
        )


class TableCardDTO(BaseModel):
    """
    Una carta sul tavolo con l'indice del giocatore che l'ha giocata.

    Sostituisce il formato tuple `[card, player_index]` con un oggetto esplicito.
    """

    card: CardDTO
    player_index: int

    @classmethod
    def from_domain(cls, card: Card, player_index: int) -> TableCardDTO:
        """Converte una tupla (Card, player_index) in TableCardDTO."""
        return cls(card=CardDTO.from_domain(card), player_index=player_index)


class PlayerInfoDTO(BaseModel):
    """
    Informazioni su un giocatore (visibili a tutti).

    Sostituisce le chiavi dinamiche `player_{n}_points`, `player_{n}_hand_size`.
    """

    index: int
    name: str
    points: int
    hand_size: int


class PlayerStateDTO(BaseModel):
    """
    Stato completo di un giocatore (debug/spectator).

    Include la mano completa e le carte raccolte; NON è adatto ad un client “fair”
    perché rivela informazione nascosta (tutte le carte).
    """

    index: int
    name: str
    points: int
    hand: list[CardDTO]
    hand_size: int
    captured_cards: list[CardDTO]


class TrickRecordDTO(BaseModel):
    """
    Una presa completata, come vista pubblicamente al tavolo.

    Informazione pubblica per costruzione (ogni carta è stata giocata scoperta):
    alimenta l'encoder v4 (opponent modeling) e permette al client/dataset di
    ricostruire la sequenza di gioco senza violare l'anti-cheat.
    """

    cards: list[TableCardDTO]
    winner_index: int
    points: int

    @classmethod
    def from_domain(cls, record: DomainTrickRecord) -> TrickRecordDTO:
        """Converte un `TrickRecord` di dominio nel DTO."""
        return cls(
            cards=[TableCardDTO.from_domain(card, player_idx) for card, player_idx in record.cards],
            winner_index=record.winner_index,
            points=record.points,
        )


class ObservationDTO(BaseModel):
    """
    Snapshot dello stato di gioco dal punto di vista di un giocatore.

    Questo è il messaggio principale inviato via WebSocket e HTTP dopo ogni azione.
    Supporta sia la modalità 2-player che 4-player (a squadre).
    """

    type: Literal["observation"] = "observation"
    server_version: int

    # Identità del giocatore
    my_index: int
    my_hand: list[CardDTO]
    my_points: int
    my_turn: bool
    first_player: int = 0

    # Stato del tavolo
    trump_card: CardDTO | None
    trump_suit: str | None
    table_cards: list[TableCardDTO]
    cards_remaining_in_deck: int

    # Azioni e stato partita
    valid_actions: list[int]
    game_over: bool
    num_players: int
    is_team_game: bool

    # Info sugli altri giocatori (sostituisce player_{n}_*)
    players: list[PlayerInfoDTO]

    # Storia pubblica (card counting lecito): one-hot su 40 carte viste (tavolo + prese + briscola).
    #
    # Nota:
    # - È informazione pubblica, quindi non viola l'anti-cheat.
    # - È opzionale per backward compatibility (client vecchi possono ignorarla).
    seen_cards_onehot: list[int] = Field(default_factory=list)

    # Carte "fuori gioco" (one-hot su 40 carte): SOLO prese + tavolo (no briscola scoperta
    # finché è pescabile/in mano). Distinta da `seen_cards_onehot`: "non più disponibile" vs
    # "vista pubblicamente". Opzionale/default per non rompere payload e dataset vecchi.
    out_of_play_cards_onehot: list[int] = Field(default_factory=list)

    # Storia ordinata e attribuita delle prese completate (pubblica; base dell'encoder v4).
    # Opzionale/default per backward compatibility con client e dataset precedenti.
    trick_history: list[TrickRecordDTO] = Field(default_factory=list)

    # Campi 4-player (opzionali, None in modalità 2-player)
    my_team: int | None = None
    teammate_index: int | None = None
    teammate_points: int | None = None
    my_team_points: int | None = None
    opponent_team_points: int | None = None


class AiCardRevealDTO(BaseModel):
    """
    Messaggio inviato quando l'IA sceglie una carta (prima di giocarla).

    Permette al frontend di mostrare la carta in mano prima di spostarla sul tavolo.
    """

    type: Literal["ai_card_reveal"] = "ai_card_reveal"
    card_index: int
    card: CardDTO
    # Ramo decisionale usato dall'agente, se disponibile. Serve solo alla presentazione/debug.
    decision_type: str | None = None


class TrickResultDTO(BaseModel):
    """
    Messaggio inviato quando una mano si completa.

    Contiene le carte giocate, il vincitore e i punti della mano.
    """

    type: Literal["trick_result"] = "trick_result"
    trick_cards: list[TableCardDTO]
    winner_index: int
    winner_name: str
    points: int
    server_version: int


class GameStateDTO(BaseModel):
    """
    Stato completo della partita (debug/spectator).

    Differenza rispetto a `ObservationDTO`:
    - include tutte le mani dei giocatori e le carte raccolte
    - include dettagli utili al debug (es. `trick_in_progress`, `expected_trick_size`)
    - NON dovrebbe essere usato dalla UI “player-facing” perché non nasconde informazioni.
    """

    type: Literal["game_state"] = "game_state"
    server_version: int

    num_players: int
    is_team_game: bool

    trump_card: CardDTO | None
    trump_suit: str | None
    # Solo debug/spectator: prossima carta pescabile dal mazzo (`deck[-1]` nel dominio).
    # Non compare in ObservationDTO, quindi gli agenti e la vista fair del giocatore non la ricevono.
    next_deck_card: CardDTO | None = None

    table_cards: list[TableCardDTO]

    current_turn: int
    first_player: int

    cards_remaining_in_deck: int
    valid_actions: list[int]

    game_over: bool
    trick_in_progress: bool
    trick_size: int
    expected_trick_size: int

    players: list[PlayerStateDTO]

    # Team-play (solo 4-player)
    teams: list[tuple[int, int]] | None = None
    team_0_points: int | None = None
    team_1_points: int | None = None


class PlayActionResultDTO(BaseModel):
    """
    Risposta HTTP per `POST /api/games/{game_id}/actions`.

    Nota didattica:
    - Il frontend NON usa questo payload per aggiornare la UI: la UI si aggiorna via WebSocket
      tramite `ObservationDTO` + messaggi evento (es. `TrickResultDTO`).
    - Questo DTO esiste per avere un contratto HTTP stabile e documentato (OpenAPI) e per
      evitare encoder JSON custom in backend.
    - In caso di azione invalida, l'endpoint risponde con errore HTTP (es. 400) usando
      `HTTPException(detail=...)` e NON con un campo `error` dentro questo DTO.
    """

    server_version: int

    played_card: CardDTO
    player: int

    trick_completed: bool
    trick_winner: int | None
    trick_size: int
    cards_dealt: bool

    # Presenti solo quando `trick_completed` è True.
    trick_cards: list[TableCardDTO] | None = None
    captured_cards: list[CardDTO] = []


class GameResultDTO(BaseModel):
    """
    Risposta HTTP per `GET /api/games/{game_id}/result`.

    Scopo:
    - esporre un contratto stabile e documentato (OpenAPI) per il risultato finale
    - evitare dizionari "ad hoc" con shape variabile a seconda della modalità di gioco

    Note:
    - in 2-player usiamo `winner_index` (0/1) quando non c'è pareggio.
    - in 4-player (a squadre) usiamo `winning_team` (0/1) quando non c'è pareggio.
    - `points` è sempre una mappa nome->punti individuali (utile per UI/debug).
    """

    type: Literal["game_result"] = "game_result"
    server_version: int

    game_in_progress: bool
    game_over: bool
    is_team_game: bool

    winner: str | None = None
    winner_index: int | None = None
    winning_team: int | None = None

    points: dict[str, int] = {}
    team_points: dict[str, int] | None = None
    point_difference: int | None = None
