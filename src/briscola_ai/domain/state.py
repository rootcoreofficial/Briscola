"""
Stato "puro" del gioco (Phase 2B).

Obiettivo:
- rendere lo stato esplicito e serializzabile
- abilitare replay deterministico e pipeline ML
- ridurre dipendenze del dominio da FastAPI/DTO/JSON

Nota storica:
in una prima iterazione `Card/Rank/Suit` vivevano fuori da `domain/`.
Ora vivono in `briscola_ai.domain.models` per rendere il dominio autosufficiente.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Card, Rank, Suit


@dataclass(frozen=True, slots=True)
class PlayerState:
    """
    Stato completo di un giocatore.

    Nota:
    - `points` è ridondante rispetto a `captured_cards`, ma tenerlo nello stato rende
      più facile consultare rapidamente lo score in un loop di simulazione.
    """

    name: str
    hand: tuple[Card, ...]
    captured_cards: tuple[Card, ...]
    points: int


@dataclass(frozen=True, slots=True)
class TrickRecord:
    """
    Una presa completata, con l'informazione PUBBLICA che la partita ha mostrato.

    Perché esiste (encoder v4 / opponent modeling):
    storicamente lo stato conservava le carte catturate solo come insieme per giocatore
    (`PlayerState.captured_cards`), perdendo ordine di gioco e attribuzione intra-presa.
    Quella perdita rendeva irrappresentabile l'inferenza comportamentale sulla mano
    avversaria ("ha tagliato a coppe", "non risponde a denari"). Qui registriamo la presa
    così come TUTTI i giocatori l'hanno vista al tavolo: è informazione pubblica per
    costruzione, quindi può entrare nell'osservazione senza violare l'anti-cheat.

    Campi:
        cards: carte nell'ordine di gioco, come coppie (Card, player_index).
        winner_index: chi ha preso.
        points: punti raccolti con questa presa.
    """

    cards: tuple[tuple[Card, int], ...]
    winner_index: int
    points: int


@dataclass(frozen=True, slots=True)
class GameState:
    """
    Stato completo della partita (debug/replay/ML).

    Convenzioni:
    - `deck` è una sequenza da cui peschiamo con `.pop()` (quindi dalla fine).
      In 2-player la briscola scoperta viene inserita in testa al mazzo (indice 0)
      per essere pescata per ultima, replicando l'implementazione attuale.
    - `table_cards` conserva l'ordine di gioco: (Card, player_index).
    - `trick_history` conserva le prese completate in ordine cronologico (vedi
      `TrickRecord`); ha default vuoto per compatibilità con i costruttori esistenti.
    """

    num_players: int
    is_team_game: bool
    teams: tuple[tuple[int, int], tuple[int, int]] | None

    players: tuple[PlayerState, ...]

    deck: tuple[Card, ...]
    trump_card: Card | None

    table_cards: tuple[tuple[Card, int], ...]
    current_turn: int
    first_player: int

    game_over: bool
    winner_index: int | None  # 2-player
    winning_team: int | None  # 4-player

    trick_history: tuple[TrickRecord, ...] = ()


def _create_deck() -> list[Card]:
    """
    Crea un mazzo completo (40 carte) nell'ordine canonico del dominio.

    Nota: l'ordine iniziale è deterministico; la casualità arriva con lo shuffle in `new_game_state`.
    """

    deck: list[Card] = []
    for suit in Suit:
        for rank in Rank:
            deck.append(Card(suit, rank))
    return deck


def new_game_state(num_players: int, player_names: list[str] | None = None, *, seed: int | None = None) -> GameState:
    """
    Crea uno stato iniziale pronto per giocare (shuffle + deal).

    Argomenti:
        num_players: 2 o 4
        player_names: lista nomi (default: Giocatore 1..N)
        seed: seed per lo shuffle del mazzo. Se `None` (default) viene estratto un seed
            casuale: chi ha bisogno di riproducibilità DEVE passarlo esplicitamente.

    Nota sul default:
    in passato il default era `seed=0`, cioè ogni chiamata senza seed produceva la STESSA
    partita. Per una pipeline dati/self-play è un footgun silenzioso (dataset degeneri di
    partite identiche senza alcun warning), quindi il default ora è la casualità esplicita.
    """

    import random

    if num_players not in (2, 4):
        raise ValueError("La Briscola supporta solo 2 o 4 giocatori")

    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**32)

    is_team_game = num_players == 4
    teams = ((0, 2), (1, 3)) if is_team_game else None

    if player_names is None:
        player_names = [f"Giocatore {i + 1}" for i in range(num_players)]
    if len(player_names) != num_players:
        raise ValueError(f"Attesi {num_players} nomi giocatore, ottenuti {len(player_names)}")

    deck = _create_deck()
    rng = random.Random(seed)
    rng.shuffle(deck)

    hands: list[list[Card]] = [[] for _ in range(num_players)]
    captured: list[list[Card]] = [[] for _ in range(num_players)]
    points = [0 for _ in range(num_players)]

    trump_card: Card | None = None

    if is_team_game:
        # 4-player: 10 carte a testa, mazzo completo distribuito
        for _ in range(10):
            for i in range(num_players):
                hands[i].append(deck.pop())
        trump_card = hands[-1][-1]
    else:
        # 2-player: 3 carte a testa + briscola scoperta reinserita in testa al deck
        if len(deck) < 7:
            raise ValueError("Carte insufficienti nel mazzo per distribuire")
        for _ in range(3):
            for i in range(num_players):
                hands[i].append(deck.pop())
        trump_card = deck.pop()
        deck.insert(0, trump_card)

    players = tuple(
        PlayerState(
            name=player_names[i],
            hand=tuple(hands[i]),
            captured_cards=tuple(captured[i]),
            points=points[i],
        )
        for i in range(num_players)
    )

    return GameState(
        num_players=num_players,
        is_team_game=is_team_game,
        teams=teams,
        players=players,
        deck=tuple(deck),
        trump_card=trump_card,
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )
