"""
Modelli del dominio (canonici) per la Briscola (Phase 2B+).

Questo modulo definisce le entità *pure* e stabili che descrivono il gioco:
- `Suit`: i semi del mazzo italiano (bastoni, coppe, denari, spade)
- `Rank`: i ranghi/valori delle carte con punteggio e ordine di forza (per vincere una mano)
- `Card`: una carta (seme + rango)

Nota didattica:
in Briscola esistono due “ordini” diversi che spesso si confondono:
- `points`: i punti che una carta vale quando viene raccolta (Asso=11, Tre=10, Re=4, Cavallo=3, Fante=2)
- `trick_strength`: quanto una carta è “forte” per vincere una mano (Asso > Tre > Re > ...)

Obiettivo architetturale:
questi modelli vivono in `domain/` perché sono parte delle regole del gioco e devono poter
essere riutilizzati senza dipendere dal backend (FastAPI) o dalla UI.
"""

from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    """
    Semi delle carte italiane.

    Internamente usiamo stringhe in inglese (`clubs`, `cups`, ...) perché sono comode
    lato frontend/API; i commenti indicano il nome italiano del seme.
    """

    CLUBS = "clubs"  # bastoni
    CUPS = "cups"  # coppe
    COINS = "coins"  # denari
    SWORDS = "swords"  # spade


class Rank(Enum):
    """
    Ranghi delle carte con:
    - `number`: numero “stampato” (1..10)
    - `points`: punti della Briscola

    Importante: l'ordine di forza in una mano non è l'ordine numerico.
    Per determinare il vincitore di una mano usare `trick_strength`.
    """

    ACE = (1, 11)  # (number, points)
    TWO = (2, 0)
    THREE = (3, 10)
    FOUR = (4, 0)
    FIVE = (5, 0)
    SIX = (6, 0)
    SEVEN = (7, 0)
    JACK = (8, 2)  # fante
    KNIGHT = (9, 3)  # cavallo
    KING = (10, 4)  # re

    def __init__(self, number: int, points: int):
        """
        Inizializza i campi associati al membro Enum.

        Argomenti:
            number: valore numerico (1..10)
            points: punti assegnati quando la carta viene raccolta
        """
        self.number = number
        self.points = points

    @property
    def trick_strength(self) -> int:
        """
        Forza relativa della carta per determinare il vincitore di una mano.

        Nota: in Briscola l'ordine di forza NON coincide con il valore numerico della carta.
        Ordine (dal più forte al più debole): Asso, Tre, Re, Cavallo, Fante, 7, 6, 5, 4, 2.

        Ritorna:
            Un intero dove un valore più alto indica una carta più forte nella mano.
        """
        strength_by_name = {
            "ACE": 10,
            "THREE": 9,
            "KING": 8,
            "KNIGHT": 7,
            "JACK": 6,
            "SEVEN": 5,
            "SIX": 4,
            "FIVE": 3,
            "FOUR": 2,
            "TWO": 1,
        }
        return strength_by_name[self.name]


@dataclass(frozen=True, slots=True)
class Card:
    """
    Rappresenta una singola carta da gioco.

    Una carta è definita da:
    - `suit`: seme
    - `rank`: rango/valore
    """

    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        """Rappresentazione breve, utile per debug/log."""
        return f"{self.rank.name} of {self.suit.value}"

    def __repr__(self) -> str:
        """Rappresentazione non ambigua (ricostruibile) per debug."""
        return f"Card({self.suit.name}, {self.rank.name})"
