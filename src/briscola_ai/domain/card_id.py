"""
Identificatore canonico (0..39) per le 40 carte del mazzo.

Perché serve
------------
Nel training ML usiamo spesso uno spazio azioni fisso "40 carte":
ogni carta del mazzo ha un id stabile in [0, 39].

Questo mapping è utile anche nel dominio per costruire feature di “card counting lecito”
(carte già viste: tavolo + prese + briscola scoperta) senza dipendere da DTO o moduli ML.

Convenzione
-----------
`card_id = suit_index * 10 + (number - 1)` dove:
- suit_index segue l'ordine di `Suit` (clubs, cups, coins, swords)
- number è `Rank.number` in [1,10]
"""

from __future__ import annotations

from .models import Card, Rank, Suit

SUIT_ORDER: tuple[Suit, ...] = (Suit.CLUBS, Suit.CUPS, Suit.COINS, Suit.SWORDS)
SUIT_TO_INDEX = {s: i for i, s in enumerate(SUIT_ORDER)}


def card_to_id(card: Card) -> int:
    """Converte una `Card` in un id canonico in [0,39]."""
    suit_index = SUIT_TO_INDEX[card.suit]
    number = int(card.rank.number)
    if number < 1 or number > 10:
        raise ValueError(f"Rank.number fuori range: {number} (atteso 1..10)")
    return suit_index * 10 + (number - 1)


def id_to_card(card_id: int) -> Card:
    """Inverso di `card_to_id`."""
    if card_id < 0 or card_id >= 40:
        raise ValueError(f"card_id fuori range: {card_id} (atteso 0..39)")
    suit_index = card_id // 10
    number = (card_id % 10) + 1
    suit = SUIT_ORDER[suit_index]
    rank = next(r for r in Rank if r.number == number)
    return Card(suit=suit, rank=rank)
