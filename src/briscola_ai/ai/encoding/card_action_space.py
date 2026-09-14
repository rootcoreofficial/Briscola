"""
Spazio azioni "canonico" per modelli (ML): 40 carte + action mask.

Motivazione
-----------
In Briscola, l'azione è "gioca una carta". Se un modello predice direttamente
l'indice nella mano (0..len(hand)-1), l'output è variabile e dipende dall'ordine
della mano.

Per un primo modello didattico (Behavior Cloning / supervised), è spesso più semplice:
- definire uno spazio azioni fisso: **una classe per ciascuna delle 40 carte**;
- usare una **action mask** che rende selezionabili solo le carte realmente in mano.

Questo modulo implementa:
- una mappatura deterministica `CardDTO -> action_id` in [0, 39]
- helper per costruire la mask a partire da una mano (lista di CardDTO)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..card_tables import POINTS_BY_NUMBER as _POINTS_BY_NUMBER
from ..card_tables import TRICK_STRENGTH_BY_NUMBER as _TRICK_STRENGTH_BY_NUMBER

# Ordine semi (coerente con `Suit` in `domain/models.py`).
SUIT_ORDER: tuple[str, ...] = ("clubs", "cups", "coins", "swords")
SUIT_TO_INDEX = {suit: i for i, suit in enumerate(SUIT_ORDER)}


def action_id_from_suit_number(*, suit: str, number: int) -> int:
    """
    Converte (suit, number) in un action id canonico [0, 39].

    Convenzione:
    - `action_id = suit_index * 10 + (number - 1)`
    - `number` è in [1, 10]
    """
    if suit not in SUIT_TO_INDEX:
        raise ValueError(f"Seme non supportato: {suit!r}. Attesi: {list(SUIT_TO_INDEX.keys())}")
    if number < 1 or number > 10:
        raise ValueError(f"Numero carta fuori range: {number} (atteso 1..10)")
    return SUIT_TO_INDEX[suit] * 10 + (number - 1)


def suit_number_from_action_id(action_id: int) -> tuple[str, int]:
    """Inverso di `action_id_from_suit_number`."""
    if action_id < 0 or action_id >= 40:
        raise ValueError(f"action_id fuori range: {action_id} (atteso 0..39)")
    suit_index = action_id // 10
    number = (action_id % 10) + 1
    return SUIT_ORDER[suit_index], number


def card_dto_to_action_id(card: dict) -> int:
    """
    Converte un `CardDTO` (JSON) in action id.

    `CardDTO` atteso (vedi `backend/dto.py`):
    - { "suit": "cups", "number": 1, ... }
    """
    if not isinstance(card, dict):
        raise TypeError(f"CardDTO atteso come dict, ottenuto: {type(card)}")
    suit = card.get("suit")
    number = card.get("number")
    if not isinstance(suit, str) or not isinstance(number, int):
        raise ValueError(f"CardDTO invalido: suit={suit!r} number={number!r}")
    return action_id_from_suit_number(suit=suit, number=number)


def action_mask_from_hand(hand: list[dict]) -> list[bool]:
    """
    Costruisce una action mask lunga 40 a partire dalla mano (`my_hand`).

    `mask[action_id] == True` se quella carta è presente in mano.
    """
    mask = [False] * 40
    for card in hand:
        mask[card_dto_to_action_id(card)] = True
    return mask


# Feature helpers: punti e "forza" usati spesso nei baseline (costanti per carta).
#
# Nota: i punti sono già disponibili nel `CardDTO`, ma tenere anche queste tabelle
# è comodo per costruire feature consistenti senza dover dipendere dai DTO.
# Le tabelle sono DERIVATE dal dominio canonico via `ai/card_tables.py` (fonte unica):
# i re-export qui sotto mantengono il nome storico usato dagli altri moduli.
POINTS_BY_NUMBER: dict[int, int] = _POINTS_BY_NUMBER
TRICK_STRENGTH_BY_NUMBER: dict[int, int] = _TRICK_STRENGTH_BY_NUMBER


@dataclass(frozen=True, slots=True)
class CardFeatures:
    """Feature per-carta precomputate (punti e forza), indicizzate per action id."""

    points_by_action_id: tuple[int, ...]
    strength_by_action_id: tuple[int, ...]


def build_card_features() -> CardFeatures:
    """Costruisce vettori per-carta (lunghezza 40) indicizzati per action id."""
    points = []
    strength = []
    for action_id in range(40):
        _, number = suit_number_from_action_id(action_id)
        points.append(POINTS_BY_NUMBER[number])
        strength.append(TRICK_STRENGTH_BY_NUMBER[number])
    return CardFeatures(points_by_action_id=tuple(points), strength_by_action_id=tuple(strength))
