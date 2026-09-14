"""
Regole pure del dominio (Phase 2B+).

Questo modulo raccoglie funzioni "piccole" e deterministiche che implementano
regole della Briscola senza dipendere da FastAPI/UI o da uno stato globale.

Perché esiste:
- rende le regole facili da testare in isolamento
- evita di duplicare logica in più punti (engine, backend, script, ...)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import Card, Suit


def who_wins_trick(table_cards: Sequence[tuple[Card, int]], trump_suit: Suit | None) -> int:
    """
    Determina il vincitore di una mano (round) dato l'insieme di carte sul tavolo.

    Regole implementate (Briscola classica):
    1. Il "seme di uscita" (leading suit) è il seme della prima carta giocata.
    2. Se è stata giocata almeno una briscola (trump suit), vince la briscola più alta.
    3. Altrimenti vince la carta più alta del seme di uscita.

    Argomenti:
        table_cards: sequenza di coppie (Card, player_index) nell'ordine di gioco.
        trump_suit: seme di briscola (None solo in casi di stato incompleto/errore).

    Ritorna:
        L'indice del giocatore che ha vinto la mano.
    """
    if not table_cards:
        raise ValueError("table_cards vuoto")

    leading_suit = table_cards[0][0].suit

    # Se ci sono briscole, vince la briscola con trick_strength maggiore.
    trump_cards = [(card, player_idx) for card, player_idx in table_cards if card.suit == trump_suit]
    if trump_cards:
        _winner_card, winner_player = max(trump_cards, key=lambda pair: pair[0].rank.trick_strength)
        return winner_player

    # Altrimenti vale il seme di uscita.
    leading_cards = [(card, player_idx) for card, player_idx in table_cards if card.suit == leading_suit]
    if leading_cards:
        _winner_card, winner_player = max(leading_cards, key=lambda pair: pair[0].rank.trick_strength)
        return winner_player

    # Caso teoricamente irraggiungibile, ma lasciamo un fallback robusto.
    return table_cards[0][1]


def trick_points(cards: Iterable[Card]) -> int:
    """
    Somma i punti di un insieme di carte raccolte in una mano.

    Nota:
    - in Briscola il totale punti del mazzo è sempre 120.
    """
    return sum(card.rank.points for card in cards)
