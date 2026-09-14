"""
Reward shaping (didattico) per training RL.

Perché esiste
-------------
Nei trainer RL (`scripts/train_a2c.py`, `scripts/train_pg.py`) il reward è parte
dell'**ambiente**: la policy continua a vedere solo `PlayerObservation` (anti-cheat),
ma possiamo definire segnali di apprendimento più densi.

Obiettivo di questo modulo
--------------------------
Fornire funzioni *pure* e testabili che calcolano piccoli termini di shaping basati
solo su informazione **lecita** (pubblica + mano del giocatore), così da:
- rendere riproducibile la logica di shaping,
- evitare di introdurre “scorciatoie” che dipendono da informazione nascosta.

Nota:
questo shaping NON cambia le regole del gioco. Serve solo a guidare l'apprendimento.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.models import Card, Suit
from ...domain.observation import PlayerObservation
from ...domain.rules import who_wins_trick


@dataclass(frozen=True, slots=True)
class TrumpOverkillInfo:
    """
    Diagnostica per "overkill briscola" in una singola decisione (secondo di mano).

    Campi:
        applicable: se il caso è applicabile (2-player e secondo di mano, briscola nota, indici validi).
        chosen_is_trump: se la carta scelta è una briscola.
        chosen_wins: se la carta scelta vince la presa.
        winning_trump_exists: se esiste almeno una briscola vincente in mano.
        is_overkill: se la briscola scelta è "più costosa del necessario" (rispetto alla briscola vincente minima).
    """

    applicable: bool
    chosen_is_trump: bool
    chosen_wins: bool
    winning_trump_exists: bool
    is_overkill: bool


def card_conservation_cost(card: Card) -> tuple[int, int]:
    """
    Costo "di conservazione" di una carta: quanto è preziosa da tenere in mano.

    Usiamo un ordine lessicografico:
    - points (carichi) prima
    - trick_strength poi

    Interpretazione:
    una carta con più punti e/o più forza è in media più "preziosa" da conservare, quindi
    usarla quando una più debole sarebbe bastata è un potenziale spreco (overkill).

    Nota architetturale: questa è la definizione CANONICA, condivisa da reward shaping,
    guard anti-overkill dei modelli (`ai/models/bc_model.py`) e metriche decision-quality
    (`ai/evaluation/decision_quality.py`). Prima esistevano tre copie identiche: cambiarne
    una sola avrebbe fatto divergere training, guard runtime e metriche in modo silenzioso.
    """
    return (int(card.rank.points), int(card.rank.trick_strength))


def _trump_cost_tuple(card: Card, *, trump_suit: Suit) -> tuple[int, int]:
    """Come `card_conservation_cost`, ma con la precondizione esplicita che sia una briscola."""
    if card.suit != trump_suit:
        raise ValueError("_trump_cost_tuple atteso su una briscola")
    return card_conservation_cost(card)


def analyze_trump_overkill_second_hand(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    low_lead_points_max: int | None = None,
) -> TrumpOverkillInfo:
    """
    Analizza una decisione e determina se è un “overkill briscola” (secondo di mano).

    Definizione operativa (didattica):
    - si applica solo quando:
      - `len(table_cards) == 1` (siamo secondi di mano)
      - la briscola è nota (`trump_card` presente)
      - `chosen_card_index` è valido
      - opzionale: la carta sul tavolo vale <= `low_lead_points_max`
    - consideriamo solo il caso in cui la carta scelta è una briscola e vince la presa
    - overkill = esiste una briscola vincente in mano con costo minore della briscola scelta.

    Anti-cheat:
    - usa solo `PlayerObservation`: mano del giocatore + carta pubblica sul tavolo + briscola pubblica.
    - non richiede informazioni su ordine del mazzo o mano avversaria.
    """
    if chosen_card_index < 0 or chosen_card_index >= len(observation.hand):
        return TrumpOverkillInfo(
            applicable=False,
            chosen_is_trump=False,
            chosen_wins=False,
            winning_trump_exists=False,
            is_overkill=False,
        )

    if len(observation.table_cards) != 1:
        return TrumpOverkillInfo(
            applicable=False,
            chosen_is_trump=False,
            chosen_wins=False,
            winning_trump_exists=False,
            is_overkill=False,
        )

    if observation.trump_card is None:
        return TrumpOverkillInfo(
            applicable=False,
            chosen_is_trump=False,
            chosen_wins=False,
            winning_trump_exists=False,
            is_overkill=False,
        )

    lead_card, lead_player = observation.table_cards[0]
    if low_lead_points_max is not None and int(lead_card.rank.points) > int(low_lead_points_max):
        return TrumpOverkillInfo(
            applicable=False,
            chosen_is_trump=False,
            chosen_wins=False,
            winning_trump_exists=False,
            is_overkill=False,
        )

    trump_suit = observation.trump_card.suit
    chosen = observation.hand[chosen_card_index]
    chosen_is_trump = chosen.suit == trump_suit
    if not chosen_is_trump:
        return TrumpOverkillInfo(
            applicable=True,
            chosen_is_trump=False,
            chosen_wins=False,
            winning_trump_exists=False,
            is_overkill=False,
        )

    trick_cards = ((lead_card, lead_player), (chosen, observation.player_index))
    chosen_wins = who_wins_trick(trick_cards, trump_suit) == observation.player_index
    if not chosen_wins:
        return TrumpOverkillInfo(
            applicable=True,
            chosen_is_trump=True,
            chosen_wins=False,
            winning_trump_exists=False,
            is_overkill=False,
        )

    winning_trump_costs: list[tuple[int, int]] = []
    for card in observation.hand:
        if card.suit != trump_suit:
            continue
        trick_cards = ((lead_card, lead_player), (card, observation.player_index))
        if who_wins_trick(trick_cards, trump_suit) == observation.player_index:
            winning_trump_costs.append(_trump_cost_tuple(card, trump_suit=trump_suit))

    if not winning_trump_costs:
        return TrumpOverkillInfo(
            applicable=True,
            chosen_is_trump=True,
            chosen_wins=True,
            winning_trump_exists=False,
            is_overkill=False,
        )

    min_cost = min(winning_trump_costs)
    chosen_cost = _trump_cost_tuple(chosen, trump_suit=trump_suit)
    is_overkill = bool(chosen_cost > min_cost)
    return TrumpOverkillInfo(
        applicable=True,
        chosen_is_trump=True,
        chosen_wins=True,
        winning_trump_exists=True,
        is_overkill=is_overkill,
    )


def trump_overkill_penalty(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    beta: float,
    low_lead_points_max: int | None = 2,
) -> float:
    """
    Ritorna una penalità (<=0) per scoraggiare overkill briscola.

    Parametri:
        beta: intensità della penalità. Se `beta <= 0`, la penalità è disattivata.
        low_lead_points_max: se non None, applichiamo la penalità solo quando la carta sul tavolo vale
            al massimo `low_lead_points_max` punti (default: 2), cioè “scarti o quasi”.

    Output:
        0.0 se non applicabile o non overkill; altrimenti `-beta`.

    Nota:
    teniamo il shaping volutamente “soft” (flat penalty) per non distruggere il segnale principale
    (delta punti) e per rendere facile fare sweep su `beta`.
    """
    if float(beta) <= 0.0:
        return 0.0

    info = analyze_trump_overkill_second_hand(
        observation,
        chosen_card_index=chosen_card_index,
        low_lead_points_max=low_lead_points_max,
    )
    if not info.applicable:
        return 0.0
    if not info.is_overkill:
        return 0.0
    return -float(beta)


def trump_overkill_gap_norm(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    low_lead_points_max: int | None = 2,
) -> float:
    """
    Ritorna un "gap" normalizzato (>=0) che misura quanto la briscola scelta è overkill.

    Definizione:
    - si applica solo quando:
      - siamo secondi di mano (`len(table_cards)==1`)
      - la briscola è nota
      - la carta scelta è una briscola e vince la presa
      - opzionale: la carta avversaria sul tavolo vale <= `low_lead_points_max`
    - calcoliamo tra le briscole vincenti in mano il costo minimo (points, strength)
    - gap = (Δpoints/11) + (Δstrength/10), dove:
      - Δpoints = chosen.points - min.points
      - Δstrength = chosen.strength - min.strength

    Interpretazione:
    - 0.0 => non overkill (o non applicabile)
    - valori più alti => briscola scelta molto più "costosa" di una vincente minima
    """
    info = analyze_trump_overkill_second_hand(
        observation,
        chosen_card_index=chosen_card_index,
        low_lead_points_max=low_lead_points_max,
    )
    if not info.applicable or not info.chosen_is_trump or not info.chosen_wins or not info.winning_trump_exists:
        return 0.0

    if observation.trump_card is None:
        return 0.0

    trump_suit = observation.trump_card.suit
    lead_card, lead_player = observation.table_cards[0]
    chosen = observation.hand[chosen_card_index]

    # Ricostruiamo i costi delle briscole vincenti (già usate in analyze, ma qui serve anche il minimo).
    winning_costs: list[tuple[int, int]] = []
    for card in observation.hand:
        if card.suit != trump_suit:
            continue
        trick_cards = ((lead_card, lead_player), (card, observation.player_index))
        if who_wins_trick(trick_cards, trump_suit) == observation.player_index:
            winning_costs.append(_trump_cost_tuple(card, trump_suit=trump_suit))

    if not winning_costs:
        return 0.0

    min_points, min_strength = min(winning_costs)
    chosen_points, chosen_strength = _trump_cost_tuple(chosen, trump_suit=trump_suit)
    dp = max(0, int(chosen_points) - int(min_points))
    ds = max(0, int(chosen_strength) - int(min_strength))
    return float(dp) / 11.0 + float(ds) / 10.0


def trump_overkill_penalty_gap(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    beta: float,
    low_lead_points_max: int | None = 2,
) -> float:
    """
    Penalità proporzionale al gap (<=0) per scoraggiare overkill briscola.

    Output:
    - 0.0 se disattivata/non applicabile
    - altrimenti `-beta * gap_norm`
    """
    if float(beta) <= 0.0:
        return 0.0
    gap = trump_overkill_gap_norm(
        observation,
        chosen_card_index=chosen_card_index,
        low_lead_points_max=low_lead_points_max,
    )
    if gap <= 0.0:
        return 0.0
    return -float(beta) * float(gap)
