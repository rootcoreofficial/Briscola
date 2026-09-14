"""
Test-àncora: le tabelle numeriche del fast path devono coincidere col dominio canonico.

Perché esistono questi test
---------------------------
Le tabelle punti/forza delle carte e la logica "chi vince la presa" sono duplicate in più
implementazioni (fast puro-Python, core Numba, solver endgame Numba, encoding) per motivi
di throughput. Nessuna copia è derivata da `domain.models.Rank`: un typo in una sola copia
corromperebbe training/dataset in modo silenzioso, senza far fallire i test comportamentali
aggregati (che verificano solo invarianti come "somma punti = 120").

Questi test ancorano OGNI copia direttamente alla fonte di verità del dominio:
- mapping carta ↔ id canonico (incluso l'helper ottimizzato `_card_to_id_fast`);
- tabelle `points` e `trick_strength` per tutti i 40 card id;
- vincitore della presa su TUTTE le coppie ordinate di carte, per ogni seme di briscola.
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.encoding.card_action_space import POINTS_BY_NUMBER, TRICK_STRENGTH_BY_NUMBER
from briscola_ai.ai.endgame.numba_solver import (
    _CARD_NUMBER_NUMBA as ENDGAME_CARD_NUMBER,
)
from briscola_ai.ai.endgame.numba_solver import (
    _CARD_POINTS_NUMBA as ENDGAME_CARD_POINTS,
)
from briscola_ai.ai.endgame.numba_solver import (
    _CARD_STRENGTH_NUMBA as ENDGAME_CARD_STRENGTH,
)
from briscola_ai.ai.endgame.numba_solver import (
    _CARD_SUIT_NUMBA as ENDGAME_CARD_SUIT,
)
from briscola_ai.ai.fast.state_2p import (
    CARD_NUMBER,
    CARD_POINTS,
    CARD_STRENGTH,
    CARD_SUIT,
    fast_who_wins_trick_2p,
)
from briscola_ai.ai.numba.core import (
    CARD_NUMBER_NUMBA,
    CARD_POINTS_NUMBA,
    CARD_STRENGTH_NUMBA,
    CARD_SUIT_NUMBA,
    _who_wins_trick_numba,
)
from briscola_ai.domain.card_id import SUIT_ORDER, SUIT_TO_INDEX, card_to_id, id_to_card
from briscola_ai.domain.models import Card, Rank
from briscola_ai.domain.observation import _card_to_id_fast
from briscola_ai.domain.rules import who_wins_trick

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba

# Tutte le 40 carte canoniche, indicizzate per card_id: la fonte di verità per ogni confronto.
ALL_CARDS: tuple[Card, ...] = tuple(id_to_card(card_id) for card_id in range(40))


def test_card_to_id_fast_matches_public_helper_on_all_cards() -> None:
    """`_card_to_id_fast` (path caldo osservazioni) deve restare identico a `card_to_id`."""
    for card_id, card in enumerate(ALL_CARDS):
        assert _card_to_id_fast(card) == card_to_id(card) == card_id


def test_fast_state_2p_tables_match_domain() -> None:
    """Le tabelle del fast path puro-Python devono derivare gli stessi valori di `Rank`."""
    for card_id, card in enumerate(ALL_CARDS):
        assert CARD_SUIT[card_id] == SUIT_TO_INDEX[card.suit]
        assert CARD_NUMBER[card_id] == card.rank.number
        assert CARD_POINTS[card_id] == card.rank.points
        assert CARD_STRENGTH[card_id] == card.rank.trick_strength


def test_numba_core_tables_match_domain() -> None:
    """Le tabelle del core Numba devono derivare gli stessi valori di `Rank`."""
    for card_id, card in enumerate(ALL_CARDS):
        assert int(CARD_SUIT_NUMBA[card_id]) == SUIT_TO_INDEX[card.suit]
        assert int(CARD_NUMBER_NUMBA[card_id]) == card.rank.number
        assert int(CARD_POINTS_NUMBA[card_id]) == card.rank.points
        assert int(CARD_STRENGTH_NUMBA[card_id]) == card.rank.trick_strength


def test_numba_endgame_solver_tables_match_domain() -> None:
    """Le tabelle (private) del solver endgame Numba devono derivare gli stessi valori di `Rank`."""
    for card_id, card in enumerate(ALL_CARDS):
        assert int(ENDGAME_CARD_SUIT[card_id]) == SUIT_TO_INDEX[card.suit]
        assert int(ENDGAME_CARD_NUMBER[card_id]) == card.rank.number
        assert int(ENDGAME_CARD_POINTS[card_id]) == card.rank.points
        assert int(ENDGAME_CARD_STRENGTH[card_id]) == card.rank.trick_strength


def test_card_action_space_tables_match_domain() -> None:
    """Le tabelle per-number dell'encoding devono derivare gli stessi valori di `Rank`."""
    for rank in Rank:
        assert POINTS_BY_NUMBER[rank.number] == rank.points
        assert TRICK_STRENGTH_BY_NUMBER[rank.number] == rank.trick_strength


@pytest.mark.parametrize("trump_suit_index", range(4))
def test_who_wins_trick_parity_on_all_ordered_pairs(trump_suit_index: int) -> None:
    """
    Il vincitore della presa 2-player deve coincidere col dominio su TUTTE le coppie ordinate.

    Copre sistematicamente i casi che i test di partita toccano solo a campione:
    briscola vs briscola, briscola vs seme di uscita, stesso seme non-briscola,
    semi diversi entrambi non-briscola. La scelta del rango della carta di briscola
    è irrilevante (conta solo il seme): usiamo il SETTE del seme.
    """
    trump_suit = SUIT_ORDER[trump_suit_index]
    trump_card_id = trump_suit_index * 10 + 6  # SETTE di briscola (number 7 -> offset 6)

    for first_id in range(40):
        for second_id in range(40):
            if first_id == second_id:
                continue
            expected = who_wins_trick(
                ((ALL_CARDS[first_id], 0), (ALL_CARDS[second_id], 1)),
                trump_suit,
            )
            assert (
                fast_who_wins_trick_2p(
                    first_card=first_id,
                    first_player=0,
                    second_card=second_id,
                    second_player=1,
                    trump_card=trump_card_id,
                )
                == expected
            )
            # Nota: dopo la de-duplicazione il solver endgame importa la stessa
            # `_who_wins_trick_numba` del core, quindi questo assert copre entrambi.
            assert _who_wins_trick_numba(first_id, 0, second_id, 1, trump_card_id) == expected
