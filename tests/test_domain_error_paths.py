"""
Path d'errore del dominio: prima coperti solo indirettamente dai percorsi "felici".

Perché esistono questi test
---------------------------
Gli errori del dominio sono parte del contratto (il backend li traduce in 400 e i loop
di simulazione ci si appoggiano): devono restare espliciti e non regredire in silenzio.
In più, `step` su input invalido deve essere PURO anche nel fallimento: lo stato passato
non deve cambiare (`after == before`), che è anche un buon esempio didattico di purezza.
"""

from __future__ import annotations

import pytest

from briscola_ai.domain.card_id import id_to_card
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.rules import who_wins_trick
from briscola_ai.domain.serialization import game_state_from_dict, game_state_to_dict
from briscola_ai.domain.state import new_game_state
from briscola_ai.versioning import get_rules_version


def test_who_wins_trick_rejects_empty_table() -> None:
    """Tavolo vuoto: non esiste un vincitore, deve essere un errore esplicito."""
    with pytest.raises(ValueError, match="table_cards vuoto"):
        who_wins_trick((), Suit.COINS)


def test_who_wins_trick_without_trump_falls_back_to_leading_suit() -> None:
    """`trump_suit=None` (stato incompleto): vince comunque il seme di uscita, senza crash."""
    table = ((Card(Suit.CUPS, Rank.KING), 0), (Card(Suit.SWORDS, Rank.ACE), 1))
    # L'Asso di spade è più forte, ma non è del seme di uscita: vince il Re di coppe.
    assert who_wins_trick(table, None) == 0


@pytest.mark.parametrize("card_id", [-1, 40, 1000])
def test_id_to_card_rejects_out_of_range(card_id: int) -> None:
    """Gli id canonici sono [0,39]: fuori range deve essere un errore esplicito."""
    with pytest.raises(ValueError, match="fuori range"):
        id_to_card(card_id)


def test_step_rejects_wrong_turn_and_leaves_state_unchanged() -> None:
    """Giocare fuori turno: errore nel result e stato di input INVARIATO (purezza)."""
    before = new_game_state(num_players=2, seed=5)
    wrong_player = 1 - before.current_turn

    after, result = step(before, PlayCardAction(player_index=wrong_player, card_index=0))

    assert result.error is not None
    assert "turno" in result.error.lower()
    assert after == before


def test_step_rejects_finished_game_and_leaves_state_unchanged() -> None:
    """Partita già terminata: nessuna azione è valida e lo stato non cambia."""
    state = new_game_state(num_players=2, seed=5)
    while not state.game_over:
        state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
        assert result.error is None

    after, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))
    assert result.error is not None
    assert "terminata" in result.error.lower()
    assert after == state


def test_game_state_from_dict_rejects_unknown_schema() -> None:
    """Il campo `schema` è verificato: un dump di uno schema futuro deve fallire esplicitamente."""
    data = game_state_to_dict(new_game_state(num_players=2, seed=3))

    data["schema"] = 999
    with pytest.raises(ValueError, match="Schema di serializzazione non supportato"):
        game_state_from_dict(data)

    del data["schema"]
    with pytest.raises(ValueError, match="Schema di serializzazione non supportato"):
        game_state_from_dict(data)


def test_rules_version_is_a_string() -> None:
    """`RULES_VERSION` è stringa per contratto (finisce in event log/dataset: '1' != 1)."""
    assert isinstance(get_rules_version(), str)
    assert get_rules_version() == "1"
