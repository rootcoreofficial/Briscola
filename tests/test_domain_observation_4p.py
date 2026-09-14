"""
Osservazione anti-cheat in modalità 4-player.

Perché esistono questi test
---------------------------
La suite storica copriva `make_player_observation` solo in 2-player. In 4p ci sono
proprietà specifiche mai verificate:
- la briscola è una carta REALE nella mano del player 3 (`hands[-1][-1]`) ed è pubblica
  (in Briscola la carta di briscola è scoperta): deve comparire in `trump_card` e in
  `seen_cards_onehot` per tutti, senza però rivelare il resto della mano del player 3;
- `teams`/`is_team_game` devono essere propagati;
- nessun leak: la mano dell'osservatore è l'unica presente, e le carte non giocate degli
  avversari non devono mai risultare "viste".
"""

from __future__ import annotations

from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state


def test_4p_observation_exposes_public_metadata_and_trump() -> None:
    """Team, briscola pubblica e dimensioni mani devono essere corretti per ogni osservatore."""
    state = new_game_state(num_players=4, seed=42)
    trump = state.trump_card
    assert trump is not None
    # In 4p la briscola è l'ultima carta distribuita al player 3 (resta nella sua mano).
    assert trump in state.players[3].hand

    for player_index in range(4):
        obs = make_player_observation(state, player_index)
        assert obs.num_players == 4
        assert obs.is_team_game is True
        assert obs.teams == ((0, 2), (1, 3))
        assert obs.player_index == player_index
        assert obs.trump_card == trump
        assert obs.deck_size == 0  # in 4p il mazzo è interamente distribuito
        assert obs.players_hand_sizes == (10, 10, 10, 10)
        # La briscola scoperta è informazione pubblica: "vista" da tutti.
        assert obs.seen_cards_onehot[card_to_id(trump)] == 1


def test_4p_observation_does_not_leak_opponent_hands() -> None:
    """L'osservazione contiene solo la mano dell'osservatore; le altre carte non sono 'viste'."""
    state = new_game_state(num_players=4, seed=7)
    trump = state.trump_card
    assert trump is not None

    for player_index in range(4):
        obs = make_player_observation(state, player_index)
        assert obs.hand == state.players[player_index].hand

        # A partita appena iniziata l'unica carta pubblica è la briscola scoperta:
        # ogni altra carta segnata "vista" sarebbe un leak delle mani avversarie.
        seen_ids = {card_id for card_id, flag in enumerate(obs.seen_cards_onehot) if flag}
        assert seen_ids == {card_to_id(trump)}

        # E niente è ancora "fuori gioco" (nessuna presa completata).
        assert sum(obs.out_of_play_cards_onehot) == 0


def test_4p_observation_tracks_public_history_after_tricks() -> None:
    """Dopo una presa completa, le 4 carte giocate diventano viste E fuori gioco per tutti."""
    state = new_game_state(num_players=4, seed=11)
    trump = state.trump_card
    assert trump is not None

    played_ids: set[int] = set()
    for _ in range(4):
        current = state.current_turn
        played_ids.add(card_to_id(state.players[current].hand[0]))
        state, result = step(state, PlayCardAction(player_index=current, card_index=0))
        assert result.error is None

    assert len(state.table_cards) == 0  # presa risolta

    for player_index in range(4):
        obs = make_player_observation(state, player_index)
        seen_ids = {card_id for card_id, flag in enumerate(obs.seen_cards_onehot) if flag}
        out_ids = {card_id for card_id, flag in enumerate(obs.out_of_play_cards_onehot) if flag}
        # Le 4 carte della presa sono viste E fuori gioco (una volta catturata, anche la
        # briscola è definitivamente fuori gioco: la distinzione "vista ma non fuori"
        # vale solo finché è esposta/in mano, non dopo la cattura).
        assert played_ids <= seen_ids
        assert out_ids == played_ids
        # Invariante di sottoinsieme: fuori gioco implica vista.
        assert out_ids <= seen_ids
