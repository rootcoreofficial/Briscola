"""
Parità encoder v3: path domain (oggetto) == fast (Python/NumPy) — Fase 5G, follow-up (a), step 1.

Oltre alla parità del vettore feature, qui blindiamo la **semantica di `out_of_play`**:
- una carta diventa "fuori gioco" appena giocata (prima dell'osservazione successiva);
- l'osservazione corrente NON include come fuori gioco le carte ancora in mano;
- la briscola scoperta resta `seen` ma NON `out_of_play` finché è pescabile/in mano.

Se `out_of_play` venisse aggiornato nel momento sbagliato, questi test devono fallire.
"""

from __future__ import annotations

import random

import pytest

from briscola_ai.ai.encoding.observation_encoder import encode_player_observation_2p
from briscola_ai.ai.fast.observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.fast.state_2p import new_fast_2p_state, step_fast_2p
from briscola_ai.ai.numba.observation import encode_fast_observation_numba_2p
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, PlayerState, new_game_state

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.numba


def _fast_v3(state_fast, *, player_index, obs):
    return encode_fast_observation_2p(
        state_fast,
        player_index=player_index,
        seen_cards_onehot=obs.seen_cards_onehot,
        out_of_play_cards_onehot=obs.out_of_play_cards_onehot,
        version="v3",
    )


def test_out_of_play_increment_semantics_matches_canonical() -> None:
    """
    Costruendo `out_of_play` incrementalmente (marca la carta appena giocata), deve coincidere
    con `make_player_observation(...).out_of_play_cards_onehot` in OGNI stato — incluse le
    transizioni a tavolo parziale e l'attraversamento della pesca finale.
    """
    canonical = new_game_state(num_players=2, seed=31)
    out_of_play: set[int] = set()
    rng = random.Random(7)

    saw_partial_table = False
    saw_deck_empty_with_unplayed_trump = False

    while not canonical.game_over:
        current = canonical.current_turn
        obs = make_player_observation(canonical, current)

        expected = [1 if cid in out_of_play else 0 for cid in range(40)]
        assert list(obs.out_of_play_cards_onehot) == expected

        # La briscola scoperta è sempre "vista"; è "fuori gioco" solo se già giocata.
        trump_id = card_to_id(canonical.trump_card)  # type: ignore[arg-type]
        assert obs.seen_cards_onehot[trump_id] == 1
        assert obs.out_of_play_cards_onehot[trump_id] == (1 if trump_id in out_of_play else 0)

        if len(canonical.table_cards) == 1:
            saw_partial_table = True
        if obs.deck_size == 0 and trump_id not in out_of_play:
            saw_deck_empty_with_unplayed_trump = True
            assert obs.out_of_play_cards_onehot[trump_id] == 0

        card_index = rng.randrange(len(canonical.players[current].hand))
        canonical, result = step(canonical, PlayCardAction(player_index=current, card_index=card_index))
        assert result.played_card is not None
        out_of_play.add(card_to_id(result.played_card))

    # Garantiamo di aver davvero esercitato i casi interessanti.
    assert saw_partial_table
    assert saw_deck_empty_with_unplayed_trump


def test_numba_v3_matches_domain_random_game() -> None:
    """Parità domain==numba (v3) lungo una partita random; tolleranza per float32 del path numba."""
    canonical = new_game_state(num_players=2, seed=23)
    fast = new_fast_2p_state(seed=23)
    rng = random.Random(5)

    while not canonical.game_over:
        current = canonical.current_turn
        obs = make_player_observation(canonical, current)
        direct = encode_player_observation_2p(obs, version="v3")
        numba = encode_fast_observation_numba_2p(
            fast,
            player_index=current,
            seen_cards_onehot=obs.seen_cards_onehot,
            out_of_play_cards_onehot=obs.out_of_play_cards_onehot,
            version="v3",
        )
        assert numba.action_mask == direct.action_mask
        assert numba.features == pytest.approx(direct.features, abs=1e-5)

        card_index = rng.randrange(len(canonical.players[current].hand))
        canonical, _ = step(canonical, PlayCardAction(player_index=current, card_index=card_index))
        step_fast_2p(fast, player_index=current, card_index=card_index)


def test_fast_v3_parity_handbuilt_endgame() -> None:
    """Parità domain==fast su uno stato di endgame costruito a mano (mazzo vuoto, prese)."""
    trump = Card(Suit.COINS, Rank.KING)
    captured0 = (Card(Suit.SWORDS, Rank.ACE), Card(Suit.CUPS, Rank.THREE))
    captured1 = (Card(Suit.CLUBS, Rank.ACE),)
    canonical = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState("P0", (Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)), captured0, 21),
            PlayerState("P1", (Card(Suit.COINS, Rank.THREE), Card(Suit.CLUBS, Rank.TWO)), captured1, 11),
        ),
        deck=(),
        trump_card=trump,
        table_cards=(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )
    obs = make_player_observation(canonical, player_index=0)

    # Stato fast equivalente: stesse mani/tavolo/briscola/deck vuoto.
    fast = new_fast_2p_state(seed=0)
    fast.hands[0] = [card_to_id(c) for c in canonical.players[0].hand]
    fast.hands[1] = [card_to_id(c) for c in canonical.players[1].hand]
    fast.table_cards = []
    fast.deck = []
    fast.trump_card = card_to_id(trump)
    fast.current_turn = 0
    fast.points = [21, 11]
    fast.game_over = False

    direct = encode_player_observation_2p(obs, version="v3")
    fast_encoded = _fast_v3(fast, player_index=0, obs=obs)
    assert fast_encoded.action_mask == direct.action_mask
    assert fast_encoded.features == direct.features


def test_fast_v3_parity_second_to_play_partial_table() -> None:
    """Parità domain==fast nel caso 'secondo di mano' (una carta sul tavolo)."""
    canonical = new_game_state(num_players=2, seed=5)
    # Avanza finché non capita uno stato con tavolo parziale.
    rng = random.Random(1)
    found = False
    fast = new_fast_2p_state(seed=5)
    while not canonical.game_over:
        current = canonical.current_turn
        if len(canonical.table_cards) == 1:
            obs = make_player_observation(canonical, current)
            direct = encode_player_observation_2p(obs, version="v3")
            fast_encoded = _fast_v3(fast, player_index=current, obs=obs)
            assert fast_encoded.action_mask == direct.action_mask
            assert fast_encoded.features == direct.features
            found = True
            break
        card_index = rng.randrange(len(canonical.players[current].hand))
        canonical, _ = step(canonical, PlayCardAction(player_index=current, card_index=card_index))
        step_fast_2p(fast, player_index=current, card_index=card_index)
    assert found
