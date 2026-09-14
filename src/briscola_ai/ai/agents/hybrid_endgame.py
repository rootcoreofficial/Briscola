"""
Agente ibrido con solver esatto nel finale.

Il solver lavora su `GameState`, ma questo modulo lo usa solo dopo aver ricostruito
lo stato endgame dalla `PlayerObservation`: nessuna lettura della mano avversaria,
nessun accesso all'ordine del mazzo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar

from ...domain.card_id import card_to_id, id_to_card
from ...domain.observation import PlayerObservation
from ...domain.state import GameState, PlayerState
from ..endgame.fast_solver import solve_endgame_fast
from ..endgame.solver import solve_endgame
from .base import Agent, AgentSpec
from .rule_based import HeuristicAgentV2

_ALL_CARD_IDS = frozenset(range(40))


def _seen_card_ids_from_observation(observation: PlayerObservation) -> set[int]:
    """
    Converte `seen_cards_onehot` in un set di card id, validando la shape pubblica.

    Il campo è parte del confine anti-cheat: se manca o ha una shape inattesa non proviamo a
    "indovinare" l'endgame, ma lasciamo che l'agente ibrido usi il fallback.
    """
    if len(observation.seen_cards_onehot) != 40:
        raise ValueError(f"seen_cards_onehot deve avere lunghezza 40, trovata {len(observation.seen_cards_onehot)}")

    seen_ids: set[int] = set()
    for card_id, seen in enumerate(observation.seen_cards_onehot):
        if seen not in (0, 1):
            raise ValueError(f"seen_cards_onehot contiene un valore non binario in posizione {card_id}: {seen!r}")
        if seen:
            seen_ids.add(card_id)
    return seen_ids


def _validate_endgame_observation_scope(observation: PlayerObservation) -> None:
    """
    Verifica che l'osservazione sia nello scope risolvibile dal solver endgame 2-player.

    L'agente deve giocare solo dal proprio punto di vista: quindi richiediamo esplicitamente che
    `current_turn == player_index`. Se non è vero, l'osservazione non rappresenta una decisione
    dell'agente e il solver non va consultato.
    """
    if observation.num_players != 2 or observation.is_team_game:
        raise ValueError("Il solver ibrido supporta solo osservazioni 2-player non a squadre")
    if observation.player_index not in (0, 1):
        raise ValueError(f"player_index fuori range: {observation.player_index}")
    if observation.current_turn not in (0, 1):
        raise ValueError(f"current_turn fuori range: {observation.current_turn}")
    if observation.current_turn != observation.player_index:
        raise ValueError("L'osservazione non è sul turno del player osservante")
    if observation.first_player not in (0, 1):
        raise ValueError(f"first_player fuori range: {observation.first_player}")
    if observation.game_over:
        raise ValueError("Partita già terminata")
    if observation.deck_size != 0:
        raise ValueError(f"Il solver endgame richiede deck_size=0, trovato {observation.deck_size}")
    if observation.trump_card is None:
        raise ValueError("Briscola assente: impossibile ricostruire il seme di briscola")
    if len(observation.players_points) != 2:
        raise ValueError(f"players_points deve avere lunghezza 2, trovata {len(observation.players_points)}")
    if len(observation.players_hand_sizes) != 2:
        raise ValueError(f"players_hand_sizes deve avere lunghezza 2, trovata {len(observation.players_hand_sizes)}")
    if any(size < 0 or size > 3 for size in observation.players_hand_sizes):
        raise ValueError(f"Dimensioni mani fuori range endgame: {observation.players_hand_sizes!r}")
    if len(observation.hand) != observation.players_hand_sizes[observation.player_index]:
        raise ValueError("La mano osservata non coincide con players_hand_sizes[player_index]")
    if len(observation.table_cards) not in (0, 1):
        raise ValueError(f"Tavolo non supportato: attese 0 o 1 carte, trovate {len(observation.table_cards)}")

    remaining = sum(observation.players_hand_sizes) + len(observation.table_cards)
    if remaining <= 0:
        raise ValueError("Osservazione non terminale senza carte residue")
    if remaining > 6:
        raise ValueError(f"Troppe carte residue per l'endgame: {remaining}")

    if not observation.table_cards:
        if observation.players_hand_sizes[0] != observation.players_hand_sizes[1]:
            raise ValueError(f"Mani sbilanciate a tavolo vuoto: {observation.players_hand_sizes!r}")
        if observation.first_player != observation.current_turn:
            raise ValueError("first_player incoerente a tavolo vuoto")
        return

    leader = observation.table_cards[0][1]
    if leader not in (0, 1):
        raise ValueError(f"Player id sul tavolo fuori range: {leader}")
    if leader == observation.current_turn:
        raise ValueError("Turno incoerente: chi ha aperto la mano non può rigiocare")
    if observation.first_player != leader:
        raise ValueError("first_player incoerente con la carta sul tavolo")
    if observation.players_hand_sizes[observation.current_turn] != observation.players_hand_sizes[leader] + 1:
        raise ValueError("Mani sbilanciate rispetto alla carta sul tavolo")


def _opponent_hand_ids_from_out_of_play(
    observation: PlayerObservation,
    *,
    my_hand_ids: set[int],
    table_ids: set[int],
    opponent_hand_size: int,
) -> set[int] | None:
    """
    Deduce la mano avversaria dal campo `out_of_play_cards_onehot`, se presente e coerente.

    È il path "pulito": le carte fuori gioco sono SOLO prese + tavolo, quindi a mazzo vuoto
    `mano_avversario = tutte − mia_mano − fuori_gioco` (la briscola non richiede trattamenti
    speciali: se è in mano avversaria non è fuori gioco, quindi ricade nel complemento).

    Ritorna `None` (→ fallback su `seen_cards_onehot`) se il campo è assente/default/incoerente:
    - lunghezza diversa da 40 o valori non binari;
    - non contiene tutte le carte sul tavolo (il tavolo è per definizione fuori gioco);
    - si sovrappone alla mano osservante (la mia mano non è fuori gioco);
    - il complemento non ha la dimensione attesa (es. campo a 40 zeri dei dataset vecchi).
    """
    raw = observation.out_of_play_cards_onehot
    if len(raw) != 40:
        return None

    out_of_play_ids: set[int] = set()
    for card_id, value in enumerate(raw):
        if value not in (0, 1):
            return None
        if value:
            out_of_play_ids.add(card_id)

    if not table_ids.issubset(out_of_play_ids):
        return None
    if my_hand_ids & out_of_play_ids:
        return None

    candidate = set(_ALL_CARD_IDS - my_hand_ids - out_of_play_ids)
    if len(candidate) != opponent_hand_size:
        return None
    return candidate


def _opponent_hand_ids_from_seen(
    observation: PlayerObservation,
    *,
    my_hand_ids: set[int],
    table_ids: set[int],
    trump_id: int,
    opponent_hand_size: int,
) -> set[int]:
    """
    Deduce la mano avversaria dalla sola `seen_cards_onehot` (path compatibile, fallback).

    `seen_cards_onehot` include sempre la briscola scoperta, anche quando è stata pescata in mano:
    è l'unico bit ambiguo, risolto contando le mani (se manca esattamente una carta ai candidati,
    quella carta è la briscola). Solleva `ValueError` se l'osservazione è incoerente.
    """
    seen_ids = _seen_card_ids_from_observation(observation)
    if trump_id not in seen_ids:
        raise ValueError("seen_cards_onehot non contiene la briscola pubblica")
    if not table_ids.issubset(seen_ids):
        raise ValueError("seen_cards_onehot non contiene tutte le carte sul tavolo")

    # La briscola scoperta è l'unica sovrapposizione lecita tra mano osservata e carte "viste".
    illegal_seen_hand_overlap = (my_hand_ids & seen_ids) - {trump_id}
    if illegal_seen_hand_overlap:
        raise ValueError("seen_cards_onehot contiene carte non-briscola ancora nella mano osservata")

    remaining_unknown_ids = set(_ALL_CARD_IDS - my_hand_ids - seen_ids)
    if len(remaining_unknown_ids) == opponent_hand_size:
        return remaining_unknown_ids
    if (
        len(remaining_unknown_ids) == opponent_hand_size - 1
        and trump_id not in my_hand_ids
        and trump_id not in table_ids
    ):
        opponent_hand_ids = set(remaining_unknown_ids)
        opponent_hand_ids.add(trump_id)
        return opponent_hand_ids
    raise ValueError(
        "Impossibile dedurre in modo univoco la mano avversaria "
        f"(candidati={len(remaining_unknown_ids)}, attesi={opponent_hand_size})"
    )


def reconstruct_endgame_state(observation: PlayerObservation) -> GameState:
    """
    Ricostruisce uno `GameState` endgame 2-player dalla sola `PlayerObservation`.

    Anti-cheat
    ----------
    La funzione non legge mai la mano avversaria dal dominio: usa solo informazione pubblica
    (`out_of_play_cards_onehot`/`seen_cards_onehot`, tavolo, dimensioni mani) e la mano propria.

    Deduzione mano avversaria
    -------------------------
    Preferisce `out_of_play_cards_onehot` (path pulito: complemento diretto) quando presente e
    coerente; altrimenti usa il fallback storico su `seen_cards_onehot`. Questo evita una
    migrazione "tutto o niente" e mantiene la compatibilità coi dataset/osservazioni vecchie.

    Punti e prese
    -------------
    Lo stato ricostruito azzera `points` e `captured_cards` di entrambi i player. È intenzionale:
    `domain.step` ricalcola i punti del vincitore da `captured_cards`, quindi copiare i punteggi
    reali senza conoscere la partizione delle prese corromperebbe il delta. La base punti è una
    costante rispetto alle mosse future, perciò non cambia la scelta ottima del solver.
    """
    _validate_endgame_observation_scope(observation)

    player_index = observation.player_index
    opponent_index = 1 - player_index
    trump_card = observation.trump_card
    if trump_card is None:
        # Ridondante rispetto alla validate, ma aiuta mypy a restringere il tipo.
        raise ValueError("Briscola assente")

    trump_id = card_to_id(trump_card)

    my_hand_ids = {card_to_id(card) for card in observation.hand}
    if len(my_hand_ids) != len(observation.hand):
        raise ValueError("La mano osservata contiene carte duplicate")

    table_ids = {card_to_id(card) for card, _player_idx in observation.table_cards}
    if len(table_ids) != len(observation.table_cards):
        raise ValueError("Il tavolo contiene carte duplicate")
    if my_hand_ids & table_ids:
        raise ValueError("Una carta non può essere insieme in mano e sul tavolo")

    opponent_hand_size = observation.players_hand_sizes[opponent_index]

    # Path pulito (out_of_play) con fallback compatibile (seen).
    opponent_hand_ids = _opponent_hand_ids_from_out_of_play(
        observation,
        my_hand_ids=my_hand_ids,
        table_ids=table_ids,
        opponent_hand_size=opponent_hand_size,
    )
    if opponent_hand_ids is None:
        opponent_hand_ids = _opponent_hand_ids_from_seen(
            observation,
            my_hand_ids=my_hand_ids,
            table_ids=table_ids,
            trump_id=trump_id,
            opponent_hand_size=opponent_hand_size,
        )

    if len(opponent_hand_ids) != opponent_hand_size:
        raise ValueError("Dimensione mano avversaria ricostruita incoerente")
    if opponent_hand_ids & my_hand_ids or opponent_hand_ids & table_ids:
        raise ValueError("Mano avversaria ricostruita sovrapposta a carte pubbliche o proprie")

    opponent_hand = tuple(id_to_card(card_id) for card_id in sorted(opponent_hand_ids))
    players = [
        PlayerState(name="P0", hand=tuple(), captured_cards=tuple(), points=0),
        PlayerState(name="P1", hand=tuple(), captured_cards=tuple(), points=0),
    ]
    players[player_index] = PlayerState(
        name=observation.player_name,
        hand=observation.hand,
        captured_cards=tuple(),
        points=0,
    )
    players[opponent_index] = PlayerState(
        name=f"P{opponent_index}",
        hand=opponent_hand,
        captured_cards=tuple(),
        points=0,
    )

    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=tuple(players),
        deck=tuple(),
        trump_card=trump_card,
        table_cards=observation.table_cards,
        current_turn=observation.current_turn,
        first_player=observation.first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def can_solve_endgame_from_observation(observation: PlayerObservation) -> bool:
    """
    Ritorna True se l'osservazione può essere ricostruita e risolta senza informazione nascosta.

    È una funzione di utilità per test/debug. `HybridEndgameAgent` usa lo stesso criterio, ma evita
    di chiamarla per non risolvere due volte lo stesso stato.
    """
    try:
        solve_endgame(reconstruct_endgame_state(observation))
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class HybridEndgameAgent:
    """
    Agente ibrido: fallback normale in mid-game, solver esatto a mazzo vuoto.

    In endgame prova a ricostruire uno stato completo dalla sola osservazione lecita; se la
    ricostruzione non è valida o il solver rifiuta lo stato, delega al fallback. Questo mantiene
    l'invariante anti-cheat: l'agente non legge mai `GameState.players[opponent].hand`.
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="hybrid_endgame",
        label="Hybrid Endgame",
        description_it=(
            "Usa l'euristica v2 durante la partita e, a mazzo vuoto, passa a un solver esatto "
            "ricostruito dalla sola osservazione pubblica."
        ),
    )

    fallback: Agent = field(default_factory=HeuristicAgentV2)
    name: str = "hybrid_endgame"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        try:
            reconstructed = reconstruct_endgame_state(observation)
            # Solver PYTHON (2026-07-07): a livello di decisione costa <1 ms (0.82 ms
            # misurati sul caso peggiore 3v3) e libera il runtime web da numba.
            return solve_endgame_fast(reconstructed).best_card_index
        except ValueError:
            return self.fallback.choose_card_index(observation, rng=rng)
