"""
Agente PIMC (Perfect-Information Monte Carlo) per prototipi offline.

Obiettivo
---------
Questo modulo implementa una prima search a inference sopra una policy esistente (es. v6):

1. parte solo da `PlayerObservation`, quindi non legge mano avversaria o ordine reale del mazzo;
2. campiona stati completi compatibili con informazione pubblica + mano del player;
3. prova ogni mossa legale su ciascuna determinizzazione;
4. completa la partita con una policy di rollout;
5. sceglie la mossa con miglior delta punti medio.

Scope
-----
È un prototipo offline, non un agente UI di default. Serve a verificare se search + modello base supera il modello puro
nel finale/semi-finale prima di investire in integrazione runtime o distillazione teacher.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from math import sqrt
from typing import ClassVar

from ...domain.card_id import card_to_id, id_to_card
from ...domain.engine import PlayCardAction, step
from ...domain.observation import PlayerObservation, make_player_observation
from ...domain.state import GameState, PlayerState
from ..encoding.observation_encoder import encode_player_observation_2p
from ..endgame.fast_solver import solve_endgame_fast
from ..models.belief_model import MLPBeliefModel, infer_belief_encoder_version
from .base import Agent, AgentSpec
from .hybrid_endgame import reconstruct_endgame_state
from .rule_based import HeuristicAgentV2

_ALL_CARD_IDS = frozenset(range(40))


@dataclass(slots=True)
class PIMCSearchStats:
    """Metriche runtime raccolte da `PIMCAgent` durante una evaluation offline."""

    total_decisions: int = 0
    search_decisions: int = 0
    fallback_decisions: int = 0
    endgame_solver_decisions: int = 0
    successful_determinizations: int = 0
    failed_determinizations: int = 0
    completed_rollouts: int = 0
    failed_rollouts: int = 0
    coerced_moves: int = 0
    search_elapsed_seconds: float = 0.0

    @property
    def seconds_per_search_decision(self) -> float:
        """Tempo medio speso nelle sole decisioni in cui PIMC ha cercato davvero."""
        if self.search_decisions <= 0:
            return 0.0
        return self.search_elapsed_seconds / self.search_decisions


@dataclass(frozen=True, slots=True)
class PIMCActionValue:
    """Valore stimato da PIMC per una singola carta giocabile."""

    card_index: int
    mean_score: float | None
    rollout_count: int


@dataclass(frozen=True, slots=True)
class PIMCSearchDiagnostics:
    """
    Diagnostica dell'ultima decisione search PIMC.

    `margin` e `margin_standard_error` sono calcolati sul confronto best-vs-second
    usando delta paired per determinizzazione. Sono pensati per dataset teacher:
    un margine alto ma con SE alta non va trattato come correzione affidabile.
    """

    best_card_index: int
    second_card_index: int | None
    best_mean_score: float
    second_mean_score: float | None
    margin: float | None
    margin_standard_error: float | None
    margin_z: float | None
    margin_ci95_low: float | None
    margin_ci95_high: float | None
    paired_margin_sample_count: int
    paired_margin_samples: tuple[float, ...]
    action_values: tuple[PIMCActionValue, ...]
    successful_determinizations: int
    failed_determinizations: int
    completed_rollouts: int
    failed_rollouts: int


def _standard_error(values: list[float]) -> float | None:
    """Errore standard campionario della media, o `None` con meno di 2 campioni."""
    n = len(values)
    if n <= 1:
        return None
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return sqrt(variance / n)


def _onehot_ids(raw: tuple[int, ...], *, name: str) -> set[int]:
    """Converte una one-hot 40 in set di card id con validazione esplicita."""
    if len(raw) != 40:
        raise ValueError(f"{name} deve avere lunghezza 40, trovata {len(raw)}")
    ids: set[int] = set()
    for card_id, value in enumerate(raw):
        if value not in (0, 1):
            raise ValueError(f"{name} contiene un valore non binario in posizione {card_id}: {value!r}")
        if value:
            ids.add(card_id)
    return ids


def _card_points(card_id: int) -> int:
    """Punti Briscola della carta canonica."""
    return int(id_to_card(card_id).rank.points)


@lru_cache(maxsize=2048)
def _subset_with_points(card_ids: tuple[int, ...], target_points: int) -> frozenset[int] | None:
    """
    Ritorna un sottoinsieme di `card_ids` con somma punti esatta.

    Serve per ricostruire `captured_cards` coerenti con `players_points`. Le carte catturate sono
    pubbliche come insieme (`out_of_play - table`), ma l'osservazione non espone la partizione per
    giocatore: una qualsiasi partizione con lo stesso punteggio è sufficiente per simulare correttamente
    punteggio corrente, card counting e futuri incrementi di punti.
    """
    if target_points < 0:
        return None
    if target_points == 0:
        return frozenset()
    if not card_ids:
        return None

    first, *rest = card_ids
    rest_tuple = tuple(rest)

    with_first = _subset_with_points(rest_tuple, target_points - _card_points(first))
    if with_first is not None:
        return frozenset({first, *with_first})

    return _subset_with_points(rest_tuple, target_points)


def _captured_cards_for_scores(
    *,
    captured_ids: set[int],
    player0_points: int,
    player1_points: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partiziona le carte catturate pubbliche in due insiemi con i punti osservati."""
    total_points = sum(_card_points(card_id) for card_id in captured_ids)
    if total_points != int(player0_points) + int(player1_points):
        raise ValueError(
            "Carte fuori gioco incoerenti con players_points: "
            f"punti_catturati={total_points}, players_points={player0_points + player1_points}"
        )

    ordered = tuple(sorted(captured_ids, key=lambda cid: (_card_points(cid), cid), reverse=True))
    player0_ids = _subset_with_points(ordered, int(player0_points))
    if player0_ids is None:
        raise ValueError(f"Impossibile partizionare le prese per ottenere {player0_points} punti a P0")

    player1_ids = captured_ids - set(player0_ids)
    if sum(_card_points(card_id) for card_id in player1_ids) != int(player1_points):
        raise ValueError(f"Impossibile partizionare le prese per ottenere {player1_points} punti a P1")

    return tuple(sorted(player0_ids)), tuple(sorted(player1_ids))


def _validate_pimc_observation(observation: PlayerObservation) -> None:
    """Verifica lo scope PIMC 2-player e il fatto che l'agente stia decidendo per sé."""
    if observation.num_players != 2 or observation.is_team_game:
        raise ValueError("PIMC supporta solo osservazioni 2-player non a squadre")
    if observation.player_index not in (0, 1):
        raise ValueError(f"player_index fuori range: {observation.player_index}")
    if observation.current_turn != observation.player_index:
        raise ValueError("PIMC può decidere solo quando current_turn == player_index")
    if observation.game_over:
        raise ValueError("Partita già terminata")
    if observation.trump_card is None:
        raise ValueError("Briscola assente: impossibile determinizzare")
    if len(observation.table_cards) not in (0, 1):
        raise ValueError(f"Tavolo non supportato: attese 0 o 1 carte, trovate {len(observation.table_cards)}")
    if len(observation.players_points) != 2:
        raise ValueError(f"players_points deve avere lunghezza 2, trovata {len(observation.players_points)}")
    if len(observation.players_hand_sizes) != 2:
        raise ValueError(f"players_hand_sizes deve avere lunghezza 2, trovata {len(observation.players_hand_sizes)}")
    if len(observation.hand) != observation.players_hand_sizes[observation.player_index]:
        raise ValueError("La mano osservata non coincide con players_hand_sizes[player_index]")


def unknown_live_card_count(observation: PlayerObservation) -> int:
    """Numero di carte vive non note al player: mano avversaria + mazzo."""
    opponent_index = 1 - observation.player_index
    return int(observation.players_hand_sizes[opponent_index]) + int(observation.deck_size)


def _safe_agent_card_index(
    agent: Agent,
    observation: PlayerObservation,
    *,
    rng: random.Random,
    metrics: PIMCSearchStats | None = None,
) -> int:
    """
    Chiede una mossa a un agente e la normalizza a un indice valido.

    In Briscola ogni carta in mano è giocabile nel dominio del progetto. Se un fallback/rollout agent
    restituisce un indice fuori range, scegliamo in modo difensivo la prima carta invece di abortire la
    determinizzazione o l'intera decisione PIMC.
    """
    if not observation.hand:
        raise ValueError("Mano vuota: nessuna azione possibile")
    try:
        card_index = int(agent.choose_card_index(observation, rng=rng))
    except Exception:
        if metrics is not None:
            metrics.coerced_moves += 1
        return 0
    if 0 <= card_index < len(observation.hand):
        return card_index
    if metrics is not None:
        metrics.coerced_moves += 1
    return 0


def belief_card_weights(
    belief_model: MLPBeliefModel,
    observation: PlayerObservation,
    *,
    uniform_mix: float = 0.10,
) -> dict[int, float]:
    """
    Calcola i pesi per-carta con cui campionare la mano avversaria nelle determinizzazioni.

    - encoda l'osservazione con l'encoder del belief model (tipicamente v4) e ne prende
      le probabilità sigmoid per le sole carte IGNOTE (non in mano mia, non fuori gioco);
    - mescola con l'uniforme (`uniform_mix`): una belief mal calibrata che assegna ~0 a una
      carta che l'avversario ha davvero renderebbe quel mondo impossibile da campionare
      (punto cieco sistematico); il mix garantisce un pavimento di esplorazione.

    Anti-cheat: input = sola osservazione lecita; la rete è un'inferenza, non una lettura.
    """
    if not 0.0 <= float(uniform_mix) <= 1.0:
        raise ValueError(f"uniform_mix fuori range: {uniform_mix}")

    encoder_version = infer_belief_encoder_version(belief_model)
    encoded = encode_player_observation_2p(observation, version=encoder_version)
    import numpy as np  # import locale: pimc.py resta importabile senza numpy nel resto del modulo

    probs = belief_model.predict_probs(np.asarray(encoded.features, dtype=np.float32))

    my_hand_ids = {card_to_id(card) for card in observation.hand}
    out_of_play_ids = {i for i, flag in enumerate(observation.out_of_play_cards_onehot) if flag}
    unknown_ids = sorted(set(range(40)) - my_hand_ids - out_of_play_ids)
    if not unknown_ids:
        return {}

    pool_probs = [max(float(probs[card_id]), 0.0) for card_id in unknown_ids]
    total = sum(pool_probs)
    n = len(unknown_ids)
    mix = float(uniform_mix)
    if total <= 0.0:
        # Belief degenerata: pesi uniformi (equivale al comportamento storico).
        return {card_id: 1.0 for card_id in unknown_ids}
    return {
        card_id: (1.0 - mix) * (p / total) + mix * (1.0 / n) for card_id, p in zip(unknown_ids, pool_probs, strict=True)
    }


def _weighted_sample_without_replacement(
    pool: list[int],
    k: int,
    weights: dict[int, float],
    rng: random.Random,
) -> list[int]:
    """
    Campiona `k` elementi da `pool` senza rimpiazzo, con probabilità proporzionale ai pesi.

    Schema "successive sampling": a ogni estrazione la probabilità è proporzionale al peso
    residuo. È l'approssimazione standard della distribuzione condizionata sulle mani
    (esatta per k=1, ottima in pratica per k piccolo come qui: 1-3 carte).

    Robustezza: pesi mancanti/negativi valgono 0; se il totale residuo è 0 si degrada
    all'uniforme sul resto del pool (mai un crash per una belief mal calibrata).
    """
    if k > len(pool):
        raise ValueError(f"Campione richiesto ({k}) maggiore del pool ({len(pool)})")
    candidates = list(pool)
    residual = [max(float(weights.get(card_id, 0.0)), 0.0) for card_id in candidates]
    chosen: list[int] = []
    for _ in range(k):
        total = sum(residual)
        if total <= 0.0:
            idx = rng.randrange(len(candidates))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = len(candidates) - 1
            for i, w in enumerate(residual):
                acc += w
                if r <= acc:
                    idx = i
                    break
        chosen.append(candidates.pop(idx))
        residual.pop(idx)
    return chosen


def determinize_observation(
    observation: PlayerObservation,
    *,
    rng: random.Random,
    card_weights: dict[int, float] | None = None,
) -> GameState:
    """
    Campiona uno `GameState` completo compatibile con una `PlayerObservation`.

    Anti-cheat: usa solo mano osservata, tavolo, dimensioni mani, deck_size e carte fuori gioco
    pubbliche (`out_of_play_cards_onehot`). Non usa mai lo stato reale nascosto.

    `card_weights` (opzionale, Fase 2 belief): pesi per card id con cui campionare la mano
    avversaria invece dell'uniforme. I pesi vengono da una belief network allenata su self-play
    (input = osservazione lecita), quindi restano anti-cheat: sono un'INFERENZA, non una lettura
    dello stato nascosto. Con `None` il comportamento è identico allo storico (uniforme),
    bit-per-bit a parità di rng.
    """
    _validate_pimc_observation(observation)

    if observation.deck_size == 0:
        return reconstruct_endgame_state(observation)

    player_index = observation.player_index
    opponent_index = 1 - player_index
    trump_card = observation.trump_card
    if trump_card is None:
        raise ValueError("Briscola assente")

    out_of_play_ids = _onehot_ids(observation.out_of_play_cards_onehot, name="out_of_play_cards_onehot")
    table_ids = {card_to_id(card) for card, _player_idx in observation.table_cards}
    my_hand_ids = [card_to_id(card) for card in observation.hand]
    if len(set(my_hand_ids)) != len(my_hand_ids):
        raise ValueError("La mano osservata contiene carte duplicate")
    if len(table_ids) != len(observation.table_cards):
        raise ValueError("Il tavolo contiene carte duplicate")
    if not table_ids.issubset(out_of_play_ids):
        raise ValueError("out_of_play_cards_onehot non contiene tutte le carte sul tavolo")
    if set(my_hand_ids) & out_of_play_ids:
        raise ValueError("out_of_play_cards_onehot si sovrappone alla mano osservata")

    trump_id = card_to_id(trump_card)
    unknown_live_ids = set(_ALL_CARD_IDS - set(my_hand_ids) - out_of_play_ids)
    opponent_hand_size = int(observation.players_hand_sizes[opponent_index])
    deck_size = int(observation.deck_size)
    if len(unknown_live_ids) != opponent_hand_size + deck_size:
        raise ValueError(
            "Conteggio carte vive incoerente: "
            f"unknown_live={len(unknown_live_ids)}, attese={opponent_hand_size + deck_size}"
        )

    deck_forced_ids: list[int] = []
    opponent_pool = set(unknown_live_ids)
    if deck_size > 0 and trump_id in opponent_pool:
        # Nella Briscola 2-player la briscola scoperta resta nel mazzo e viene pescata per ultima.
        deck_forced_ids.append(trump_id)
        opponent_pool.remove(trump_id)
    if len(deck_forced_ids) > deck_size:
        raise ValueError("Deck size incoerente con la briscola pubblica")

    if card_weights is None:
        opponent_hand_ids = set(rng.sample(sorted(opponent_pool), opponent_hand_size))
    else:
        opponent_hand_ids = set(
            _weighted_sample_without_replacement(sorted(opponent_pool), opponent_hand_size, card_weights, rng)
        )
    deck_rest_ids = list(opponent_pool - opponent_hand_ids)
    rng.shuffle(deck_rest_ids)
    if len(deck_rest_ids) != deck_size - len(deck_forced_ids):
        raise ValueError("Determinizzazione incoerente: dimensione deck errata")

    # `domain.step` pesca da `deck.pop()`: mettendo la briscola in testa la rendiamo l'ultima pescata.
    deck_ids = tuple(deck_forced_ids + deck_rest_ids)

    captured_ids = set(out_of_play_ids - table_ids)
    p0_captured_ids, p1_captured_ids = _captured_cards_for_scores(
        captured_ids=captured_ids,
        player0_points=int(observation.players_points[0]),
        player1_points=int(observation.players_points[1]),
    )

    players = [
        PlayerState(
            name="P0",
            hand=tuple(),
            captured_cards=tuple(id_to_card(card_id) for card_id in p0_captured_ids),
            points=int(observation.players_points[0]),
        ),
        PlayerState(
            name="P1",
            hand=tuple(),
            captured_cards=tuple(id_to_card(card_id) for card_id in p1_captured_ids),
            points=int(observation.players_points[1]),
        ),
    ]
    players[player_index] = PlayerState(
        name=observation.player_name,
        hand=observation.hand,
        captured_cards=players[player_index].captured_cards,
        points=players[player_index].points,
    )
    players[opponent_index] = PlayerState(
        name=f"P{opponent_index}",
        hand=tuple(id_to_card(card_id) for card_id in sorted(opponent_hand_ids)),
        captured_cards=players[opponent_index].captured_cards,
        points=players[opponent_index].points,
    )

    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=tuple(players),
        deck=tuple(id_to_card(card_id) for card_id in deck_ids),
        trump_card=trump_card,
        table_cards=observation.table_cards,
        current_turn=observation.current_turn,
        first_player=observation.first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
        # La storia delle prese e' informazione PUBBLICA dell'osservazione: va preservata
        # nello stato determinizzato, cosi' policy/value v4 simulano CON la memoria reale.
        trick_history=observation.trick_history,
    )


def rollout_to_terminal(
    state: GameState,
    *,
    rollout_agent: Agent,
    rng: random.Random,
    use_endgame_solver: bool = True,
    metrics: PIMCSearchStats | None = None,
    max_steps: int = 128,
) -> GameState:
    """Completa una partita determinizzata con la policy di rollout."""
    cursor = state
    steps = 0
    while not cursor.game_over:
        if steps >= max_steps:
            raise RuntimeError("Rollout PIMC non terminato entro il limite di sicurezza")
        steps += 1

        observation = make_player_observation(cursor, cursor.current_turn)
        if use_endgame_solver and len(cursor.deck) == 0:
            # Endgame a informazione perfetta (lo stato determinizzato è completo): si
            # risolve UNA volta sola e si segue la principal variation fino al termine,
            # invece di ri-risolvere a ogni carta come si faceva col kernel numba.
            # Ogni linea ottima raggiunge lo stesso delta minimax, quindi il valore del
            # rollout è identico; con ~1/6 delle chiamate il solver PYTHON (~0.8 ms sul
            # caso peggiore) diventa sostenibile nel percorso caldo — è ciò che rende il
            # runtime web zero-numba (2026-07-07).
            try:
                solution = solve_endgame_fast(cursor)
            except ValueError:
                card_index = _safe_agent_card_index(rollout_agent, observation, rng=rng, metrics=metrics)
            else:
                for mover, move_index in solution.principal_variation:
                    cursor, result = step(cursor, PlayCardAction(player_index=mover, card_index=move_index))
                    if result.error:
                        raise RuntimeError(f"Errore durante rollout PIMC (PV endgame): {result.error}")
                continue
        else:
            card_index = _safe_agent_card_index(rollout_agent, observation, rng=rng, metrics=metrics)

        cursor, result = step(cursor, PlayCardAction(player_index=cursor.current_turn, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore durante rollout PIMC: {result.error}")
    return cursor


@dataclass(frozen=True)
class PIMCAgent:
    """
    Agente PIMC con fallback e rollout configurabili.

    Il fallback viene usato quando lo stato è fuori scope, quando ci sono troppe carte ignote o quando
    una determinizzazione fallisce. Il rollout agent rappresenta la policy approssimata per entrambi i lati.

    Concorrenza (assunzione di contratto): UNA istanza per partita, mai condivisa tra
    richieste concorrenti. La dataclass è `frozen` ma `metrics` e `last_search_diagnostics`
    sono stato mutabile non thread-safe (il backend li legge dopo ogni mossa); la factory
    `build_agent` crea un'istanza nuova per mossa/partita, e va mantenuto così.
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="pimc",
        label="PIMC",
        description_it=(
            "Prototipo offline: campiona stati compatibili con l'informazione pubblica e usa una policy "
            "di rollout per scegliere nel finale. Non è esposto come agente UI di default."
        ),
    )

    rollout_agent: Agent = field(default_factory=HeuristicAgentV2)
    fallback: Agent = field(default_factory=HeuristicAgentV2)
    num_determinizations: int = 32
    max_unknown_cards: int = 10
    use_endgame_solver: bool = True
    # Fase 2 (belief): se presente, le determinizzazioni campionano la mano avversaria
    # con i pesi della belief network invece che uniformemente.
    belief_model: MLPBeliefModel | None = None
    belief_uniform_mix: float = 0.10
    # Search JIT (kernel `ai/numba/pimc.py`): stessa semantica della search python
    # (~20-50x piu' veloce). Richiede un rollout_agent MLP (`BCModelAgent`); su stati
    # non determinizzabili il kernel ritorna -1 e si degrada alla search python.
    use_numba_search: bool = False
    name: str = "pimc"
    metrics: PIMCSearchStats = field(default_factory=PIMCSearchStats, repr=False, compare=False)
    last_search_diagnostics: PIMCSearchDiagnostics | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def _choose_with_numba_search(self, observation: PlayerObservation, *, rng: random.Random) -> int | None:
        """
        Path JIT della search: costruisce gli array numerici dall'osservazione e delega
        al kernel `choose_pimc_card_numba_arrays`. Ritorna None quando il kernel non e'
        applicabile (rollout non-MLP, stato non determinizzabile): il chiamante prosegue
        con la search python, identica in semantica.
        """
        from ..models.bc_model import BCModelAgent, MLPBCModel

        rollout = self.rollout_agent
        if not isinstance(rollout, BCModelAgent) or not isinstance(rollout.model, MLPBCModel):
            return None
        if rollout.model.has_belief_input:
            return None  # policy 409: il rollout JIT ibrido non supporta l'input belief

        import numpy as np

        from ..numba.pimc import choose_pimc_card_numba_arrays

        my_index = observation.player_index
        my_hand = np.full(3, -1, dtype=np.int64)
        for i, card in enumerate(observation.hand):
            my_hand[i] = card_to_id(card)

        table_cards = np.full(2, -1, dtype=np.int64)
        table_players = np.full(2, -1, dtype=np.int64)
        for i, (card, seat) in enumerate(observation.table_cards):
            table_cards[i] = card_to_id(card)
            table_players[i] = int(seat)

        trick_hist = np.zeros((20, 5), dtype=np.int64)
        num_tricks = min(len(observation.trick_history), 20)
        for i in range(num_tricks):
            record = observation.trick_history[i]
            (lead_card, lead_player), (resp_card, _resp_player) = record.cards
            trick_hist[i, 0] = card_to_id(lead_card)
            trick_hist[i, 1] = int(lead_player)
            trick_hist[i, 2] = card_to_id(resp_card)
            trick_hist[i, 3] = int(record.winner_index)
            trick_hist[i, 4] = int(record.points)

        weights = np.ones(40, dtype=np.float64)
        if self.belief_model is not None:
            weight_map = belief_card_weights(self.belief_model, observation, uniform_mix=self.belief_uniform_mix)
            weights[:] = 0.0
            for card_id, w in weight_map.items():
                weights[card_id] = w

        trump_card = observation.trump_card
        assert trump_card is not None  # gia' validato dal chiamante

        search_started = time.perf_counter()
        card_index = int(
            choose_pimc_card_numba_arrays(
                rollout.model.w1,
                rollout.model.b1,
                rollout.model.w2,
                rollout.model.b2,
                bool(rollout.overkill_guard_enabled),
                weights,
                my_hand,
                len(observation.hand),
                int(observation.players_hand_sizes[1 - my_index]),
                int(observation.deck_size),
                table_cards,
                table_players,
                len(observation.table_cards),
                int(my_index),
                card_to_id(trump_card),
                np.asarray(observation.players_points, dtype=np.int64),
                np.asarray(observation.out_of_play_cards_onehot, dtype=np.int64),
                np.asarray(observation.seen_cards_onehot, dtype=np.int64),
                trick_hist,
                num_tricks,
                max(1, int(self.num_determinizations)),
                rng.randrange(0, 2**31),
            )
        )
        if card_index < 0:
            return None  # stato non determinizzabile: search python (che sapra' fallire con metrica)
        determinations = max(1, int(self.num_determinizations))
        self.metrics.successful_determinizations += determinations
        self.metrics.completed_rollouts += determinations * len(observation.hand)
        self.metrics.search_elapsed_seconds += time.perf_counter() - search_started
        return card_index

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        if not observation.hand:
            raise ValueError("Mano vuota: nessuna azione possibile")
        object.__setattr__(self, "last_search_diagnostics", None)
        self.metrics.total_decisions += 1
        try:
            _validate_pimc_observation(observation)
        except ValueError:
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        if observation.deck_size == 0 and self.use_endgame_solver:
            try:
                card_index = solve_endgame_fast(reconstruct_endgame_state(observation)).best_card_index
                if 0 <= card_index < len(observation.hand):
                    self.metrics.endgame_solver_decisions += 1
                    return card_index
            except ValueError:
                pass
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        if unknown_live_card_count(observation) > int(self.max_unknown_cards):
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        self.metrics.search_decisions += 1
        if self.use_numba_search:
            fast_index = self._choose_with_numba_search(observation, rng=rng)
            if fast_index is not None:
                return fast_index
        search_started = time.perf_counter()
        legal_indices = list(range(len(observation.hand)))
        scores = [0.0 for _ in legal_indices]
        counts = [0 for _ in legal_indices]
        per_determinization_scores: list[list[float | None]] = []
        local_successful_determinizations = 0
        local_failed_determinizations = 0
        local_completed_rollouts = 0
        local_failed_rollouts = 0

        determinizations = max(1, int(self.num_determinizations))
        card_weights: dict[int, float] | None = None
        if self.belief_model is not None:
            card_weights = belief_card_weights(self.belief_model, observation, uniform_mix=self.belief_uniform_mix)
        try:
            for sample_index in range(determinizations):
                sample_rng = random.Random(rng.randrange(0, 2**32) ^ (sample_index * 0x9E3779B9))
                try:
                    sampled_state = determinize_observation(observation, rng=sample_rng, card_weights=card_weights)
                except ValueError:
                    self.metrics.failed_determinizations += 1
                    local_failed_determinizations += 1
                    continue
                self.metrics.successful_determinizations += 1
                local_successful_determinizations += 1
                sample_scores: list[float | None] = [None for _ in legal_indices]

                for local_pos, card_index in enumerate(legal_indices):
                    next_state, result = step(
                        sampled_state,
                        PlayCardAction(player_index=sampled_state.current_turn, card_index=card_index),
                    )
                    if result.error:
                        continue
                    rollout_rng = random.Random(sample_rng.randrange(0, 2**32) ^ (card_index * 0x85EBCA6B))
                    try:
                        final_state = rollout_to_terminal(
                            next_state,
                            rollout_agent=self.rollout_agent,
                            rng=rollout_rng,
                            use_endgame_solver=self.use_endgame_solver,
                            metrics=self.metrics,
                        )
                    except RuntimeError:
                        self.metrics.failed_rollouts += 1
                        local_failed_rollouts += 1
                        continue
                    self.metrics.completed_rollouts += 1
                    local_completed_rollouts += 1
                    player_points = final_state.players[observation.player_index].points
                    opponent_points = final_state.players[1 - observation.player_index].points
                    score = float(player_points - opponent_points)
                    sample_scores[local_pos] = score
                    scores[local_pos] += score
                    counts[local_pos] += 1
                per_determinization_scores.append(sample_scores)
        finally:
            self.metrics.search_elapsed_seconds += time.perf_counter() - search_started

        if not any(counts):
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        valid_positions = [pos for pos, count in enumerate(counts) if count > 0]
        best_pos = max(
            valid_positions,
            key=lambda pos: (scores[pos] / counts[pos], -legal_indices[pos]),
        )
        second_pos = (
            max(
                (pos for pos in valid_positions if pos != best_pos),
                key=lambda pos: (scores[pos] / counts[pos], -legal_indices[pos]),
                default=None,
            )
            if len(valid_positions) >= 2
            else None
        )
        action_values = tuple(
            PIMCActionValue(
                card_index=card_index,
                mean_score=(scores[pos] / counts[pos]) if counts[pos] else None,
                rollout_count=counts[pos],
            )
            for pos, card_index in enumerate(legal_indices)
        )
        best_mean = scores[best_pos] / counts[best_pos]
        second_mean = (scores[second_pos] / counts[second_pos]) if second_pos is not None else None
        margin = best_mean - second_mean if second_mean is not None else None
        paired_margin_samples: list[float] = []
        if second_pos is not None:
            for sample_scores in per_determinization_scores:
                best_sample = sample_scores[best_pos]
                second_sample = sample_scores[second_pos]
                if best_sample is not None and second_sample is not None:
                    paired_margin_samples.append(best_sample - second_sample)

        margin_se = _standard_error(paired_margin_samples)
        margin_z = (margin / margin_se) if margin is not None and margin_se is not None and margin_se != 0.0 else None
        margin_ci95_low = margin - 1.96 * margin_se if margin is not None and margin_se is not None else None
        margin_ci95_high = margin + 1.96 * margin_se if margin is not None and margin_se is not None else None
        object.__setattr__(
            self,
            "last_search_diagnostics",
            PIMCSearchDiagnostics(
                best_card_index=legal_indices[best_pos],
                second_card_index=legal_indices[second_pos] if second_pos is not None else None,
                best_mean_score=best_mean,
                second_mean_score=second_mean,
                margin=margin,
                margin_standard_error=margin_se,
                margin_z=margin_z,
                margin_ci95_low=margin_ci95_low,
                margin_ci95_high=margin_ci95_high,
                paired_margin_sample_count=len(paired_margin_samples),
                paired_margin_samples=tuple(paired_margin_samples),
                action_values=action_values,
                successful_determinizations=local_successful_determinizations,
                failed_determinizations=local_failed_determinizations,
                completed_rollouts=local_completed_rollouts,
                failed_rollouts=local_failed_rollouts,
            ),
        )
        return legal_indices[best_pos]
