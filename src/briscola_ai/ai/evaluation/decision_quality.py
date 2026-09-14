"""
Metriche di “qualità decisionale” per agenti Briscola (didattico).

Perché servono
--------------
Le metriche standard (win-rate, avg diff punti) dicono *quanto* un agente è forte,
ma non sempre spiegano *come* gioca.

In pratica, quando osservi una policy forte che però “spreca briscole” (es. usa una
briscola alta per prendere uno scarto), vuoi una metrica che renda quel comportamento
misurabile e quindi ottimizzabile.

Questo modulo definisce due metriche semplici (2-player):
- **trump_waste_rate**: quante volte, da secondi di mano, l'agente gioca una briscola
  pur avendo almeno una risposta vincente non-briscola.
- **trump_overkill_rate**: quante volte, quando l'agente *vince* giocando una briscola,
  usa una briscola “più costosa del necessario” rispetto alla briscola vincente minima
  disponibile in mano.

Nota anti-cheat
---------------
Le metriche sono calcolate usando lo stato completo del dominio (per simulare),
ma la policy continua a ricevere solo `PlayerObservation`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from ...domain.engine import PlayCardAction, step
from ...domain.observation import make_player_observation
from ...domain.rules import who_wins_trick
from ...domain.state import GameState, new_game_state
from ..agents import Agent
from ..fast.evaluation import FAST_EVALUATION_AGENT_NAMES
from ..models.bc_model import BCModelAgent, MLPBCModel
from ..numba.mlp import evaluate_mlp_policy_quality_numba_2p
from ..training.reward_shaping import card_conservation_cost
from .match import SeatFairStats, _winner_index_2p  # noqa: PLC2701 (didattico, reuse interno)


@dataclass(frozen=True, slots=True)
class DecisionQualityStats:
    """
    Metriche aggregate di qualità decisionale (2-player).

    Convenzioni:
    - Consideriamo solo decisioni in cui l'agente è **secondo di mano**
      (table_cards ha lunghezza 1).
    - `trump_waste_*` è definito solo quando esiste almeno una risposta vincente.
    """

    num_second_hand_decisions: int
    num_second_hand_with_winning_reply: int
    num_trump_waste: int
    num_second_hand_trump_wins: int
    num_trump_overkill: int
    num_second_hand_trump_wins_low_lead_points: int
    num_trump_overkill_low_lead_points: int

    @property
    def trump_waste_rate(self) -> float:
        """Frazione di sprechi briscola (0..1)."""
        if self.num_second_hand_with_winning_reply <= 0:
            return 0.0
        return float(self.num_trump_waste) / float(self.num_second_hand_with_winning_reply)

    @property
    def trump_overkill_rate(self) -> float:
        """
        Frazione di “overkill briscola” (0..1).

        Definizione:
        - consideriamo solo decisioni (secondo di mano) in cui la scelta dell'agente:
          - è una briscola
          - vince la mano
        - overkill = esisteva una briscola vincente “più economica” in mano.
        """
        if self.num_second_hand_trump_wins <= 0:
            return 0.0
        return float(self.num_trump_overkill) / float(self.num_second_hand_trump_wins)

    @property
    def trump_overkill_rate_low_lead_points(self) -> float:
        """
        Come `trump_overkill_rate`, ma solo quando la carta avversaria sul tavolo vale pochi punti.

        Scopo:
        rendere più “mirata” la diagnosi del caso tipico:
        “uso briscole alte per prendere scarti (0–2 punti)”.
        """
        if self.num_second_hand_trump_wins_low_lead_points <= 0:
            return 0.0
        return float(self.num_trump_overkill_low_lead_points) / float(self.num_second_hand_trump_wins_low_lead_points)


def _card_cost_for_conservation(*, card, trump_suit) -> tuple[int, int, int]:
    """
    Funzione di costo (euristica) per stimare quanto una carta sia “preziosa da conservare”.

    Usiamo un ordinamento lessicografico:
    - prima: briscola vs non-briscola (conservare briscole è spesso importante)
    - poi: punti della carta (conservare carichi può essere utile in molte fasi)
    - poi: forza nella mano (trick_strength)

    Nota:
    questa non è una regola di gioco: è solo una *metrica* per misurare lo stile di decisione.
    """
    is_trump = 1 if (trump_suit is not None and card.suit == trump_suit) else 0
    # Il costo (points, strength) è la definizione canonica condivisa: vedi
    # `training/reward_shaping.card_conservation_cost` (stesso ordinamento di guard e shaping).
    return (is_trump, *card_conservation_cost(card))


def _is_trump_overkill_second_hand(*, state: GameState, player_index: int, chosen_card_index: int) -> bool | None:
    """
    Ritorna True/False se la scelta è un “overkill briscola”, oppure None se non applicabile.

    Applicabile solo in 2-player quando:
    - è il turno di `player_index`
    - sul tavolo c'è già una carta (siamo secondi di mano)
    - `chosen_card_index` è valido

    Definizione operativa (didattica):
    - se l'agente NON vince la mano giocando una briscola -> non applichiamo (None)
    - altrimenti, tra le briscole in mano che vincerebbero la mano, troviamo quella con costo minimo
      (`_card_cost_for_conservation` ristretto alle briscole).
    - overkill = la briscola scelta ha un costo maggiore del minimo.
    """
    if state.num_players != 2:
        return None
    if state.game_over:
        return None
    if state.current_turn != player_index:
        return None
    if len(state.table_cards) != 1:
        return None

    hand = state.players[player_index].hand
    if chosen_card_index < 0 or chosen_card_index >= len(hand):
        return None

    trump_suit = state.trump_card.suit if state.trump_card else None
    if trump_suit is None:
        return None

    lead_card, lead_player = state.table_cards[0]
    chosen = hand[chosen_card_index]

    if chosen.suit != trump_suit:
        return None

    trick_cards = ((lead_card, lead_player), (chosen, player_index))
    if who_wins_trick(trick_cards, trump_suit) != player_index:
        return None

    winning_trumps: list[tuple[int, int]] = []
    for card in hand:
        if card.suit != trump_suit:
            continue
        trick_cards = ((lead_card, lead_player), (card, player_index))
        if who_wins_trick(trick_cards, trump_suit) == player_index:
            # Costo ristretto ai trumps: (points, strength)
            winning_trumps.append((int(card.rank.points), int(card.rank.trick_strength)))

    if not winning_trumps:
        return None

    min_cost = min(winning_trumps)
    chosen_cost = (int(chosen.rank.points), int(chosen.rank.trick_strength))
    return bool(chosen_cost > min_cost)


def _is_trump_waste_second_hand(*, state: GameState, player_index: int, chosen_card_index: int) -> bool | None:
    """
    Ritorna True/False se la scelta è uno “spreco briscola”, oppure None se non applicabile.

    Applicabile solo in 2-player quando:
    - è il turno di `player_index`
    - sul tavolo c'è già una carta (siamo secondi di mano)
    - esiste almeno una carta che vince la presa
    """
    if state.num_players != 2:
        return None
    if state.game_over:
        return None
    if state.current_turn != player_index:
        return None
    if len(state.table_cards) != 1:
        return None

    hand = state.players[player_index].hand
    if chosen_card_index < 0 or chosen_card_index >= len(hand):
        return None

    trump_suit = state.trump_card.suit if state.trump_card else None
    lead_card, lead_player = state.table_cards[0]

    winning_non_trump_exists = False
    winning_any_exists = False
    for card in hand:
        trick_cards = ((lead_card, lead_player), (card, player_index))
        winner = who_wins_trick(trick_cards, trump_suit)
        if winner != player_index:
            continue
        winning_any_exists = True
        if trump_suit is None or card.suit != trump_suit:
            winning_non_trump_exists = True

    if not winning_any_exists:
        return None

    chosen_card = hand[chosen_card_index]
    chosen_is_trump = trump_suit is not None and chosen_card.suit == trump_suit
    return bool(chosen_is_trump and winning_non_trump_exists)


def play_one_game_2p_collect_quality(
    agent0: Agent,
    agent1: Agent,
    *,
    tracked_agent_index: int,
    rng: random.Random,
    game_seed: int,
) -> tuple[GameState, DecisionQualityStats]:
    """
    Simula una singola partita 2-player e colleziona `DecisionQualityStats` per un agente.

    `tracked_agent_index` indica quale player (0 o 1) vogliamo misurare in questa partita.
    """
    if tracked_agent_index not in (0, 1):
        raise ValueError("tracked_agent_index deve essere 0 o 1")

    state = new_game_state(num_players=2, seed=game_seed)

    num_second = 0
    num_second_with_win = 0
    num_waste = 0
    num_trump_wins = 0
    num_trump_overkill = 0
    num_trump_wins_low = 0
    num_trump_overkill_low = 0

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        agent = agent0 if current == 0 else agent1
        observation = make_player_observation(state, current)
        card_index = agent.choose_card_index(observation, rng=rng)

        if current == tracked_agent_index and len(state.table_cards) == 1:
            num_second += 1
            waste = _is_trump_waste_second_hand(state=state, player_index=current, chosen_card_index=card_index)
            if waste is not None:
                num_second_with_win += 1
                if waste:
                    num_waste += 1

            # Overkill briscola: applicabile solo se la scelta è una briscola vincente.
            trump_suit = state.trump_card.suit if state.trump_card else None
            lead_card, lead_player = state.table_cards[0]
            chosen = state.players[current].hand[card_index]
            lead_points = int(lead_card.rank.points)
            low_lead = lead_points <= 2  # 0 oppure 2: “scarto” o quasi

            if trump_suit is not None and chosen.suit == trump_suit:
                trick_cards = ((lead_card, lead_player), (chosen, current))
                if who_wins_trick(trick_cards, trump_suit) == current:
                    num_trump_wins += 1
                    if low_lead:
                        num_trump_wins_low += 1

                    overkill = _is_trump_overkill_second_hand(
                        state=state, player_index=current, chosen_card_index=card_index
                    )
                    if overkill:
                        num_trump_overkill += 1
                        if low_lead:
                            num_trump_overkill_low += 1

        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

    return (
        state,
        DecisionQualityStats(
            num_second_hand_decisions=num_second,
            num_second_hand_with_winning_reply=num_second_with_win,
            num_trump_waste=num_waste,
            num_second_hand_trump_wins=num_trump_wins,
            num_trump_overkill=num_trump_overkill,
            num_second_hand_trump_wins_low_lead_points=num_trump_wins_low,
            num_trump_overkill_low_lead_points=num_trump_overkill_low,
        ),
    )


@dataclass(frozen=True, slots=True)
class SeatFairStatsWithQuality:
    """Seat-fair stats + metriche qualità per l'agente A."""

    match: SeatFairStats
    quality: DecisionQualityStats


def evaluate_bc_model_seat_fair_match_2p_with_quality_numba(
    model_agent: BCModelAgent,
    opponent_name: str,
    *,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
    opponent_agent: BCModelAgent | None = None,
) -> SeatFairStatsWithQuality:
    """
    Valuta decision-quality con core Numba per `bc_model` MLP vs baseline fast-compatible.

    La semantica è quella del path dominio:
    - Agent A è sempre il modello;
    - valutazione seat-fair;
    - policy deterministica argmax;
    - eventuale guard anti-overkill letto dal `BCModelAgent`.
    """
    if int(num_games) % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari.")
    if opponent_agent is None and opponent_name not in FAST_EVALUATION_AGENT_NAMES:
        supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
        raise ValueError(f"`engine=numba` supporta opponent: {supported}. Ottenuto: {opponent_name!r}")

    model = model_agent.model
    if not isinstance(model, MLPBCModel):
        raise ValueError("`engine=numba` supporta solo modelli `.npz` MLP con chiavi w1/b1/w2/b2.")
    opponent_model: MLPBCModel | None = None
    opponent_label = opponent_name
    if opponent_agent is not None:
        if not isinstance(opponent_agent.model, MLPBCModel):
            raise ValueError("`engine=numba` supporta solo opponent `.npz` MLP con chiavi w1/b1/w2/b2.")
        opponent_model = opponent_agent.model
        opponent_label = opponent_agent.name

    rng_game = random.Random(seed)
    num_pairs = int(num_games) // 2
    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_pairs)]
    if len(seeds) < num_pairs:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_pairs}, ottenuti {len(seeds)}")

    summary = evaluate_mlp_policy_quality_numba_2p(
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        opponent_name=opponent_label,
        num_games=int(num_games),
        seed=int(seed),
        game_seeds=seeds[:num_pairs],
        policy_overkill_guard=bool(model_agent.overkill_guard_enabled),
        parallel=True,
        opponent_w1=opponent_model.w1 if opponent_model is not None else None,
        opponent_b1=opponent_model.b1 if opponent_model is not None else None,
        opponent_w2=opponent_model.w2 if opponent_model is not None else None,
        opponent_b2=opponent_model.b2 if opponent_model is not None else None,
        opponent_overkill_guard=bool(opponent_agent.overkill_guard_enabled) if opponent_agent is not None else False,
        policy_name=model_agent.name,
    )
    return SeatFairStatsWithQuality(
        match=summary.to_seat_fair_stats(),
        quality=DecisionQualityStats(
            num_second_hand_decisions=summary.num_second_hand_decisions,
            num_second_hand_with_winning_reply=summary.num_second_hand_with_winning_reply,
            num_trump_waste=summary.num_trump_waste,
            num_second_hand_trump_wins=summary.num_second_hand_trump_wins,
            num_trump_overkill=summary.num_trump_overkill,
            num_second_hand_trump_wins_low_lead_points=summary.num_second_hand_trump_wins_low_lead_points,
            num_trump_overkill_low_lead_points=summary.num_trump_overkill_low_lead_points,
        ),
    )


@dataclass(frozen=True, slots=True)
class _SeatFairQualityTotals:
    """Somme interne, più comode da aggregare tra processi."""

    num_games: int
    agent_a_name: str
    agent_b_name: str
    wins_a: int
    wins_b: int
    draws: int
    sum_a: int
    sum_b: int
    sum_diff: int
    q_num_second: int
    q_num_second_with_win: int
    q_num_waste: int
    q_num_trump_wins: int
    q_num_trump_overkill: int
    q_num_trump_wins_low: int
    q_num_trump_overkill_low: int


def _pair_action_rng(*, seed: int, pair_index: int) -> random.Random:
    """
    RNG azioni indipendente per coppia seat-fair.

    La coppia e' l'unita' minima del confronto seat-fair: assegnarle uno stream stabile
    rende il risultato indipendente dal numero di worker e dal modo in cui le coppie
    vengono distribuite tra processi. La seconda partita prosegue lo stream della prima,
    mantenendo esplicito l'ordine interno della coppia.
    """
    mixed = (int(seed) ^ 0x9E3779B9 ^ ((int(pair_index) + 1) * 0x85EBCA6B)) & 0xFFFFFFFF
    return random.Random(mixed)


def _evaluate_quality_seed_pairs_independent_rng(
    *,
    agent_a: Agent,
    agent_b: Agent,
    seeds: Sequence[int],
    seed: int,
    pair_offset: int,
) -> _SeatFairQualityTotals:
    """
    Valuta un sottoinsieme di coppie seat-fair con RNG indipendente per coppia.

    E' il core condiviso dai percorsi seriale e parallelo: ogni chunk puo' girare in un
    processo diverso e poi essere sommato senza dipendenze dal partizionamento.
    """
    wins_a = 0
    wins_b = 0
    draws = 0
    sum_a = 0
    sum_b = 0
    sum_diff = 0

    q_num_second = 0
    q_num_second_with_win = 0
    q_num_waste = 0
    q_num_trump_wins = 0
    q_num_trump_overkill = 0
    q_num_trump_wins_low = 0
    q_num_trump_overkill_low = 0

    for local_i, game_seed in enumerate(seeds):
        pair_index = int(pair_offset) + local_i
        rng_action = _pair_action_rng(seed=int(seed), pair_index=pair_index)

        # Game 1: A=P0, B=P1
        s1, q1 = play_one_game_2p_collect_quality(
            agent_a, agent_b, tracked_agent_index=0, rng=rng_action, game_seed=int(game_seed)
        )
        p0, p1 = s1.players[0].points, s1.players[1].points
        sum_a += p0
        sum_b += p1
        sum_diff += p0 - p1
        w = _winner_index_2p(s1)
        if w is None:
            draws += 1
        elif w == 0:
            wins_a += 1
        else:
            wins_b += 1

        q_num_second += q1.num_second_hand_decisions
        q_num_second_with_win += q1.num_second_hand_with_winning_reply
        q_num_waste += q1.num_trump_waste
        q_num_trump_wins += q1.num_second_hand_trump_wins
        q_num_trump_overkill += q1.num_trump_overkill
        q_num_trump_wins_low += q1.num_second_hand_trump_wins_low_lead_points
        q_num_trump_overkill_low += q1.num_trump_overkill_low_lead_points

        # Game 2: A=P1, B=P0. Usiamo la stessa RNG della coppia, in sequenza.
        s2, q2 = play_one_game_2p_collect_quality(
            agent_b, agent_a, tracked_agent_index=1, rng=rng_action, game_seed=int(game_seed)
        )
        p0, p1 = s2.players[0].points, s2.players[1].points
        sum_b += p0
        sum_a += p1
        sum_diff += p1 - p0
        w = _winner_index_2p(s2)
        if w is None:
            draws += 1
        elif w == 0:
            wins_b += 1
        else:
            wins_a += 1

        q_num_second += q2.num_second_hand_decisions
        q_num_second_with_win += q2.num_second_hand_with_winning_reply
        q_num_waste += q2.num_trump_waste
        q_num_trump_wins += q2.num_second_hand_trump_wins
        q_num_trump_overkill += q2.num_trump_overkill
        q_num_trump_wins_low += q2.num_second_hand_trump_wins_low_lead_points
        q_num_trump_overkill_low += q2.num_trump_overkill_low_lead_points

    return _SeatFairQualityTotals(
        num_games=len(seeds) * 2,
        agent_a_name=agent_a.name,
        agent_b_name=agent_b.name,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        sum_a=sum_a,
        sum_b=sum_b,
        sum_diff=sum_diff,
        q_num_second=q_num_second,
        q_num_second_with_win=q_num_second_with_win,
        q_num_waste=q_num_waste,
        q_num_trump_wins=q_num_trump_wins,
        q_num_trump_overkill=q_num_trump_overkill,
        q_num_trump_wins_low=q_num_trump_wins_low,
        q_num_trump_overkill_low=q_num_trump_overkill_low,
    )


def _evaluate_quality_chunk(args: tuple[Agent, Agent, list[int], int, int]) -> _SeatFairQualityTotals:
    """Wrapper top-level per `ProcessPoolExecutor` (serve una funzione picklable)."""
    agent_a, agent_b, seeds, seed, pair_offset = args
    return _evaluate_quality_seed_pairs_independent_rng(
        agent_a=agent_a,
        agent_b=agent_b,
        seeds=seeds,
        seed=seed,
        pair_offset=pair_offset,
    )


def _merge_quality_totals(parts: Sequence[_SeatFairQualityTotals]) -> SeatFairStatsWithQuality:
    """Somma chunk paralleli in un unico output pubblico."""
    if not parts:
        raise ValueError("Nessun risultato da aggregare")

    num_games = sum(p.num_games for p in parts)
    first = parts[0]
    sum_a = sum(p.sum_a for p in parts)
    sum_b = sum(p.sum_b for p in parts)
    sum_diff = sum(p.sum_diff for p in parts)

    match = SeatFairStats(
        num_games=num_games,
        agent_a_name=first.agent_a_name,
        agent_b_name=first.agent_b_name,
        wins_agent_a=sum(p.wins_a for p in parts),
        wins_agent_b=sum(p.wins_b for p in parts),
        draws=sum(p.draws for p in parts),
        avg_points_agent_a=sum_a / num_games if num_games else 0.0,
        avg_points_agent_b=sum_b / num_games if num_games else 0.0,
        avg_point_diff_agent_a_minus_agent_b=sum_diff / num_games if num_games else 0.0,
    )
    quality = DecisionQualityStats(
        num_second_hand_decisions=sum(p.q_num_second for p in parts),
        num_second_hand_with_winning_reply=sum(p.q_num_second_with_win for p in parts),
        num_trump_waste=sum(p.q_num_waste for p in parts),
        num_second_hand_trump_wins=sum(p.q_num_trump_wins for p in parts),
        num_trump_overkill=sum(p.q_num_trump_overkill for p in parts),
        num_second_hand_trump_wins_low_lead_points=sum(p.q_num_trump_wins_low for p in parts),
        num_trump_overkill_low_lead_points=sum(p.q_num_trump_overkill_low for p in parts),
    )
    return SeatFairStatsWithQuality(match=match, quality=quality)


def evaluate_seat_fair_match_2p_with_quality(
    agent_a: Agent,
    agent_b: Agent,
    *,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
) -> SeatFairStatsWithQuality:
    """
    Valuta A vs B (seat-fair) e colleziona quality metrics per A.

    Note:
    - in seat-fair, A gioca meta' partite come P0 e meta' come P1;
    - ogni coppia riceve uno stream RNG delle azioni derivato da `seed` e dal proprio
      indice. Lo stesso comando produce quindi gli stessi aggregati anche se la versione
      parallela usa un numero diverso di worker.
    """
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari (giochiamo a coppie).")

    rng_game = random.Random(seed)
    num_pairs = num_games // 2

    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_pairs)]
    if len(seeds) < num_pairs:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_pairs}, ottenuti {len(seeds)}")
    totals = _evaluate_quality_seed_pairs_independent_rng(
        agent_a=agent_a,
        agent_b=agent_b,
        seeds=seeds[:num_pairs],
        seed=int(seed),
        pair_offset=0,
    )
    return _merge_quality_totals([totals])


def evaluate_seat_fair_match_2p_with_quality_parallel(
    agent_a: Agent,
    agent_b: Agent,
    *,
    num_games: int,
    seed: int,
    workers: int,
    game_seeds: Sequence[int] | None = None,
) -> SeatFairStatsWithQuality:
    """
    Valuta A vs B con quality metrics usando piu' processi.

    Regola:
    - `workers <= 1` delega alla funzione seriale;
    - `workers > 1` divide le coppie seat-fair in chunk indipendenti.

    Nota RNG:
    seriale e parallelo usano lo stesso RNG azioni indipendente per coppia seat-fair.
    `workers` cambia quindi il tempo di esecuzione, non il risultato, anche con agenti
    stocastici.
    """
    if workers <= 1:
        return evaluate_seat_fair_match_2p_with_quality(
            agent_a,
            agent_b,
            num_games=num_games,
            seed=seed,
            game_seeds=game_seeds,
        )
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari (giochiamo a coppie).")

    rng_game = random.Random(seed)
    num_pairs = num_games // 2
    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_pairs)]
    if len(seeds) < num_pairs:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_pairs}, ottenuti {len(seeds)}")
    seeds = seeds[:num_pairs]
    if num_pairs == 0:
        return evaluate_seat_fair_match_2p_with_quality(
            agent_a,
            agent_b,
            num_games=0,
            seed=seed,
            game_seeds=seeds,
        )

    worker_count = max(1, min(int(workers), num_pairs))
    chunk_size = (num_pairs + worker_count - 1) // worker_count
    chunks: list[tuple[Agent, Agent, list[int], int, int]] = []
    for start in range(0, num_pairs, chunk_size):
        chunk = seeds[start : start + chunk_size]
        if chunk:
            chunks.append((agent_a, agent_b, chunk, int(seed), start))

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        parts = list(executor.map(_evaluate_quality_chunk, chunks))
    return _merge_quality_totals(parts)
