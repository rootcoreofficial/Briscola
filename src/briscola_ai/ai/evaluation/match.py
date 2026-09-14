"""
Valutazione offline di agenti (dominio-only).

Perché serve?
-------------
Quando modifichiamo un agente (euristica, rete neurale, ecc.), vogliamo un modo
semplice e riproducibile per rispondere a:
- “È meglio di random?”
- “Quanto è stabile su seed diversi?”
- “Quante partite vince e con che margine?”

Questa valutazione non usa HTTP/WS né UI:
lavora direttamente su `GameState + step()` per essere veloce e deterministica.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ...domain.engine import PlayCardAction, step
from ...domain.observation import make_player_observation
from ...domain.state import GameState, new_game_state
from ..agents import Agent


@dataclass(frozen=True)
class MatchStats:
    """
    Risultati aggregati di una valutazione.

    Nota:
    in 2-player, ogni partita ha 120 punti totali; quindi è comodo usare anche
    il check `total_points == 120 * num_games` come sanity check.
    """

    num_games: int
    agent0_name: str
    agent1_name: str
    wins_agent0: int
    wins_agent1: int
    draws: int
    avg_points_agent0: float
    avg_points_agent1: float
    avg_point_diff_agent0_minus_agent1: float


@dataclass(frozen=True)
class SeatFairStats:
    """
    Risultati aggregati “seat-fair” (l'agente A gioca metà partite come P0 e metà come P1).

    In pratica, per ogni seed di partita giochiamo due game:
    1) A = player 0, B = player 1
    2) B = player 0, A = player 1

    Questo riduce drasticamente il bias dovuto a “chi inizia” (player 0 nel dominio).

    Nota statistica (importante per le CI):
    le due partite di una coppia condividono lo stesso mazzo, quindi NON sono campioni
    indipendenti — con agenti deterministici sono anzi fortemente correlate. L'unità
    statistica indipendente è la COPPIA, non la partita. I campi `sum_sq_pair_*` servono
    a calcolare CI corrette sull'unità coppia; le CI per-partita basate su
    `sum_sq_point_diff_*` sovrastimano il campione effettivo e risultano troppo strette
    (anti-conservative). Restano come fallback per i path che non tracciano le coppie.
    """

    num_games: int
    agent_a_name: str
    agent_b_name: str
    wins_agent_a: int
    wins_agent_b: int
    draws: int
    avg_points_agent_a: float
    avg_points_agent_b: float
    avg_point_diff_agent_a_minus_agent_b: float
    sum_sq_point_diff_agent_a_minus_agent_b: float | None = None
    # Somma su ogni coppia di (d1 + d2)^2, dove d = punti_A - punti_B in ciascun game.
    sum_sq_pair_point_diff_agent_a_minus_agent_b: float | None = None
    # Somma su ogni coppia di s^2, dove s = (score_g1 + score_g2) / 2 e score = 1/0.5/0 (win/draw/loss di A).
    sum_sq_pair_score_agent_a: float | None = None

    @property
    def num_pairs(self) -> int:
        """Numero di coppie seat-fair (2 partite per coppia)."""
        return self.num_games // 2

    @property
    def sample_std_point_diff_agent_a_minus_agent_b(self) -> float | None:
        """
        Deviazione standard campionaria del diff punti PER PARTITA, se disponibile.

        Attenzione: tratta le partite come indipendenti, ma le due partite di una coppia
        seat-fair non lo sono. Per le CI usare l'unità coppia (`sum_sq_pair_*`) quando
        disponibile; questa property resta per compatibilità e diagnostica.
        """
        if self.sum_sq_point_diff_agent_a_minus_agent_b is None or self.num_games <= 1:
            return None
        sum_diff = self.avg_point_diff_agent_a_minus_agent_b * self.num_games
        numerator = self.sum_sq_point_diff_agent_a_minus_agent_b - (sum_diff * sum_diff / self.num_games)
        variance = max(0.0, numerator / (self.num_games - 1))
        return variance**0.5


def _winner_index_2p(state: GameState) -> int | None:
    """Ritorna 0/1 se c'è un vincitore, altrimenti None (pareggio)."""
    p0 = state.players[0].points
    p1 = state.players[1].points
    if p0 > p1:
        return 0
    if p1 > p0:
        return 1
    return None


def play_one_game_2p(
    agent0: Agent,
    agent1: Agent,
    *,
    rng: random.Random,
    game_seed: int,
) -> GameState:
    """
    Simula una singola partita 2-player.

    - `agent0` controlla player 0
    - `agent1` controlla player 1
    - `rng` controlla la stochasticity dell'agente (scelte) per riproducibilità
    - `game_seed` controlla lo shuffle del mazzo (dominio) per riproducibilità
    """
    state = new_game_state(num_players=2, seed=game_seed)

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        agent = agent0 if current == 0 else agent1
        observation = make_player_observation(state, current)
        card_index = agent.choose_card_index(observation, rng=rng)

        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

    return state


def evaluate_match_2p(
    agent0: Agent,
    agent1: Agent,
    *,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
) -> MatchStats:
    """
    Valuta `agent0` vs `agent1` in 2-player.

    Scelta di design:
    - Separiamo RNG “game” (shuffle) da RNG “action” (scelte agenti), così cambiamenti nella policy
      non alterano la sequenza di shuffle.
    - Se `game_seeds` è fornito, usiamo quella suite (random ma fissata) per gli shuffle.
    """
    rng_game = random.Random(seed)
    rng_action = random.Random(seed ^ 0x9E3779B9)

    wins0 = 0
    wins1 = 0
    draws = 0
    sum0 = 0
    sum1 = 0
    sum_diff = 0

    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_games)]
    if len(seeds) < num_games:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_games}, ottenuti {len(seeds)}")

    for i in range(num_games):
        game_seed = seeds[i]
        final_state = play_one_game_2p(agent0, agent1, rng=rng_action, game_seed=game_seed)

        p0 = final_state.players[0].points
        p1 = final_state.players[1].points
        sum0 += p0
        sum1 += p1
        sum_diff += p0 - p1

        winner = _winner_index_2p(final_state)
        if winner is None:
            draws += 1
        elif winner == 0:
            wins0 += 1
        else:
            wins1 += 1

    return MatchStats(
        num_games=num_games,
        agent0_name=agent0.name,
        agent1_name=agent1.name,
        wins_agent0=wins0,
        wins_agent1=wins1,
        draws=draws,
        avg_points_agent0=sum0 / num_games if num_games else 0.0,
        avg_points_agent1=sum1 / num_games if num_games else 0.0,
        avg_point_diff_agent0_minus_agent1=sum_diff / num_games if num_games else 0.0,
    )


def evaluate_seat_fair_match_2p(
    agent_a: Agent,
    agent_b: Agent,
    *,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
) -> SeatFairStats:
    """
    Valuta `agent_a` vs `agent_b` in 2-player eliminando il bias di posto.

    `num_games` deve essere pari, perché lo interpretiamo come:
    - `num_pairs = num_games // 2`
    - per ogni pair giochiamo 2 partite con lo stesso `game_seed` ma agenti scambiati.
    """
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari (giochiamo a coppie).")

    rng_game = random.Random(seed)
    rng_action = random.Random(seed ^ 0x9E3779B9)
    num_pairs = num_games // 2

    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_pairs)]
    if len(seeds) < num_pairs:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_pairs}, ottenuti {len(seeds)}")

    wins_a = 0
    wins_b = 0
    draws = 0
    sum_a = 0
    sum_b = 0
    sum_diff = 0
    sum_sq_diff = 0
    sum_sq_pair_diff = 0
    sum_sq_pair_score = 0.0

    for i in range(num_pairs):
        game_seed = seeds[i]

        # Game 1: A=P0, B=P1
        s1 = play_one_game_2p(agent_a, agent_b, rng=rng_action, game_seed=game_seed)
        p0, p1 = s1.players[0].points, s1.players[1].points
        sum_a += p0
        sum_b += p1
        diff1 = p0 - p1
        sum_diff += diff1
        sum_sq_diff += diff1 * diff1
        w = _winner_index_2p(s1)
        if w is None:
            draws += 1
            score1 = 0.5
        elif w == 0:
            wins_a += 1
            score1 = 1.0
        else:
            wins_b += 1
            score1 = 0.0

        # Game 2: B=P0, A=P1 (swap)
        s2 = play_one_game_2p(agent_b, agent_a, rng=rng_action, game_seed=game_seed)
        p0, p1 = s2.players[0].points, s2.players[1].points
        # Qui A è player 1.
        sum_a += p1
        sum_b += p0
        diff2 = p1 - p0
        sum_diff += diff2
        sum_sq_diff += diff2 * diff2
        w = _winner_index_2p(s2)
        if w is None:
            draws += 1
            score2 = 0.5
        elif w == 0:
            # winner = player 0 = agent B
            wins_b += 1
            score2 = 0.0
        else:
            wins_a += 1
            score2 = 1.0

        # Accumulo per COPPIA (l'unità statistica indipendente: stesso mazzo, seat scambiati).
        pair_diff = diff1 + diff2
        sum_sq_pair_diff += pair_diff * pair_diff
        pair_score = (score1 + score2) / 2.0
        sum_sq_pair_score += pair_score * pair_score

    return SeatFairStats(
        num_games=num_games,
        agent_a_name=agent_a.name,
        agent_b_name=agent_b.name,
        wins_agent_a=wins_a,
        wins_agent_b=wins_b,
        draws=draws,
        avg_points_agent_a=sum_a / num_games if num_games else 0.0,
        avg_points_agent_b=sum_b / num_games if num_games else 0.0,
        avg_point_diff_agent_a_minus_agent_b=sum_diff / num_games if num_games else 0.0,
        sum_sq_point_diff_agent_a_minus_agent_b=float(sum_sq_diff),
        sum_sq_pair_point_diff_agent_a_minus_agent_b=float(sum_sq_pair_diff),
        sum_sq_pair_score_agent_a=float(sum_sq_pair_score),
    )
