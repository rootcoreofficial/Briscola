"""
Self-play veloce basato sul fast path 2-player, senza event log completo.

Questo modulo serve per misurare e riusare roll-out 2-player ad alto throughput quando non serve
serializzare ogni osservazione/azione nel DB SQLite. È pensato come ponte verso training più veloce:
prima validiamo seed, policy semplici e aggregazione; poi possiamo integrare policy neurali.

Limite intenzionale:
- supporta solo gli agenti già tradotti nel path fast (`random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`);
- non produce dataset BC completo, perché non salva `PlayerObservation` step-by-step.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..evaluation import MatchStats
from .evaluation import FAST_EVALUATION_AGENT_NAMES, play_one_fast_game_2p
from .state_2p import Fast2PState


@dataclass(frozen=True, slots=True)
class FastSelfPlayGameSummary:
    """
    Riassunto minimale di una partita fast.

    Contiene solo dati riproducibili e aggregabili: seed, agenti, punti finali e vincitore.
    Non include osservazioni, azioni o carte intermedie, quindi resta leggero anche su run lunghe.
    """

    game_index: int
    game_seed: int
    action_seed: int
    agent0_name: str
    agent1_name: str
    points0: int
    points1: int
    winner_index: int | None

    def to_json_dict(self) -> dict[str, int | str | None]:
        """Rappresentazione JSON-friendly e stabile del riassunto partita."""
        return {
            "schema": "fast_self_play_summary_v1",
            "game_index": self.game_index,
            "game_seed": self.game_seed,
            "action_seed": self.action_seed,
            "agent0_name": self.agent0_name,
            "agent1_name": self.agent1_name,
            "points0": self.points0,
            "points1": self.points1,
            "winner_index": self.winner_index,
        }

    def to_json_line(self) -> str:
        """Serializza il riassunto come una riga JSONL."""
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":")) + "\n"


@dataclass(slots=True)
class FastSelfPlayAccumulator:
    """
    Accumulatore streaming per risultati self-play.

    Permette di processare run lunghe senza conservare tutti i `FastSelfPlayGameSummary` in memoria.
    """

    agent0_name: str
    agent1_name: str
    num_games: int = 0
    wins_agent0: int = 0
    wins_agent1: int = 0
    draws: int = 0
    sum_points0: int = 0
    sum_points1: int = 0

    def add(self, summary: FastSelfPlayGameSummary) -> None:
        """Aggiunge una partita all'aggregato."""
        if summary.agent0_name != self.agent0_name or summary.agent1_name != self.agent1_name:
            raise ValueError("Summary con agenti diversi dall'accumulatore")

        self.num_games += 1
        self.sum_points0 += int(summary.points0)
        self.sum_points1 += int(summary.points1)
        if summary.winner_index is None:
            self.draws += 1
        elif summary.winner_index == 0:
            self.wins_agent0 += 1
        elif summary.winner_index == 1:
            self.wins_agent1 += 1
        else:
            raise ValueError(f"winner_index non valido: {summary.winner_index}")

    def to_match_stats(self) -> MatchStats:
        """Converte l'aggregato nel DTO statistico già usato da evaluation."""
        num_games = int(self.num_games)
        return MatchStats(
            num_games=num_games,
            agent0_name=self.agent0_name,
            agent1_name=self.agent1_name,
            wins_agent0=int(self.wins_agent0),
            wins_agent1=int(self.wins_agent1),
            draws=int(self.draws),
            avg_points_agent0=self.sum_points0 / num_games if num_games else 0.0,
            avg_points_agent1=self.sum_points1 / num_games if num_games else 0.0,
            avg_point_diff_agent0_minus_agent1=(self.sum_points0 - self.sum_points1) / num_games if num_games else 0.0,
        )


def _winner_index_from_fast_state(state: Fast2PState) -> int | None:
    """Determina il vincitore finale dallo stato fast."""
    if state.points[0] > state.points[1]:
        return 0
    if state.points[1] > state.points[0]:
        return 1
    return None


def _validate_fast_self_play_agent(agent_name: str) -> None:
    """Valida un agente supportato dal self-play fast."""
    if agent_name not in FAST_EVALUATION_AGENT_NAMES:
        supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
        raise ValueError(f"Self-play fast supporta solo: {supported}. Ottenuto: {agent_name!r}")


def iter_fast_self_play_2p(
    *,
    agent0_name: str,
    agent1_name: str,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
) -> Iterator[FastSelfPlayGameSummary]:
    """
    Genera riassunti partita usando il fast path 2-player.

    Seed:
    - `game_seed` controlla lo shuffle;
    - `action_seed` controlla le scelte agenti ed è indipendente per ogni partita.

    Questa indipendenza rende i roll-out più facili da parallelizzare in futuro, perché una partita non
    dipende dallo stream RNG consumato da quelle precedenti.
    """
    _validate_fast_self_play_agent(agent0_name)
    _validate_fast_self_play_agent(agent1_name)
    if num_games < 0:
        raise ValueError("--num-games deve essere >= 0")

    rng_game = random.Random(seed)
    rng_action_master = random.Random(seed ^ 0x9E3779B9)
    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_games)]
    if len(seeds) < num_games:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_games}, ottenuti {len(seeds)}")

    for game_index in range(num_games):
        game_seed = int(seeds[game_index])
        action_seed = int(rng_action_master.randrange(0, 2**32))
        state = play_one_fast_game_2p(
            agent0_name,
            agent1_name,
            rng=random.Random(action_seed),
            game_seed=game_seed,
        )
        yield FastSelfPlayGameSummary(
            game_index=game_index,
            game_seed=game_seed,
            action_seed=action_seed,
            agent0_name=agent0_name,
            agent1_name=agent1_name,
            points0=int(state.points[0]),
            points1=int(state.points[1]),
            winner_index=_winner_index_from_fast_state(state),
        )


def run_fast_self_play_2p(
    *,
    agent0_name: str,
    agent1_name: str,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
) -> MatchStats:
    """
    Esegue self-play fast e ritorna solo statistiche aggregate.

    Usa streaming interno, quindi la memoria resta costante rispetto a `num_games`.
    """
    accumulator = FastSelfPlayAccumulator(agent0_name=agent0_name, agent1_name=agent1_name)
    for summary in iter_fast_self_play_2p(
        agent0_name=agent0_name,
        agent1_name=agent1_name,
        num_games=num_games,
        seed=seed,
        game_seeds=game_seeds,
    ):
        accumulator.add(summary)
    return accumulator.to_match_stats()
