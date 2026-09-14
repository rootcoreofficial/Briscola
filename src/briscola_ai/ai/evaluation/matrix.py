"""
Valutazione “a matrice” per modelli (seed suite × avversari).

Motivazione
-----------
Quando alleniamo molti modelli (.npz) è facile fare benchmark “a mano” in modo incoerente
o dimenticare un confronto (es. holdout). Una *evaluation matrix* standard:
- riduce errori manuali;
- rende confronti ripetibili nel tempo;
- fornisce una fotografia di robustezza (vs avversari diversi + holdout seed).

Questa implementazione è offline (no HTTP/WS). Il path di default riusa il dominio canonico;
quando `engine="numba"` usa il rollout MLP full-JIT per modelli `.npz` MLP, avversari fast-compatible
e un eventuale opponent `bc_model` MLP.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ..agents import build_agent
from ..fast.evaluation import FAST_EVALUATION_AGENT_NAMES
from ..models.bc_model import BCModelAgent, MLPBCModel
from ..numba.mlp import evaluate_mlp_policy_numba_2p
from .match import SeatFairStats, evaluate_seat_fair_match_2p

BenchmarkName = Literal["small", "medium", "big"]
EvaluationEngine = Literal["domain", "numba"]


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """Spec di una seed suite generata via range (per big/holdout)."""

    name: str
    range_start: int
    range_step: int
    num_seeds: int


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """Un risultato della matrice: (suite, opponent) -> stats."""

    suite: SuiteSpec
    opponent: str
    stats: SeatFairStats


@dataclass(frozen=True, slots=True)
class EvaluationMatrix:
    """Risultato completo (righe) + metadati di esecuzione."""

    model_path: str
    benchmark: BenchmarkName
    num_games: int
    seed: int
    rows: list[MatrixRow]
    engine: EvaluationEngine = "domain"

    def to_json_dict(self) -> dict:
        """Ritorna una struttura JSON-serializzabile (stabile) per persistere i risultati."""
        return {
            "model_path": self.model_path,
            "engine": self.engine,
            "benchmark": self.benchmark,
            "num_games": self.num_games,
            "seed": self.seed,
            "rows": [
                {
                    "suite": asdict(r.suite),
                    "opponent": r.opponent,
                    "stats": asdict(r.stats),
                }
                for r in self.rows
            ],
        }

    def to_json_text(self) -> str:
        """JSON pretty-printed."""
        return json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def benchmark_num_games(benchmark: BenchmarkName) -> int:
    """Taglie benchmark coerenti con `scripts/evaluate_agents.py` (tutte seat-fair)."""
    if benchmark == "small":
        return 2000
    if benchmark == "medium":
        return 10000
    if benchmark == "big":
        return 100000
    raise ValueError(f"Benchmark non supportato: {benchmark!r}")


def make_range_seed_suite(*, start: int, step: int, count: int) -> list[int]:
    """
    Genera una suite di seed tramite range aritmetico.

    Nota:
    normalizziamo i seed su 32 bit (coerente con uso tipico di RNG).
    """
    if count < 0:
        raise ValueError(f"count deve essere >= 0, ottenuto {count}")
    if step <= 0:
        raise ValueError(f"step deve essere > 0, ottenuto {step}")
    return [((start + i * step) & 0xFFFFFFFF) for i in range(count)]


def build_suites_for_benchmark(
    *,
    benchmark: BenchmarkName,
    standard_start: int = 0,
    holdout_start: int = 1_000_000,
    step: int = 1,
) -> list[SuiteSpec]:
    """
    Ritorna le due suite standard per la matrice:
    - `standard`: start=0
    - `holdout`: start=1_000_000 (default)
    """
    num_games = benchmark_num_games(benchmark)
    needed_seeds = num_games // 2
    return [
        SuiteSpec(name="standard", range_start=standard_start, range_step=step, num_seeds=needed_seeds),
        SuiteSpec(name="holdout", range_start=holdout_start, range_step=step, num_seeds=needed_seeds),
    ]


def _evaluate_numba_model_vs_opponent(
    *,
    model_agent: BCModelAgent,
    opponent_name: str,
    opponent_agent: BCModelAgent | None = None,
    num_games: int,
    seed: int,
    game_seeds: list[int],
) -> SeatFairStats:
    """
    Valuta un modello MLP `.npz` con il core Numba e ritorna lo stesso DTO del path dominio.

    Il modello sotto test è sempre Agent A. L'opponent può essere una baseline fast-compatible
    oppure un secondo modello MLP (`bc_model`) caricato dal caller.
    """
    if opponent_agent is None and opponent_name not in FAST_EVALUATION_AGENT_NAMES:
        supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
        raise ValueError(f"`engine=numba` supporta opponent: {supported} oppure bc_model. Ottenuto: {opponent_name!r}")

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

    summary = evaluate_mlp_policy_numba_2p(
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        opponent_name=opponent_label,
        num_games=int(num_games),
        seed=int(seed),
        seat_fair=True,
        game_seeds=game_seeds,
        deterministic=True,
        policy_overkill_guard=bool(model_agent.overkill_guard_enabled),
        parallel=True,
        opponent_w1=opponent_model.w1 if opponent_model is not None else None,
        opponent_b1=opponent_model.b1 if opponent_model is not None else None,
        opponent_w2=opponent_model.w2 if opponent_model is not None else None,
        opponent_b2=opponent_model.b2 if opponent_model is not None else None,
        opponent_overkill_guard=bool(opponent_agent.overkill_guard_enabled) if opponent_agent is not None else False,
        policy_name=model_agent.name,
    )
    return summary.to_seat_fair_stats()


def _build_domain_matrix_opponent(opponent_name: str, opponent_model_path: str):
    """Costruisce l'opponent del path dominio, incluso `bc_model` con path esplicito."""
    if opponent_name == "bc_model":
        if not opponent_model_path:
            raise ValueError("`bc_model` nella matrix richiede `opponent_model_path`.")
        return build_agent("bc_model", model_path=Path(opponent_model_path))
    return build_agent(opponent_name)


def _evaluate_matrix_row_job(args: tuple[str, str, str, int, SuiteSpec, list[int], EvaluationEngine]) -> MatrixRow:
    """Valuta una singola riga della matrix. Funzione top-level per multiprocessing."""
    model_path, opponent_name, opponent_model_path, seed, suite, seeds, engine = args
    model = BCModelAgent.from_npz(model_path)
    opponent_agent = BCModelAgent.from_npz(opponent_model_path) if opponent_name == "bc_model" else None
    if engine == "numba":
        stats = _evaluate_numba_model_vs_opponent(
            model_agent=model,
            opponent_name=opponent_name,
            opponent_agent=opponent_agent,
            num_games=len(seeds) * 2,
            seed=int(seed),
            game_seeds=seeds,
        )
    else:
        opponent = _build_domain_matrix_opponent(opponent_name, opponent_model_path)
        stats = evaluate_seat_fair_match_2p(
            model,
            opponent,
            num_games=len(seeds) * 2,
            seed=int(seed),
            game_seeds=seeds,
        )
    return MatrixRow(suite=suite, opponent=stats.agent_b_name, stats=stats)


def evaluate_model_matrix(
    *,
    model_path: str | Path,
    opponents: list[str],
    benchmark: BenchmarkName,
    seed: int,
    standard_start: int = 0,
    holdout_start: int = 1_000_000,
    range_step: int = 1,
    workers: int = 1,
    engine: EvaluationEngine = "domain",
    opponent_model_path: str | Path | None = None,
) -> EvaluationMatrix:
    """
    Valuta un modello `.npz` contro una lista di avversari su 2 suite (standard + holdout).

    Scelte:
    - sempre seat-fair (riduce bias “chi inizia”)
    - suite generate via range (evitiamo file enormi)
    """
    num_games = benchmark_num_games(benchmark)

    suites = build_suites_for_benchmark(
        benchmark=benchmark,
        standard_start=standard_start,
        holdout_start=holdout_start,
        step=range_step,
    )

    if engine not in ("domain", "numba"):
        raise ValueError(f"engine non supportato: {engine!r}")
    opponent_model_path_str = str(Path(opponent_model_path)) if opponent_model_path is not None else ""
    if "bc_model" in opponents and not opponent_model_path_str:
        raise ValueError("`bc_model` negli opponent della matrix richiede `opponent_model_path`.")
    if opponent_model_path_str and "bc_model" not in opponents:
        raise ValueError("`opponent_model_path` è valido solo se `opponents` contiene `bc_model`.")

    effective_workers = 1 if engine == "numba" else int(workers)

    jobs: list[tuple[str, str, str, int, SuiteSpec, list[int], EvaluationEngine]] = []
    for suite in suites:
        seeds = make_range_seed_suite(start=suite.range_start, step=suite.range_step, count=suite.num_seeds)
        for opp_name in opponents:
            jobs.append((str(Path(model_path)), opp_name, opponent_model_path_str, int(seed), suite, seeds, engine))

    if effective_workers <= 1:
        model = BCModelAgent.from_npz(model_path)
        rows = []
        opponent_agent = BCModelAgent.from_npz(opponent_model_path_str) if opponent_model_path_str else None
        for _, opp_name, opp_model_path, _, suite, seeds, _ in jobs:
            if engine == "numba":
                stats = _evaluate_numba_model_vs_opponent(
                    model_agent=model,
                    opponent_name=opp_name,
                    opponent_agent=opponent_agent if opp_name == "bc_model" else None,
                    num_games=num_games,
                    seed=seed,
                    game_seeds=seeds,
                )
            else:
                opponent = _build_domain_matrix_opponent(opp_name, opp_model_path)
                stats = evaluate_seat_fair_match_2p(
                    model,
                    opponent,
                    num_games=num_games,
                    seed=seed,
                    game_seeds=seeds,
                )
            rows.append(MatrixRow(suite=suite, opponent=stats.agent_b_name, stats=stats))
    else:
        worker_count = max(1, min(effective_workers, len(jobs)))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(_evaluate_matrix_row_job, jobs))

    return EvaluationMatrix(
        model_path=str(Path(model_path)),
        benchmark=benchmark,
        num_games=num_games,
        seed=int(seed),
        rows=rows,
        engine=engine,
    )


def default_opponents() -> list[str]:
    """Avversari baseline consigliati per una matrice minima."""
    return ["heuristic_v1", "random", "greedy_points"]


def format_matrix_table(matrix: EvaluationMatrix) -> str:
    """
    Ritorna una tabella testuale compatta.

    Mostriamo:
    - opponent
    - suite
    - avg_point_diff (A-B) dove A = modello, B = avversario
    - win/loss/draw
    """
    lines = []
    lines.append(
        "Evaluation matrix | "
        f"model={Path(matrix.model_path).name} | "
        f"engine={matrix.engine} | "
        f"benchmark={matrix.benchmark} games={matrix.num_games}"
    )
    lines.append("opponent,suite,avg_diff,wins_model,wins_opp,draws")
    for r in matrix.rows:
        lines.append(
            f"{r.opponent},{r.suite.name},{r.stats.avg_point_diff_agent_a_minus_agent_b:.2f},"
            f"{r.stats.wins_agent_a},{r.stats.wins_agent_b},{r.stats.draws}"
        )
    return "\n".join(lines) + "\n"
