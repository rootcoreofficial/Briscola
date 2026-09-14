#!/usr/bin/env python3
"""Confronta training A2C seriale e realmente paired su tre seed.

Il protocollo distingue due domande che non vanno confuse:

1. a pari partite/update, il pairing dimezza i mazzi distinti;
2. a pari mazzi distinti, il pairing usa il doppio delle partite e degli update.

Ogni modello viene valutato sulla stessa suite seat-fair contro v14. Per ciascun seed
confrontiamo inoltre direttamente i due modelli paired con il controllo seriale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from briscola_ai.ai.evaluation.match import SeatFairStats
from briscola_ai.ai.evaluation.round_robin import (
    seat_fair_avg_point_diff_ci,
    seat_fair_score_rate_ci,
)
from briscola_ai.ai.training.game_schedule import (
    ScheduledTrainingGame,
    build_training_game_schedule,
    training_schedule_sha256,
)
from briscola_ai.ai.training.opponent_mix import parse_opponent_mix
from briscola_ai.versioning import get_code_version, get_rules_version

SCHEMA = "briscola.a2c_paired_schedule_probe.v1"
RECEIPT_SCHEMA = "briscola.a2c_training_receipt.v1"
EVALUATION_RECEIPT_SCHEMA = "briscola.a2c_evaluation_receipt.v1"
DEFAULT_SEEDS = (20260717, 20260718, 20260719)
DEFAULT_OPPONENT_MIX = (
    "bc_model:0.15,bc_model_pimc_belief:0.40,"
    "bc_model_value_lookahead_8x8:0.20,heuristic_trump_saver:0.12,"
    "heuristic_v1:0.04,heuristic_v2:0.06,random:0.03"
)


@dataclass(frozen=True, slots=True)
class Regime:
    """Una delle tre condizioni preregistrate del confronto."""

    name: str
    schedule: Literal["serial", "paired"]
    game_multiplier: int


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Soglie di screening, non gate di promozione."""

    direct_median_point_diff_min: float = 0.0
    direct_nonnegative_seed_count_min: int = 2
    strength_between_seed_std_ratio_max: float = 1.0
    gradient_cv_ratio_max: float = 1.0
    stop_direct_median_point_diff_max: float = -0.25
    stop_negative_seed_count_min: int = 2
    stop_variance_ratio_min: float = 1.10


REGIMES = (
    Regime("serial_same_games", "serial", 1),
    Regime("paired_same_games", "paired", 1),
    Regime("paired_same_decks", "paired", 2),
)


def _repo_root() -> Path:
    """Ritorna la root del checkout indipendentemente dalla cwd."""
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    """Calcola SHA-256 streaming per identificare input e output."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path) -> str:
    """Preferisce un path relativo al repository nei report."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path, *, root: Path) -> dict[str, str | int]:
    """Descrive un artefatto senza incorporarne il contenuto."""
    return {
        "path": _display_path(path, root=root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    """Carica un oggetto JSON rigoroso e rifiuta NaN/Infinity."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Costante JSON non standard {value!r} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"Oggetto JSON atteso in {path}")
    return payload


def _model_metadata(path: Path) -> dict[str, Any]:
    """Legge soltanto i metadati JSON di un modello `.npz`."""
    with np.load(path, allow_pickle=False) as archive:
        payload = json.loads(str(archive["metadata_json"]))
    if not isinstance(payload, dict):
        raise ValueError(f"Metadati modello non validi: {path}")
    return payload


def _parse_seeds(raw: str) -> tuple[int, ...]:
    """Converte una lista CSV in tre seed distinti."""
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--seeds richiede interi separati da virgola") from exc
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Il probe richiede esattamente tre seed distinti")
    return seeds


def _expected_schedule(
    *,
    seed: int,
    regime: Regime,
    base_games: int,
    update_every: int,
    opponent_mix: str,
) -> tuple[ScheduledTrainingGame, ...]:
    """Ricostruisce indipendentemente la schedule che il trainer deve dichiarare."""
    return build_training_game_schedule(
        num_games=base_games * regime.game_multiplier,
        update_every=update_every,
        mode=regime.schedule,
        seat_fair=True,
        default_opponent_name="unused_with_mix",
        opponent_mix=parse_opponent_mix(opponent_mix),
        rng_game=np.random.default_rng(seed ^ 0x9E3779B9),
        rng_opponent=np.random.default_rng(seed ^ 0xA5A5A5A5),
    )


def environment_sequence(schedule: tuple[ScheduledTrainingGame, ...]) -> tuple[tuple[int, str], ...]:
    """Riduce una schedule alle estrazioni indipendenti di mazzo e opponent."""
    if schedule and schedule[0].pair_index is not None:
        return tuple((schedule[index].game_seed, schedule[index].opponent_name) for index in range(0, len(schedule), 2))
    return tuple((game.game_seed, game.opponent_name) for game in schedule)


def validate_environment_alignment(schedules: dict[str, tuple[ScheduledTrainingGame, ...]]) -> dict[str, Any]:
    """Prova che i confronti riusino esattamente i pool dichiarati."""
    serial = environment_sequence(schedules["serial_same_games"])
    paired_games = environment_sequence(schedules["paired_same_games"])
    paired_decks = environment_sequence(schedules["paired_same_decks"])
    same_games_uses_serial_prefix = paired_games == serial[: len(paired_games)]
    same_decks_exact = paired_decks == serial
    if not same_games_uses_serial_prefix or not same_decks_exact:
        raise ValueError("Le schedule non rispettano l'allineamento preregistrato degli ambienti")
    return {
        "paired_same_games_uses_serial_environment_prefix": same_games_uses_serial_prefix,
        "paired_same_decks_matches_serial_environments": same_decks_exact,
        "serial_environment_count": len(serial),
        "paired_same_games_environment_count": len(paired_games),
        "paired_same_decks_environment_count": len(paired_decks),
    }


def _training_command(
    *,
    root: Path,
    regime: Regime,
    model_out: Path,
    diagnostics_out: Path,
    init_model: Path,
    opponent_model: Path,
    belief_model: Path,
    value_model: Path,
    opponent_mix: str,
    base_games: int,
    update_every: int,
    seed: int,
) -> list[str]:
    """Costruisce la ricetta v14 cambiando soltanto schedule e budget dichiarato."""
    num_games = base_games * regime.game_multiplier
    command = [
        sys.executable,
        str(root / "scripts/train_a2c.py"),
        "--out",
        str(model_out),
        "--diagnostics-json",
        str(diagnostics_out),
        "--init",
        str(init_model),
        "--encoder-version",
        "v4",
        "--rollout-engine",
        "fast",
        "--fast-rollout",
        "numba",
        "--training-schedule",
        regime.schedule,
        "--opponent-mix",
        opponent_mix,
        "--opponent-model",
        str(opponent_model),
        "--opponent-belief-model",
        str(belief_model),
        "--opponent-pimc-determinizations",
        "16",
        "--opponent-value-model",
        str(value_model),
        "--opponent-value-max-unknown-cards",
        "8",
        "--bc-anchor",
        str(init_model),
        "--bc-anchor-beta",
        "0.01",
        "--overkill-penalty-mode",
        "gap",
        "--overkill-penalty-beta",
        "0.3",
        "--num-games",
        str(num_games),
        "--update-every",
        str(update_every),
        "--log-every",
        str(max(1, num_games // update_every // 10)),
        "--metrics-mode",
        "summary",
        "--seed",
        str(seed),
    ]
    if regime.schedule == "serial":
        command.append("--seat-fair")
    return command


def _evaluation_command(
    *,
    root: Path,
    agent_a_model: Path,
    agent_b_model: Path,
    out_json: Path,
    eval_games: int,
    eval_seed_start: int,
) -> list[str]:
    """Confronta due MLP sulla stessa suite deterministica e seat-fair."""
    return [
        sys.executable,
        str(root / "scripts/evaluate_agents.py"),
        "--engine",
        "numba",
        "--num-games",
        str(eval_games),
        "--seat-fair",
        "--seed-suite-range-start",
        str(eval_seed_start),
        "--agent0",
        "bc_model",
        "--agent0-model",
        str(agent_a_model),
        "--agent1",
        "bc_model",
        "--agent1-model",
        str(agent_b_model),
        "--out-json",
        str(out_json),
    ]


def _write_receipt(
    path: Path,
    *,
    elapsed_seconds: float,
    command: list[str],
    model: Path,
    diagnostics: Path,
    root: Path,
) -> None:
    """Salva il costo del sottoprocesso, altrimenti perso in una ripresa parziale."""
    payload = {
        "schema": RECEIPT_SCHEMA,
        "elapsed_seconds": elapsed_seconds,
        "command": command,
        "artifacts": {
            "model": _artifact(model, root=root),
            "diagnostics": _artifact(diagnostics, root=root),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _run_training(
    *,
    command: list[str],
    model: Path,
    diagnostics: Path,
    receipt: Path,
    root: Path,
    resume: bool,
) -> dict[str, Any]:
    """Esegue o riprende un training solo quando tutti gli artefatti sono coerenti."""
    exists = (model.exists(), diagnostics.exists(), receipt.exists())
    if any(exists) and not all(exists):
        raise FileExistsError(f"Training parziale: model/diagnostics/receipt={exists}")
    if all(exists):
        if not resume:
            raise FileExistsError(f"Training già presente: {model}; usa --resume")
        payload = _load_json(receipt)
        if payload.get("schema") != RECEIPT_SCHEMA:
            raise ValueError(f"Receipt incompatibile: {receipt}")
        artifacts = payload.get("artifacts")
        if (
            not isinstance(artifacts, dict)
            or not isinstance(artifacts.get("model"), dict)
            or artifacts["model"].get("sha256") != _sha256(model)
            or not isinstance(artifacts.get("diagnostics"), dict)
            or artifacts["diagnostics"].get("sha256") != _sha256(diagnostics)
        ):
            raise ValueError(f"Artefatti cambiati rispetto al receipt: {receipt}")
        print(f"resume: {model.name}", flush=True)
        return payload

    print("\n+ " + " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True)
    elapsed = time.perf_counter() - started
    _write_receipt(
        receipt,
        elapsed_seconds=elapsed,
        command=command,
        model=model,
        diagnostics=diagnostics,
        root=root,
    )
    return _load_json(receipt)


def _validate_training(
    *,
    model: Path,
    diagnostics: Path,
    expected_schedule: tuple[ScheduledTrainingGame, ...],
    regime: Regime,
    seed: int,
    num_games: int,
    update_every: int,
    init_sha256: str,
) -> None:
    """Impedisce che un artefatto di un'altra ricetta entri nel confronto."""
    metadata = _model_metadata(model)
    train = metadata.get("train")
    report = _load_json(diagnostics)
    config = report.get("config")
    artifacts = report.get("artifacts")
    if not isinstance(train, dict) or not isinstance(config, dict) or not isinstance(artifacts, dict):
        raise ValueError(f"Metadati training incompleti: {model}")
    schedule_metadata = train.get("training_schedule")
    init_artifact = artifacts.get("init")
    expected_digest = training_schedule_sha256(expected_schedule)
    if (
        metadata.get("seed") != seed
        or train.get("num_games") != num_games
        or train.get("update_every") != update_every
        or not isinstance(schedule_metadata, dict)
        or schedule_metadata.get("mode") != regime.schedule
        or schedule_metadata.get("sha256") != expected_digest
        or config.get("training_schedule") != regime.schedule
        or config.get("training_schedule_sha256") != expected_digest
        or not isinstance(init_artifact, dict)
        or init_artifact.get("sha256") != init_sha256
    ):
        raise ValueError(f"Training incompatibile con il protocollo: {model}")


def _run_evaluation(
    *,
    command: list[str],
    out_json: Path,
    root: Path,
    resume: bool,
    eval_games: int,
    eval_seed_start: int,
    agent_a_model: Path,
    agent_b_model: Path,
) -> dict[str, Any]:
    """Esegue o riprende una valutazione sulla suite comune."""
    receipt_path = out_json.with_name(f"{out_json.stem}.receipt.json")
    exists = (out_json.exists(), receipt_path.exists())
    if any(exists) and not all(exists):
        raise FileExistsError(f"Valutazione parziale output/receipt={exists}: {out_json}")
    if all(exists):
        if not resume:
            raise FileExistsError(f"Valutazione già presente: {out_json}; usa --resume")
        payload = _load_json(out_json)
        receipt = _load_json(receipt_path)
        models = receipt.get("models")
        if (
            receipt.get("schema") != EVALUATION_RECEIPT_SCHEMA
            or not isinstance(models, dict)
            or not isinstance(models.get("agent_a"), dict)
            or models["agent_a"].get("sha256") != _sha256(agent_a_model)
            or not isinstance(models.get("agent_b"), dict)
            or models["agent_b"].get("sha256") != _sha256(agent_b_model)
        ):
            raise ValueError(f"Receipt valutazione incompatibile: {receipt_path}")
        print(f"resume: {out_json.name}", flush=True)
    else:
        print("\n+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)
        payload = _load_json(out_json)
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": EVALUATION_RECEIPT_SCHEMA,
                    "command": command,
                    "models": {
                        "agent_a": _artifact(agent_a_model, root=root),
                        "agent_b": _artifact(agent_b_model, root=root),
                    },
                    "evaluation": _artifact(out_json, root=root),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    suite = payload.get("seed_suite")
    if (
        payload.get("mode") != "seat_fair"
        or payload.get("engine") != "numba"
        or payload.get("num_games") != eval_games
        or not isinstance(suite, dict)
        or suite.get("range_start") != eval_seed_start
    ):
        raise ValueError(f"Valutazione incompatibile: {out_json}")
    return payload


def summarize_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggiunge CI corrette per coppia alle statistiche di evaluate_agents."""
    raw_stats = payload.get("stats")
    if not isinstance(raw_stats, dict):
        raise ValueError("Valutazione senza stats")
    stats = SeatFairStats(**raw_stats)
    point_ci = seat_fair_avg_point_diff_ci(stats, confidence=0.95)
    score_ci = seat_fair_score_rate_ci(stats, confidence=0.95)
    return {
        "avg_point_diff_agent_a_minus_agent_b": stats.avg_point_diff_agent_a_minus_agent_b,
        "avg_point_diff_ci95": asdict(point_ci) if point_ci is not None else None,
        "score_rate_agent_a": (stats.wins_agent_a + 0.5 * stats.draws) / stats.num_games,
        "score_rate_ci95": asdict(score_ci),
        "wins_agent_a": stats.wins_agent_a,
        "wins_agent_b": stats.wins_agent_b,
        "draws": stats.draws,
        "num_games": stats.num_games,
    }


def summarize_diagnostics(payload: dict[str, Any]) -> dict[str, float | int]:
    """Estrae proxy di rumore e salute dagli update completi."""
    raw_updates = payload.get("updates")
    if not isinstance(raw_updates, list) or not raw_updates:
        raise ValueError("Diagnostica senza update")
    gradients = np.asarray([float(row["global_gradient_l2"]) for row in raw_updates], dtype=np.float64)
    advantage_means = np.asarray(
        [float(row["signals"]["advantage_mean"]) for row in raw_updates],
        dtype=np.float64,
    )
    late = raw_updates[len(raw_updates) // 2 :]
    critic_values = np.asarray(
        [
            float(row["signals"]["critic_explained_variance"])
            for row in late
            if row["signals"]["critic_explained_variance"] is not None
        ],
        dtype=np.float64,
    )
    if not (
        bool(np.all(np.isfinite(gradients)))
        and bool(np.all(np.isfinite(advantage_means)))
        and bool(np.all(np.isfinite(critic_values)))
    ):
        raise ValueError("Diagnostica con valori non finiti")
    gradient_mean = float(np.mean(gradients))
    return {
        "updates": len(raw_updates),
        "global_gradient_cv": float(np.std(gradients) / gradient_mean) if gradient_mean > 1e-12 else 0.0,
        "global_gradient_p95_over_median": (
            float(np.quantile(gradients, 0.95) / np.median(gradients)) if float(np.median(gradients)) > 1e-12 else 0.0
        ),
        "advantage_mean_std": float(np.std(advantage_means)),
        "late_critic_explained_variance_median": (float(np.median(critic_values)) if critic_values.size else 0.0),
    }


def actor_relative_delta(model_path: Path, init_path: Path) -> float:
    """Misura quanto i quattro array della policy si sono mossi dall'init."""
    delta_sq = 0.0
    init_sq = 0.0
    with np.load(model_path, allow_pickle=False) as model, np.load(init_path, allow_pickle=False) as init:
        for name in ("w1", "b1", "w2", "b2"):
            current = np.asarray(model[name], dtype=np.float64)
            initial = np.asarray(init[name], dtype=np.float64)
            delta_sq += float(np.sum((current - initial) ** 2, dtype=np.float64))
            init_sq += float(np.sum(initial**2, dtype=np.float64))
    return float(np.sqrt(delta_sq) / np.sqrt(init_sq))


def _distribution(values: list[float]) -> dict[str, float | int]:
    """Riassume tre seed senza nascondere i valori individuali."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def decide(
    *,
    regime_summaries: dict[str, dict[str, Any]],
    direct_same_games: list[dict[str, Any]],
    thresholds: DecisionThresholds,
) -> dict[str, Any]:
    """Applica il gate primario al confronto paired-vs-serial a pari partite."""
    direct_diffs = [float(row["avg_point_diff_agent_a_minus_agent_b"]) for row in direct_same_games]
    direct_distribution = _distribution(direct_diffs)
    nonnegative = sum(value >= 0.0 for value in direct_diffs)
    negative = sum(value < 0.0 for value in direct_diffs)
    serial_strength_std = float(regime_summaries["serial_same_games"]["vs_v14_point_diff"]["sample_std"])
    paired_strength_std = float(regime_summaries["paired_same_games"]["vs_v14_point_diff"]["sample_std"])
    serial_gradient_cv = float(regime_summaries["serial_same_games"]["global_gradient_cv"]["median"])
    paired_gradient_cv = float(regime_summaries["paired_same_games"]["global_gradient_cv"]["median"])
    strength_std_ratio = paired_strength_std / serial_strength_std if serial_strength_std > 1e-12 else None
    gradient_cv_ratio = paired_gradient_cv / serial_gradient_cv if serial_gradient_cv > 1e-12 else None
    variance_signal = (
        strength_std_ratio is not None
        and strength_std_ratio <= thresholds.strength_between_seed_std_ratio_max
        and gradient_cv_ratio is not None
        and gradient_cv_ratio <= thresholds.gradient_cv_ratio_max
    )
    go = (
        direct_distribution["median"] >= thresholds.direct_median_point_diff_min
        and nonnegative >= thresholds.direct_nonnegative_seed_count_min
        and variance_signal
    )
    stop_strength = (
        direct_distribution["median"] <= thresholds.stop_direct_median_point_diff_max
        and negative >= thresholds.stop_negative_seed_count_min
    )
    stop_variance = (
        strength_std_ratio is not None
        and gradient_cv_ratio is not None
        and strength_std_ratio >= thresholds.stop_variance_ratio_min
        and gradient_cv_ratio >= thresholds.stop_variance_ratio_min
    )
    if go:
        verdict = "go_longer_paired_screen"
        explanation = (
            "A pari partite il paired non regredisce nella maggioranza dei seed e riduce sia la dispersione "
            "di forza sia quella dei gradienti: è autorizzato un solo screen più lungo."
        )
    elif stop_strength or stop_variance:
        verdict = "stop_paired_schedule"
        explanation = (
            "Il paired peggiora materialmente la forza o aumenta entrambe le misure di variabilità: "
            "la schedule seriale resta il default."
        )
    else:
        verdict = "inconclusive_keep_serial"
        explanation = (
            "I tre seed non mostrano insieme non-regressione e riduzione della variabilità: "
            "manca evidenza per sostituire la schedule seriale."
        )
    return {
        "verdict": verdict,
        "explanation_it": explanation,
        "direct_same_games_point_diff": direct_distribution,
        "direct_nonnegative_seed_count": nonnegative,
        "strength_between_seed_std_ratio_paired_over_serial": strength_std_ratio,
        "gradient_cv_ratio_paired_over_serial": gradient_cv_ratio,
        "variance_signal": variance_signal,
        "stop_strength": stop_strength,
        "stop_variance": stop_variance,
    }


def _git_commit() -> str | None:
    """Commit corrente best-effort per la riproducibilità."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError, subprocess.SubprocessError:
        return None


def _parse_args() -> argparse.Namespace:
    """Definisce il protocollo lungo e i path degli asset ufficiali."""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="A/B multi-seed della training schedule A2C paired")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=root / "benchmarks/experiments/a2c_paired_schedule_v0_20260714",
    )
    parser.add_argument("--init-model", type=Path, default=root / "data/models/best_a2c_v14.npz")
    parser.add_argument("--opponent-model", type=Path, default=root / "data/models/best_a2c_v14.npz")
    parser.add_argument(
        "--belief-model",
        type=Path,
        default=root / "data/models/belief_v0_h128_50k_seed20260702.npz",
    )
    parser.add_argument(
        "--value-model",
        type=Path,
        default=root / "data/models/value_v1_v4_fullgame_h128_seed20260718.npz",
    )
    parser.add_argument("--opponent-mix", default=DEFAULT_OPPONENT_MIX)
    parser.add_argument("--base-games", type=int, default=20_000)
    parser.add_argument("--update-every", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=4_000)
    parser.add_argument("--eval-seed-start", type=int, default=4_000_000)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Esegue training, valutazioni e decisione preregistrata."""
    args = _parse_args()
    root = _repo_root()
    seeds = _parse_seeds(str(args.seeds))
    base_games = int(args.base_games)
    update_every = int(args.update_every)
    eval_games = int(args.eval_games)
    if (
        base_games <= 0
        or update_every <= 0
        or base_games % update_every != 0
        or base_games % 2 != 0
        or update_every % 2 != 0
    ):
        raise ValueError("--base-games deve essere pari e multiplo di --update-every, anch'esso pari")
    if eval_games <= 0 or eval_games % 2 != 0:
        raise ValueError("--eval-games deve essere positivo e pari")
    input_paths = (args.init_model, args.opponent_model, args.belief_model, args.value_model)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    final_path = work_dir / f"a2c_paired_schedule_g{base_games}_3seeds.json"
    if final_path.exists() and not args.resume:
        raise FileExistsError(f"Report già presente: {final_path}; usa --resume")
    thresholds = DecisionThresholds()
    init_sha256 = _sha256(args.init_model)
    models: dict[tuple[int, str], Path] = {}
    runs: list[dict[str, Any]] = []
    alignment_by_seed: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        expected_by_regime = {
            regime.name: _expected_schedule(
                seed=seed,
                regime=regime,
                base_games=base_games,
                update_every=update_every,
                opponent_mix=str(args.opponent_mix),
            )
            for regime in REGIMES
        }
        alignment_by_seed[str(seed)] = validate_environment_alignment(expected_by_regime)
        for regime in REGIMES:
            num_games = base_games * regime.game_multiplier
            stem = f"{regime.name}_g{num_games}_seed{seed}"
            model = work_dir / f"{stem}.npz"
            diagnostics = work_dir / f"{stem}.diagnostics.json"
            receipt = work_dir / f"{stem}.receipt.json"
            command = _training_command(
                root=root,
                regime=regime,
                model_out=model,
                diagnostics_out=diagnostics,
                init_model=args.init_model,
                opponent_model=args.opponent_model,
                belief_model=args.belief_model,
                value_model=args.value_model,
                opponent_mix=str(args.opponent_mix),
                base_games=base_games,
                update_every=update_every,
                seed=seed,
            )
            receipt_payload = _run_training(
                command=command,
                model=model,
                diagnostics=diagnostics,
                receipt=receipt,
                root=root,
                resume=bool(args.resume),
            )
            _validate_training(
                model=model,
                diagnostics=diagnostics,
                expected_schedule=expected_by_regime[regime.name],
                regime=regime,
                seed=seed,
                num_games=num_games,
                update_every=update_every,
                init_sha256=init_sha256,
            )
            models[(seed, regime.name)] = model
            runs.append(
                {
                    "seed": seed,
                    "regime": regime.name,
                    "schedule": regime.schedule,
                    "num_games": num_games,
                    "environment_draws": len(environment_sequence(expected_by_regime[regime.name])),
                    "schedule_sha256": training_schedule_sha256(expected_by_regime[regime.name]),
                    "elapsed_seconds": float(receipt_payload["elapsed_seconds"]),
                    "actor_relative_delta_from_init": actor_relative_delta(model, args.init_model),
                    "diagnostics": summarize_diagnostics(_load_json(diagnostics)),
                    "artifacts": {
                        "model": _artifact(model, root=root),
                        "diagnostics": _artifact(diagnostics, root=root),
                        "receipt": _artifact(receipt, root=root),
                    },
                }
            )

    reference_evaluations: dict[tuple[int, str], dict[str, Any]] = {}
    direct_evaluations: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in seeds:
        for regime in REGIMES:
            model = models[(seed, regime.name)]
            out_json = work_dir / f"eval_{regime.name}_seed{seed}_vs_v14.json"
            payload = _run_evaluation(
                command=_evaluation_command(
                    root=root,
                    agent_a_model=model,
                    agent_b_model=args.init_model,
                    out_json=out_json,
                    eval_games=eval_games,
                    eval_seed_start=int(args.eval_seed_start),
                ),
                out_json=out_json,
                root=root,
                resume=bool(args.resume),
                eval_games=eval_games,
                eval_seed_start=int(args.eval_seed_start),
                agent_a_model=model,
                agent_b_model=args.init_model,
            )
            reference_evaluations[(seed, regime.name)] = {
                "summary": summarize_evaluation(payload),
                "artifact": _artifact(out_json, root=root),
            }

        serial_model = models[(seed, "serial_same_games")]
        for paired_name in ("paired_same_games", "paired_same_decks"):
            out_json = work_dir / f"eval_{paired_name}_vs_serial_seed{seed}.json"
            payload = _run_evaluation(
                command=_evaluation_command(
                    root=root,
                    agent_a_model=models[(seed, paired_name)],
                    agent_b_model=serial_model,
                    out_json=out_json,
                    eval_games=eval_games,
                    eval_seed_start=int(args.eval_seed_start),
                ),
                out_json=out_json,
                root=root,
                resume=bool(args.resume),
                eval_games=eval_games,
                eval_seed_start=int(args.eval_seed_start),
                agent_a_model=models[(seed, paired_name)],
                agent_b_model=serial_model,
            )
            direct_evaluations[(seed, paired_name)] = {
                "summary": summarize_evaluation(payload),
                "artifact": _artifact(out_json, root=root),
            }

    regime_summaries: dict[str, dict[str, Any]] = {}
    for regime in REGIMES:
        regime_runs = [row for row in runs if row["regime"] == regime.name]
        regime_summaries[regime.name] = {
            "vs_v14_point_diff": _distribution(
                [
                    float(reference_evaluations[(seed, regime.name)]["summary"]["avg_point_diff_agent_a_minus_agent_b"])
                    for seed in seeds
                ]
            ),
            "global_gradient_cv": _distribution(
                [float(row["diagnostics"]["global_gradient_cv"]) for row in regime_runs]
            ),
            "advantage_mean_std": _distribution(
                [float(row["diagnostics"]["advantage_mean_std"]) for row in regime_runs]
            ),
            "actor_relative_delta_from_init": _distribution(
                [float(row["actor_relative_delta_from_init"]) for row in regime_runs]
            ),
            "elapsed_seconds": _distribution([float(row["elapsed_seconds"]) for row in regime_runs]),
        }

    direct_same_games = [direct_evaluations[(seed, "paired_same_games")]["summary"] for seed in seeds]
    direct_same_decks = [direct_evaluations[(seed, "paired_same_decks")]["summary"] for seed in seeds]
    report = {
        "schema": SCHEMA,
        "protocol": {
            "seeds": list(seeds),
            "base_games": base_games,
            "update_every": update_every,
            "eval_games": eval_games,
            "eval_seed_start": int(args.eval_seed_start),
            "regimes": [asdict(regime) for regime in REGIMES],
            "thresholds": asdict(thresholds),
            "primary_comparison": "paired_same_games_vs_serial_same_games",
            "same_decks_is_supportive_only": True,
        },
        "recipe": {
            "encoder_version": "v4",
            "rollout_engine": "fast",
            "fast_rollout": "numba",
            "opponent_mix": str(args.opponent_mix),
            "opponent_pimc_determinizations": 16,
            "opponent_value_max_unknown_cards": 8,
            "bc_anchor_beta": 0.01,
            "overkill_penalty_mode": "gap",
            "overkill_penalty_beta": 0.3,
        },
        "inputs": {
            "init_model": _artifact(args.init_model, root=root),
            "opponent_model": _artifact(args.opponent_model, root=root),
            "belief_model": _artifact(args.belief_model, root=root),
            "value_model": _artifact(args.value_model, root=root),
        },
        "environment_alignment": alignment_by_seed,
        "runs": runs,
        "reference_evaluations": [
            {"seed": seed, "regime": regime.name, **reference_evaluations[(seed, regime.name)]}
            for seed in seeds
            for regime in REGIMES
        ],
        "direct_evaluations": {
            "paired_same_games_vs_serial": [
                {"seed": seed, **direct_evaluations[(seed, "paired_same_games")]} for seed in seeds
            ],
            "paired_same_decks_vs_serial": [
                {"seed": seed, **direct_evaluations[(seed, "paired_same_decks")]} for seed in seeds
            ],
        },
        "aggregate": {
            "regimes": regime_summaries,
            "direct_same_decks_point_diff": _distribution(
                [float(row["avg_point_diff_agent_a_minus_agent_b"]) for row in direct_same_decks]
            ),
        },
        "decision": decide(
            regime_summaries=regime_summaries,
            direct_same_games=direct_same_games,
            thresholds=thresholds,
        ),
        "versions": {
            "code": get_code_version(),
            "rules": get_rules_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": _git_commit(),
        },
    }
    final_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved paired schedule probe: {final_path}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
