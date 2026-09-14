#!/usr/bin/env python3
"""Esegue e riassume uno smoke diagnostico A2C su tre seed indipendenti.

Il protocollo non confronta la forza di modelli e non promuove artefatti. Mantiene la
ricetta A2C corrente, attiva soltanto la telemetria passiva di ``train_a2c.py`` e usa
soglie fissate in anticipo per scegliere una singola ablation successiva.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.versioning import get_code_version, get_rules_version

SCHEMA = "briscola.a2c_health_probe.v1"
DIAGNOSTIC_SCHEMA = "briscola.a2c_training_diagnostics.v1"
DEFAULT_SEEDS = (20260714, 20260715, 20260716)
DEFAULT_OPPONENT_MIX = (
    "bc_model:0.15,bc_model_pimc_belief:0.40,"
    "bc_model_value_lookahead_8x8:0.20,heuristic_trump_saver:0.12,"
    "heuristic_v1:0.04,heuristic_v2:0.06,random:0.03"
)


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """Soglie di instradamento, non gate di promozione di un modello."""

    critic_explained_variance_median_min: float = 0.10
    critic_negative_explained_variance_fraction_max: float = 0.25
    advantage_bias_ratio_median_max: float = 0.25
    global_gradient_p95_over_median_max: float = 5.0
    trunk_relative_update_p95_max: float = 0.01
    actor_head_relative_update_p95_max: float = 0.01
    hidden_units_never_active_fraction_max: float = 0.75
    hidden_activation_rate_mean_min: float = 0.02
    hidden_activation_rate_mean_max: float = 0.98


def _repo_root() -> Path:
    """Ritorna la root del checkout indipendentemente dalla cwd."""
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    """Calcola l'identità contenutistica di un artefatto senza caricarlo in memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path) -> str:
    """Preferisce path relativi al repository nei report condivisibili."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path, *, root: Path) -> dict[str, str | int]:
    """Descrive un file sufficiente a rilevare riprese incompatibili."""
    return {
        "path": _display_path(path, root=root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    """Carica JSON rigoroso: NaN/Infinity e root non-oggetto sono errori."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Costante JSON non standard {value!r} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"Oggetto JSON atteso in {path}")
    return payload


def _nested_float(row: dict[str, Any], *keys: str, allow_none: bool = False) -> float | None:
    """Legge una metrica annidata rifiutando bool, stringhe e valori non finiti."""
    value: Any = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Metrica mancante: {'.'.join(keys)}")
        value = value[key]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Metrica non numerica {'.'.join(keys)}={value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Metrica non finita {'.'.join(keys)}={result!r}")
    return result


def _percentile(values: list[float], quantile: float) -> float:
    """Percentile float64 su una lista che deve contenere almeno un elemento."""
    if not values:
        raise ValueError("Distribuzione diagnostica vuota")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def summarize_run(
    report: dict[str, Any],
    *,
    expected_seed: int,
    expected_num_games: int,
    expected_update_every: int,
    expected_init_sha256: str,
    thresholds: HealthThresholds,
) -> dict[str, Any]:
    """Valida un report per-seed e applica le soglie alla metà finale degli update."""
    if report.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError(f"Schema diagnostico inatteso: {report.get('schema')!r}")
    config = report.get("config")
    artifacts = report.get("artifacts")
    initialization = report.get("initialization")
    if not isinstance(config, dict) or not isinstance(artifacts, dict) or not isinstance(initialization, dict):
        raise ValueError("Report diagnostico privo di config/artifacts/initialization")
    init_artifact = artifacts.get("init")
    if not isinstance(init_artifact, dict):
        raise ValueError("Lo health probe richiede un init identificato")
    if (
        config.get("seed") != expected_seed
        or config.get("num_games") != expected_num_games
        or config.get("update_every") != expected_update_every
        or init_artifact.get("sha256") != expected_init_sha256
    ):
        raise ValueError(f"Report seed {expected_seed} incompatibile con il protocollo richiesto")
    hidden_dim = config.get("hidden_dim")
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError(f"hidden_dim non valido nel report: {hidden_dim!r}")

    raw_updates = report.get("updates")
    if not isinstance(raw_updates, list) or not raw_updates or not all(isinstance(row, dict) for row in raw_updates):
        raise ValueError("Report diagnostico senza update validi")
    updates: list[dict[str, Any]] = raw_updates
    expected_updates = expected_num_games // expected_update_every
    if len(updates) != expected_updates:
        raise ValueError(f"Update inattesi per seed {expected_seed}: {len(updates)}/{expected_updates}")
    late_updates = updates[len(updates) // 2 :]

    critic_values_with_none = [
        _nested_float(row, "signals", "critic_explained_variance", allow_none=True) for row in late_updates
    ]
    critic_values = [float(value) for value in critic_values_with_none if value is not None]
    critic_coverage = len(critic_values) / len(late_updates)
    critic_median = _percentile(critic_values, 0.50) if critic_values else None
    critic_negative_fraction = (
        float(np.mean(np.asarray(critic_values, dtype=np.float64) < 0.0)) if critic_values else None
    )
    critic_mse = [float(_nested_float(row, "signals", "critic_mean_squared_error")) for row in late_updates]

    advantage_bias_ratios: list[float] = []
    for row in late_updates:
        mean = float(_nested_float(row, "signals", "advantage_mean"))
        std = float(_nested_float(row, "signals", "advantage_std"))
        if std > 1e-12:
            advantage_bias_ratios.append(abs(mean) / std)
    advantage_bias_coverage = len(advantage_bias_ratios) / len(late_updates)
    advantage_bias_median = _percentile(advantage_bias_ratios, 0.50) if advantage_bias_ratios else None

    global_gradients = [float(_nested_float(row, "global_gradient_l2")) for row in updates]
    gradient_median = _percentile(global_gradients, 0.50)
    gradient_spike_ratio = _percentile(global_gradients, 0.95) / gradient_median if gradient_median > 1e-12 else None
    trunk_updates = [float(_nested_float(row, "trunk_relative_update")) for row in updates]
    actor_updates = [float(_nested_float(row, "actor_head_relative_update")) for row in updates]
    hidden_inactive_fractions = [
        float(_nested_float(row, "signals", "hidden_units_never_active")) / hidden_dim for row in late_updates
    ]
    hidden_activation_rates = [
        float(_nested_float(row, "signals", "hidden_activation_rate_mean")) for row in late_updates
    ]

    metrics = {
        "updates": len(updates),
        "late_updates": len(late_updates),
        "critic_explained_variance_coverage": critic_coverage,
        "critic_explained_variance_median": critic_median,
        "critic_negative_explained_variance_fraction": critic_negative_fraction,
        "critic_mean_squared_error_median": _percentile(critic_mse, 0.50),
        "advantage_bias_ratio_coverage": advantage_bias_coverage,
        "advantage_bias_ratio_median": advantage_bias_median,
        "global_gradient_p95_over_median": gradient_spike_ratio,
        "trunk_relative_update_p95": _percentile(trunk_updates, 0.95),
        "actor_head_relative_update_p95": _percentile(actor_updates, 0.95),
        "hidden_units_never_active_fraction_max": max(hidden_inactive_fractions),
        "hidden_activation_rate_mean_median": _percentile(hidden_activation_rates, 0.50),
    }
    gates = {
        "integrity": (
            report.get("method", {}).get("passive") is True
            and initialization.get("critic_mode") == "reset_zero"
            and initialization.get("init_critic_used") is False
        ),
        "critic": (
            critic_coverage == 1.0
            and critic_median is not None
            and critic_median >= thresholds.critic_explained_variance_median_min
            and critic_negative_fraction is not None
            and critic_negative_fraction <= thresholds.critic_negative_explained_variance_fraction_max
        ),
        "advantage": (
            advantage_bias_coverage == 1.0
            and advantage_bias_median is not None
            and advantage_bias_median <= thresholds.advantage_bias_ratio_median_max
        ),
        "gradient": (
            gradient_spike_ratio is not None and gradient_spike_ratio <= thresholds.global_gradient_p95_over_median_max
        ),
        "update": (
            metrics["trunk_relative_update_p95"] <= thresholds.trunk_relative_update_p95_max
            and metrics["actor_head_relative_update_p95"] <= thresholds.actor_head_relative_update_p95_max
        ),
        "hidden": (
            metrics["hidden_units_never_active_fraction_max"] <= thresholds.hidden_units_never_active_fraction_max
            and thresholds.hidden_activation_rate_mean_min
            <= metrics["hidden_activation_rate_mean_median"]
            <= thresholds.hidden_activation_rate_mean_max
        ),
    }
    return {"seed": expected_seed, "metrics": metrics, "gates": gates}


def route_next_experiment(run_summaries: list[dict[str, Any]]) -> dict[str, str]:
    """Sceglie una sola ablation seguendo la priorità congelata del protocollo."""
    if not run_summaries:
        raise ValueError("Nessun run da valutare")
    gate_names = ("integrity", "critic", "advantage", "gradient", "update", "hidden")
    gates = {name: all(bool(run["gates"].get(name)) for run in run_summaries) for name in gate_names}
    if not gates["integrity"]:
        return {
            "verdict": "invalid_probe",
            "next_experiment": "repeat_health_probe_after_fixing_integrity",
            "explanation_it": "Almeno un report non dimostra che la sonda fosse passiva: il probe va ripetuto.",
        }
    if not gates["critic"]:
        return {
            "verdict": "critic_first",
            "next_experiment": "critic_reset_vs_reuse",
            "explanation_it": (
                "Il valutatore interno non spiega abbastanza i risultati nella parte finale: "
                "il prossimo test deve confrontare reset e riuso del critic, lasciando invariato il resto."
            ),
        }
    if not gates["advantage"]:
        return {
            "verdict": "advantage_first",
            "next_experiment": "batch_advantage_normalization",
            "explanation_it": (
                "Il segnale che premia o penalizza le mosse resta troppo sbilanciato: "
                "il prossimo test deve normalizzarlo sull'intero update."
            ),
        }
    if not gates["gradient"] or not gates["update"]:
        return {
            "verdict": "optimizer_stability_first",
            "next_experiment": "global_gradient_clipping",
            "explanation_it": (
                "Gradienti o passi Adam mostrano picchi eccessivi: il prossimo test deve limitare "
                "la norma globale senza cambiare gli altri segnali."
            ),
        }
    if not gates["hidden"]:
        return {
            "verdict": "representation_health_first",
            "next_experiment": "inspect_hidden_activation_during_training",
            "explanation_it": (
                "Durante gli update troppe unità restano spente o l'attivazione media è estrema: "
                "prima di cambiare l'ottimizzatore va verificata la rappresentazione."
            ),
        }
    return {
        "verdict": "signals_healthy",
        "next_experiment": "paired_training_schedule",
        "explanation_it": (
            "I segnali interni sono stabili su tutti i seed: non emerge una correzione numerica prioritaria; "
            "il test successivo può isolare la schedule davvero appaiata."
        ),
    }


def _aggregate_gates(run_summaries: list[dict[str, Any]]) -> dict[str, bool]:
    """Richiede che ogni seed superi ogni controllo, senza nascondere outlier nella media."""
    return {
        name: all(bool(run["gates"][name]) for run in run_summaries)
        for name in ("integrity", "critic", "advantage", "gradient", "update", "hidden")
    }


def _parse_seeds(raw: str) -> tuple[int, ...]:
    """Converte una lista CSV in seed unici e ordinati come richiesti."""
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--seeds richiede interi separati da virgola") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds deve contenere almeno un seed e nessun duplicato")
    return seeds


def _git_commit() -> str | None:
    """Commit corrente best-effort, utile anche quando il probe gira da un checkout sporco."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError, subprocess.SubprocessError:
        return None


def _build_training_command(
    *,
    root: Path,
    model_out: Path,
    diagnostics_out: Path,
    init_model: Path,
    opponent_model: Path,
    belief_model: Path,
    value_model: Path,
    opponent_mix: str,
    num_games: int,
    update_every: int,
    seed: int,
) -> list[str]:
    """Costruisce la ricetta corrente senza introdurre correzioni sperimentali."""
    return [
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
        "serial",
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
        str(max(1, num_games // update_every // 4)),
        "--metrics-mode",
        "summary",
        "--seat-fair",
        "--seed",
        str(seed),
    ]


def _parse_args() -> argparse.Namespace:
    """Definisce path e dimensione del probe mantenendo stabili i default ufficiali."""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Diagnostica passiva A2C su più seed")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=root / "benchmarks/experiments/a2c_health_v14_v0_20260714",
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
    parser.add_argument("--num-games", type=int, default=2_000)
    parser.add_argument("--update-every", type=int, default=20)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Esegue i run mancanti, valida ogni report e salva il verdetto aggregato."""
    args = _parse_args()
    root = _repo_root()
    seeds = _parse_seeds(str(args.seeds))
    num_games = int(args.num_games)
    update_every = int(args.update_every)
    if num_games <= 0 or update_every <= 0 or num_games % update_every != 0:
        raise ValueError("--num-games e --update-every devono essere positivi e divisibili senza resto")
    input_paths = (args.init_model, args.opponent_model, args.belief_model, args.value_model)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    work_dir = args.work_dir
    summary_path = work_dir / f"a2c_health_v14_g{num_games}_{len(seeds)}seeds.json"
    if summary_path.exists() and not args.resume:
        raise FileExistsError(f"Report già presente: {summary_path}. Usa --resume o un'altra --work-dir.")
    work_dir.mkdir(parents=True, exist_ok=True)
    thresholds = HealthThresholds()
    init_sha256 = _sha256(args.init_model)
    runs: list[dict[str, Any]] = []

    for seed in seeds:
        model_out = work_dir / f"a2c_health_seed{seed}.npz"
        diagnostics_out = work_dir / f"a2c_health_seed{seed}.diagnostics.json"
        exists = (model_out.exists(), diagnostics_out.exists())
        if any(exists) and not all(exists):
            raise FileExistsError(f"Run seed {seed} incompleto: model={exists[0]} diagnostics={exists[1]}")
        if all(exists):
            if not args.resume:
                raise FileExistsError(f"Artefatti seed {seed} già presenti; usa --resume")
            print(f"resume: seed {seed} già presente", flush=True)
        else:
            command = _build_training_command(
                root=root,
                model_out=model_out,
                diagnostics_out=diagnostics_out,
                init_model=args.init_model,
                opponent_model=args.opponent_model,
                belief_model=args.belief_model,
                value_model=args.value_model,
                opponent_mix=str(args.opponent_mix),
                num_games=num_games,
                update_every=update_every,
                seed=seed,
            )
            print("\n+ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=root, check=True)

        diagnostic_payload = _load_json(diagnostics_out)
        run = summarize_run(
            diagnostic_payload,
            expected_seed=seed,
            expected_num_games=num_games,
            expected_update_every=update_every,
            expected_init_sha256=init_sha256,
            thresholds=thresholds,
        )
        run["artifacts"] = {
            "model": _artifact(model_out, root=root),
            "diagnostics": _artifact(diagnostics_out, root=root),
        }
        run["initialization"] = diagnostic_payload["initialization"]
        runs.append(run)

    aggregate_gates = _aggregate_gates(runs)
    report = {
        "schema": SCHEMA,
        "protocol": {
            "purpose": "route_one_next_a2c_ablation_not_model_promotion",
            "seeds": list(seeds),
            "num_games_per_seed": num_games,
            "update_every": update_every,
            "late_window": "last_half_of_optimizer_updates_per_seed",
            "whole_run_metrics": [
                "global_gradient_p95_over_median",
                "trunk_relative_update_p95",
                "actor_head_relative_update_p95",
            ],
            "thresholds": asdict(thresholds),
            "rule": (
                "every_seed_must_pass_each_gate; routing priority is integrity, critic, advantage, optimizer, hidden"
            ),
        },
        "recipe": {
            "encoder_version": "v4",
            "rollout_engine": "fast",
            "fast_rollout": "numba",
            "training_schedule": "serial",
            "seat_fair": True,
            "opponent_mix": str(args.opponent_mix),
            "opponent_pimc_determinizations": 16,
            "opponent_value_max_unknown_cards": 8,
            "bc_anchor_beta": 0.01,
            "overkill_penalty_mode": "gap",
            "overkill_penalty_beta": 0.3,
            "metrics_mode": "summary",
        },
        "inputs": {
            "init_model": _artifact(args.init_model, root=root),
            "opponent_model": _artifact(args.opponent_model, root=root),
            "belief_model": _artifact(args.belief_model, root=root),
            "value_model": _artifact(args.value_model, root=root),
        },
        "runs": runs,
        "aggregate": {
            "gates": aggregate_gates,
            "all_gates_pass_all_seeds": all(aggregate_gates.values()),
        },
        "decision": route_next_experiment(runs),
        "versions": {
            "code": get_code_version(),
            "rules": get_rules_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": _git_commit(),
        },
    }
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved health probe: {summary_path}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
