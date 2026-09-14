#!/usr/bin/env python3
"""Confronta v14 continuata con copie che riattivano 8 o 16 unità dormienti.

Lo screening usa lo stesso dataset, shuffle, augmentation e ottimizzatore per tutte le
varianti. Le unità candidate devono essere risultate esattamente inattive sia nella
diagnostica originale sia nel holdout causale; il loro ordine è congelato da un ranking
SHA-256. La variante viene scelta sulla validation e confermata sul test per partita.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.evaluation.hidden_units import reinitialize_mlp_hidden_units
from briscola_ai.ai.models import MLPBCModel, load_bc_model_npz
from briscola_ai.ai.training.suit_distillation import (
    DistillationTrainResult,
    SplitName,
    SuitDistillationDataset,
    load_suit_distillation_dataset,
    train_suit_distillation,
)
from briscola_ai.versioning import get_code_version, get_rules_version

SCHEMA = "briscola.dormant_reinitialization_screen.v1"
DEFAULT_VARIANTS = (0, 8, 16)
INITIAL_ACTION_AGREEMENT_MIN = 0.999
VALIDATION_KL_RELATIVE_IMPROVEMENT_MIN = 0.01
ARGMAX_AGREEMENT_DELTA_MIN = -0.0005
ACTIVE_UNIT_RATE_MIN = 0.01
ACTIVE_UNIT_FRACTION_MIN = 0.75
OUTGOING_CENTERED_L2_MIN = 0.001
LEARNED_OUTGOING_FRACTION_MIN = 0.75


def _repo_root() -> Path:
    """Ritorna la root del checkout."""
    return Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    """Calcola SHA-256 a blocchi per modelli, dataset ed evidenze."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path) -> str:
    """Usa path relativi al repository quando possibile."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _git_commit(root: Path) -> str | None:
    """Legge il commit corrente senza rendere Git un requisito del calcolo."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _git_worktree_dirty(root: Path) -> bool | None:
    """Registra se il run include modifiche non committate."""
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _load_json(path: Path, *, expected_schema: str) -> dict[str, Any]:
    """Carica un'evidenza e ne verifica lo schema prima di usarla per selezione."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"Schema inatteso in {path}: {payload.get('schema')!r}")
    return payload


def _selection_pool(
    diagnostic_path: Path,
    holdout_path: Path,
    *,
    source_model_sha256: str,
) -> tuple[list[int], dict[str, Any]]:
    """Interseca le unità con attivazione esattamente zero nelle due evidenze congelate."""
    diagnostic = _load_json(diagnostic_path, expected_schema="briscola.hidden_unit_diagnostic.v1")
    holdout = _load_json(holdout_path, expected_schema="briscola.dormant_unit_ablation.v1")
    diagnostic_sha256 = _sha256_file(diagnostic_path)

    diagnostic_model_sha256 = diagnostic.get("artifacts", {}).get("primary", {}).get("sha256")
    holdout_model_sha256 = holdout.get("artifacts", {}).get("source_model", {}).get("sha256")
    if diagnostic_model_sha256 != source_model_sha256 or holdout_model_sha256 != source_model_sha256:
        raise ValueError("Le evidenze non si riferiscono al modello sorgente richiesto")
    recorded_diagnostic_sha256 = holdout.get("artifacts", {}).get("source_evidence", {}).get("sha256")
    if recorded_diagnostic_sha256 != diagnostic_sha256:
        raise ValueError("Il holdout non deriva dall'evidenza diagnostica fornita")

    rows = diagnostic.get("models", {}).get("primary", {}).get("units")
    holdout_zero = holdout.get("policy_agreement", {}).get("selected_units_never_active")
    if not isinstance(rows, list) or not isinstance(holdout_zero, list):
        raise ValueError("Le evidenze non contengono le liste di attività richieste")
    diagnostic_zero = {
        int(row["unit"]) for row in rows if isinstance(row, dict) and float(row.get("activation_rate", -1.0)) == 0.0
    }
    holdout_zero_set = {int(unit) for unit in holdout_zero}
    pool = sorted(diagnostic_zero & holdout_zero_set)
    if not pool:
        raise ValueError("Nessuna unità è esattamente inattiva in entrambe le evidenze")
    provenance = {
        "diagnostic_zero_count": len(diagnostic_zero),
        "holdout_zero_count": len(holdout_zero_set),
        "intersection_count": len(pool),
        "intersection_units": pool,
    }
    return pool, provenance


def _rank_units(pool: list[int], *, seed: int) -> list[int]:
    """Ordina le unità con hash stabile per ottenere varianti annidate non adattive."""

    def key(unit: int) -> tuple[bytes, int]:
        material = f"briscola.dormant_selection.v1:{int(seed)}:{unit}".encode("ascii")
        return hashlib.sha256(material).digest(), unit

    return sorted(pool, key=key)


def _policy_agreement(
    original: MLPBCModel,
    candidate: MLPBCModel,
    dataset: SuitDistillationDataset,
    *,
    split: SplitName,
    batch_size: int = 4096,
) -> dict[str, int | float]:
    """Confronta gli argmax legali dei due modelli sullo stesso split."""
    indices = dataset.indices(split)
    disagreements = 0
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        features = dataset.features[batch_indices]
        masks = dataset.action_masks[batch_indices]
        original_actions = np.argmax(np.where(masks, original.logits(features), -np.inf), axis=1)
        candidate_actions = np.argmax(np.where(masks, candidate.logits(features), -np.inf), axis=1)
        disagreements += int(np.sum(original_actions != candidate_actions))
    return {
        "examples": int(indices.size),
        "disagreements": disagreements,
        "agreement_rate": 1.0 - disagreements / int(indices.size),
    }


def _unit_activity(
    model: MLPBCModel,
    features: np.ndarray,
    unit_indices: list[int],
    *,
    activation_epsilon: float,
) -> dict[str, Any]:
    """Misura attivazione e peso uscente delle sole unità congelate dal protocollo."""
    if not unit_indices:
        return {
            "unit_count": 0,
            "active_at_least_one_percent_count": 0,
            "active_at_least_one_percent_fraction": 0.0,
            "learned_outgoing_count": 0,
            "learned_outgoing_fraction": 0.0,
            "units": [],
        }
    selected = np.asarray(unit_indices, dtype=np.intp)
    hidden = np.maximum(features @ model.w1[:, selected] + model.b1[selected], 0.0)
    activation_rates = np.mean(hidden > activation_epsilon, axis=0)
    outgoing = np.asarray(model.w2[selected], dtype=np.float64)
    outgoing_centered_l2 = np.linalg.norm(outgoing - np.mean(outgoing, axis=1, keepdims=True), axis=1)
    active = activation_rates >= ACTIVE_UNIT_RATE_MIN
    learned = outgoing_centered_l2 >= OUTGOING_CENTERED_L2_MIN
    rows = [
        {
            "unit": int(unit),
            "activation_rate": float(activation_rates[index]),
            "mean_activation": float(np.mean(hidden[:, index], dtype=np.float64)),
            "p95_activation": float(np.quantile(hidden[:, index], 0.95)),
            "max_activation": float(np.max(hidden[:, index])),
            "outgoing_centered_l2": float(outgoing_centered_l2[index]),
        }
        for index, unit in enumerate(unit_indices)
    ]
    return {
        "unit_count": len(unit_indices),
        "active_at_least_one_percent_count": int(np.sum(active)),
        "active_at_least_one_percent_fraction": float(np.mean(active)),
        "learned_outgoing_count": int(np.sum(learned)),
        "learned_outgoing_fraction": float(np.mean(learned)),
        "units": rows,
    }


def _model_from_result(result: DistillationTrainResult, metadata: dict[str, Any]) -> MLPBCModel:
    """Materializza il checkpoint scelto dal trainer con metadati sintetici."""
    return MLPBCModel(
        w1=result.w1,
        b1=result.b1,
        w2=result.w2,
        b2=result.b2,
        metadata=metadata,
    )


def _save_model(path: Path, model: MLPBCModel) -> None:
    """Salva soltanto pesi e metadati essenziali del candidato locale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        metadata_json=np.asarray(json.dumps(model.metadata, ensure_ascii=True, sort_keys=True)),
    )


def _variant_key(count: int) -> str:
    """Nome JSON e basename stabile per una dimensione dello screening."""
    return "control" if count == 0 else f"reset_{count}"


def _decision(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sceglie sulla validation e applica sul vincitore i gate preregistrati."""
    control = variants["control"]
    reset_variants = [row for row in variants.values() if int(row["reinitialized_unit_count"]) > 0]
    winner = min(
        reset_variants,
        key=lambda row: (float(row["best_validation"]["kl_divergence"]), int(row["reinitialized_unit_count"])),
    )
    control_validation_kl = float(control["best_validation"]["kl_divergence"])
    winner_validation_kl = float(winner["best_validation"]["kl_divergence"])
    relative_improvement = (
        (control_validation_kl - winner_validation_kl) / control_validation_kl if control_validation_kl > 0.0 else 0.0
    )
    validation_agreement_delta = float(winner["best_validation"]["argmax_agreement"]) - float(
        control["best_validation"]["argmax_agreement"]
    )
    test_kl_delta = float(winner["test"]["kl_divergence"]) - float(control["test"]["kl_divergence"])
    test_agreement_delta = float(winner["test"]["argmax_agreement"]) - float(control["test"]["argmax_agreement"])
    activity = winner["post_training_reinitialized_activity"]
    initial_agreement = min(
        float(winner["initial_policy_agreement"]["validation"]["agreement_rate"]),
        float(winner["initial_policy_agreement"]["test"]["agreement_rate"]),
    )
    checks = {
        "initial_action_agreement": initial_agreement >= INITIAL_ACTION_AGREEMENT_MIN,
        "validation_kl_relative_improvement": relative_improvement >= VALIDATION_KL_RELATIVE_IMPROVEMENT_MIN,
        "validation_argmax_no_regression": validation_agreement_delta >= ARGMAX_AGREEMENT_DELTA_MIN,
        "test_kl_no_regression": test_kl_delta <= 0.0,
        "test_argmax_no_regression": test_agreement_delta >= ARGMAX_AGREEMENT_DELTA_MIN,
        "units_became_active": float(activity["active_at_least_one_percent_fraction"]) >= ACTIVE_UNIT_FRACTION_MIN,
        "units_learned_nonconstant_output": float(activity["learned_outgoing_fraction"])
        >= LEARNED_OUTGOING_FRACTION_MIN,
    }
    return {
        "verdict": "go_evaluate_reinitialized_candidate" if all(checks.values()) else "stop_dormant_reinitialization",
        "selected_by": "minimum validation KL among reset variants; test excluded from selection",
        "selected_variant": winner["key"],
        "selected_unit_count": winner["reinitialized_unit_count"],
        "checks": checks,
        "comparisons_vs_control": {
            "initial_action_agreement_min": initial_agreement,
            "validation_kl_relative_improvement": relative_improvement,
            "validation_argmax_agreement_delta": validation_agreement_delta,
            "test_kl_delta": test_kl_delta,
            "test_argmax_agreement_delta": test_agreement_delta,
        },
        "note_it": (
            "Un GO autorizza solo sonde di simmetria e direct match del candidato scelto; "
            "non autorizza promozione, modifica di v14 live o interpretazioni causali sui singoli neuroni."
        ),
    }


def _parse_args() -> argparse.Namespace:
    """Definisce il protocollo ufficiale e consente smoke test ridotti."""
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/distillation/suit_teacher_v13_10k_seed20260711.npz")
    parser.add_argument("--init", type=Path, default=root / "data/models/best_a2c_v14.npz")
    parser.add_argument(
        "--diagnostic-evidence",
        type=Path,
        default=root / "docs/reports/evidence/hidden_units_v14.v1.json",
    )
    parser.add_argument(
        "--holdout-evidence",
        type=Path,
        default=root / "docs/reports/evidence/dormant_unit_ablation_v14.v1.json",
    )
    parser.add_argument("--variants", type=int, nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--selection-seed", type=int, default=20260712)
    parser.add_argument("--reinit-seed", type=int, default=20260712)
    parser.add_argument("--training-seed", type=int, default=20260712)
    parser.add_argument("--expected-pool-size", type=int, default=79)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "benchmarks/experiments/dormant_reinitialization_v14_v0/models",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=root / "docs/reports/evidence/dormant_reinitialization_screen_v14.v1.json",
    )
    return parser.parse_args()


def main() -> int:
    """Esegue le tre continuazioni controllate e salva il verdetto riproducibile."""
    args = _parse_args()
    root = _repo_root()
    paths = {
        "data": args.data.resolve(),
        "init": args.init.resolve(),
        "diagnostic": args.diagnostic_evidence.resolve(),
        "holdout": args.holdout_evidence.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    variants = tuple(args.variants)
    if not variants or variants[0] != 0 or len(set(variants)) != len(variants):
        raise ValueError("--variants deve iniziare con 0 e contenere valori unici")
    if any(count < 0 for count in variants) or len(variants) < 2:
        raise ValueError("Servono il controllo 0 e almeno una variante positiva")

    source = load_bc_model_npz(paths["init"])
    if not isinstance(source, MLPBCModel) or source.has_belief_input:
        raise ValueError("--init deve essere una MLP senza belief embedded")
    source_sha256 = _sha256_file(paths["init"])
    pool, pool_provenance = _selection_pool(
        paths["diagnostic"],
        paths["holdout"],
        source_model_sha256=source_sha256,
    )
    if len(pool) != args.expected_pool_size:
        raise ValueError(f"Pool inattivo atteso {args.expected_pool_size}, ottenuto {len(pool)}")
    ranked = _rank_units(pool, seed=args.selection_seed)
    max_variant = max(variants)
    if max_variant > len(ranked):
        raise ValueError(f"Variante {max_variant} oltre il pool di {len(ranked)} unità")

    dataset = load_suit_distillation_dataset(paths["data"])
    activation_epsilon = float(
        _load_json(paths["diagnostic"], expected_schema="briscola.hidden_unit_diagnostic.v1")["thresholds"][
            "activation_epsilon"
        ]
    )
    validation_features = dataset.features[dataset.indices("validation")]
    tracked_units = ranked[:max_variant]
    variant_reports: dict[str, dict[str, Any]] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for count in variants:
        key = _variant_key(count)
        selected = ranked[:count]
        initialized = source if count == 0 else reinitialize_mlp_hidden_units(source, selected, seed=args.reinit_seed)
        initial_policy_agreement = {
            split: _policy_agreement(source, initialized, dataset, split=split) for split in ("validation", "test")
        }
        initial_selected_activity = _unit_activity(
            initialized,
            validation_features,
            selected,
            activation_epsilon=activation_epsilon,
        )
        print(f"{key}: training con {count} unità reinizializzate...", flush=True)
        started = time.perf_counter()
        result = train_suit_distillation(
            dataset,
            initialized,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            seed=args.training_seed,
            paired_augmentation=True,
        )
        elapsed = time.perf_counter() - started
        metadata = dict(source.metadata)
        metadata.update(
            {
                "label": f"v14 screening riattivazione {count} unità",
                "description_it": (
                    "Candidato diagnostico locale: continuazione della distillazione v14 con "
                    f"{count} unità dormienti reinizializzate; non promosso."
                ),
                "dormant_reinitialization_screen": {
                    "format": SCHEMA,
                    "source_model_sha256": source_sha256,
                    "reinitialized_unit_count": count,
                    "reinitialized_units": selected,
                    "selection_seed": int(args.selection_seed),
                    "reinit_seed": int(args.reinit_seed),
                    "training_seed": int(args.training_seed),
                    "best_epoch": int(result.best_epoch),
                    "validation_kl": result.best_validation.kl_divergence,
                    "validation_argmax_agreement": result.best_validation.argmax_agreement,
                    "test_kl": result.test.kl_divergence,
                    "test_argmax_agreement": result.test.argmax_agreement,
                },
            }
        )
        trained = _model_from_result(result, metadata)
        model_path = args.out_dir / f"v14_dormant_reinit_{key}_screen_v0.npz"
        _save_model(model_path, trained)
        selected_activity = _unit_activity(
            trained,
            validation_features,
            selected,
            activation_epsilon=activation_epsilon,
        )
        tracked_activity = _unit_activity(
            trained,
            validation_features,
            tracked_units,
            activation_epsilon=activation_epsilon,
        )
        variant_reports[key] = {
            "key": key,
            "reinitialized_unit_count": count,
            "reinitialized_units": selected,
            "elapsed_seconds": elapsed,
            "initial_policy_agreement": initial_policy_agreement,
            "initial_reinitialized_activity": initial_selected_activity,
            "before_validation": asdict(result.before_validation),
            "before_test": asdict(result.before_test),
            "best_epoch": result.best_epoch,
            "best_validation": asdict(result.best_validation),
            "test": asdict(result.test),
            "post_training_reinitialized_activity": selected_activity,
            "post_training_tracked_activity": tracked_activity,
            "epochs": [asdict(epoch) for epoch in result.epochs],
            "artifact": {
                "path_local": _display_path(model_path, root=root),
                "sha256": _sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
            },
        }
        print(
            f"{key}: val KL {result.best_validation.kl_divergence:.6f}, "
            f"test KL {result.test.kl_divergence:.6f}, "
            f"test agreement {result.test.argmax_agreement:.4%}",
            flush=True,
        )

    selection_manifest = "\n".join(str(unit) for unit in ranked).encode("ascii")
    report = {
        "schema": SCHEMA,
        "method": {
            "anti_cheat": "dataset e teacher contengono solo feature derivate da PlayerObservation",
            "variant_selection": "minimum validation KL; test excluded from selection",
            "same_dataset_shuffle_augmentation_optimizer": True,
            "paired_suit_augmentation": True,
            "initialization": "He per w1 della singola unità, b1=0, riga w2=0",
            "split_unit": "game",
        },
        "gates": {
            "initial_action_agreement_min": INITIAL_ACTION_AGREEMENT_MIN,
            "validation_kl_relative_improvement_min": VALIDATION_KL_RELATIVE_IMPROVEMENT_MIN,
            "argmax_agreement_delta_min": ARGMAX_AGREEMENT_DELTA_MIN,
            "test_kl_delta_max": 0.0,
            "active_unit_rate_min": ACTIVE_UNIT_RATE_MIN,
            "active_unit_fraction_min": ACTIVE_UNIT_FRACTION_MIN,
            "outgoing_centered_l2_min": OUTGOING_CENTERED_L2_MIN,
            "learned_outgoing_fraction_min": LEARNED_OUTGOING_FRACTION_MIN,
        },
        "config": {
            "variants": list(variants),
            "selection_seed": int(args.selection_seed),
            "reinit_seed": int(args.reinit_seed),
            "training_seed": int(args.training_seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "versions": {
            "code": get_code_version(),
            "rules": get_rules_version(),
            "git_commit": _git_commit(root),
            "git_worktree_dirty": _git_worktree_dirty(root),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "artifacts": {
            "source_model": {
                "path": _display_path(paths["init"], root=root),
                "sha256": source_sha256,
                "size_bytes": paths["init"].stat().st_size,
            },
            "dataset": {
                "path_local": _display_path(paths["data"], root=root),
                "sha256": _sha256_file(paths["data"]),
                "size_bytes": paths["data"].stat().st_size,
                "metadata": dataset.metadata,
            },
            "diagnostic_evidence": {
                "path": _display_path(paths["diagnostic"], root=root),
                "sha256": _sha256_file(paths["diagnostic"]),
            },
            "holdout_evidence": {
                "path": _display_path(paths["holdout"], root=root),
                "sha256": _sha256_file(paths["holdout"]),
            },
        },
        "selection": {
            **pool_provenance,
            "ranking": "SHA-256 of schema, selection seed and unit index",
            "ranked_units": ranked,
            "tracked_units": tracked_units,
            "ranking_manifest_sha256": hashlib.sha256(selection_manifest).hexdigest(),
        },
        "variants": variant_reports,
        "decision": _decision(variant_reports),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report: {_display_path(args.out_json, root=root)}")
    print(f"Verdetto: {report['decision']['verdict']} ({report['decision']['selected_variant']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
