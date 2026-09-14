#!/usr/bin/env python3
"""Aggrega i fold leave-one-opponent-out della belief v1 e applica gate fissi.

Ogni modello fold prodotto da ``train_belief.py`` contiene gia' candidate e belief v0
valutate sullo stesso holdout. Questo script verifica che dataset, baseline e roster
siano comuni, calcola macro-medie non pesate per stile e produce un JSON rigoroso.

Gate offline predefiniti
------------------------
- miglioramento BCE macro relativo >= 1%;
- top-k macro non regressivo;
- Brier ed ECE macro non regressivi;
- nessun singolo stile peggiora la BCE di oltre il 2% relativo.

Un GO autorizza solo l'addestramento all-styles e il confronto runtime PIMC v0-v1. Non
autorizza promozione o deploy. ``--pilot`` registra gli stessi numeri ma restituisce il
verdetto ``pilot_pipeline_validated``: un dataset smoke non puo' passare un gate di forza.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.models.belief_model import load_belief_model_npz
from briscola_ai.versioning import get_code_version, get_rules_version

METRIC_NAMES = ("bce", "topk_recall", "brier", "ece")


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    """Metriche e artefatti di un singolo stile escluso."""

    holdout: str
    candidate: dict[str, float]
    baseline: dict[str, float]
    model_artifact: dict[str, Any]
    dataset_artifact: dict[str, Any]
    baseline_artifact: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FoldGates:
    """Soglie preregistrate del gate offline."""

    min_macro_bce_relative_improvement: float = 0.01
    min_macro_topk_delta: float = 0.0
    max_macro_brier_delta: float = 0.0
    max_macro_ece_delta: float = 0.0
    max_worst_fold_bce_relative_regression: float = 0.02


def _sha256(path: Path) -> str:
    """SHA-256 streaming di un modello fold."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    """Descrive un file locale senza leggerne payload sensibili."""
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _git_commit() -> str | None:
    """Commit corrente best-effort."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError, subprocess.SubprocessError:
        return None


def _metric_dict(raw: Any, *, label: str) -> dict[str, float]:
    """Estrae il contratto minimo di metriche da metadata JSON."""
    if not isinstance(raw, dict):
        raise ValueError(f"Metriche {label} mancanti")
    metrics: dict[str, float] = {}
    for name in METRIC_NAMES:
        value = raw.get(name)
        if not isinstance(value, int | float) or not np.isfinite(float(value)):
            raise ValueError(f"Metrica {label}.{name} non valida: {value!r}")
        metrics[name] = float(value)
    return metrics


def load_fold(path: Path) -> FoldMetrics:
    """Carica e valida un modello addestrato con holdout di stile."""
    model = load_belief_model_npz(path)
    train = model.metadata.get("train")
    if not isinstance(train, dict):
        raise ValueError(f"{path}: metadata.train mancante")
    split = train.get("split")
    if not isinstance(split, dict) or split.get("strategy") != "leave_one_opponent_out":
        raise ValueError(f"{path}: non e' un fold leave_one_opponent_out")
    holdout = split.get("holdout_opponent")
    if not isinstance(holdout, str) or not holdout:
        raise ValueError(f"{path}: holdout_opponent mancante")
    validation = train.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{path}: metadata.train.validation mancante")
    baseline = validation.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"{path}: confronto baseline mancante")
    dataset_artifact = model.metadata.get("dataset_artifact")
    baseline_artifact = baseline.get("artifact")
    if not isinstance(dataset_artifact, dict) or not isinstance(baseline_artifact, dict):
        raise ValueError(f"{path}: artefatti dataset/baseline mancanti")
    return FoldMetrics(
        holdout=holdout,
        candidate=_metric_dict(validation.get("candidate"), label="candidate"),
        baseline=_metric_dict(baseline.get("metrics"), label="baseline"),
        model_artifact=_artifact(path),
        dataset_artifact=dict(dataset_artifact),
        baseline_artifact=dict(baseline_artifact),
    )


def _expected_roster_ids(path: Path) -> tuple[str, ...]:
    """Legge gli id dal roster congelato, mantenendone l'ordine documentale."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"Roster senza items: {path}")
    ids = tuple(str(item.get("id", "")).strip() for item in items if isinstance(item, dict))
    if len(ids) != len(items) or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Roster con id mancanti o duplicati: {path}")
    return ids


def summarize_folds(
    model_paths: tuple[Path, ...],
    *,
    roster_path: Path,
    gates: FoldGates,
    pilot: bool,
) -> dict[str, Any]:
    """Valida la matrice fold e costruisce decisione e report aggregato."""
    if not model_paths:
        raise ValueError("Nessun modello fold")
    folds = [load_fold(path) for path in model_paths]
    by_holdout = {fold.holdout: fold for fold in folds}
    if len(by_holdout) != len(folds):
        raise ValueError("Holdout duplicato tra i modelli fold")
    expected_ids = _expected_roster_ids(roster_path)
    if set(by_holdout) != set(expected_ids):
        raise ValueError(f"Fold incompleti: attesi={list(expected_ids)} trovati={sorted(by_holdout)}")

    dataset_hashes = {str(fold.dataset_artifact.get("sha256")) for fold in folds}
    baseline_hashes = {str(fold.baseline_artifact.get("sha256")) for fold in folds}
    if len(dataset_hashes) != 1 or len(baseline_hashes) != 1:
        raise ValueError("I fold non condividono lo stesso dataset o la stessa baseline")

    ordered = [by_holdout[opponent_id] for opponent_id in expected_ids]
    candidate_macro = {metric: float(np.mean([fold.candidate[metric] for fold in ordered])) for metric in METRIC_NAMES}
    baseline_macro = {metric: float(np.mean([fold.baseline[metric] for fold in ordered])) for metric in METRIC_NAMES}
    deltas = {metric: candidate_macro[metric] - baseline_macro[metric] for metric in METRIC_NAMES}
    bce_relative_improvement = (baseline_macro["bce"] - candidate_macro["bce"]) / baseline_macro["bce"]
    fold_bce_relative_change = {
        fold.holdout: (fold.candidate["bce"] - fold.baseline["bce"]) / fold.baseline["bce"] for fold in ordered
    }
    worst_fold_bce_relative_regression = max(fold_bce_relative_change.values())

    checks = {
        "macro_bce_relative_improvement": (bce_relative_improvement >= gates.min_macro_bce_relative_improvement),
        "macro_topk_non_regression": deltas["topk_recall"] >= gates.min_macro_topk_delta,
        "macro_brier_non_regression": deltas["brier"] <= gates.max_macro_brier_delta,
        "macro_ece_non_regression": deltas["ece"] <= gates.max_macro_ece_delta,
        "worst_fold_bce_regression": (
            worst_fold_bce_relative_regression <= gates.max_worst_fold_bce_relative_regression
        ),
    }
    if pilot:
        verdict = "pilot_pipeline_validated"
        note = "Il pilot verifica solo integrita' e metriche; non autorizza training finale o runtime A/B."
    elif all(checks.values()):
        verdict = "go_train_all_styles_candidate"
        note = "Il GO autorizza solo candidato all-styles e A/B PIMC v0-v1; nessuna promozione."
    else:
        verdict = "stop_offline_gate"
        note = "Almeno un gate offline e' fallito: non eseguire il confronto runtime come candidato di promozione."

    return {
        "schema": "briscola.belief_leave_one_out.v1",
        "pilot": bool(pilot),
        "roster": {
            "artifact": _artifact(roster_path),
            "opponent_ids": list(expected_ids),
            "folds": len(ordered),
        },
        "artifacts": {
            "dataset": ordered[0].dataset_artifact,
            "baseline": ordered[0].baseline_artifact,
            "fold_models": [fold.model_artifact for fold in ordered],
        },
        "folds": {
            fold.holdout: {
                "candidate": fold.candidate,
                "baseline": fold.baseline,
                "bce_relative_change_candidate_minus_baseline": fold_bce_relative_change[fold.holdout],
            }
            for fold in ordered
        },
        "macro": {
            "candidate": candidate_macro,
            "baseline": baseline_macro,
            "delta_candidate_minus_baseline": deltas,
            "bce_relative_improvement": bce_relative_improvement,
            "worst_fold_bce_relative_regression": worst_fold_bce_relative_regression,
        },
        "gates": {
            "thresholds": {
                "min_macro_bce_relative_improvement": gates.min_macro_bce_relative_improvement,
                "min_macro_topk_delta": gates.min_macro_topk_delta,
                "max_macro_brier_delta": gates.max_macro_brier_delta,
                "max_macro_ece_delta": gates.max_macro_ece_delta,
                "max_worst_fold_bce_relative_regression": gates.max_worst_fold_bce_relative_regression,
            },
            "checks": checks,
        },
        "decision": {"verdict": verdict, "note_it": note},
        "versions": {
            "code": get_code_version(),
            "rules": get_rules_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": _git_commit(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggrega gate leave-one-opponent-out belief v1")
    parser.add_argument("--model", action="append", required=True, help="Modello fold .npz; ripetibile.")
    parser.add_argument("--roster", required=True, help="Roster JSON congelato.")
    parser.add_argument("--out-json", required=True, help="Report aggregato JSON.")
    parser.add_argument("--pilot", action="store_true", help="Marca una run smoke non probatoria.")
    parser.add_argument("--min-macro-bce-improvement", type=float, default=0.01)
    parser.add_argument("--min-macro-topk-delta", type=float, default=0.0)
    parser.add_argument("--max-macro-brier-delta", type=float, default=0.0)
    parser.add_argument("--max-macro-ece-delta", type=float, default=0.0)
    parser.add_argument("--max-worst-fold-bce-regression", type=float, default=0.02)
    args = parser.parse_args()

    report = summarize_folds(
        tuple(Path(path) for path in args.model),
        roster_path=Path(args.roster),
        gates=FoldGates(
            min_macro_bce_relative_improvement=float(args.min_macro_bce_improvement),
            min_macro_topk_delta=float(args.min_macro_topk_delta),
            max_macro_brier_delta=float(args.max_macro_brier_delta),
            max_macro_ece_delta=float(args.max_macro_ece_delta),
            max_worst_fold_bce_relative_regression=float(args.max_worst_fold_bce_regression),
        ),
        pilot=bool(args.pilot),
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    macro = report["macro"]
    print(
        f"fold={report['roster']['folds']} | BCE candidate={macro['candidate']['bce']:.4f} "
        f"baseline={macro['baseline']['bce']:.4f} rel_improvement={macro['bce_relative_improvement'] * 100:.2f}%"
    )
    print(
        f"topk_delta={macro['delta_candidate_minus_baseline']['topk_recall']:+.4f} "
        f"brier_delta={macro['delta_candidate_minus_baseline']['brier']:+.4f} "
        f"ece_delta={macro['delta_candidate_minus_baseline']['ece']:+.4f}"
    )
    print(f"verdict={report['decision']['verdict']} | report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
