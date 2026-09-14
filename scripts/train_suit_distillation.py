#!/usr/bin/env python3
"""Distilla il teacher simmetrizzato in una singola MLP v4 warm-start da v13."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from briscola_ai.ai.models import MLPBCModel, load_bc_model_npz
from briscola_ai.ai.training.suit_distillation import (
    load_suit_distillation_dataset,
    train_suit_distillation,
)


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset creato da generate_suit_distillation_dataset.py")
    parser.add_argument("--init", required=True, help="MLP v4 usata come warm-start")
    parser.add_argument("--out", required=True, help="Modello `.npz` output")
    parser.add_argument("--report-json", default="", help="Report dettagliato opzionale")
    parser.add_argument("--epochs", type=int, default=5, help="Epoche massime")
    parser.add_argument("--batch-size", type=int, default=1024, help="Esempi originali per minibatch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="L2 sui pesi")
    parser.add_argument("--seed", type=int, default=20260711, help="Seed shuffle e augmentation")
    parser.add_argument(
        "--no-paired-augmentation",
        action="store_true",
        help="Disabilita la copia supervisionata con una rinomina non identità per minibatch",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    init_path = Path(args.init)
    out_path = Path(args.out)
    dataset = load_suit_distillation_dataset(data_path)
    init_model = load_bc_model_npz(init_path)
    if not isinstance(init_model, MLPBCModel):
        raise ValueError("--init deve essere una MLP con w1/b1/w2/b2")

    started = time.perf_counter()
    result = train_suit_distillation(
        dataset,
        init_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        paired_augmentation=not args.no_paired_augmentation,
    )
    elapsed = time.perf_counter() - started

    for row in result.epochs:
        print(
            f"epoch {row.epoch:02d} | train CE {row.train_cross_entropy:.6f} "
            f"agree {row.train_argmax_agreement:.4f} | "
            f"val KL {row.validation.kl_divergence:.6f} "
            f"agree {row.validation.argmax_agreement:.4f}"
        )
    print(
        f"best epoch {result.best_epoch} | val KL {result.best_validation.kl_divergence:.6f} "
        f"agree {result.best_validation.argmax_agreement:.4f} | "
        f"test KL {result.test.kl_divergence:.6f} agree {result.test.argmax_agreement:.4f}"
    )

    paired_augmentation = not args.no_paired_augmentation
    metadata = {
        "format": "mlp_bc_v1",
        "label": "Distillazione simmetrica v0",
        "description_it": (
            "MLP v4 sperimentale distillata dalla media v13 sulle 24 rinomine dei semi; "
            "non promossa nel catalogo ufficiale."
        ),
        "feature_dim": int(result.w1.shape[0]),
        "hidden_dim": int(result.w1.shape[1]),
        "action_dim": 40,
        "encoder": "encode_observation_2p:v4",
        "encoder_version": "v4",
        "inference_overkill_guard": False,
        "distillation": {
            "format": "suit_symmetry_logits_mean_24x_v1",
            "dataset_path": str(data_path),
            "init_path": str(init_path),
            "split_unit": "game",
            "seed": int(args.seed),
            "paired_augmentation": paired_augmentation,
            "epochs_requested": int(args.epochs),
            "best_epoch": int(result.best_epoch),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "validation_kl": result.best_validation.kl_divergence,
            "validation_argmax_agreement": result.best_validation.argmax_agreement,
            "test_kl": result.test.kl_divergence,
            "test_argmax_agreement": result.test.argmax_agreement,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        w1=result.w1,
        b1=result.b1,
        w2=result.w2,
        b2=result.b2,
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )

    report = {
        "schema_version": 1,
        "data": str(data_path),
        "init": str(init_path),
        "out": str(out_path),
        "elapsed_seconds": elapsed,
        "config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "paired_augmentation": paired_augmentation,
        },
        "dataset_metadata": dataset.metadata,
        "before_validation": asdict(result.before_validation),
        "before_test": asdict(result.before_test),
        "best_epoch": int(result.best_epoch),
        "best_validation": asdict(result.best_validation),
        "test": asdict(result.test),
        "epochs": [asdict(row) for row in result.epochs],
    }
    if args.report_json.strip():
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"Saved model: {out_path} ({out_path.stat().st_size} bytes, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
