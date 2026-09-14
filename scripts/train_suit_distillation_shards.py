#!/usr/bin/env python3
"""Distilla un corpus teacher 24x sharded in una singola MLP v4."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from briscola_ai.ai.models import MLPBCModel, load_bc_model_npz
from briscola_ai.ai.training.a2c_checkpoint import atomic_savez
from briscola_ai.ai.training.suit_distillation_shards import (
    load_sharded_suit_distillation_dataset,
    sha256_file,
    train_suit_distillation_sharded,
)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Scrive il report soltanto dopo una serializzazione JSON rigorosa completa."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(raw_tmp)
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    """Entry point CLI del trainer streaming."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest completo creato dal generatore sharded")
    parser.add_argument("--init", required=True, help="MLP v4 usata come warm-start")
    parser.add_argument("--out", required=True, help="Modello candidato `.npz`")
    parser.add_argument("--report-json", required=True, help="Report rigoroso del training")
    parser.add_argument("--epochs", type=int, default=5, help="Epoche massime")
    parser.add_argument("--batch-size", type=int, default=1024, help="Esempi originali per minibatch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="L2 sui pesi")
    parser.add_argument("--seed", type=int, default=20260724, help="Seed ordine shard, righe e augmentation")
    parser.add_argument("--label", default="Distillazione teacher 20M 250k", help="Label interna del candidato")
    parser.add_argument(
        "--description-it",
        default=(
            "MLP v4 sperimentale distillata dal teacher 20M sulle 24 rinomine dei semi; "
            "non promossa nel catalogo ufficiale."
        ),
        help="Descrizione interna del candidato",
    )
    parser.add_argument(
        "--no-paired-augmentation",
        action="store_true",
        help="Disabilita la copia con una rinomina non identità per minibatch",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    init_path = Path(args.init)
    out_path = Path(args.out)
    report_path = Path(args.report_json)
    corpus = load_sharded_suit_distillation_dataset(manifest_path, verify_hashes=True)
    init_model = load_bc_model_npz(init_path)
    if not isinstance(init_model, MLPBCModel):
        raise ValueError("--init deve essere una MLP con w1/b1/w2/b2")

    started = time.perf_counter()
    paired_augmentation = not args.no_paired_augmentation
    result = train_suit_distillation_sharded(
        corpus,
        init_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        paired_augmentation=paired_augmentation,
    )
    elapsed = time.perf_counter() - started

    for row in result.epochs:
        print(
            f"epoch {row.epoch:02d} | train CE {row.train_cross_entropy:.6f} "
            f"agree {row.train_argmax_agreement:.4f} | "
            f"val KL {row.validation.kl_divergence:.6f} "
            f"agree {row.validation.argmax_agreement:.4f}",
            flush=True,
        )
    print(
        f"best epoch {result.best_epoch} | val KL {result.best_validation.kl_divergence:.6f} "
        f"agree {result.best_validation.argmax_agreement:.4f} | "
        f"test KL {result.test.kl_divergence:.6f} agree {result.test.argmax_agreement:.4f}",
        flush=True,
    )

    manifest_receipt = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }
    init_receipt = {
        "path": str(init_path),
        "sha256": sha256_file(init_path),
        "size_bytes": init_path.stat().st_size,
    }
    teacher_model = corpus.payload["teacher_model"]
    metadata = {
        "format": "mlp_bc_v1",
        "label": args.label,
        "description_it": args.description_it,
        "feature_dim": int(result.w1.shape[0]),
        "hidden_dim": int(result.w1.shape[1]),
        "action_dim": 40,
        "encoder": "encode_observation_2p:v4",
        "encoder_version": "v4",
        "inference_overkill_guard": False,
        "distillation": {
            "format": "suit_symmetry_logits_mean_24x_sharded_v1",
            "manifest": manifest_receipt,
            "manifest_config_fingerprint": corpus.config_fingerprint,
            "teacher_model": teacher_model,
            "init_model": init_receipt,
            "split_unit": "game",
            "shard_order": "shuffle_per_epoch_then_rows_within_shard",
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
    atomic_savez(
        out_path,
        w1=result.w1,
        b1=result.b1,
        w2=result.w2,
        b2=result.b2,
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False),
    )
    model_receipt = {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    report: dict[str, object] = {
        "schema": "briscola.suit_distillation_sharded_train.v1",
        "manifest": manifest_receipt,
        "manifest_config_fingerprint": corpus.config_fingerprint,
        "init": init_receipt,
        "out": model_receipt,
        "elapsed_seconds": elapsed,
        "config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "paired_augmentation": paired_augmentation,
            "streaming_order": "shuffle_shards_then_rows_within_each_shard",
        },
        "dataset": corpus.dataset_metadata,
        "teacher_model": teacher_model,
        "before_validation": asdict(result.before_validation),
        "before_test": asdict(result.before_test),
        "best_epoch": int(result.best_epoch),
        "best_validation": asdict(result.best_validation),
        "test": asdict(result.test),
        "epochs": [asdict(row) for row in result.epochs],
    }
    _atomic_write_json(report_path, report)
    print(f"Saved model: {out_path} ({out_path.stat().st_size} bytes, {elapsed:.1f}s)", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
