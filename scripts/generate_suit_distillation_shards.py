#!/usr/bin/env python3
"""Genera un corpus teacher 24x sharded, deterministico e riprendibile."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from generate_suit_distillation_dataset import (
    DEFAULT_OPPONENT_MIX,
    SuitDatasetGenerationConfig,
    generate_suit_distillation_dataset,
)

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.ai.training.a2c_checkpoint import config_fingerprint
from briscola_ai.ai.training.opponent_mix import parse_opponent_mix
from briscola_ai.ai.training.suit_distillation import (
    DATASET_FORMAT,
    load_suit_distillation_dataset,
    make_game_split_ids,
)
from briscola_ai.ai.training.suit_distillation_shards import (
    SHARDED_MANIFEST_SCHEMA,
    SHARDED_MANIFEST_STATUS_COMPLETE,
    SHARDED_MANIFEST_STATUS_IN_PROGRESS,
    SuitDistillationShard,
    derive_shard_seed,
    load_sharded_suit_distillation_dataset,
    load_strict_json,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "manifest.json"


def _repo_relative(path: Path) -> str:
    """Usa path relativi al repository quando possibile, altrimenti conserva l'assoluto."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _file_receipt(path: Path) -> dict[str, object]:
    """Crea la ricevuta portabile di un file di input o sorgente."""
    return {
        "path": _repo_relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _git_commit() -> str:
    """Registra il commit come contesto; gli hash sorgente restano l'identità eseguibile."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Pubblica il manifest con replace atomico e JSON rigoroso/deterministico."""
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


def _atomic_save_dataset(path: Path, dataset) -> None:
    """Comprime uno shard in un temporaneo e lo rende visibile solo quando completo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(raw_tmp)
    try:
        dataset.save(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _split_counts(split_ids: np.ndarray) -> dict[str, int]:
    """Conta partite train/validation/test da un vettore globale o da una sua slice."""
    return {
        "train": int(np.sum(split_ids == 0)),
        "validation": int(np.sum(split_ids == 1)),
        "test": int(np.sum(split_ids == 2)),
    }


def _build_shard_record(
    *,
    path: Path,
    manifest_path: Path,
    dataset,
    shard_index: int,
    shard_seed: int,
) -> SuitDistillationShard:
    """Trasforma uno shard già validato nella ricevuta inclusa nel manifest."""
    metadata = dataset.metadata
    try:
        relative_path = str(path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError as exc:
        raise ValueError(f"Lo shard deve stare sotto {manifest_path.parent}: {path}") from exc
    return SuitDistillationShard(
        index=shard_index,
        path=relative_path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        seed=shard_seed,
        game_id_start=int(metadata["game_id_start"]),
        game_id_stop=int(metadata["game_id_stop"]),
        num_games=int(metadata["num_games"]),
        num_examples=int(metadata["num_examples"]),
        split_game_counts={str(key): int(value) for key, value in metadata["split_game_counts"].items()},
        opponent_game_counts={str(key): int(value) for key, value in metadata["opponent_game_counts"].items()},
    )


def _aggregate_counts(records: list[SuitDistillationShard], field: str) -> dict[str, int]:
    """Somma i contatori registrati negli shard già completati."""
    total: Counter[str] = Counter()
    for record in records:
        total.update(getattr(record, field))
    return dict(sorted(total.items()))


def _build_manifest_base(
    *,
    model_path: Path,
    num_games: int,
    games_per_shard: int,
    seed: int,
    temperature: float,
    opponent_mix: str,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[dict[str, object], np.ndarray]:
    """Costruisce configurazione globale e fingerprint prima di generare il primo byte."""
    if num_games < 3 or games_per_shard < 3 or games_per_shard > num_games:
        raise ValueError("num_games e games_per_shard devono essere >= 3, con shard non più grande del corpus")
    if num_games % games_per_shard != 0:
        raise ValueError("Per questo protocollo num_games deve essere divisibile per games_per_shard")
    if temperature <= 0.0:
        raise ValueError("temperature deve essere > 0")
    mix = parse_opponent_mix(opponent_mix)
    split_seed = seed ^ 0x51A17
    split_by_game = make_game_split_ids(
        num_games,
        seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    num_shards = num_games // games_per_shard
    dataset_config: dict[str, object] = {
        "format": DATASET_FORMAT,
        "schema_version": 1,
        "encoder_version": "v4",
        "feature_dim": int(FEATURE_DIM_2P_V4),
        "action_dim": 40,
        "num_games": num_games,
        "num_examples": num_games * 38,
        "examples_per_game": 38,
        "num_shards": num_shards,
        "games_per_shard": games_per_shard,
        "seed": seed,
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "split_unit": "game",
        "split_game_counts": _split_counts(split_by_game),
        "teacher_temperature": temperature,
        "teacher_target": "mean_logits_over_24_suit_permutations",
        "opponent_mix": [{"name": item.name, "prob": item.prob} for item in mix],
        "base_seat": "alternating",
        "forced_decisions_excluded_per_game": 2,
        "shard_seed_derivation": "sha256:suit-distillation-shard-v1",
        "shard_order": "ascending_index",
    }
    teacher_model = _file_receipt(model_path)
    source_paths = (
        Path(__file__),
        ROOT / "scripts/generate_suit_distillation_dataset.py",
        ROOT / "src/briscola_ai/ai/training/suit_distillation.py",
        ROOT / "src/briscola_ai/ai/training/suit_distillation_shards.py",
        ROOT / "src/briscola_ai/ai/agents/suit_symmetrized.py",
    )
    source_files = [_file_receipt(path) for path in source_paths]
    fingerprint_material: dict[str, object] = {
        "dataset": dataset_config,
        "teacher_model": teacher_model,
        "source_files": source_files,
    }
    manifest: dict[str, object] = {
        "schema": SHARDED_MANIFEST_SCHEMA,
        "status": SHARDED_MANIFEST_STATUS_IN_PROGRESS,
        "config_fingerprint": config_fingerprint(fingerprint_material),
        "dataset": dataset_config,
        "teacher_model": teacher_model,
        "provenance": {
            "git_commit_at_start": _git_commit(),
            "source_files": source_files,
        },
        "shards": [],
        "completed": {
            "num_shards": 0,
            "num_games": 0,
            "num_examples": 0,
            "split_game_counts": {"train": 0, "validation": 0, "test": 0},
            "opponent_game_counts": {},
        },
    }
    return manifest, split_by_game


def _manifest_with_records(
    base: dict[str, object],
    records: list[SuitDistillationShard],
    *,
    complete: bool,
) -> dict[str, object]:
    """Aggiorna il solo stato progressivo senza cambiare la configurazione congelata."""
    payload = dict(base)
    payload["status"] = SHARDED_MANIFEST_STATUS_COMPLETE if complete else SHARDED_MANIFEST_STATUS_IN_PROGRESS
    payload["shards"] = [record.to_payload() for record in records]
    payload["completed"] = {
        "num_shards": len(records),
        "num_games": sum(record.num_games for record in records),
        "num_examples": sum(record.num_examples for record in records),
        "split_game_counts": _aggregate_counts(records, "split_game_counts"),
        "opponent_game_counts": _aggregate_counts(records, "opponent_game_counts"),
    }
    return payload


def _load_resume_records(
    manifest_path: Path,
    *,
    expected_fingerprint: str,
) -> tuple[dict[str, Any], list[SuitDistillationShard]]:
    """Valida un prefisso esistente prima di saltare qualsiasi lavoro."""
    existing = load_strict_json(manifest_path)
    if existing.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("Il manifest esistente appartiene a una configurazione o a sorgenti diverse")
    validated = load_sharded_suit_distillation_dataset(
        manifest_path,
        require_complete=False,
        verify_hashes=True,
    )
    return existing, list(validated.shards)


def main() -> int:
    """Entry point CLI con resume al confine atomico fra shard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Policy v4 base del teacher simmetrizzato")
    parser.add_argument("--out-dir", required=True, help="Directory contenente manifest e shard")
    parser.add_argument("--num-games", type=int, default=250_000, help="Partite globali")
    parser.add_argument("--games-per-shard", type=int, default=25_000, help="Partite per shard")
    parser.add_argument("--seed", type=int, default=20260724, help="Seed globale raccolta e split")
    parser.add_argument("--opponent-mix", default=DEFAULT_OPPONENT_MIX, help="Mix name:weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperatura target soft")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=1_000, help="Frequenza log interna a ogni shard")
    parser.add_argument("--resume", action="store_true", help="Verifica e continua un prefisso esistente")
    parser.add_argument(
        "--stop-after-shards",
        type=int,
        default=0,
        help="Stop tecnico dopo N shard globali completati; 0 completa il corpus",
    )
    parser.add_argument("--verify-only", action="store_true", help="Valida hash e contenuto di un corpus completo")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise ValueError(f"Modello teacher mancante: {model_path}")
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    expected_manifest, split_by_game = _build_manifest_base(
        model_path=model_path,
        num_games=args.num_games,
        games_per_shard=args.games_per_shard,
        seed=args.seed,
        temperature=args.temperature,
        opponent_mix=args.opponent_mix,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    expected_fingerprint = str(expected_manifest["config_fingerprint"])

    if args.verify_only:
        corpus = load_sharded_suit_distillation_dataset(manifest_path, verify_hashes=True)
        for descriptor in corpus.shards:
            shard = corpus.load_shard(descriptor)
            del shard
            gc.collect()
        print(json.dumps({"verified": True, "manifest": str(manifest_path), "shards": len(corpus.shards)}))
        return 0

    records: list[SuitDistillationShard]
    if manifest_path.exists():
        if not args.resume:
            raise ValueError(f"Manifest già esistente; usare --resume senza sovrascriverlo: {manifest_path}")
        existing_manifest, records = _load_resume_records(
            manifest_path,
            expected_fingerprint=expected_fingerprint,
        )
        expected_manifest["provenance"] = existing_manifest["provenance"]
        if existing_manifest.get("status") == SHARDED_MANIFEST_STATUS_COMPLETE:
            print(json.dumps(existing_manifest["completed"], ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    else:
        existing_npz = list(out_dir.glob("shards/*.npz")) if out_dir.exists() else []
        if existing_npz:
            raise ValueError("Shard presenti senza manifest: spostare gli artefatti prima di iniziare")
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        _atomic_write_json(manifest_path, _manifest_with_records(expected_manifest, records, complete=False))

    base_agent = BCModelAgent.from_npz(model_path)
    if base_agent.encoder_version != "v4":
        raise ValueError("Il corpus sharded richiede una policy base v4")
    mix = parse_opponent_mix(args.opponent_mix)
    opponents: dict[str, Agent] = {}
    for item in mix:
        opponents[item.name] = base_agent if item.name == "mirror" else build_agent(item.name)

    num_shards = int(expected_manifest["dataset"]["num_shards"])
    started = time.perf_counter()
    for shard_index in range(len(records), num_shards):
        game_start = shard_index * args.games_per_shard
        game_stop = min(args.num_games, game_start + args.games_per_shard)
        shard_games = game_stop - game_start
        shard_seed = derive_shard_seed(args.seed, shard_index)
        shard_path = out_dir / "shards" / f"shard-{shard_index:05d}-of-{num_shards:05d}.npz"

        if shard_path.exists():
            if not args.resume:
                raise ValueError(f"Shard già esistente: {shard_path}")
            dataset = load_suit_distillation_dataset(shard_path)
            expected_orphan_metadata = {
                "manifest_config_fingerprint": expected_fingerprint,
                "shard_index": shard_index,
                "shard_seed": shard_seed,
                "game_id_start": game_start,
                "game_id_stop": game_stop,
            }
            for key, expected in expected_orphan_metadata.items():
                if dataset.metadata.get(key) != expected:
                    raise ValueError(f"Shard orfano con {key} diverso: {shard_path}")
            record = _build_shard_record(
                path=shard_path,
                manifest_path=manifest_path,
                dataset=dataset,
                shard_index=shard_index,
                shard_seed=shard_seed,
            )
            print(f"recovered shard {shard_index + 1}/{num_shards}: {shard_path}", flush=True)
        else:
            config = SuitDatasetGenerationConfig(
                out_path=shard_path,
                num_games=shard_games,
                seed=shard_seed,
                opponent_mix=args.opponent_mix,
                temperature=args.temperature,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                progress_every=args.progress_every,
            )
            dataset, counters = generate_suit_distillation_dataset(
                config,
                base_agent=base_agent,
                opponents=opponents,
                teacher_model_sha256=sha256_file(model_path),
                game_id_offset=game_start,
                split_by_game=split_by_game[game_start:game_stop],
                metadata_overrides={
                    "global_seed": args.seed,
                    "manifest_config_fingerprint": expected_fingerprint,
                    "shard_index": shard_index,
                    "shard_count": num_shards,
                    "shard_seed": shard_seed,
                },
            )
            _atomic_save_dataset(shard_path, dataset)
            record = _build_shard_record(
                path=shard_path,
                manifest_path=manifest_path,
                dataset=dataset,
                shard_index=shard_index,
                shard_seed=shard_seed,
            )
            print(
                f"saved shard {shard_index + 1}/{num_shards}: games={shard_games} "
                f"examples={record.num_examples} bytes={record.size_bytes} "
                f"generate_s={float(counters['elapsed_seconds_before_save']):.1f}",
                flush=True,
            )

        records.append(record)
        complete = len(records) == num_shards
        current_manifest = _manifest_with_records(expected_manifest, records, complete=complete)
        _atomic_write_json(manifest_path, current_manifest)
        del dataset
        gc.collect()
        if args.stop_after_shards > 0 and len(records) >= args.stop_after_shards and not complete:
            print(f"technical stop after {len(records)} shard; resume with the same command and --resume", flush=True)
            break

    elapsed = time.perf_counter() - started
    final_payload = load_strict_json(manifest_path)
    load_sharded_suit_distillation_dataset(
        manifest_path,
        require_complete=final_payload.get("status") == SHARDED_MANIFEST_STATUS_COMPLETE,
        verify_hashes=False,
    )
    summary = {
        "status": final_payload["status"],
        "manifest": str(manifest_path),
        "elapsed_seconds_this_invocation": elapsed,
        "completed": final_payload["completed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
