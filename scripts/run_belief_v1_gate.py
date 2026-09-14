#!/usr/bin/env python3
"""Esegue il gate belief v1 multi-stile senza saltare passaggi decisionali.

Il job e' volutamente sequenziale: i sette fold condividono un dataset grande e
allenarli in parallelo moltiplicherebbe memoria e contesa BLAS. La pipeline:

1. genera un dataset mirror secondo il roster congelato;
2. allena un fold per ogni stile, lasciandolo interamente fuori dal training;
3. applica i gate offline preregistrati;
4. allena il candidato su tutti gli stili solo dopo un GO offline.

Il candidato finale resta locale e non viene promosso. Il confronto PIMC v0-v1 e'
un esperimento successivo, stampato come comando pronto al termine del job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.models.belief_model import load_belief_model_npz

ROSTER_SCHEMA = "briscola.belief_roster.v1"
GO_VERDICT = "go_train_all_styles_candidate"


@dataclass(frozen=True, slots=True)
class ProtocolPaths:
    """Path deterministici di dataset, fold, report e candidato locale."""

    dataset: Path
    folds: tuple[Path, ...]
    summary: Path
    candidate: Path
    runtime_report: Path


def _repo_root() -> Path:
    """Ritorna la root del checkout anche se il comando parte da un'altra cwd."""
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    """SHA-256 streaming usato per validare gli artefatti ripresi."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path) -> str:
    """Preferisce un path relativo al repository, ma accetta work dir esterne."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    """Carica un oggetto JSON e rifiuta payload di tipo inatteso."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Oggetto JSON atteso in {path}")
    return payload


def load_roster_ids(path: Path) -> tuple[tuple[str, ...], int]:
    """Valida lo schema minimo del roster e ritorna id ordinati e peso totale."""
    payload = _load_json(path)
    if payload.get("schema") != ROSTER_SCHEMA:
        raise ValueError(f"Schema roster inatteso in {path}: {payload.get('schema')!r}")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Roster senza items: {path}")

    opponent_ids: list[str] = []
    total_weight = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"roster.items[{index}] deve essere un oggetto")
        opponent_id = str(item.get("id", "")).strip()
        weight = item.get("weight", 1)
        if not opponent_id or not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"roster.items[{index}] richiede id e weight intero positivo")
        opponent_ids.append(opponent_id)
        total_weight += weight
    if len(opponent_ids) != len(set(opponent_ids)):
        raise ValueError(f"Roster con id duplicati: {opponent_ids}")
    return tuple(opponent_ids), total_weight


def _safe_id(value: str) -> str:
    """Converte un id validato in un basename portabile e non ambiguo."""
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"ID stile non valido per un path: {value!r}")
    return safe


def build_protocol_paths(
    work_dir: Path,
    opponent_ids: tuple[str, ...],
    *,
    num_games: int,
    seed: int,
    hidden_dim: int,
) -> ProtocolPaths:
    """Costruisce nomi stabili, cosi' ``--resume`` riprende gli stessi file."""
    stem = f"g{num_games}_seed{seed}"
    safe_ids = tuple(_safe_id(item) for item in opponent_ids)
    if len(safe_ids) != len(set(safe_ids)):
        raise ValueError(f"Gli id roster collidono dopo la normalizzazione dei path: {opponent_ids}")
    return ProtocolPaths(
        dataset=work_dir / f"belief_v1_multistyle_{stem}.npz",
        folds=tuple(work_dir / "folds" / f"belief_v1_holdout_{item}_{stem}.npz" for item in safe_ids),
        summary=work_dir / f"belief_v1_leave_one_out_{stem}.json",
        candidate=work_dir / f"belief_v1_all_styles_h{hidden_dim}_{stem}.npz",
        runtime_report=work_dir / f"belief_v1_vs_v0_pimc16x8_screen_{stem}.json",
    )


def _run(command: list[str], *, root: Path) -> None:
    """Esegue un sottopasso mostrando nel log il comando esatto."""
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _dataset_metadata(path: Path) -> dict[str, Any]:
    """Legge solo i metadati del dataset ripreso, senza materializzare le matrici."""
    with np.load(path) as data:
        if "metadata_json" not in data:
            raise ValueError(f"Dataset senza metadata_json: {path}")
        raw = str(data["metadata_json"])
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadati dataset non validi: {path}")
    return payload


def _validate_resumed_dataset(
    path: Path,
    *,
    roster_path: Path,
    num_games: int,
    seed: int,
) -> None:
    """Impedisce che ``--resume`` mescoli dataset creati con protocolli diversi."""
    metadata = _dataset_metadata(path)
    roster = metadata.get("roster")
    if (
        metadata.get("format") != "belief_dataset_v2"
        or metadata.get("num_games") != num_games
        or metadata.get("seed") != seed
        or not isinstance(roster, dict)
        or roster.get("source_sha256") != _sha256(roster_path)
    ):
        raise ValueError(f"Dataset esistente incompatibile con il protocollo richiesto: {path}")


def _validate_resumed_model(
    path: Path,
    *,
    dataset_path: Path,
    baseline_path: Path,
    holdout: str | None,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    """Verifica dati, split e iperparametri prima di saltare un training esistente."""
    model = load_belief_model_npz(path)
    train = model.metadata.get("train")
    dataset_artifact = model.metadata.get("dataset_artifact")
    if not isinstance(train, dict) or not isinstance(dataset_artifact, dict):
        raise ValueError(f"Modello ripreso senza metadati del protocollo: {path}")
    split = train.get("split")
    validation = train.get("validation")
    baseline = validation.get("baseline") if isinstance(validation, dict) else None
    baseline_artifact = baseline.get("artifact") if isinstance(baseline, dict) else None
    expected_strategy = "leave_one_opponent_out" if holdout is not None else "game_modulo"
    if (
        model.hidden_dim != hidden_dim
        or dataset_artifact.get("sha256") != _sha256(dataset_path)
        or not isinstance(baseline_artifact, dict)
        or baseline_artifact.get("sha256") != _sha256(baseline_path)
        or not isinstance(split, dict)
        or split.get("strategy") != expected_strategy
        or split.get("holdout_opponent") != holdout
        or train.get("epochs") != epochs
        or train.get("batch_size") != batch_size
        or train.get("seed") != seed
        or not np.isclose(float(train.get("lr", float("nan"))), learning_rate)
    ):
        raise ValueError(f"Modello esistente incompatibile con il protocollo richiesto: {path}")


def _prepare_output(path: Path, *, resume: bool) -> bool:
    """Ritorna True se un artefatto puo' essere ripreso; altrimenti prepara la directory."""
    if path.exists():
        if not resume:
            raise FileExistsError(f"Artefatto gia' presente: {path}. Usare --resume o un altro --work-dir.")
        print(f"resume: trovato {path}", flush=True)
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    return False


def _training_command(
    *,
    root: Path,
    data_path: Path,
    out_path: Path,
    baseline_path: Path,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    ece_bins: int,
    holdout: str | None,
) -> list[str]:
    """Costruisce il comando comune ai fold e al candidato all-styles."""
    command = [
        sys.executable,
        str(root / "scripts/train_belief.py"),
        "--data",
        str(data_path),
        "--out",
        str(out_path),
        "--baseline-model",
        str(baseline_path),
        "--hidden-dim",
        str(hidden_dim),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(learning_rate),
        "--seed",
        str(seed),
        "--ece-bins",
        str(ece_bins),
    ]
    if holdout is not None:
        command.extend(["--holdout-opponent", holdout])
    return command


def _print_runtime_command(
    *,
    root: Path,
    candidate_path: Path,
    baseline_path: Path,
    report_path: Path,
    seed: int,
) -> None:
    """Stampa il solo passo successivo autorizzato dal GO offline."""
    command = [
        "uv",
        "run",
        "python",
        "scripts/evaluate_pimc.py",
        "--model",
        "data/models/best_a2c_v14.npz",
        "--belief-model",
        _display_path(candidate_path, root=root),
        "--opponent",
        "pimc",
        "--opponent-belief-model",
        _display_path(baseline_path, root=root),
        "--determinizations",
        "16",
        "--max-unknown-cards",
        "8",
        "--opponent-determinizations",
        "16",
        "--opponent-max-unknown-cards",
        "8",
        "--num-games",
        "2000",
        "--seed",
        str(seed + 1),
        "--out-json",
        _display_path(report_path, root=root),
    ]
    print("\nGO offline. Prossimo passo, NON avviato automaticamente:", flush=True)
    print(" ".join(command), flush=True)


def _parse_args() -> argparse.Namespace:
    """Definisce il protocollo ufficiale e le sole riduzioni utili agli smoke test."""
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=root / "docs/plans/belief-v1-roster-2026-07-14.json")
    parser.add_argument("--work-dir", type=Path, default=root / "data/belief/belief_v1_gate_20260714")
    parser.add_argument("--baseline-model", type=Path, default=root / "data/models/belief_v0_h128_50k_seed20260702.npz")
    parser.add_argument("--num-games", type=int, default=66000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-unknown-cards", type=int, default=10)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Riprende solo artefatti compatibili gia' completi.")
    return parser.parse_args()


def main() -> int:
    """Esegue il protocollo e si ferma automaticamente al primo gate fallito."""
    args = _parse_args()
    root = _repo_root()
    roster_path = args.roster.resolve()
    baseline_path = args.baseline_model.resolve()
    work_dir = args.work_dir.resolve()
    if not roster_path.is_file() or not baseline_path.is_file():
        raise FileNotFoundError(f"Roster o baseline mancanti: {roster_path}, {baseline_path}")
    if min(args.num_games, args.hidden_dim, args.epochs, args.batch_size) <= 0:
        raise ValueError("num-games, hidden-dim, epochs e batch-size devono essere > 0")
    if not 0.0 <= args.epsilon <= 1.0 or args.max_unknown_cards < 0 or args.ece_bins <= 1:
        raise ValueError("epsilon, max-unknown-cards o ece-bins non validi")

    opponent_ids, weight_total = load_roster_ids(roster_path)
    if args.num_games % weight_total != 0:
        raise ValueError(f"--num-games deve essere multiplo del peso roster {weight_total}")
    paths = build_protocol_paths(
        work_dir,
        opponent_ids,
        num_games=args.num_games,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
    )

    if _prepare_output(paths.dataset, resume=args.resume):
        _validate_resumed_dataset(
            paths.dataset,
            roster_path=roster_path,
            num_games=args.num_games,
            seed=args.seed,
        )
    else:
        _run(
            [
                sys.executable,
                str(root / "scripts/generate_belief_dataset.py"),
                "--out",
                str(paths.dataset),
                "--roster",
                str(roster_path),
                "--num-games",
                str(args.num_games),
                "--seed",
                str(args.seed),
                "--epsilon",
                str(args.epsilon),
                "--max-unknown-cards",
                str(args.max_unknown_cards),
            ],
            root=root,
        )

    for opponent_id, fold_path in zip(opponent_ids, paths.folds, strict=True):
        if _prepare_output(fold_path, resume=args.resume):
            _validate_resumed_model(
                fold_path,
                dataset_path=paths.dataset,
                baseline_path=baseline_path,
                holdout=opponent_id,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
            )
            continue
        _run(
            _training_command(
                root=root,
                data_path=paths.dataset,
                out_path=fold_path,
                baseline_path=baseline_path,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
                ece_bins=args.ece_bins,
                holdout=opponent_id,
            ),
            root=root,
        )

    # Il summary e' economico e viene sempre rigenerato dai fold correnti. In questo
    # modo un resume non puo' riusare per errore un verdetto precedente ai modelli.
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_command = [
        sys.executable,
        str(root / "scripts/summarize_belief_folds.py"),
        "--roster",
        str(roster_path),
        "--out-json",
        str(paths.summary),
    ]
    for fold_path in paths.folds:
        summary_command.extend(["--model", str(fold_path)])
    _run(summary_command, root=root)

    summary = _load_json(paths.summary)
    verdict = summary.get("decision", {}).get("verdict")
    if verdict != GO_VERDICT:
        print(f"\nSTOP offline ({verdict}). Nessun candidato all-styles e nessun test PIMC.", flush=True)
        return 0

    if _prepare_output(paths.candidate, resume=args.resume):
        _validate_resumed_model(
            paths.candidate,
            dataset_path=paths.dataset,
            baseline_path=baseline_path,
            holdout=None,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )
    else:
        _run(
            _training_command(
                root=root,
                data_path=paths.dataset,
                out_path=paths.candidate,
                baseline_path=baseline_path,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
                ece_bins=args.ece_bins,
                holdout=None,
            ),
            root=root,
        )
    _print_runtime_command(
        root=root,
        candidate_path=paths.candidate,
        baseline_path=baseline_path,
        report_path=paths.runtime_report,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
