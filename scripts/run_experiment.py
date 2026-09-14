#!/usr/bin/env python3
"""
Pipeline “training + evaluation” riproducibile per modelli Briscola AI.

Perché esiste
-------------
Allenare modelli RL richiede molte iterazioni. Se lanci comandi “a mano” è facile:
- dimenticare un benchmark (es. holdout);
- perdere la configurazione con cui hai allenato un `.npz`;
- sovrascrivere file o creare nomi poco descrittivi.

Questo script orchestri:
1) training (A2C o PG REINFORCE) -> salva un `.npz` in `data/models/`
2) evaluation matrix (`medium` e/o `big`, include holdout) -> salva JSON in `benchmarks/experiments/<name>/`
3) manifest JSON (config + versioni + percorsi) per riproducibilità
4) (opzionale) aggiorna un “best model” locale: `data/models/best_<algo>.npz`

Nota:
gli artefatti in `data/` e `benchmarks/` sono locali (gitignored).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.experiment_pipeline import (
    AlgoName,
    build_experiment_name,
    extract_best_metric_from_matrix_json,
    read_json,
    utc_now_iso,
    write_json,
)
from briscola_ai.ai.training.curriculum import CurriculumPreset, build_curriculum_stages
from briscola_ai.versioning import get_code_version, get_rules_version


def _run(cmd: list[str], *, log_path: Path, env_overrides: dict[str, str] | None = None) -> None:
    """
    Esegue un comando e fa tee su stdout + file.

    Nota didattica:
    salviamo i log su file per poter ricostruire *esattamente* cosa è successo
    anche dopo giorni (utile quando alleni decine di modelli).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # `buffering=1` -> line buffering (in text mode): aiuta ad avere log “live” anche su file.
    with log_path.open("w", encoding="utf-8", buffering=1) as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()

        # Importante: quando il processo figlio scrive su stdout e noi lo pipiamo (stdout=PIPE),
        # Python *bufferizza* le print (non essendo un TTY). Impostiamo quindi un env unbuffered,
        # così vediamo i log “live” (utile per capire se il training diverge).
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if env_overrides:
            env.update(env_overrides)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Comando fallito (exit={rc}): {' '.join(cmd)}")


def _parse_model_metadata_json(raw: Any) -> dict[str, Any]:
    """
    Parsa i metadati salvati nel `.npz`.

    Il loader runtime è best-effort; qui usiamo la stessa filosofia perché la
    compattazione non deve rendere inutilizzabile un modello solo per metadati
    non standard.
    """
    text = str(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_metadata_json": text}
    if isinstance(parsed, dict):
        return parsed
    return {"raw_metadata_json": text}


def _compact_best_model_metadata(*, metadata: dict[str, Any], source_model_path: Path) -> dict[str, Any]:
    """
    Rimuove i record metriche raw dai metadati del best model.

    I trainer salvano una riga metrica ogni pochi update; su run lunghi questi
    record possono pesare centinaia di MB pur non servendo al runtime. Il modello
    completo resta nella cartella esperimento, mentre il best ufficiale conserva
    solo un riassunto sufficiente per audit rapido e UI.
    """
    metrics = metadata.pop("metrics", None)
    if not isinstance(metrics, list):
        if metrics is not None:
            metadata["metrics"] = metrics
        return {
            "best_model_metrics_stripped": False,
            "num_training_metric_records": None,
            "full_metrics_source_model_path": str(source_model_path),
        }

    metadata["metrics_summary"] = {
        "stripped_for_best_model": True,
        "num_records": len(metrics),
        "first_record": metrics[0] if metrics else None,
        "last_record": metrics[-1] if metrics else None,
        "full_metrics_source_model_path": str(source_model_path),
    }
    return {
        "best_model_metrics_stripped": True,
        "num_training_metric_records": len(metrics),
        "full_metrics_source_model_path": str(source_model_path),
    }


def _write_compact_best_model(*, model_path: Path, best_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Scrive `best_path` come copia runtime compatta di `model_path`.

    Se il modello non contiene `metadata_json`, manteniamo il comportamento
    storico e facciamo una copia byte-per-byte: non c'è nulla da compattare.
    """
    with np.load(model_path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            shutil.copy2(model_path, best_path)
            return {
                "best_model_metrics_stripped": False,
                "num_training_metric_records": None,
                "full_metrics_source_model_path": str(model_path),
                "reason": "metadata_json_missing",
            }, None

        arrays = {key: np.asarray(data[key]).copy() for key in data.files if key != "metadata_json"}
        metadata = _parse_model_metadata_json(data["metadata_json"])

    compaction = _compact_best_model_metadata(metadata=metadata, source_model_path=model_path)
    tmp_path = best_path.with_name(best_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        **arrays,
        metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    os.replace(tmp_path, best_path)
    return compaction, metadata


def _copy_best_model(*, model_path: Path, best_path: Path, score: float, meta_path: Path, manifest: dict) -> bool:
    """
    Aggiorna il best model se lo score è migliore.

    Persistiamo anche un JSON a fianco, così il best non è “magico”.
    Il `.npz` ufficiale viene compattato per evitare che lunghe serie di
    `metadata_json.metrics` gonfino inutilmente il modello usato da UI/runtime.
    """
    previous_score: float | None = None
    if meta_path.exists():
        try:
            prev = read_json(meta_path)
            raw = prev.get("score")
            if isinstance(raw, (int, float)):
                previous_score = float(raw)
        except Exception:
            previous_score = None

    if previous_score is not None and score <= previous_score:
        print(f"Best model invariato: score={score:+.2f} <= best={previous_score:+.2f}")
        return False

    best_path.parent.mkdir(parents=True, exist_ok=True)
    compaction, compacted_metadata = _write_compact_best_model(model_path=model_path, best_path=best_path)
    payload = {
        "score": float(score),
        # Nota: se vuoi tenere `data/models/` minimal, questo path deve puntare al best
        # (non al modello “run-specific”, che potrebbe essere eliminato).
        "model_path": str(best_path),
        "source_model_path": str(model_path),
        "updated_utc": utc_now_iso(),
        "metadata_compaction": compaction,
        "manifest": manifest,
    }
    if compacted_metadata is not None and isinstance(compacted_metadata.get("inference_overkill_guard"), bool):
        payload["inference_overkill_guard"] = bool(compacted_metadata["inference_overkill_guard"])
    write_json(meta_path, payload)
    print(f"Updated best model: {best_path} (score={score:+.2f})")
    return True


def _prune_models_dir(*, models_dir: Path, algo: str) -> None:
    """
    Mantiene in `models_dir` solo i file “best” per l’algoritmo indicato.

    Regola conservativa:
    - tocchiamo solo file nella root di `models_dir` (niente ricorsione);
    - tocchiamo solo `.npz` e `.json`;
    - preserviamo `best_<algo>.npz` e `best_<algo>.json`.
    """
    keep = {f"best_{algo}.npz", f"best_{algo}.json"}
    for path in models_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in keep:
            continue
        if path.suffix.lower() not in {".npz", ".json"}:
            continue
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Esegue training + evaluation matrix + manifest (riproducibile).")
    parser.add_argument("--algo", choices=["a2c", "pg"], default="a2c", help="Algoritmo training (default: a2c).")
    parser.add_argument("--num-games", type=int, default=200000, help="Numero partite training.")
    parser.add_argument("--train-seed", type=int, default=0, help="Seed training.")
    parser.add_argument("--eval-seed", type=int, default=0, help="Seed evaluation (RNG agent).")
    parser.add_argument("--init", default="", help="Warm-start da un `.npz` (opzionale).")
    parser.add_argument("--opponent", default="heuristic_v1", help="Avversario (se non usi --opponent-mix).")
    parser.add_argument(
        "--opponent-mix",
        default="heuristic_v1:0.7,random:0.2,greedy_points:0.1",
        help="Miscela avversari (se valorizzata, sovrascrive --opponent).",
    )
    parser.add_argument(
        "--curriculum",
        choices=["", "easy_standard_hard"],
        default="",
        help=(
            "Se valorizzato, esegue un curriculum multi-stage (easy→standard→hard) "
            "e ignora `--opponent/--opponent-mix` (li usa solo se `--curriculum` è vuoto)."
        ),
    )
    parser.add_argument("--seat-fair", action="store_true", help="Seat-fair durante training (consigliato).")
    parser.add_argument(
        "--benchmarks",
        default="medium,big",
        help="Benchmark da eseguire in evaluation matrix (CSV: small,medium,big). Default: medium,big.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Numero processi per parallelizzare le righe di `evaluate_matrix.py` (default: 1).",
    )
    parser.add_argument(
        "--eval-engine",
        choices=["domain", "numba"],
        default="domain",
        help=(
            "Engine per `evaluate_matrix.py`. `numba` accelera la valutazione di modelli MLP contro opponent "
            "fast-compatible (default: domain)."
        ),
    )
    parser.add_argument(
        "--rollout-engine",
        choices=["domain", "fast"],
        default="domain",
        help=(
            "Engine rollout training per A2C. `domain` usa il motore canonico; `fast` abilita il path "
            "ottimizzato 2-player (default: domain)."
        ),
    )
    parser.add_argument(
        "--fast-encoder",
        choices=["python", "numba"],
        default="python",
        help=(
            "Encoder osservazione usato da A2C con `--rollout-engine fast`. "
            "`numba` valida il wrapper JIT dell'encoder (default: python)."
        ),
    )
    parser.add_argument(
        "--fast-rollout",
        choices=["python", "numba"],
        default="python",
        help=(
            "Collector rollout usato da A2C con `--rollout-engine fast`. "
            "`numba` usa il batch collector full-JIT parallelo (default: python)."
        ),
    )
    parser.add_argument(
        "--name",
        default="",
        help="Nome esperimento (se vuoto, viene costruito in modo deterministico dai parametri).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffisso opzionale per distinguere run con stessa config (entra nel nome).",
    )
    parser.add_argument(
        "--models-dir",
        default="./data/models",
        help="Directory output modelli (default: ./data/models).",
    )
    parser.add_argument(
        "--experiments-dir",
        default="./benchmarks/experiments",
        help="Directory output esperimenti (default: ./benchmarks/experiments).",
    )
    parser.add_argument(
        "--train-extra",
        nargs=argparse.REMAINDER,
        help=(
            "DEPRECATO: usa argomenti posizionali dopo `--` (vedi `trainer_args`). "
            "Esempio: --train-extra --lr 3e-4 --entropy-beta 1e-3"
        ),
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help=(
            "Argomenti extra pass-through al trainer. "
            "Usa `--` per separarli dalle opzioni della pipeline. "
            "Esempio: python scripts/run_experiment.py --algo a2c -- --lr 3e-4 --entropy-beta 1e-3"
        ),
    )
    parser.add_argument(
        "--update-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Aggiorna `best_<algo>.npz` se lo score migliora (default: true). Usa `--no-update-best` per disabilitare."
        ),
    )
    parser.add_argument(
        "--minimal-data",
        action="store_true",
        help=(
            "Mantiene `data/models/` minimale: conserva solo `best_<algo>.npz` + `best_<algo>.json` "
            "e rimuove gli altri file. Per riproducibilità, copia comunque il modello finale dentro "
            "`benchmarks/experiments/<name>/model.npz`."
        ),
    )
    args = parser.parse_args()

    algo: AlgoName = args.algo
    if int(args.num_games) <= 0:
        raise ValueError("--num-games deve essere > 0")
    if int(args.eval_workers) <= 0:
        raise ValueError("--eval-workers deve essere > 0")
    if algo != "a2c" and (
        args.rollout_engine != "domain" or args.fast_encoder != "python" or args.fast_rollout != "python"
    ):
        raise ValueError("I flag rollout veloci/Numba sono supportati solo con --algo a2c.")
    if args.rollout_engine != "fast" and (args.fast_encoder != "python" or args.fast_rollout != "python"):
        raise ValueError("--fast-encoder e --fast-rollout richiedono --rollout-engine fast.")

    curriculum_raw = str(args.curriculum).strip()
    curriculum: CurriculumPreset | None = None
    if curriculum_raw:
        if curriculum_raw != "easy_standard_hard":
            raise ValueError(f"--curriculum non supportato: {curriculum_raw!r}")
        curriculum = "easy_standard_hard"

    opponent_mix = args.opponent_mix.strip()
    opponent = args.opponent.strip()
    if curriculum is not None:
        opponent_for_name = None
        mix_for_name = f"curriculum:{curriculum}"
    else:
        if opponent_mix:
            opponent_for_name = None
            mix_for_name = opponent_mix
        else:
            opponent_for_name = opponent
            mix_for_name = None

    name = args.name.strip()
    if not name:
        name = build_experiment_name(
            algo=algo,
            num_games=int(args.num_games),
            seed=int(args.train_seed),
            opponent=opponent_for_name,
            opponent_mix=mix_for_name,
            tag=args.tag.strip() or None,
        )

    models_dir = Path(args.models_dir)
    experiments_dir = Path(args.experiments_dir) / name
    model_path = models_dir / f"{name}.npz"

    train_log = experiments_dir / "train.log"
    manifest_path = experiments_dir / "manifest.json"

    # --- Training ---
    trainer_script = "scripts/train_a2c.py" if algo == "a2c" else "scripts/train_pg.py"
    env_overrides = {"BRISCOLA_MODELS_DIR": str(models_dir)}

    train_extra = list(args.train_extra or [])
    positional_extra = list(args.trainer_args or [])
    if train_extra and positional_extra:
        raise ValueError("Usa o `--train-extra` oppure gli argomenti dopo `--`, non entrambi.")

    if positional_extra:
        train_extra = positional_extra

    # Argparse include spesso `--` dentro `REMAINDER` quando usiamo il separatore.
    # Per pass-through al trainer, lo togliamo.
    if train_extra and train_extra[0] == "--":
        train_extra = train_extra[1:]
    forbidden = {"--out", "--data"}
    if any(tok in forbidden for tok in train_extra):
        raise ValueError(f"`--train-extra` non può includere {sorted(forbidden)} (gestiti dalla pipeline).")
    stage_records: list[dict] = []
    primary_train_cmd: list[str] | None = None

    def _build_train_cmd(
        *, out: Path, init: str, opponent_mix_arg: str | None, opponent_arg: str | None, num: int
    ) -> list[str]:
        cmd = [sys.executable, trainer_script, "--out", str(out)]
        if init:
            cmd += ["--init", init]
        if opponent_mix_arg:
            cmd += ["--opponent-mix", opponent_mix_arg]
        elif opponent_arg:
            cmd += ["--opponent", opponent_arg]
        else:
            raise ValueError("Serve --opponent o --opponent-mix")
        cmd += ["--num-games", str(int(num)), "--seed", str(int(args.train_seed))]
        if bool(args.seat_fair):
            cmd.append("--seat-fair")
        if algo == "a2c":
            cmd += [
                "--rollout-engine",
                str(args.rollout_engine),
                "--fast-encoder",
                str(args.fast_encoder),
                "--fast-rollout",
                str(args.fast_rollout),
            ]
        cmd += list(train_extra)
        return cmd

    if curriculum is None:
        # Training single-stage.
        train_cmd = _build_train_cmd(
            out=model_path,
            init=args.init.strip(),
            opponent_mix_arg=opponent_mix if opponent_mix else None,
            opponent_arg=None if opponent_mix else opponent,
            num=int(args.num_games),
        )
        primary_train_cmd = train_cmd
        _run(train_cmd, log_path=train_log, env_overrides=env_overrides)
        if not model_path.exists():
            raise RuntimeError(f"Training completato ma il modello non esiste: {model_path}")
        stage_records.append(
            {
                "name": "single",
                "num_games": int(args.num_games),
                "opponent": opponent if not opponent_mix else None,
                "opponent_mix": opponent_mix if opponent_mix else None,
                "out_path": str(model_path),
                "log_path": str(train_log),
                "cmd": train_cmd,
            }
        )
    else:
        # Curriculum multi-stage: salviamo gli stage dentro l'esperimento e poi copiamo nel path finale.
        stages = build_curriculum_stages(preset=curriculum, total_games=int(args.num_games))
        stages_dir = experiments_dir / "stages"
        stages_dir.mkdir(parents=True, exist_ok=True)

        init_path = args.init.strip()
        stage_out_paths: list[Path] = []
        for i, stage in enumerate(stages, start=1):
            out_path = stages_dir / f"stage_{i:02d}_{stage.name}.npz"
            log_path = experiments_dir / f"train_stage_{i:02d}_{stage.name}.log"
            cmd = _build_train_cmd(
                out=out_path,
                init=init_path,
                opponent_mix_arg=stage.opponent_mix,
                opponent_arg=None,
                num=int(stage.num_games),
            )
            primary_train_cmd = cmd
            stage_records.append(
                {
                    "name": stage.name,
                    "num_games": int(stage.num_games),
                    "opponent_mix": stage.opponent_mix,
                    "out_path": str(out_path),
                    "log_path": str(log_path),
                    "cmd": cmd,
                }
            )
            _run(cmd, log_path=log_path, env_overrides=env_overrides)
            if not out_path.exists():
                raise RuntimeError(f"Stage completato ma il modello non esiste: {out_path}")
            stage_out_paths.append(out_path)
            init_path = str(out_path)

        # Copiamo il risultato finale nello stesso output “storico” in data/models/.
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_out_paths[-1], model_path)

        # In modalità minimal, rimuoviamo gli stage intermedi (manteniamo solo il finale).
        if bool(args.minimal_data):
            for p in stage_out_paths:
                with contextlib.suppress(FileNotFoundError):
                    p.unlink()

    # Se vogliamo mantenere `data/models/` minimal, copiamo subito una copia del modello
    # dentro la cartella dell'esperimento (così il manifest resta “auto-consistente” anche
    # se poi eliminiamo il file da `data/models/`).
    experiment_model_path: Path | None = None
    if bool(args.minimal_data):
        experiment_model_path = experiments_dir / "model.npz"
        shutil.copy2(model_path, experiment_model_path)

    # --- Evaluation matrix ---
    benchmarks = [b.strip() for b in str(args.benchmarks).split(",") if b.strip()]
    allowed = {"small", "medium", "big"}
    if any(b not in allowed for b in benchmarks):
        raise ValueError(f"--benchmarks deve essere subset di {sorted(allowed)} (ottenuto: {benchmarks})")

    matrix_paths: dict[str, Path] = {}
    requested_eval_workers = int(args.eval_workers)
    effective_eval_workers = 1 if str(args.eval_engine) == "numba" else requested_eval_workers
    eval_parallelism = "numba_threads" if str(args.eval_engine) == "numba" else "process_workers"
    for b in benchmarks:
        out_json = experiments_dir / f"matrix_{b}.json"
        log_path = experiments_dir / f"matrix_{b}.log"
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_matrix.py",
            "--model",
            str(model_path),
            "--engine",
            str(args.eval_engine),
            "--benchmark",
            b,
            "--seed",
            str(int(args.eval_seed)),
            "--out-json",
            str(out_json),
            "--format",
            "csv",
            "--workers",
            str(effective_eval_workers),
        ]
        _run(eval_cmd, log_path=log_path, env_overrides=env_overrides)
        matrix_paths[b] = out_json

    if primary_train_cmd is None:
        raise RuntimeError("Errore interno: primary_train_cmd non impostato")

    # --- Manifest + score ---
    manifest: dict = {
        "name": name,
        "created_utc": utc_now_iso(),
        "code_version": get_code_version(),
        "rules_version": get_rules_version(),
        "algo": algo,
        "model_path": str(experiment_model_path or model_path),
        "curriculum": str(curriculum) if curriculum is not None else None,
        "train_stages": stage_records,
        "train": {
            "cmd": primary_train_cmd,
            "rollout": {
                "engine": str(args.rollout_engine) if algo == "a2c" else None,
                "fast_encoder": str(args.fast_encoder) if algo == "a2c" else None,
                "fast_rollout": str(args.fast_rollout) if algo == "a2c" else None,
            },
        },
        "eval": [
            {
                "benchmark": b,
                "matrix_json": str(p),
                "engine": str(args.eval_engine),
                "workers": effective_eval_workers,
                "requested_workers": requested_eval_workers,
                "parallelism": eval_parallelism,
            }
            for b, p in matrix_paths.items()
        ],
    }

    score: float | None = None
    best_metric: dict | None = None

    # Preferiamo big holdout vs heuristic_v1 se disponibile.
    preferred = "big" if "big" in matrix_paths else ("medium" if "medium" in matrix_paths else benchmarks[0])
    matrix_json = read_json(matrix_paths[preferred])
    metric = extract_best_metric_from_matrix_json(matrix_json, benchmark=preferred)
    score = float(metric.avg_diff)
    best_metric = {
        "opponent": metric.opponent,
        "suite": metric.suite,
        "benchmark": metric.benchmark,
        "avg_diff": metric.avg_diff,
    }
    manifest["best_metric"] = best_metric

    write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")

    best_updated = False
    if bool(args.update_best) and score is not None:
        best_path = models_dir / f"best_{algo}.npz"
        best_meta = models_dir / f"best_{algo}.json"
        best_updated = _copy_best_model(
            model_path=model_path, best_path=best_path, score=score, meta_path=best_meta, manifest=manifest
        )

    if bool(args.minimal_data):
        # In modalità “minimal”, vogliamo mantenere `data/models/` il più pulito possibile.
        #
        # Caso tipico:
        # - abbiamo una copia stabile in `benchmarks/experiments/<name>/model.npz`
        # - abbiamo (già) un `best_<algo>.npz` locale, o lo aggiorniamo in questo run
        #
        # Nota:
        # supportiamo `--minimal-data` anche con `--no-update-best` per fare “screening” veloce:
        # - se esiste già `best_<algo>.npz`, possiamo rimuovere il modello run-specific e mantenere solo i best
        # - se NON esiste ancora un best e non vogliamo aggiornarlo, lasciamo il modello run-specific per comodità
        best_path = models_dir / f"best_{algo}.npz"
        best_present = best_path.exists()

        if bool(args.update_best) or best_present:
            with contextlib.suppress(FileNotFoundError):
                model_path.unlink()
            _prune_models_dir(models_dir=models_dir, algo=algo)
        else:
            # Primo run “minimal” senza best: preserviamo solo il modello run-specific.
            for path in models_dir.iterdir():
                if not path.is_file():
                    continue
                if path == model_path:
                    continue
                if path.suffix.lower() not in {".npz", ".json"}:
                    continue
                path.unlink()

        # Piccola nota nel manifest per ricordare che il file in `data/models/` è stato rimosso.
        manifest["minimal_data"] = {
            "enabled": True,
            "model_copied_to": str(experiment_model_path) if experiment_model_path else None,
            "best_updated": bool(best_updated),
            "update_best_enabled": bool(args.update_best),
        }
        write_json(manifest_path, manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
