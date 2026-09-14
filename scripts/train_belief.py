#!/usr/bin/env python3
"""
Allena la belief network MLP e misura generalizzazione e calibrazione.

Lo split storico per partita resta il default. I dataset belief v2 aggiungono
``opponent_id`` come metadato e abilitano ``--holdout-opponent``: tutte le partite di
uno stile diventano validation e nessuna loro traiettoria entra nel training.
``opponent_id`` non viene mai concatenato a ``x`` e quindi non puo' diventare una
scorciatoia a inference.

Metriche offline
----------------
- BCE mascherata sulle sole carte ignote;
- top-k recall con k pari alla dimensione reale della mano avversaria;
- Brier score mascherato;
- ECE (Expected Calibration Error) sulle probabilita' delle carte ignote;
- confronto opzionale con una belief baseline sullo stesso identico holdout.

Le metriche offline autorizzano soltanto il successivo A/B nel runtime PIMC: la sigmoid
per-carta non impone la cardinalita' esatta della mano e una BCE migliore non garantisce
automaticamente decisioni di search migliori.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.models.belief_model import load_belief_model_npz
from briscola_ai.versioning import get_code_version


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Maschere record e descrizione auditabile dello split."""

    train_mask: np.ndarray
    val_mask: np.ndarray
    strategy: str
    holdout_opponent: str | None
    train_opponents: tuple[str, ...]
    val_opponents: tuple[str, ...]


def _sha256(path: Path) -> str:
    """SHA-256 streaming di un asset baseline."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bce_masked(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """BCE media sulle sole posizioni mascherate (carte ignote)."""
    eps = 1e-7
    p = np.clip(probs, eps, 1.0 - eps)
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)) * mask
    denom = float(mask.sum())
    return float(losses.sum() / denom) if denom > 0 else 0.0


def _brier_masked(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Errore quadratico medio delle probabilita' sulle carte ignote."""
    denom = float(mask.sum())
    if denom <= 0:
        return 0.0
    return float((((probs - y) ** 2) * mask).sum() / denom)


def _ece_masked(probs: np.ndarray, y: np.ndarray, mask: np.ndarray, *, bins: int) -> float:
    """Expected Calibration Error pesato, con bin uniformi in probabilita'."""
    if bins <= 1:
        raise ValueError("ece bins deve essere > 1")
    selected = mask.astype(bool)
    p = np.asarray(probs[selected], dtype=np.float64)
    targets = np.asarray(y[selected], dtype=np.float64)
    if p.size == 0:
        return 0.0
    bin_ids = np.minimum((np.clip(p, 0.0, 1.0) * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for bin_index in range(bins):
        in_bin = bin_ids == bin_index
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(p[in_bin].mean())
        frequency = float(targets[in_bin].mean())
        ece += (count / p.size) * abs(confidence - frequency)
    return float(ece)


def _uniform_probs(mask: np.ndarray, opp_hand_size: np.ndarray) -> np.ndarray:
    """Baseline uniforme: p = carte in mano / carte ignote per ogni record."""
    n_unknown = mask.sum(axis=1, keepdims=True).astype(np.float64)
    p = np.divide(
        opp_hand_size[:, None].astype(np.float64),
        n_unknown,
        out=np.zeros_like(n_unknown),
        where=n_unknown > 0,
    )
    return p * mask


def _topk_recall(probs: np.ndarray, y: np.ndarray, mask: np.ndarray, opp_hand_size: np.ndarray) -> float:
    """Frazione delle carte vere tra le prime k predette, con k=dimensione mano."""
    scores = np.where(mask > 0, probs, -1.0)
    order = np.argsort(-scores, axis=1)
    hits = 0.0
    total = 0.0
    for index in range(probs.shape[0]):
        k = int(opp_hand_size[index])
        if k <= 0:
            continue
        topk = order[index, :k]
        hits += float(y[index, topk].sum())
        total += k
    return hits / total if total > 0 else 0.0


def metric_bundle(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    opp_hand_size: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, float | int]:
    """Calcola tutte le metriche su uno stesso sottoinsieme di record."""
    return {
        "records": int(probs.shape[0]),
        "masked_cards": int(mask.sum()),
        "bce": _bce_masked(probs, y, mask),
        "topk_recall": _topk_recall(probs, y, mask, opp_hand_size),
        "brier": _brier_masked(probs, y, mask),
        "ece": _ece_masked(probs, y, mask, bins=ece_bins),
        "ece_bins": int(ece_bins),
    }


def _forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward MLP: ritorna probabilita' e hidden per il backward."""
    hidden = np.maximum(x @ w1 + b1, 0.0)
    logits = hidden @ w2 + b2
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs, hidden


def _opponent_labels(dataset_meta: dict[str, Any], opponent_id: np.ndarray) -> tuple[str, ...]:
    """Risolve gli indici compatti del dataset in identificatori leggibili."""
    raw = dataset_meta.get("opponent_index")
    if isinstance(raw, list) and raw and all(isinstance(item, str) and item for item in raw):
        labels = tuple(str(item) for item in raw)
    else:
        labels = ("mirror",)
    if opponent_id.size and (int(opponent_id.min()) < 0 or int(opponent_id.max()) >= len(labels)):
        raise ValueError("opponent_id contiene un indice non presente in metadata.opponent_index")
    return labels


def build_split(
    game_index: np.ndarray,
    opponent_id: np.ndarray,
    opponent_labels: tuple[str, ...],
    *,
    holdout_opponent: str | None,
    val_frac: float,
) -> DatasetSplit:
    """Costruisce uno split per partita, opzionalmente leave-one-opponent-out."""
    if game_index.shape != opponent_id.shape:
        raise ValueError("game_index e opponent_id devono avere la stessa shape")
    game_to_opponent: dict[int, int] = {}
    for game, opponent in zip(game_index, opponent_id, strict=True):
        previous = game_to_opponent.setdefault(int(game), int(opponent))
        if previous != int(opponent):
            raise ValueError(
                f"La partita {int(game)} attraversa piu' opponent_id ({previous}, {int(opponent)}): "
                "leave-one-out non sarebbe ermetico"
            )

    if holdout_opponent is not None:
        if holdout_opponent not in opponent_labels:
            raise ValueError(f"Holdout {holdout_opponent!r} non presente: {list(opponent_labels)}")
        held_index = opponent_labels.index(holdout_opponent)
        val_mask = opponent_id == held_index
        strategy = "leave_one_opponent_out"
    else:
        if not 0.0 < float(val_frac) < 1.0:
            raise ValueError("val_frac deve essere in (0,1)")
        modulo = max(2, round(1.0 / float(val_frac)))
        val_mask = (game_index % modulo) == 0
        strategy = "game_modulo"
    train_mask = ~val_mask
    if not bool(train_mask.any()) or not bool(val_mask.any()):
        raise ValueError(
            f"Split vuoto: train={int(train_mask.sum())} val={int(val_mask.sum())} holdout={holdout_opponent!r}"
        )

    train_opponents = tuple(
        label for index, label in enumerate(opponent_labels) if bool(np.any(opponent_id[train_mask] == index))
    )
    val_opponents = tuple(
        label for index, label in enumerate(opponent_labels) if bool(np.any(opponent_id[val_mask] == index))
    )
    return DatasetSplit(
        train_mask=train_mask,
        val_mask=val_mask,
        strategy=strategy,
        holdout_opponent=holdout_opponent,
        train_opponents=train_opponents,
        val_opponents=val_opponents,
    )


def _metrics_by_opponent(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    opp_hand_size: np.ndarray,
    opponent_id: np.ndarray,
    labels: tuple[str, ...],
    *,
    ece_bins: int,
) -> dict[str, dict[str, float | int]]:
    """Metriche separate per stile presente nella validation."""
    result: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        selected = opponent_id == index
        if bool(selected.any()):
            result[label] = metric_bundle(
                probs[selected],
                y[selected],
                mask[selected],
                opp_hand_size[selected],
                ece_bins=ece_bins,
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Allena belief network su dataset .npz")
    parser.add_argument("--data", required=True, help="Dataset .npz da generate_belief_dataset.py")
    parser.add_argument("--out", required=True, help="Path belief model .npz")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Neuroni hidden (default 128)")
    parser.add_argument("--epochs", type=int, default=20, help="Epoche (default 20)")
    parser.add_argument("--batch-size", type=int, default=512, help="Minibatch (default 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam (default 1e-3)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation per-partita senza holdout")
    parser.add_argument("--seed", type=int, default=0, help="Seed init/shuffle")
    parser.add_argument(
        "--holdout-opponent",
        default="",
        help="ID stile escluso interamente dal training (leave-one-opponent-out).",
    )
    parser.add_argument(
        "--baseline-model",
        default="",
        help="Belief .npz di confronto, valutata sullo stesso holdout.",
    )
    parser.add_argument("--ece-bins", type=int, default=10, help="Bin calibrazione ECE (default 10)")
    args = parser.parse_args()

    data_path = Path(args.data)
    with np.load(data_path) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        mask = np.asarray(data["unknown"], dtype=np.float32)
        opp_hand_size = np.asarray(data["opp_hand_size"], dtype=np.int64)
        game_index = np.asarray(data["game_index"], dtype=np.int64)
        opponent_id = (
            np.asarray(data["opponent_id"], dtype=np.int64)
            if "opponent_id" in data
            else np.zeros(game_index.shape, dtype=np.int64)
        )
        dataset_meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}

    if x.shape[0] == 0:
        raise ValueError("Dataset vuoto")
    expected_rows = x.shape[0]
    arrays = {"y": y, "unknown": mask, "opp_hand_size": opp_hand_size, "game_index": game_index}
    for name, array in arrays.items():
        if array.shape[0] != expected_rows:
            raise ValueError(f"Dataset incoerente: x={expected_rows} {name}={array.shape[0]}")
    feature_dim = int(x.shape[1])
    opponent_labels = _opponent_labels(dataset_meta, opponent_id)
    holdout = str(args.holdout_opponent).strip() or None
    split = build_split(
        game_index,
        opponent_id,
        opponent_labels,
        holdout_opponent=holdout,
        val_frac=float(args.val_frac),
    )

    x_tr, y_tr, m_tr = x[split.train_mask], y[split.train_mask], mask[split.train_mask]
    x_va, y_va, m_va = x[split.val_mask], y[split.val_mask], mask[split.val_mask]
    k_va = opp_hand_size[split.val_mask]
    opponent_va = opponent_id[split.val_mask]
    print(
        f"record: train={x_tr.shape[0]} val={x_va.shape[0]} | feature_dim={feature_dim} | "
        f"split={split.strategy} val_opponents={list(split.val_opponents)}"
    )

    rng = np.random.default_rng(args.seed)
    hidden = int(args.hidden_dim)
    if hidden <= 0 or int(args.epochs) <= 0 or int(args.batch_size) <= 0:
        raise ValueError("hidden-dim, epochs e batch-size devono essere > 0")
    w1 = rng.normal(0.0, np.sqrt(2.0 / feature_dim), size=(feature_dim, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = np.zeros((hidden, 40), dtype=np.float32)
    b2 = np.zeros(40, dtype=np.float32)

    params = [w1, b1, w2, b2]
    m_state = [np.zeros_like(param) for param in params]
    v_state = [np.zeros_like(param) for param in params]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    adam_t = 0

    uniform_va = _uniform_probs(m_va, k_va)
    uniform_metrics = metric_bundle(uniform_va, y_va, m_va, k_va, ece_bins=int(args.ece_bins))
    print(
        f"uniforme | bce={uniform_metrics['bce']:.4f} topk={uniform_metrics['topk_recall']:.4f} "
        f"brier={uniform_metrics['brier']:.4f} ece={uniform_metrics['ece']:.4f}"
    )

    baseline_path = Path(args.baseline_model) if str(args.baseline_model).strip() else None
    baseline_metrics: dict[str, Any] | None = None
    baseline_by_opponent: dict[str, Any] | None = None
    if baseline_path is not None:
        baseline = load_belief_model_npz(baseline_path)
        if baseline.feature_dim != feature_dim:
            raise ValueError(f"Baseline feature_dim={baseline.feature_dim}, dataset={feature_dim}")
        baseline_probs = np.asarray(baseline.predict_probs(x_va), dtype=np.float64)
        baseline_metrics = metric_bundle(
            baseline_probs,
            y_va,
            m_va,
            k_va,
            ece_bins=int(args.ece_bins),
        )
        baseline_by_opponent = _metrics_by_opponent(
            baseline_probs,
            y_va,
            m_va,
            k_va,
            opponent_va,
            opponent_labels,
            ece_bins=int(args.ece_bins),
        )
        print(
            f"baseline {baseline_path.name} | bce={baseline_metrics['bce']:.4f} "
            f"topk={baseline_metrics['topk_recall']:.4f} brier={baseline_metrics['brier']:.4f} "
            f"ece={baseline_metrics['ece']:.4f}"
        )

    best_epoch = -1
    best_bce = float("inf")
    best_weights: list[np.ndarray] | None = None
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        order = rng.permutation(x_tr.shape[0])
        for start in range(0, len(order), int(args.batch_size)):
            indexes = order[start : start + int(args.batch_size)]
            xb, yb, mb = x_tr[indexes], y_tr[indexes], m_tr[indexes]
            probs, hidden_values = _forward(xb, w1, b1, w2, b2)

            denom = max(float(mb.sum()), 1.0)
            dlogits = (probs - yb) * mb / denom
            dw2 = hidden_values.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ w2.T
            dhidden[hidden_values <= 0.0] = 0.0
            dw1 = xb.T @ dhidden
            db1 = dhidden.sum(axis=0)

            adam_t += 1
            for param, gradient, m_acc, v_acc in zip(
                params,
                [dw1, db1, dw2, db2],
                m_state,
                v_state,
                strict=True,
            ):
                m_acc[:] = beta1 * m_acc + (1 - beta1) * gradient
                v_acc[:] = beta2 * v_acc + (1 - beta2) * (gradient * gradient)
                m_hat = m_acc / (1 - beta1**adam_t)
                v_hat = v_acc / (1 - beta2**adam_t)
                param -= args.lr * m_hat / (np.sqrt(v_hat) + epsilon)

        probs_va, _ = _forward(x_va, w1, b1, w2, b2)
        val_bce = _bce_masked(probs_va, y_va, m_va)
        val_recall = _topk_recall(probs_va, y_va, m_va, k_va)
        marker = ""
        if val_bce < best_bce:
            best_epoch = epoch
            best_bce = val_bce
            best_weights = [param.copy() for param in params]
            marker = " *best"
        print(f"epoch {epoch:02d} | val_bce={val_bce:.4f} | topk_recall={val_recall:.4f}{marker}")

    if best_weights is None:
        raise RuntimeError("Training senza checkpoint valido")
    w1, b1, w2, b2 = best_weights
    elapsed = time.perf_counter() - started
    candidate_probs, _ = _forward(x_va, w1, b1, w2, b2)
    candidate_metrics = metric_bundle(
        candidate_probs,
        y_va,
        m_va,
        k_va,
        ece_bins=int(args.ece_bins),
    )
    candidate_by_opponent = _metrics_by_opponent(
        candidate_probs,
        y_va,
        m_va,
        k_va,
        opponent_va,
        opponent_labels,
        ece_bins=int(args.ece_bins),
    )

    validation: dict[str, Any] = {
        "candidate": candidate_metrics,
        "candidate_by_opponent": candidate_by_opponent,
        "uniform": uniform_metrics,
    }
    if baseline_path is not None and baseline_metrics is not None and baseline_by_opponent is not None:
        validation["baseline"] = {
            "artifact": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
                "size_bytes": baseline_path.stat().st_size,
            },
            "metrics": baseline_metrics,
            "metrics_by_opponent": baseline_by_opponent,
        }

    metadata = {
        "format": "belief_mlp_v1",
        "encoder_version": str(dataset_meta.get("encoder_version", "v4")),
        "feature_dim": feature_dim,
        "hidden_dim": hidden,
        "target": "opponent_hand",
        "iteration": "belief_v1_multi_style",
        "train": {
            "optimizer": "adam",
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "best_epoch": int(best_epoch),
            "split": {
                "strategy": split.strategy,
                "val_frac": float(args.val_frac) if split.strategy == "game_modulo" else None,
                "holdout_opponent": split.holdout_opponent,
                "train_opponents": list(split.train_opponents),
                "val_opponents": list(split.val_opponents),
                "train_records": int(split.train_mask.sum()),
                "val_records": int(split.val_mask.sum()),
                "unit": "game",
            },
            "validation": validation,
        },
        "dataset": dataset_meta,
        "dataset_artifact": {
            "path": str(data_path),
            "sha256": _sha256(data_path),
            "size_bytes": data_path.stat().st_size,
        },
        "anti_cheat": "opponent_id usato solo per split/metriche; input policy = x da PlayerObservation",
        "code_version": get_code_version(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(metadata, ensure_ascii=False))
    print(
        f"Salvato {out_path} | best_epoch={best_epoch} | bce={candidate_metrics['bce']:.4f} "
        f"topk={candidate_metrics['topk_recall']:.4f} brier={candidate_metrics['brier']:.4f} "
        f"ece={candidate_metrics['ece']:.4f} | {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
