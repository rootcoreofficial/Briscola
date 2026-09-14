#!/usr/bin/env python3
"""
Allena un value model su foglie PIMC con loss decision-aligned.

Il dataset atteso è prodotto da `generate_pimc_leaf_value_dataset.py`: ogni gruppo contiene
le foglie generate dalle carte candidate della stessa posizione root. La loss combina:

- regressione Huber/MSE sul target residuale scalato;
- ranking pairwise dentro lo stesso gruppo, usando i valori PIMC root-level come ordinamento.

Il modello salvato usa lo stesso formato `value_mlp_v1` di `train_value.py`, quindi può essere
usato direttamente da `ValueLookaheadAgent`. Lo split usa la partita sorgente, non la sola
root: tutte le posizioni correlate della stessa partita restano nello stesso insieme.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from briscola_ai.ai.models import load_value_model_npz
from briscola_ai.ai.training.dataset_split import make_grouped_dataset_split

LossName = str


@dataclass(frozen=True, slots=True)
class LeafValueDataset:
    """Dataset leaf-value caricato da `.npz`."""

    x: np.ndarray
    y: np.ndarray
    current_delta: np.ndarray
    final_delta: np.ndarray
    root_sign: np.ndarray
    root_id: np.ndarray
    action_value: np.ndarray
    game_ids: np.ndarray


@dataclass
class PairwiseMetrics:
    """Metriche per epoca."""

    epoch: int
    train_loss: float
    pair_loss: float
    val_mae_points: float
    val_pairwise_accuracy: float
    val_top1_accuracy: float


@dataclass
class AdamState:
    """Stato Adam per un tensore."""

    m: np.ndarray
    v: np.ndarray


def _adam_init(param: np.ndarray) -> AdamState:
    return AdamState(m=np.zeros_like(param), v=np.zeros_like(param))


def _adam_update(
    param: np.ndarray,
    grad: np.ndarray,
    *,
    state: AdamState,
    lr: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Step Adam in-place."""
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * (grad * grad)
    m_hat = state.m / (1.0 - beta1**t)
    v_hat = state.v / (1.0 - beta2**t)
    param -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)


def load_leaf_value_dataset(path: Path) -> LeafValueDataset:
    """Carica dataset prodotto da `generate_pimc_leaf_value_dataset.py`."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "x",
            "y",
            "current_delta",
            "final_delta",
            "root_sign",
            "root_id",
            "action_value",
            "game_ids",
        }
        missing = required - set(data.keys())
        if missing:
            detail = ""
            if "game_ids" in missing:
                detail = " Rigenera il dataset: lo split per partita richiede game_ids."
            raise ValueError(f"Dataset leaf-value invalido: mancano {sorted(missing)}.{detail}")
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        current = np.asarray(data["current_delta"], dtype=np.float32)
        final = np.asarray(data["final_delta"], dtype=np.float32)
        root_sign = np.asarray(data["root_sign"], dtype=np.float32)
        root_id = np.asarray(data["root_id"], dtype=np.int64)
        action_value = np.asarray(data["action_value"], dtype=np.float32)
        game_ids = np.asarray(data["game_ids"])
    if x.ndim != 2:
        raise ValueError(f"x deve essere 2D, ottenuto {x.shape}")
    n = x.shape[0]
    for name, arr in {
        "y": y,
        "current_delta": current,
        "final_delta": final,
        "root_sign": root_sign,
        "root_id": root_id,
        "action_value": action_value,
        "game_ids": game_ids,
    }.items():
        if arr.shape != (n,):
            raise ValueError(f"{name} shape={arr.shape}, atteso {(n,)}")
    return LeafValueDataset(
        x=x,
        y=y,
        current_delta=current,
        final_delta=final,
        root_sign=root_sign,
        root_id=root_id,
        action_value=action_value,
        game_ids=game_ids,
    )


def _predict(
    x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward MLP scalare; ritorna `(pred_scaled, z1, hidden)`."""
    z1 = x @ w1 + b1
    h = np.maximum(z1, 0.0)
    pred = h @ w2 + float(b2)
    return pred.astype(np.float32), z1, h


def _backward_from_dpred(
    x: np.ndarray,
    z1: np.ndarray,
    h: np.ndarray,
    w2: np.ndarray,
    dpred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Backprop MLP dato `dL/dpred_scaled` già normalizzato."""
    grad_w2 = h.T @ dpred
    grad_b2 = float(np.sum(dpred))
    dh = dpred[:, None] @ w2[None, :]
    dz1 = dh * (z1 > 0.0)
    grad_w1 = x.T @ dz1
    grad_b1 = np.sum(dz1, axis=0)
    return grad_w1, grad_b1, grad_w2, grad_b2


def _regression_loss_grad(
    x: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
    *,
    loss_name: str,
    huber_delta: float,
    weight_decay: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Loss regressione e gradienti."""
    pred, z1, h = _predict(x, w1, b1, w2, b2)
    err = pred - y
    n = max(1, x.shape[0])
    if loss_name == "huber":
        abs_err = np.abs(err)
        delta = float(huber_delta)
        quadratic = abs_err <= delta
        per_example = np.where(quadratic, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
        dpred = np.where(quadratic, err, delta * np.sign(err)).astype(np.float32)
    else:
        per_example = 0.5 * err * err
        dpred = err.astype(np.float32)
    loss = float(np.mean(per_example))
    if weight_decay > 0.0:
        loss += 0.5 * float(weight_decay) * (float(np.sum(w1 * w1)) + float(np.sum(w2 * w2)))
    dpred /= float(n)
    gw1, gb1, gw2, gb2 = _backward_from_dpred(x, z1, h, w2, dpred)
    if weight_decay > 0.0:
        gw1 += float(weight_decay) * w1
        gw2 += float(weight_decay) * w2
    return loss, gw1, gb1, gw2, gb2


def _build_pairs(
    root_id: np.ndarray,
    action_value: np.ndarray,
    *,
    min_margin: float,
) -> np.ndarray:
    """Costruisce coppie `(better_idx, worse_idx)` dentro lo stesso root group."""
    pairs: list[tuple[int, int]] = []
    for group in np.unique(root_id):
        idx = np.where(root_id == group)[0]
        if idx.shape[0] < 2:
            continue
        for i in idx:
            for j in idx:
                diff = float(action_value[i] - action_value[j])
                if diff >= float(min_margin):
                    pairs.append((int(i), int(j)))
    return np.asarray(pairs, dtype=np.int64)


def _pairwise_loss_grad(
    data: LeafValueDataset,
    pairs: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
    *,
    temperature: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Logistic pairwise loss su coppie better>worse."""
    if pairs.shape[0] == 0:
        return 0.0, np.zeros_like(w1), np.zeros_like(b1), np.zeros_like(w2), 0.0
    better = pairs[:, 0]
    worse = pairs[:, 1]
    x_pair = np.concatenate([data.x[better], data.x[worse]], axis=0)
    pred, z1, h = _predict(x_pair, w1, b1, w2, b2)
    n = better.shape[0]
    pred_better = pred[:n]
    pred_worse = pred[n:]
    root_better = data.root_sign[better] * (data.current_delta[better] / 120.0 + pred_better)
    root_worse = data.root_sign[worse] * (data.current_delta[worse] / 120.0 + pred_worse)
    temp = max(1e-6, float(temperature))
    margin = (root_better - root_worse) / temp
    # log(1 + exp(-margin)) stabile.
    losses = np.logaddexp(0.0, -margin)
    sigmoid_neg = 1.0 / (1.0 + np.exp(margin))
    d_margin = -(sigmoid_neg / temp) / float(n)

    dpred = np.zeros((2 * n,), dtype=np.float32)
    dpred[:n] = d_margin * data.root_sign[better]
    dpred[n:] = -d_margin * data.root_sign[worse]
    gw1, gb1, gw2, gb2 = _backward_from_dpred(x_pair, z1, h, w2, dpred)
    return float(np.mean(losses)), gw1, gb1, gw2, gb2


def _root_predictions_scaled(data: LeafValueDataset, pred_scaled: np.ndarray) -> np.ndarray:
    """Predizione finale scalata dal punto di vista del root group."""
    return data.root_sign * (data.current_delta / 120.0 + pred_scaled)


def evaluate_pairwise(
    data: LeafValueDataset, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: float, *, min_margin: float
) -> dict[str, float]:
    """Metriche decision-aligned sul dataset."""
    pred, _z1, _h = _predict(data.x, w1, b1, w2, b2)
    pred_final = data.current_delta + pred * 120.0
    mae = float(np.mean(np.abs(pred_final - data.final_delta)))
    root_pred = _root_predictions_scaled(data, pred)
    pairs = _build_pairs(data.root_id, data.action_value, min_margin=min_margin)
    # SIM108 soppresso: il ternario su questa espressione numpy sarebbe meno leggibile.
    if pairs.shape[0] > 0:  # noqa: SIM108
        pair_acc = float(np.mean(root_pred[pairs[:, 0]] > root_pred[pairs[:, 1]]))
    else:
        pair_acc = 0.0

    top_total = 0
    top_match = 0
    for group in np.unique(data.root_id):
        idx = np.where(data.root_id == group)[0]
        if idx.shape[0] < 2:
            continue
        top_total += 1
        pred_best = int(idx[np.argmax(root_pred[idx])])
        teacher_best = int(idx[np.argmax(data.action_value[idx])])
        if pred_best == teacher_best:
            top_match += 1
    return {
        "mae_points": mae,
        "pairwise_accuracy": pair_acc,
        "top1_accuracy": float(top_match / top_total) if top_total else 0.0,
        "pair_count": float(pairs.shape[0]),
        "root_count": float(top_total),
    }


def _subset(data: LeafValueDataset, idx: np.ndarray) -> LeafValueDataset:
    return LeafValueDataset(
        x=data.x[idx],
        y=data.y[idx],
        current_delta=data.current_delta[idx],
        final_delta=data.final_delta[idx],
        root_sign=data.root_sign[idx],
        root_id=data.root_id[idx],
        action_value=data.action_value[idx],
        game_ids=data.game_ids[idx],
    )


def train_pairwise(args: argparse.Namespace) -> dict:
    data = load_leaf_value_dataset(Path(args.data))
    rng = np.random.default_rng(int(args.seed))
    split = make_grouped_dataset_split(
        data.game_ids,
        validation_fraction=float(args.val_frac),
        test_fraction=float(args.test_frac),
        seed=int(args.seed),
        group_key="game_ids",
    )
    train_idx = split.train_indices
    val_idx = split.validation_indices
    test_idx = split.test_indices
    if not val_idx.size:
        raise ValueError("train_value_pairwise richiede --val-frac > 0 per scegliere il checkpoint")
    print(
        "dataset_split | unit=game "
        f"groups={split.provenance['group_counts']} records={split.provenance['record_counts']} "
        f"sha256={split.provenance['assignment_sha256']}"
    )
    train = _subset(data, train_idx)
    val = _subset(data, val_idx)
    test = _subset(data, test_idx)
    train_pairs = _build_pairs(train.root_id, train.action_value, min_margin=float(args.pair_min_margin))

    n, d = train.x.shape
    hdim = int(args.hidden_dim)
    if str(args.init_value_model).strip():
        init_model = load_value_model_npz(Path(str(args.init_value_model).strip()))
        if init_model.feature_dim != d:
            raise ValueError(
                f"--init-value-model feature_dim={init_model.feature_dim}, ma il dataset ha feature_dim={d}"
            )
        if init_model.hidden_dim != hdim:
            raise ValueError(f"--init-value-model hidden_dim={init_model.hidden_dim}, ma --hidden-dim={hdim}")
        w1 = np.asarray(init_model.w1, dtype=np.float32).copy()
        b1 = np.asarray(init_model.b1, dtype=np.float32).copy()
        w2 = np.asarray(init_model.w2, dtype=np.float32).reshape(-1).copy()
        b2 = np.asarray([float(init_model.b2)], dtype=np.float32)
    else:
        w1 = rng.normal(0.0, 0.02, size=(d, hdim)).astype(np.float32)
        b1 = np.zeros(hdim, dtype=np.float32)
        w2 = rng.normal(0.0, 0.02, size=hdim).astype(np.float32)
        b2 = np.asarray([0.0], dtype=np.float32)
    st_w1, st_b1, st_w2, st_b2 = _adam_init(w1), _adam_init(b1), _adam_init(w2), _adam_init(b2)

    best_score = -1.0
    best_epoch = 0
    best_eval: dict | None = None
    best_snapshot = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
    metrics: list[PairwiseMetrics] = []
    t = 0

    for epoch in range(1, int(args.epochs) + 1):
        idx = np.arange(n)
        rng.shuffle(idx)
        reg_losses: list[float] = []
        pair_losses: list[float] = []
        for start in range(0, n, int(args.batch_size)):
            batch_idx = idx[start : start + int(args.batch_size)]
            t += 1
            loss, gw1, gb1, gw2, gb2 = _regression_loss_grad(
                train.x[batch_idx],
                train.y[batch_idx],
                w1,
                b1,
                w2,
                float(b2[0]),
                loss_name=str(args.loss),
                huber_delta=float(args.huber_delta),
                weight_decay=float(args.weight_decay),
            )
            if train_pairs.shape[0] > 0 and float(args.pairwise_beta) > 0.0:
                pair_take = min(int(args.pair_batch_size), int(train_pairs.shape[0]))
                pair_idx = rng.choice(train_pairs.shape[0], size=pair_take, replace=train_pairs.shape[0] < pair_take)
                pair_loss, pgw1, pgb1, pgw2, pgb2 = _pairwise_loss_grad(
                    train,
                    train_pairs[pair_idx],
                    w1,
                    b1,
                    w2,
                    float(b2[0]),
                    temperature=float(args.pair_temperature),
                )
                beta = float(args.pairwise_beta)
                gw1 += beta * pgw1
                gb1 += beta * pgb1
                gw2 += beta * pgw2
                gb2 += beta * pgb2
                pair_losses.append(float(pair_loss))
            _adam_update(w1, gw1, state=st_w1, lr=float(args.lr), t=t)
            _adam_update(b1, gb1, state=st_b1, lr=float(args.lr), t=t)
            _adam_update(w2, gw2, state=st_w2, lr=float(args.lr), t=t)
            _adam_update(b2, np.asarray([gb2], dtype=np.float32), state=st_b2, lr=float(args.lr), t=t)
            reg_losses.append(float(loss))

        val_eval = evaluate_pairwise(val, w1, b1, w2, float(b2[0]), min_margin=float(args.pair_min_margin))
        row = PairwiseMetrics(
            epoch=epoch,
            train_loss=float(np.mean(reg_losses)),
            pair_loss=float(np.mean(pair_losses)) if pair_losses else 0.0,
            val_mae_points=float(val_eval["mae_points"]),
            val_pairwise_accuracy=float(val_eval["pairwise_accuracy"]),
            val_top1_accuracy=float(val_eval["top1_accuracy"]),
        )
        metrics.append(row)
        score = float(val_eval["pairwise_accuracy"]) + 0.25 * float(val_eval["top1_accuracy"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_eval = val_eval
            best_snapshot = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
        print(
            f"epoch {epoch:02d} | train loss {row.train_loss:.5f} | pair loss {row.pair_loss:.5f} | "
            f"val MAE {row.val_mae_points:.2f} | pair {row.val_pairwise_accuracy:.3f} | "
            f"top1 {row.val_top1_accuracy:.3f}"
        )

    w1, b1, w2, b2 = best_snapshot
    test_eval = None
    if test_idx.size:
        test_eval = evaluate_pairwise(test, w1, b1, w2, float(b2[0]), min_margin=float(args.pair_min_margin))
        print(
            f"test finale | MAE {test_eval['mae_points']:.2f} | "
            f"pair {test_eval['pairwise_accuracy']:.3f} | top1 {test_eval['top1_accuracy']:.3f}"
        )
    payload = {
        "format": "value_mlp_v1",
        "feature_dim": int(d),
        "hidden_dim": int(hdim),
        "encoder_version": "v3",
        "target": "residual",
        "target_scale": 120.0,
        "loss": str(args.loss),
        "huber_delta": float(args.huber_delta),
        "data_path": str(args.data),
        "init_value_model": str(args.init_value_model).strip() or None,
        "seed": int(args.seed),
        "dataset_split": split.provenance,
        "train": {
            "optimizer": "adam",
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "pairwise_beta": float(args.pairwise_beta),
            "pair_temperature": float(args.pair_temperature),
            "pair_min_margin": float(args.pair_min_margin),
            "pair_batch_size": int(args.pair_batch_size),
            "weight_decay": float(args.weight_decay),
        },
        "metrics": [asdict(metric) for metric in metrics],
        "best_epoch": int(best_epoch),
        "best_eval": best_eval,
        "test_eval": test_eval,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved value model: {out_path} (best epoch {best_epoch}, pair/top score {best_score:.4f})")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Allena value MLP con loss pairwise su leaf PIMC.")
    parser.add_argument(
        "--data", required=True, help="Dataset NPZ foglie decision-aligned con root_id/action_value/game_ids"
    )
    parser.add_argument("--out", required=True, help="Path del value model .npz da salvare")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Neuroni hidden layer della MLP (default 128)")
    parser.add_argument(
        "--init-value-model",
        default="",
        help="Value model `.npz` da cui inizializzare i pesi; richiede stesso feature_dim/hidden_dim.",
    )
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber", help="Loss di regressione (default huber)")
    parser.add_argument("--huber-delta", type=float, default=0.25, help="Soglia delta della Huber loss (default 0.25)")
    parser.add_argument("--epochs", type=int, default=30, help="Epoche di training (default 30)")
    parser.add_argument("--batch-size", type=int, default=512, help="Dimensione minibatch regressione (default 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay L2 (default 0: disattivo)")
    parser.add_argument(
        "--val-frac", type=float, default=0.1, help="Frazione di PARTITE per la validation (default 0.1)"
    )
    parser.add_argument("--test-frac", type=float, default=0.1, help="Frazione di PARTITE per il test (default 0.1)")
    parser.add_argument(
        "--pairwise-beta", type=float, default=1.0, help="Peso del termine ranking pairwise nella loss (default 1.0)"
    )
    parser.add_argument(
        "--pair-temperature", type=float, default=0.10, help="Temperatura della logistic pairwise (default 0.10)"
    )
    parser.add_argument(
        "--pair-min-margin",
        type=float,
        default=2.0,
        help="Margine minimo (punti) tra due foglie per formare una coppia di ranking (default 2.0)",
    )
    parser.add_argument("--pair-batch-size", type=int, default=512, help="Minibatch delle coppie ranking (default 512)")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG per split/shuffle (riproducibilita')")
    args = parser.parse_args()
    train_pairwise(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
