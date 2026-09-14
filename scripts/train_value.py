#!/usr/bin/env python3
"""
Allena una rete di valore scalare da dataset `value_observation`.

Il target consigliato per lo Stage 0 V-lookahead e' il residuo normalizzato:

    (final_score_delta - current_score_delta) / 120

Questo rende esplicita la baseline "delta corrente" e chiede alla rete di predire solo il
valore futuro residuo della posizione.

Lo split train/validation/test e' assegnato per partita intera (default 80/10/10). I dataset
devono quindi conservare un ``game_id`` per record; il formato Numba v2 aggiunge sia
``game_ids`` sia i ``game_seeds`` di provenance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from briscola_ai.ai.encoding.observation_encoder import EncoderVersion, encode_observation_2p_with_version
from briscola_ai.ai.numba.value_dataset import phase_name_from_code
from briscola_ai.ai.training.dataset_split import make_grouped_dataset_split

TargetMode = Literal["residual", "absolute"]
LossName = Literal["huber", "mse"]


@dataclass(frozen=True, slots=True)
class ValueBatch:
    """Mini-batch per regressione di valore."""

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class ValueDataset:
    """Dataset value caricato in memoria, con provenance per partita."""

    x: np.ndarray
    y: np.ndarray
    current_delta: np.ndarray
    final_delta: np.ndarray
    phases: np.ndarray
    game_ids: np.ndarray


@dataclass
class ValueMetrics:
    """Metriche per epoca."""

    epoch: int
    train_loss: float
    val_loss: float | None
    val_mae_points: float | None
    val_baseline_mae_points: float | None
    val_sign_acc: float | None
    val_baseline_sign_acc: float | None


def _iter_jsonl(path: Path):
    """Itera record JSONL non vuoti."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _target_scaled(record: dict, *, target: TargetMode) -> float:
    """Estrae il target scalato dalla riga value_observation."""
    if target == "residual":
        raw = record.get("target_residual_scaled")
        if raw is not None:
            return float(raw)
        return (float(record["final_score_delta"]) - float(record["current_score_delta"])) / 120.0
    raw = record.get("target_final_scaled")
    if raw is not None:
        return float(raw)
    return float(record["final_score_delta"]) / 120.0


def _load_value_dataset_jsonl(path: Path, *, encoder_version: EncoderVersion, target: TargetMode) -> ValueDataset:
    """Carica un dataset JSONL `value_observation` in array NumPy."""
    features: list[list[float]] = []
    targets: list[float] = []
    current_delta: list[float] = []
    final_delta: list[float] = []
    phases: list[str] = []
    game_ids: list[str | int] = []

    for rec in _iter_jsonl(path):
        if rec.get("dataset_kind") != "value_observation":
            continue
        obs = rec.get("observation")
        if not isinstance(obs, dict) or obs.get("num_players") != 2:
            continue
        game_id = rec.get("game_id")
        if isinstance(game_id, bool) or not isinstance(game_id, (str, int)):
            raise ValueError(
                "Record value valido senza game_id stringa/intero: lo split per partita e' obbligatorio. "
                "Rigenera il dataset con generate_value_dataset.py aggiornato."
            )
        if isinstance(game_id, str) and not game_id.strip():
            raise ValueError("Record value valido con game_id vuoto")
        encoded = encode_observation_2p_with_version(obs, version=encoder_version)
        features.append(encoded.features)
        targets.append(_target_scaled(rec, target=target))
        current_delta.append(float(rec["current_score_delta"]))
        final_delta.append(float(rec["final_score_delta"]))
        phases.append(str(rec.get("phase", "unknown")))
        game_ids.append(game_id)

    if not features:
        raise ValueError("Nessun record value_observation valido trovato")

    return ValueDataset(
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        current_delta=np.asarray(current_delta, dtype=np.float32),
        final_delta=np.asarray(final_delta, dtype=np.float32),
        phases=np.asarray(phases, dtype=object),
        game_ids=np.asarray(game_ids, dtype=object),
    )


def _load_value_dataset_npz(path: Path, *, target: TargetMode) -> ValueDataset:
    """Carica un dataset compatto `.npz` prodotto da `generate_value_dataset_numba.py`."""
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.keys())
        missing = {"x", "current_delta", "final_delta", "game_ids"} - keys
        if missing:
            detail = ""
            if "game_ids" in missing:
                detail = (
                    " Il formato storico non consente uno split sicuro per partita: "
                    "rigenera il dataset con generate_value_dataset_numba.py aggiornato."
                )
            raise ValueError(f"Dataset value .npz invalido: mancano chiavi {sorted(missing)}.{detail}")

        x = np.asarray(data["x"], dtype=np.float32)
        current_delta = np.asarray(data["current_delta"], dtype=np.float32)
        final_delta = np.asarray(data["final_delta"], dtype=np.float32)
        game_ids = np.asarray(data["game_ids"])
        if x.ndim != 2:
            raise ValueError(f"Dataset value .npz invalido: x deve essere 2D, ottenuto {x.shape}")
        if (
            current_delta.shape != (x.shape[0],)
            or final_delta.shape != (x.shape[0],)
            or game_ids.shape != (x.shape[0],)
        ):
            raise ValueError(
                "Dataset value .npz invalido: current_delta/final_delta/game_ids devono avere shape "
                f"{(x.shape[0],)}, ottenuto {current_delta.shape}/{final_delta.shape}/{game_ids.shape}"
            )

        if target == "residual":
            if "residual_delta" in data:
                raw_target = np.asarray(data["residual_delta"], dtype=np.float32)
            else:
                raw_target = final_delta - current_delta
            y = raw_target / 120.0
        else:
            y = final_delta / 120.0

        if "phase" in data:
            raw_phase = np.asarray(data["phase"])
            phases = np.asarray([phase_name_from_code(int(code)) for code in raw_phase.tolist()], dtype=object)
        else:
            phases = np.full((x.shape[0],), "unknown", dtype=object)

    return ValueDataset(
        x=x,
        y=np.asarray(y, dtype=np.float32),
        current_delta=current_delta,
        final_delta=final_delta,
        phases=phases,
        game_ids=game_ids,
    )


def load_value_dataset(path: Path, *, encoder_version: EncoderVersion, target: TargetMode) -> ValueDataset:
    """Carica un dataset value da JSONL canonico oppure da `.npz` compatto Numba."""
    if path.suffix.lower() == ".npz":
        return _load_value_dataset_npz(path, target=target)
    return _load_value_dataset_jsonl(path, encoder_version=encoder_version, target=target)


def _subset_value_dataset(dataset: ValueDataset, indices: np.ndarray) -> ValueDataset:
    """Crea una vista indicizzata mantenendo allineati target, fasi e game_id."""
    return ValueDataset(
        x=dataset.x[indices],
        y=dataset.y[indices],
        current_delta=dataset.current_delta[indices],
        final_delta=dataset.final_delta[indices],
        phases=dataset.phases[indices],
        game_ids=dataset.game_ids[indices],
    )


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _predict(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward MLP scalare; ritorna `(pred, hidden)`."""
    h = _relu(x @ w1 + b1)
    pred = h @ w2 + float(b2)
    return pred.astype(np.float32), h


def _loss_grad(
    x: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
    *,
    loss_name: LossName,
    huber_delta: float,
    weight_decay: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Loss regressione + gradienti MLP."""
    pred, h = _predict(x, w1, b1, w2, b2)
    err = pred - y
    bsz = max(1, x.shape[0])

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

    dpred /= float(bsz)
    grad_w2 = h.T @ dpred
    grad_b2 = float(np.sum(dpred))

    dh = dpred[:, None] @ w2[None, :]
    z1 = x @ w1 + b1
    dz1 = dh * (z1 > 0.0)
    grad_w1 = x.T @ dz1
    grad_b1 = np.sum(dz1, axis=0)

    if weight_decay > 0.0:
        grad_w1 += float(weight_decay) * w1
        grad_w2 += float(weight_decay) * w2

    return loss, grad_w1, grad_b1, grad_w2, grad_b2


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


def _iter_minibatches(x: np.ndarray, y: np.ndarray, *, batch_size: int, rng: np.random.Generator):
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        yield ValueBatch(x=x[batch_idx], y=y[batch_idx])


def _predicted_final_points(
    pred_scaled: np.ndarray,
    *,
    current_delta: np.ndarray,
    target: TargetMode,
) -> np.ndarray:
    """Converte predizione scalata in final_score_delta in punti."""
    if target == "residual":
        return current_delta + pred_scaled * 120.0
    return pred_scaled * 120.0


def _phase_metrics(
    *,
    pred_final: np.ndarray,
    current_delta: np.ndarray,
    final_delta: np.ndarray,
    phases: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Metriche MAE per fase."""
    out: dict[str, dict[str, float | int]] = {}
    for phase in sorted(set(str(p) for p in phases.tolist())):
        mask = phases == phase
        if not bool(np.any(mask)):
            continue
        err = np.abs(pred_final[mask] - final_delta[mask])
        base_err = np.abs(current_delta[mask] - final_delta[mask])
        out[phase] = {
            "n": int(np.sum(mask)),
            "mae_points": float(np.mean(err)),
            "baseline_mae_points": float(np.mean(base_err)),
        }
    return out


def evaluate_predictions(
    *,
    pred_scaled: np.ndarray,
    dataset: ValueDataset,
    target: TargetMode,
) -> dict[str, float | dict[str, dict[str, float | int]]]:
    """Metriche value su scala punti."""
    pred_final = _predicted_final_points(pred_scaled, current_delta=dataset.current_delta, target=target)
    abs_err = np.abs(pred_final - dataset.final_delta)
    baseline_abs_err = np.abs(dataset.current_delta - dataset.final_delta)
    sign_acc = np.mean(np.sign(pred_final) == np.sign(dataset.final_delta))
    baseline_sign_acc = np.mean(np.sign(dataset.current_delta) == np.sign(dataset.final_delta))
    return {
        "mae_points": float(np.mean(abs_err)),
        "baseline_mae_points": float(np.mean(baseline_abs_err)),
        "rmse_points": float(np.sqrt(np.mean((pred_final - dataset.final_delta) ** 2))),
        "sign_acc": float(sign_acc),
        "baseline_sign_acc": float(baseline_sign_acc),
        "phase": _phase_metrics(
            pred_final=pred_final,
            current_delta=dataset.current_delta,
            final_delta=dataset.final_delta,
            phases=dataset.phases,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Allena value MLP da dataset value_observation")
    parser.add_argument("--data", required=True, help="Dataset value-observation: JSONL canonico o .npz compatto")
    parser.add_argument("--out", required=True, help="Path del value model .npz da salvare")
    parser.add_argument(
        "--encoder-version",
        choices=["v1", "v2", "v3", "v4"],
        default="v3",
        help=(
            "Versione encoder osservazioni (default v3). Per dataset .npz pre-encodati "
            "viene INFERITA dalla dimensione delle feature se incoerente col flag."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Neuroni hidden layer della MLP (default 128)")
    parser.add_argument(
        "--target",
        choices=["residual", "absolute"],
        default="residual",
        help="Target: residuo (final-current)/120 (consigliato) oppure delta assoluto scalato",
    )
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber", help="Loss di regressione (default huber)")
    parser.add_argument("--huber-delta", type=float, default=0.25, help="Soglia delta della Huber loss (default 0.25)")
    parser.add_argument("--epochs", type=int, default=30, help="Epoche di training (default 30)")
    parser.add_argument("--batch-size", type=int, default=512, help="Dimensione minibatch (default 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam-like (default 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay L2 (default 0: disattivo)")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Frazione di PARTITE per la validation (default 0.1)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Frazione di PARTITE per il test finale (default 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG per split/shuffle (riproducibilita')")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    encoder_version: EncoderVersion = str(args.encoder_version)  # type: ignore[assignment]
    target: TargetMode = str(args.target)  # type: ignore[assignment]
    loss_name: LossName = str(args.loss)  # type: ignore[assignment]

    dataset = load_value_dataset(data_path, encoder_version=encoder_version, target=target)
    # Dataset .npz pre-encodato: la dimensione delle feature e' la verita'; se il flag
    # non coincide, inferiamo la versione corretta (metadata coerenti nel modello salvato).
    from briscola_ai.ai.encoding.observation_encoder import feature_dim_for_encoder_version

    actual_dim = int(dataset.x.shape[1])
    if actual_dim != int(feature_dim_for_encoder_version(encoder_version)):
        for candidate in ("v1", "v2", "v3", "v4"):
            if actual_dim == int(feature_dim_for_encoder_version(candidate)):  # type: ignore[arg-type]
                print(f"encoder-version inferita dal dataset: {candidate} (feature_dim={actual_dim})")
                encoder_version = candidate  # type: ignore[assignment]
                break
        else:
            raise ValueError(f"feature_dim del dataset non riconosciuta: {actual_dim}")
    rng = np.random.default_rng(int(args.seed))
    _n, d = dataset.x.shape
    split = make_grouped_dataset_split(
        dataset.game_ids,
        validation_fraction=float(args.val_frac),
        test_fraction=float(args.test_frac),
        seed=int(args.seed),
        group_key="game_id",
    )
    train_idx = split.train_indices
    val_idx = split.validation_indices
    test_idx = split.test_indices
    print(
        "dataset_split | unit=game "
        f"groups={split.provenance['group_counts']} records={split.provenance['record_counts']} "
        f"sha256={split.provenance['assignment_sha256']}"
    )

    x_train, y_train = dataset.x[train_idx], dataset.y[train_idx]
    val_dataset = _subset_value_dataset(dataset, val_idx)
    test_dataset = _subset_value_dataset(dataset, test_idx)

    hdim = int(args.hidden_dim)
    if hdim <= 0:
        raise ValueError("--hidden-dim deve essere > 0")
    w1 = rng.normal(0.0, 0.02, size=(d, hdim)).astype(np.float32)
    b1 = np.zeros((hdim,), dtype=np.float32)
    w2 = rng.normal(0.0, 0.02, size=(hdim,)).astype(np.float32)
    b2 = np.asarray([0.0], dtype=np.float32)

    st_w1 = _adam_init(w1)
    st_b1 = _adam_init(b1)
    st_w2 = _adam_init(w2)
    st_b2 = _adam_init(b2)
    t = 0
    metrics: list[ValueMetrics] = []
    # Teniamo i pesi della MIGLIORE val_loss, non quelli dell'ultima epoca: una regressione di
    # valore può iniziare a overfittare (val che risale) e il gate gira sul modello salvato.
    best_val_loss = float("inf")
    best_eval: dict | None = None
    best_epoch = 0
    best_snapshot: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        train_losses: list[float] = []
        for batch in _iter_minibatches(x_train, y_train, batch_size=int(args.batch_size), rng=rng):
            t += 1
            loss, gw1, gb1, gw2, gb2 = _loss_grad(
                batch.x,
                batch.y,
                w1,
                b1,
                w2,
                float(b2[0]),
                loss_name=loss_name,
                huber_delta=float(args.huber_delta),
                weight_decay=float(args.weight_decay),
            )
            _adam_update(w1, gw1, state=st_w1, lr=float(args.lr), t=t)
            _adam_update(b1, gb1, state=st_b1, lr=float(args.lr), t=t)
            _adam_update(w2, gw2, state=st_w2, lr=float(args.lr), t=t)
            _adam_update(b2, np.asarray([gb2], dtype=np.float32), state=st_b2, lr=float(args.lr), t=t)
            train_losses.append(float(loss))

        val_loss: float | None = None
        eval_metrics: dict[str, float | dict[str, dict[str, float | int]]] | None = None
        if val_idx.size:
            val_pred, _ = _predict(val_dataset.x, w1, b1, w2, float(b2[0]))
            val_loss, *_ = _loss_grad(
                val_dataset.x,
                val_dataset.y,
                w1,
                b1,
                w2,
                float(b2[0]),
                loss_name=loss_name,
                huber_delta=float(args.huber_delta),
                weight_decay=0.0,
            )
            eval_metrics = evaluate_predictions(pred_scaled=val_pred, dataset=val_dataset, target=target)
        row = ValueMetrics(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_loss=float(val_loss) if val_loss is not None else None,
            val_mae_points=float(eval_metrics["mae_points"]) if eval_metrics is not None else None,
            val_baseline_mae_points=(float(eval_metrics["baseline_mae_points"]) if eval_metrics is not None else None),
            val_sign_acc=float(eval_metrics["sign_acc"]) if eval_metrics is not None else None,
            val_baseline_sign_acc=(float(eval_metrics["baseline_sign_acc"]) if eval_metrics is not None else None),
        )
        metrics.append(row)
        # Snapshot del miglior checkpoint per val_loss (copie, perché i pesi mutano in-place).
        if val_loss is not None and float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            best_eval = eval_metrics
            best_epoch = epoch
            best_snapshot = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
        if row.val_loss is None:
            print(f"epoch {epoch:02d} | train loss {row.train_loss:.5f} | validation disabilitata")
        else:
            assert row.val_mae_points is not None
            assert row.val_baseline_mae_points is not None
            assert row.val_sign_acc is not None
            assert row.val_baseline_sign_acc is not None
            print(
                f"epoch {epoch:02d} | train loss {row.train_loss:.5f} | val loss {row.val_loss:.5f} | "
                f"MAE {row.val_mae_points:.2f} vs baseline {row.val_baseline_mae_points:.2f} | "
                f"sign {row.val_sign_acc:.3f} vs baseline {row.val_baseline_sign_acc:.3f}"
            )

    # Salviamo i pesi del miglior checkpoint per val_loss (fallback: ultimi, es. val_frac=0).
    has_validation = best_snapshot is not None
    if has_validation:
        assert best_snapshot is not None
        w1, b1, w2, b2 = best_snapshot
    else:
        # Nessuna validation utile (es. --val-frac 0): teniamo l'ultima epoca, senza fingere un "best".
        best_epoch = len(metrics)
        best_eval = None

    test_eval: dict[str, object] | None = None
    if test_idx.size:
        test_pred, _ = _predict(test_dataset.x, w1, b1, w2, float(b2[0]))
        test_loss, *_ = _loss_grad(
            test_dataset.x,
            test_dataset.y,
            w1,
            b1,
            w2,
            float(b2[0]),
            loss_name=loss_name,
            huber_delta=float(args.huber_delta),
            weight_decay=0.0,
        )
        test_eval = {
            "examples": int(test_idx.size),
            "loss": float(test_loss),
            **evaluate_predictions(pred_scaled=test_pred, dataset=test_dataset, target=target),
        }
        print(
            f"test finale | examples {test_eval['examples']} loss {test_eval['loss']:.5f} "
            f"MAE {test_eval['mae_points']:.2f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "value_mlp_v1",
        "feature_dim": int(d),
        "hidden_dim": int(hdim),
        "encoder_version": encoder_version,
        "target": target,
        "target_scale": 120.0,
        "loss": loss_name,
        "huber_delta": float(args.huber_delta),
        "data_path": str(data_path),
        "seed": int(args.seed),
        "dataset_split": split.provenance,
        "train": {
            "optimizer": "adam",
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "weight_decay": float(args.weight_decay),
        },
        "metrics": [asdict(metric) for metric in metrics],
        "best_epoch": int(best_epoch),
        "best_eval": best_eval,
        "test_eval": test_eval,
    }
    np.savez(
        out_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        metadata_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    if has_validation:
        print(f"Saved value model: {out_path} (best epoch {best_epoch}, val_loss {best_val_loss:.5f})")
    else:
        print(f"Saved value model: {out_path} (nessuna validation: tenuta ultima epoca {best_epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
