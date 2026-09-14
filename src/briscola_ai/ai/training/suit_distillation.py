"""
Dataset numerico e trainer per distillare il teacher simmetrizzato sui semi.

La distillazione usa target soft prodotti dalla media dei logits di v13 sulle 24
rinomine. Gli split sono assegnati per partita, non per singola decisione: stati
correlati della stessa partita non possono quindi comparire sia nel training sia
nella validazione o nel test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ..encoding.observation_encoder import EncoderVersion, feature_dim_for_encoder_version
from ..evaluation.suit_symmetry import all_suit_permutations
from ..models.bc_model import MLPBCModel
from .suit_augmentation import (
    permute_action_ids,
    permute_action_masks,
    permute_action_vectors,
    permute_encoded_features,
)

DATASET_FORMAT = "suit_distillation_v1"
SPLIT_TRAIN = np.uint8(0)
SPLIT_VALIDATION = np.uint8(1)
SPLIT_TEST = np.uint8(2)
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SuitDistillationDataset:
    """Array necessari alla distillazione, con provenienza per partita."""

    features: np.ndarray
    action_masks: np.ndarray
    target_probs: np.ndarray
    target_action_ids: np.ndarray
    game_ids: np.ndarray
    split_ids: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        """Valida shape, dtype, probabilità, mask e assenza di leakage fra split."""
        arrays = (
            self.features,
            self.action_masks,
            self.target_probs,
            self.target_action_ids,
            self.game_ids,
            self.split_ids,
        )
        if any(array.ndim == 0 for array in arrays):
            raise ValueError("Gli array del dataset devono avere almeno una dimensione")
        num_examples = int(self.features.shape[0])
        if num_examples <= 0 or any(int(array.shape[0]) != num_examples for array in arrays):
            raise ValueError("Numero esempi vuoto o incoerente fra gli array")
        if self.features.ndim != 2:
            raise ValueError(f"features deve essere 2D, ottenuto {self.features.shape}")
        if self.action_masks.shape != (num_examples, 40) or self.action_masks.dtype != np.bool_:
            raise ValueError(f"action_masks invalida: {self.action_masks.shape}/{self.action_masks.dtype}")
        if self.target_probs.shape != (num_examples, 40):
            raise ValueError(f"target_probs invalida: {self.target_probs.shape}")
        if not np.issubdtype(self.target_action_ids.dtype, np.integer):
            raise ValueError("target_action_ids deve avere dtype intero")
        if not np.issubdtype(self.game_ids.dtype, np.integer):
            raise ValueError("game_ids deve avere dtype intero")
        if self.split_ids.dtype != np.uint8:
            raise ValueError("split_ids deve avere dtype uint8")
        if bool(np.any(~np.isfinite(self.features))) or bool(np.any(~np.isfinite(self.target_probs))):
            raise ValueError("Il dataset contiene valori non finiti")
        if bool(np.any(self.target_probs < 0.0)):
            raise ValueError("Il dataset contiene probabilità negative")
        if not bool(np.allclose(np.sum(self.target_probs, axis=1), 1.0, atol=1e-5)):
            raise ValueError("Le probabilità target non sommano a uno")
        if bool(np.any(self.target_probs[~self.action_masks] != 0.0)):
            raise ValueError("Le probabilità target assegnano massa ad azioni illegali")
        if bool(np.any(self.target_action_ids < 0)) or bool(np.any(self.target_action_ids >= 40)):
            raise ValueError("target_action_ids fuori da [0, 39]")
        rows = np.arange(num_examples)
        if not bool(np.all(self.action_masks[rows, self.target_action_ids.astype(np.intp)])):
            raise ValueError("Un target argmax non è legale nella propria action mask")
        if not bool(np.array_equal(np.argmax(self.target_probs, axis=1), self.target_action_ids)):
            raise ValueError("target_action_ids non coincide con argmax(target_probs)")
        if not set(np.unique(self.split_ids).tolist()).issubset({0, 1, 2}):
            raise ValueError("split_ids contiene valori diversi da train/validation/test")

        order = np.argsort(self.game_ids, kind="stable")
        sorted_games = self.game_ids[order]
        sorted_splits = self.split_ids[order]
        boundaries = np.flatnonzero(np.diff(sorted_games)) + 1
        for group in np.split(sorted_splits, boundaries):
            if np.unique(group).size != 1:
                raise ValueError("Leakage: la stessa partita compare in più split")

    @property
    def encoder_version(self) -> EncoderVersion:
        """Versione encoder dichiarata nei metadati."""
        raw = self.metadata.get("encoder_version")
        if raw not in {"v1", "v2", "v3", "v4"}:
            raise ValueError(f"encoder_version dataset non valida: {raw!r}")
        return cast(EncoderVersion, raw)

    def indices(self, split: SplitName) -> np.ndarray:
        """Indici degli esempi appartenenti allo split richiesto."""
        split_id = {"train": SPLIT_TRAIN, "validation": SPLIT_VALIDATION, "test": SPLIT_TEST}[split]
        return np.flatnonzero(self.split_ids == split_id)

    def save(self, path: Path) -> None:
        """Salva il dataset compresso dopo una validazione completa."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=np.asarray(self.features, dtype=np.float32),
            action_masks=np.asarray(self.action_masks, dtype=bool),
            target_probs=np.asarray(self.target_probs, dtype=np.float32),
            target_action_ids=np.asarray(self.target_action_ids, dtype=np.int16),
            game_ids=np.asarray(self.game_ids, dtype=np.int32),
            split_ids=np.asarray(self.split_ids, dtype=np.uint8),
            metadata_json=json.dumps(self.metadata, ensure_ascii=False, sort_keys=True),
        )


def load_suit_distillation_dataset(path: Path) -> SuitDistillationDataset:
    """Carica un dataset `.npz` senza pickle e ne verifica formato e invarianti."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "features",
            "action_masks",
            "target_probs",
            "target_action_ids",
            "game_ids",
            "split_ids",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Dataset incompleto: mancano {sorted(missing)}")
        metadata = json.loads(str(data["metadata_json"].item()))
        if not isinstance(metadata, dict) or metadata.get("format") != DATASET_FORMAT:
            raise ValueError(f"Formato dataset non supportato: {metadata!r}")
        dataset = SuitDistillationDataset(
            features=np.asarray(data["features"], dtype=np.float32),
            action_masks=np.asarray(data["action_masks"], dtype=bool),
            target_probs=np.asarray(data["target_probs"], dtype=np.float32),
            target_action_ids=np.asarray(data["target_action_ids"], dtype=np.int16),
            game_ids=np.asarray(data["game_ids"], dtype=np.int32),
            split_ids=np.asarray(data["split_ids"], dtype=np.uint8),
            metadata=metadata,
        )
    dataset.validate()
    expected_dim = int(feature_dim_for_encoder_version(dataset.encoder_version))
    if dataset.features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dim dataset {dataset.features.shape[1]} incoerente con {dataset.encoder_version} ({expected_dim})"
        )
    return dataset


def make_game_split_ids(
    num_games: int,
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> np.ndarray:
    """Assegna deterministicamente ogni partita a train, validation o test."""
    if num_games < 3:
        raise ValueError("Servono almeno tre partite per creare i tre split")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("Le frazioni train/validation devono essere in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction deve essere < 1")

    train_count = int(round(num_games * train_fraction))
    validation_count = int(round(num_games * validation_fraction))
    train_count = min(max(train_count, 1), num_games - 2)
    validation_count = min(max(validation_count, 1), num_games - train_count - 1)

    shuffled = np.arange(num_games, dtype=np.int32)
    np.random.default_rng(seed).shuffle(shuffled)
    split_by_game = np.full(num_games, SPLIT_TEST, dtype=np.uint8)
    split_by_game[shuffled[:train_count]] = SPLIT_TRAIN
    split_by_game[shuffled[train_count : train_count + validation_count]] = SPLIT_VALIDATION
    return split_by_game


def masked_softmax_batch(logits: np.ndarray, masks: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Softmax stabile sulle azioni legali per un batch di logits."""
    values = np.asarray(logits, dtype=np.float64)
    legal = np.asarray(masks, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 40 or legal.shape != values.shape:
        raise ValueError(f"Shape logits/mask invalida: {values.shape}/{legal.shape}")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature deve essere finita e > 0")
    if bool(np.any(~np.any(legal, axis=1))):
        raise ValueError("Action mask vuota nel batch")
    masked = np.where(legal, values / float(temperature), -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp = np.where(legal, np.exp(shifted), 0.0)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class DistillationMetrics:
    """Metriche teacher-student su uno split."""

    cross_entropy: float
    teacher_entropy: float
    kl_divergence: float
    argmax_agreement: float
    examples: int


@dataclass(frozen=True, slots=True)
class DistillationEpoch:
    """Metriche aggregate di una epoca."""

    epoch: int
    train_cross_entropy: float
    train_argmax_agreement: float
    validation: DistillationMetrics


@dataclass(frozen=True, slots=True)
class DistillationTrainResult:
    """Pesi migliori sulla validation e diagnostica completa del training."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    best_epoch: int
    before_validation: DistillationMetrics
    before_test: DistillationMetrics
    best_validation: DistillationMetrics
    test: DistillationMetrics
    epochs: tuple[DistillationEpoch, ...]


@dataclass(slots=True)
class _AdamState:
    """Momenti Adam per un tensore."""

    m: np.ndarray
    v: np.ndarray


def _evaluate_arrays(
    *,
    features: np.ndarray,
    masks: np.ndarray,
    target_probs: np.ndarray,
    target_action_ids: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    batch_size: int,
) -> DistillationMetrics:
    """Valuta CE, entropia, KL e agreement senza creare logits per tutto lo split."""
    ce_sum = 0.0
    entropy_sum = 0.0
    agreements = 0
    num_examples = int(features.shape[0])
    if num_examples == 0:
        raise ValueError("Split vuoto")
    for start in range(0, num_examples, batch_size):
        stop = min(start + batch_size, num_examples)
        x = features[start:stop]
        mask = masks[start:stop]
        target = target_probs[start:stop].astype(np.float64)
        logits = np.maximum(x @ w1 + b1, 0.0) @ w2 + b2
        predicted = masked_softmax_batch(logits, mask).astype(np.float64)
        ce_sum += float(np.sum(-target * np.log(predicted + 1e-12)))
        entropy_sum += float(np.sum(-target * np.log(target + 1e-12)))
        agreements += int(np.sum(np.argmax(predicted, axis=1) == target_action_ids[start:stop]))
    cross_entropy = ce_sum / num_examples
    teacher_entropy = entropy_sum / num_examples
    return DistillationMetrics(
        cross_entropy=cross_entropy,
        teacher_entropy=teacher_entropy,
        kl_divergence=max(0.0, cross_entropy - teacher_entropy),
        argmax_agreement=agreements / num_examples,
        examples=num_examples,
    )


def evaluate_distilled_model(
    dataset: SuitDistillationDataset,
    model: MLPBCModel,
    *,
    split: SplitName,
    batch_size: int = 4096,
) -> DistillationMetrics:
    """Valuta un modello MLP su uno split del dataset."""
    indices = dataset.indices(split)
    return _evaluate_arrays(
        features=dataset.features[indices],
        masks=dataset.action_masks[indices],
        target_probs=dataset.target_probs[indices],
        target_action_ids=dataset.target_action_ids[indices],
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        batch_size=batch_size,
    )


def _adam_update(
    parameter: np.ndarray,
    gradient: np.ndarray,
    state: _AdamState,
    *,
    learning_rate: float,
    step: int,
) -> None:
    """Aggiornamento Adam in-place con iperparametri standard."""
    state.m *= 0.9
    state.m += 0.1 * gradient
    state.v *= 0.999
    state.v += 0.001 * (gradient * gradient)
    m_hat = state.m / (1.0 - 0.9**step)
    v_hat = state.v / (1.0 - 0.999**step)
    parameter -= float(learning_rate) * m_hat / (np.sqrt(v_hat) + 1e-8)


def _paired_batch(
    x: np.ndarray,
    masks: np.ndarray,
    targets: np.ndarray,
    target_ids: np.ndarray,
    *,
    version: EncoderVersion,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Affianca al batch una copia con una rinomina non identità campionata."""
    permutations = all_suit_permutations()[1:]
    permutation = permutations[int(rng.integers(0, len(permutations)))]
    return (
        np.concatenate([x, permute_encoded_features(x, version=version, permutation=permutation)]),
        np.concatenate([masks, permute_action_masks(masks, permutation=permutation)]),
        np.concatenate([targets, permute_action_vectors(targets, permutation=permutation)]),
        np.concatenate([target_ids, permute_action_ids(target_ids, permutation=permutation)]),
    )


def train_suit_distillation(
    dataset: SuitDistillationDataset,
    init_model: MLPBCModel,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    paired_augmentation: bool = True,
) -> DistillationTrainResult:
    """
    Fine-tuning supervisionato della MLP sui target soft del teacher simmetrizzato.

    Il checkpoint scelto è quello con KL minima sulla validation. Il test rimane
    intatto fino alla selezione conclusa e viene valutato soltanto su modello iniziale
    e checkpoint migliore.
    """
    dataset.validate()
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("epochs/batch/lr devono essere positivi e weight_decay non negativo")
    if init_model.has_belief_input:
        raise ValueError("La distillazione non supporta policy con belief embedded")
    if init_model.feature_dim != dataset.features.shape[1]:
        raise ValueError("Feature dim diversa fra init model e dataset")

    train_indices = dataset.indices("train")
    validation_indices = dataset.indices("validation")
    test_indices = dataset.indices("test")
    if min(train_indices.size, validation_indices.size, test_indices.size) <= 0:
        raise ValueError("Train, validation e test devono essere non vuoti")

    w1 = init_model.w1.copy()
    b1 = init_model.b1.copy()
    w2 = init_model.w2.copy()
    b2 = init_model.b2.copy()
    states = [_AdamState(np.zeros_like(param), np.zeros_like(param)) for param in (w1, b1, w2, b2)]
    rng = np.random.default_rng(seed)

    def evaluate_indices(indices: np.ndarray) -> DistillationMetrics:
        return _evaluate_arrays(
            features=dataset.features[indices],
            masks=dataset.action_masks[indices],
            target_probs=dataset.target_probs[indices],
            target_action_ids=dataset.target_action_ids[indices],
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            batch_size=max(batch_size, 4096),
        )

    before_validation = evaluate_indices(validation_indices)
    before_test = evaluate_indices(test_indices)
    best_validation = before_validation
    best_epoch = 0
    best_parameters = tuple(param.copy() for param in (w1, b1, w2, b2))
    history: list[DistillationEpoch] = []
    optimizer_step = 0

    for epoch in range(1, epochs + 1):
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        train_ce_sum = 0.0
        train_agreements = 0
        train_examples = 0
        for start in range(0, shuffled.size, batch_size):
            batch_indices = shuffled[start : start + batch_size]
            x = dataset.features[batch_indices]
            masks = dataset.action_masks[batch_indices]
            targets = dataset.target_probs[batch_indices]
            target_ids = dataset.target_action_ids[batch_indices]
            if paired_augmentation:
                x, masks, targets, target_ids = _paired_batch(
                    x,
                    masks,
                    targets,
                    target_ids,
                    version=dataset.encoder_version,
                    rng=rng,
                )

            z1 = x @ w1 + b1
            hidden = np.maximum(z1, 0.0)
            logits = hidden @ w2 + b2
            predicted = masked_softmax_batch(logits, masks)
            batch_count = int(x.shape[0])
            train_ce_sum += float(np.sum(-targets.astype(np.float64) * np.log(predicted.astype(np.float64) + 1e-12)))
            train_agreements += int(np.sum(np.argmax(predicted, axis=1) == target_ids))
            train_examples += batch_count

            dlogits = (predicted - targets) / float(batch_count)
            dlogits[~masks] = 0.0
            grad_w2 = hidden.T @ dlogits + float(weight_decay) * w2
            grad_b2 = np.sum(dlogits, axis=0)
            dz1 = (dlogits @ w2.T) * (z1 > 0.0)
            grad_w1 = x.T @ dz1 + float(weight_decay) * w1
            grad_b1 = np.sum(dz1, axis=0)

            optimizer_step += 1
            for parameter, gradient, state in zip(
                (w1, b1, w2, b2),
                (grad_w1, grad_b1, grad_w2, grad_b2),
                states,
                strict=True,
            ):
                _adam_update(
                    parameter,
                    gradient.astype(parameter.dtype, copy=False),
                    state,
                    learning_rate=learning_rate,
                    step=optimizer_step,
                )

        validation = evaluate_indices(validation_indices)
        history.append(
            DistillationEpoch(
                epoch=epoch,
                train_cross_entropy=train_ce_sum / train_examples,
                train_argmax_agreement=train_agreements / train_examples,
                validation=validation,
            )
        )
        if validation.kl_divergence < best_validation.kl_divergence:
            best_validation = validation
            best_epoch = epoch
            best_parameters = tuple(param.copy() for param in (w1, b1, w2, b2))

    w1, b1, w2, b2 = best_parameters
    test = _evaluate_arrays(
        features=dataset.features[test_indices],
        masks=dataset.action_masks[test_indices],
        target_probs=dataset.target_probs[test_indices],
        target_action_ids=dataset.target_action_ids[test_indices],
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        batch_size=max(batch_size, 4096),
    )
    return DistillationTrainResult(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        best_epoch=best_epoch,
        before_validation=before_validation,
        before_test=before_test,
        best_validation=best_validation,
        test=test,
        epochs=tuple(history),
    )


__all__ = [
    "DATASET_FORMAT",
    "DistillationEpoch",
    "DistillationMetrics",
    "DistillationTrainResult",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VALIDATION",
    "SuitDistillationDataset",
    "evaluate_distilled_model",
    "load_suit_distillation_dataset",
    "make_game_split_ids",
    "masked_softmax_batch",
    "train_suit_distillation",
]
