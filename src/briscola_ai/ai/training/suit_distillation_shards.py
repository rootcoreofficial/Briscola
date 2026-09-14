"""
Dataset sharded e training streaming per la distillazione della simmetria dei semi.

Il formato monolitico usato per v14 carica tutti gli esempi in RAM. Questo modulo mantiene
lo stesso contenuto numerico, ma lo divide in shard verificabili tramite SHA-256. Il
manifest assegna intervalli contigui e non sovrapposti di ``game_id``; il trainer apre un
solo shard alla volta e calcola validation/test globali senza unire gli array.

Il confine anti-cheat non cambia: gli shard contengono soltanto feature già derivate da
``PlayerObservation``, action mask e target soft prodotti dal teacher.
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..encoding.observation_encoder import EncoderVersion, feature_dim_for_encoder_version
from ..models.bc_model import MLPBCModel
from .suit_distillation import (
    DATASET_FORMAT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    DistillationEpoch,
    DistillationMetrics,
    DistillationTrainResult,
    SuitDistillationDataset,
    _adam_update,
    _AdamState,
    _paired_batch,
    load_suit_distillation_dataset,
    make_game_split_ids,
    masked_softmax_batch,
)

SHARDED_MANIFEST_SCHEMA = "briscola.suit_distillation_shards.v1"
SHARDED_MANIFEST_STATUS_IN_PROGRESS = "in_progress"
SHARDED_MANIFEST_STATUS_COMPLETE = "complete"


def sha256_file(path: Path) -> str:
    """Calcola SHA-256 a blocchi senza caricare il file intero in memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strict_json(path: Path) -> dict[str, Any]:
    """Legge un oggetto JSON rifiutando NaN/Infinity e payload non-oggetto."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Costante JSON non valida in {path}: {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Manifest JSON non leggibile: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Il manifest deve essere un oggetto JSON: {path}")
    return payload


def derive_shard_seed(global_seed: int, shard_index: int) -> int:
    """Deriva un seed uint32 stabile, indipendente dai tentativi di resume."""
    if global_seed < 0 or shard_index < 0:
        raise ValueError("global_seed e shard_index devono essere non negativi")
    material = f"suit-distillation-shard-v1:{global_seed}:{shard_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


@dataclass(frozen=True, slots=True)
class SuitDistillationShard:
    """Ricevuta immutabile di uno shard dichiarato dal manifest."""

    index: int
    path: str
    sha256: str
    size_bytes: int
    seed: int
    game_id_start: int
    game_id_stop: int
    num_games: int
    num_examples: int
    split_game_counts: dict[str, int]
    opponent_game_counts: dict[str, int]

    @classmethod
    def from_payload(cls, payload: object) -> SuitDistillationShard:
        """Converte una riga del manifest controllando campi e valori basilari."""
        if not isinstance(payload, dict):
            raise ValueError("Ogni shard del manifest deve essere un oggetto")
        required = {
            "index",
            "path",
            "sha256",
            "size_bytes",
            "seed",
            "game_id_start",
            "game_id_stop",
            "num_games",
            "num_examples",
            "split_game_counts",
            "opponent_game_counts",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Shard incompleto: mancano {sorted(missing)}")

        def integer(name: str) -> int:
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} dello shard deve essere un intero non negativo")
            return int(value)

        def counts(name: str) -> dict[str, int]:
            raw = payload[name]
            if not isinstance(raw, dict) or any(
                not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) or value < 0
                for key, value in raw.items()
            ):
                raise ValueError(f"{name} dello shard non valido")
            return {str(key): int(value) for key, value in raw.items()}

        path = payload["path"]
        digest = payload["sha256"]
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError("Il path dello shard deve essere relativo e non vuoto")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("SHA-256 dello shard non valido")
        return cls(
            index=integer("index"),
            path=path,
            sha256=digest,
            size_bytes=integer("size_bytes"),
            seed=integer("seed"),
            game_id_start=integer("game_id_start"),
            game_id_stop=integer("game_id_stop"),
            num_games=integer("num_games"),
            num_examples=integer("num_examples"),
            split_game_counts=counts("split_game_counts"),
            opponent_game_counts=counts("opponent_game_counts"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serializza la ricevuta in ordine logico; il writer ordina poi le chiavi."""
        return {
            "index": self.index,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "seed": self.seed,
            "game_id_start": self.game_id_start,
            "game_id_stop": self.game_id_stop,
            "num_games": self.num_games,
            "num_examples": self.num_examples,
            "split_game_counts": self.split_game_counts,
            "opponent_game_counts": self.opponent_game_counts,
        }


@dataclass(frozen=True, slots=True)
class ShardedSuitDistillationDataset:
    """Manifest validato e accesso lazy ai relativi shard NPZ."""

    manifest_path: Path
    payload: dict[str, Any]
    shards: tuple[SuitDistillationShard, ...]

    @property
    def dataset_metadata(self) -> dict[str, Any]:
        """Metadati globali del corpus."""
        raw = self.payload.get("dataset")
        if not isinstance(raw, dict):
            raise ValueError("Sezione dataset mancante nel manifest")
        return raw

    @property
    def encoder_version(self) -> EncoderVersion:
        """Versione encoder comune a tutti gli shard."""
        raw = self.dataset_metadata.get("encoder_version")
        if raw not in {"v1", "v2", "v3", "v4"}:
            raise ValueError(f"encoder_version sharded non valida: {raw!r}")
        return cast(EncoderVersion, raw)

    @property
    def config_fingerprint(self) -> str:
        """Fingerprint che lega gli shard alla configurazione di raccolta."""
        raw = self.payload.get("config_fingerprint")
        if not isinstance(raw, str) or len(raw) != 64:
            raise ValueError("config_fingerprint mancante o non valido")
        return raw

    def shard_path(self, shard: SuitDistillationShard) -> Path:
        """Risolvi un path relativo impedendo traversal fuori dalla directory del manifest."""
        root = self.manifest_path.parent.resolve()
        target = (root / shard.path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Shard fuori dalla directory del manifest: {shard.path}")
        return target

    def verify_shard_file(self, shard: SuitDistillationShard) -> None:
        """Controlla esistenza, dimensione e digest della ricevuta."""
        path = self.shard_path(shard)
        if not path.is_file():
            raise ValueError(f"Shard mancante: {path}")
        if path.stat().st_size != shard.size_bytes:
            raise ValueError(f"Dimensione shard diversa dal manifest: {path}")
        if sha256_file(path) != shard.sha256:
            raise ValueError(f"SHA-256 shard diverso dal manifest: {path}")

    def load_shard(self, shard: SuitDistillationShard, *, verify_hash: bool = False) -> SuitDistillationDataset:
        """Carica e valida uno shard, controllandone anche intervallo e split per partita."""
        if verify_hash:
            self.verify_shard_file(shard)
        dataset = load_suit_distillation_dataset(self.shard_path(shard))
        metadata = dataset.metadata
        expected_metadata = {
            "manifest_config_fingerprint": self.config_fingerprint,
            "shard_index": shard.index,
            "game_id_start": shard.game_id_start,
            "game_id_stop": shard.game_id_stop,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Metadato {key} incoerente nello shard {shard.index}")
        if dataset.encoder_version != self.encoder_version:
            raise ValueError(f"Encoder incoerente nello shard {shard.index}")
        if dataset.features.shape[0] != shard.num_examples:
            raise ValueError(f"Numero esempi incoerente nello shard {shard.index}")

        games, first_rows, examples_per_game = np.unique(
            dataset.game_ids,
            return_index=True,
            return_counts=True,
        )
        expected_games = np.arange(shard.game_id_start, shard.game_id_stop, dtype=np.int32)
        if not np.array_equal(games, expected_games):
            raise ValueError(f"Intervallo game_id incoerente nello shard {shard.index}")
        if not bool(np.all(examples_per_game == 38)):
            raise ValueError(f"Ogni partita deve avere 38 esempi nello shard {shard.index}")
        split_counts_raw = np.bincount(dataset.split_ids[first_rows], minlength=3)
        split_counts = {
            "train": int(split_counts_raw[int(SPLIT_TRAIN)]),
            "validation": int(split_counts_raw[int(SPLIT_VALIDATION)]),
            "test": int(split_counts_raw[int(SPLIT_TEST)]),
        }
        if split_counts != shard.split_game_counts:
            raise ValueError(f"Conteggi split incoerenti nello shard {shard.index}")
        global_split = make_game_split_ids(
            int(self.dataset_metadata["num_games"]),
            seed=int(self.dataset_metadata["split_seed"]),
            train_fraction=float(self.dataset_metadata["train_fraction"]),
            validation_fraction=float(self.dataset_metadata["validation_fraction"]),
        )
        expected_splits = global_split[shard.game_id_start : shard.game_id_stop]
        if not np.array_equal(dataset.split_ids[first_rows], expected_splits):
            raise ValueError(f"Assegnazione split globale incoerente nello shard {shard.index}")
        return dataset


def load_sharded_suit_distillation_dataset(
    manifest_path: Path,
    *,
    require_complete: bool = True,
    verify_hashes: bool = True,
) -> ShardedSuitDistillationDataset:
    """
    Carica il manifest e verifica ordine, copertura, totali e ricevute disponibili.

    Con ``require_complete=False`` accetta il prefisso contiguo prodotto durante una
    raccolta riprendibile. Il trainer usa sempre il default completo.
    """
    payload = load_strict_json(manifest_path)
    if payload.get("schema") != SHARDED_MANIFEST_SCHEMA:
        raise ValueError(f"Schema manifest non supportato: {payload.get('schema')!r}")
    status = payload.get("status")
    if status not in {SHARDED_MANIFEST_STATUS_IN_PROGRESS, SHARDED_MANIFEST_STATUS_COMPLETE}:
        raise ValueError(f"Stato manifest non valido: {status!r}")
    if require_complete and status != SHARDED_MANIFEST_STATUS_COMPLETE:
        raise ValueError("Il corpus sharded non è ancora completo")
    fingerprint = payload.get("config_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("config_fingerprint mancante o non valido")

    metadata = payload.get("dataset")
    if not isinstance(metadata, dict) or metadata.get("format") != DATASET_FORMAT:
        raise ValueError("Sezione dataset assente o formato numerico non supportato")
    integer_fields = (
        "num_games",
        "num_examples",
        "num_shards",
        "games_per_shard",
        "feature_dim",
        "action_dim",
        "seed",
        "split_seed",
    )
    if any(isinstance(metadata.get(name), bool) or not isinstance(metadata.get(name), int) for name in integer_fields):
        raise ValueError("Conteggi globali del manifest mancanti o non interi")
    num_games = int(metadata["num_games"])
    num_examples = int(metadata["num_examples"])
    num_shards = int(metadata["num_shards"])
    games_per_shard = int(metadata["games_per_shard"])
    if min(num_games, num_examples, num_shards, games_per_shard) <= 0 or num_examples != num_games * 38:
        raise ValueError("Totali globali del manifest non validi")
    encoder = metadata.get("encoder_version")
    if encoder not in {"v1", "v2", "v3", "v4"}:
        raise ValueError("encoder_version globale non valida")
    if int(metadata["feature_dim"]) != int(feature_dim_for_encoder_version(cast(EncoderVersion, encoder))):
        raise ValueError("feature_dim globale incoerente con l'encoder")
    if int(metadata["action_dim"]) != 40:
        raise ValueError("action_dim globale deve essere 40")
    train_fraction = metadata.get("train_fraction")
    validation_fraction = metadata.get("validation_fraction")
    if not isinstance(train_fraction, int | float) or not isinstance(validation_fraction, int | float):
        raise ValueError("Frazioni di split globali mancanti")
    expected_global_split = make_game_split_ids(
        num_games,
        seed=int(metadata["split_seed"]),
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
    )
    expected_global_counts = {
        "train": int(np.sum(expected_global_split == SPLIT_TRAIN)),
        "validation": int(np.sum(expected_global_split == SPLIT_VALIDATION)),
        "test": int(np.sum(expected_global_split == SPLIT_TEST)),
    }
    if metadata.get("split_game_counts") != expected_global_counts:
        raise ValueError("I conteggi split globali non corrispondono al seed dichiarato")

    teacher_model = payload.get("teacher_model")
    if (
        not isinstance(teacher_model, dict)
        or not isinstance(teacher_model.get("path"), str)
        or not isinstance(teacher_model.get("sha256"), str)
        or len(teacher_model["sha256"]) != 64
        or isinstance(teacher_model.get("size_bytes"), bool)
        or not isinstance(teacher_model.get("size_bytes"), int)
        or teacher_model["size_bytes"] <= 0
    ):
        raise ValueError("Ricevuta teacher_model mancante o non valida")

    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("La lista shards manca dal manifest")
    shards = tuple(SuitDistillationShard.from_payload(item) for item in raw_shards)
    if len(shards) > num_shards or (status == SHARDED_MANIFEST_STATUS_COMPLETE and len(shards) != num_shards):
        raise ValueError("Numero shard incoerente con lo stato del manifest")

    expected_start = 0
    seen_paths: set[str] = set()
    aggregate_splits = {"train": 0, "validation": 0, "test": 0}
    corpus = ShardedSuitDistillationDataset(manifest_path, payload, shards)
    for expected_index, shard in enumerate(shards):
        expected_games = min(games_per_shard, num_games - expected_start)
        if shard.index != expected_index:
            raise ValueError("Gli shard devono essere un prefisso ordinato da indice zero")
        if shard.path in seen_paths:
            raise ValueError(f"Path shard duplicato: {shard.path}")
        seen_paths.add(shard.path)
        if shard.seed != derive_shard_seed(int(metadata["seed"]), shard.index):
            raise ValueError(f"Seed derivato incoerente nello shard {shard.index}")
        if (
            shard.game_id_start != expected_start
            or shard.num_games != expected_games
            or shard.game_id_stop != expected_start + expected_games
            or shard.num_examples != expected_games * 38
            or sum(shard.split_game_counts.values()) != expected_games
            or shard.size_bytes <= 0
        ):
            raise ValueError(f"Copertura o conteggi incoerenti nello shard {shard.index}")
        for name in aggregate_splits:
            aggregate_splits[name] += shard.split_game_counts.get(name, 0)
        if verify_hashes:
            corpus.verify_shard_file(shard)
        expected_start = shard.game_id_stop

    opponent_names = {name for shard in shards for name in shard.opponent_game_counts}
    aggregate_opponents = {
        name: sum(shard.opponent_game_counts.get(name, 0) for shard in shards) for name in opponent_names
    }
    expected_completed = {
        "num_shards": len(shards),
        "num_games": sum(shard.num_games for shard in shards),
        "num_examples": sum(shard.num_examples for shard in shards),
        "split_game_counts": aggregate_splits,
        "opponent_game_counts": dict(sorted(aggregate_opponents.items())),
    }
    if payload.get("completed") != expected_completed:
        raise ValueError("La ricevuta completed non corrisponde agli shard presenti")

    if status == SHARDED_MANIFEST_STATUS_COMPLETE:
        if expected_start != num_games:
            raise ValueError("Gli shard completi non coprono tutte le partite")
        global_splits = metadata.get("split_game_counts")
        if not isinstance(global_splits, dict) or aggregate_splits != global_splits:
            raise ValueError("I conteggi split degli shard non sommano ai totali globali")
    return ShardedSuitDistillationDataset(manifest_path=manifest_path, payload=payload, shards=shards)


@dataclass(slots=True)
class _MetricSums:
    """Accumulatori additivi per metriche calcolate su più shard."""

    cross_entropy: float = 0.0
    teacher_entropy: float = 0.0
    agreements: int = 0
    examples: int = 0

    def finish(self) -> DistillationMetrics:
        """Normalizza gli aggregati soltanto dopo avere visitato tutti gli shard."""
        if self.examples <= 0:
            raise ValueError("Split sharded vuoto")
        cross_entropy = self.cross_entropy / self.examples
        teacher_entropy = self.teacher_entropy / self.examples
        return DistillationMetrics(
            cross_entropy=cross_entropy,
            teacher_entropy=teacher_entropy,
            kl_divergence=max(0.0, cross_entropy - teacher_entropy),
            argmax_agreement=self.agreements / self.examples,
            examples=self.examples,
        )


def _accumulate_metrics(
    sums: _MetricSums,
    *,
    dataset: SuitDistillationDataset,
    indices: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
) -> None:
    """Aggiunge uno split di uno shard senza materializzarne una copia completa."""
    w1, b1, w2, b2 = parameters
    for start in range(0, int(indices.size), batch_size):
        batch_indices = indices[start : start + batch_size]
        x = dataset.features[batch_indices]
        masks = dataset.action_masks[batch_indices]
        targets = dataset.target_probs[batch_indices].astype(np.float64)
        target_ids = dataset.target_action_ids[batch_indices]
        logits = np.maximum(x @ w1 + b1, 0.0) @ w2 + b2
        predicted = masked_softmax_batch(logits, masks).astype(np.float64)
        sums.cross_entropy += float(np.sum(-targets * np.log(predicted + 1e-12)))
        sums.teacher_entropy += float(np.sum(-targets * np.log(targets + 1e-12)))
        sums.agreements += int(np.sum(np.argmax(predicted, axis=1) == target_ids))
        sums.examples += int(batch_indices.size)


def _evaluate_sharded(
    dataset: ShardedSuitDistillationDataset,
    *,
    split: str,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
) -> DistillationMetrics:
    """Valuta uno split globale aprendo gli shard in ordine canonico."""
    split_id = {"train": SPLIT_TRAIN, "validation": SPLIT_VALIDATION, "test": SPLIT_TEST}.get(split)
    if split_id is None:
        raise ValueError(f"Split sharded sconosciuto: {split}")
    sums = _MetricSums()
    for descriptor in dataset.shards:
        shard = dataset.load_shard(descriptor)
        indices = np.flatnonzero(shard.split_ids == split_id)
        _accumulate_metrics(
            sums,
            dataset=shard,
            indices=indices,
            parameters=parameters,
            batch_size=batch_size,
        )
        del shard, indices
        gc.collect()
    return sums.finish()


def train_suit_distillation_sharded(
    dataset: ShardedSuitDistillationDataset,
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
    Distilla un manifest completo mantenendo in RAM un solo shard alla volta.

    A ogni epoca l'ordine degli shard e quello degli esempi train interni vengono
    rimescolati dallo stesso RNG. Validation e test seguono invece sempre l'ordine
    canonico del manifest. Il test viene letto prima del training e poi soltanto sul
    checkpoint scelto dalla KL di validation.
    """
    if dataset.payload.get("status") != SHARDED_MANIFEST_STATUS_COMPLETE:
        raise ValueError("Il trainer richiede un manifest completo")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("epochs/batch/lr devono essere positivi e weight_decay non negativo")
    if init_model.has_belief_input:
        raise ValueError("La distillazione non supporta policy con belief embedded")
    expected_dim = int(dataset.dataset_metadata["feature_dim"])
    if init_model.feature_dim != expected_dim:
        raise ValueError("Feature dim diversa fra init model e manifest")
    split_counts = dataset.dataset_metadata.get("split_game_counts")
    if not isinstance(split_counts, dict) or any(
        int(split_counts.get(name, 0)) <= 0 for name in ("train", "validation", "test")
    ):
        raise ValueError("Train, validation e test globali devono essere non vuoti")

    w1 = init_model.w1.copy()
    b1 = init_model.b1.copy()
    w2 = init_model.w2.copy()
    b2 = init_model.b2.copy()
    states = [_AdamState(np.zeros_like(param), np.zeros_like(param)) for param in (w1, b1, w2, b2)]
    rng = np.random.default_rng(seed)
    evaluation_batch_size = max(batch_size, 4096)

    parameters = (w1, b1, w2, b2)
    before_validation = _evaluate_sharded(
        dataset,
        split="validation",
        parameters=parameters,
        batch_size=evaluation_batch_size,
    )
    before_test = _evaluate_sharded(
        dataset,
        split="test",
        parameters=parameters,
        batch_size=evaluation_batch_size,
    )
    best_validation = before_validation
    best_epoch = 0
    best_parameters = tuple(param.copy() for param in parameters)
    history: list[DistillationEpoch] = []
    optimizer_step = 0

    for epoch in range(1, epochs + 1):
        shard_order = np.arange(len(dataset.shards), dtype=np.int32)
        rng.shuffle(shard_order)
        train_ce_sum = 0.0
        train_agreements = 0
        train_examples = 0
        for shard_index in shard_order:
            shard = dataset.load_shard(dataset.shards[int(shard_index)])
            train_indices = np.flatnonzero(shard.split_ids == SPLIT_TRAIN)
            rng.shuffle(train_indices)
            for start in range(0, int(train_indices.size), batch_size):
                batch_indices = train_indices[start : start + batch_size]
                x = shard.features[batch_indices]
                masks = shard.action_masks[batch_indices]
                targets = shard.target_probs[batch_indices]
                target_ids = shard.target_action_ids[batch_indices]
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
                train_ce_sum += float(
                    np.sum(-targets.astype(np.float64) * np.log(predicted.astype(np.float64) + 1e-12))
                )
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
                    parameters,
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
            if any(bool(np.any(~np.isfinite(parameter))) for parameter in parameters):
                raise FloatingPointError(f"Pesi non finiti dopo lo shard {int(shard_index)} dell'epoca {epoch}")
            del shard, train_indices
            gc.collect()

        validation = _evaluate_sharded(
            dataset,
            split="validation",
            parameters=parameters,
            batch_size=evaluation_batch_size,
        )
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
            best_parameters = tuple(param.copy() for param in parameters)

    w1, b1, w2, b2 = best_parameters
    test = _evaluate_sharded(
        dataset,
        split="test",
        parameters=(w1, b1, w2, b2),
        batch_size=evaluation_batch_size,
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
    "SHARDED_MANIFEST_SCHEMA",
    "SHARDED_MANIFEST_STATUS_COMPLETE",
    "SHARDED_MANIFEST_STATUS_IN_PROGRESS",
    "ShardedSuitDistillationDataset",
    "SuitDistillationShard",
    "derive_shard_seed",
    "load_sharded_suit_distillation_dataset",
    "load_strict_json",
    "sha256_file",
    "train_suit_distillation_sharded",
]
