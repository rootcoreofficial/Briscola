"""Diagnostica delle unità ReLU di una policy MLP a un solo livello nascosto.

Il modulo non modifica i pesi e non usa stato nascosto del gioco. Riceve matrici di
feature già codificate e action mask, ricostruisce il forward esatto della policy e
misura tre aspetti complementari:

* utilizzo: frequenza e intensità di attivazione di ogni unità;
* ridondanza: correlazioni e rango effettivo della matrice delle attivazioni;
* causalità locale: effetto sulle decisioni quando una singola unità viene azzerata.

L'ablation è esatta per questa architettura: togliere l'unità ``j`` equivale a sottrarre
``hidden[:, j] * w2[j, :]`` dai logits. Il file `.npz` resta immutato.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.bc_model import MLPBCModel


@dataclass(frozen=True, slots=True)
class HiddenUnitThresholds:
    """Soglie descrittive preregistrate per rendere confrontabili report successivi."""

    activation_epsilon: float = 1e-8
    dead_activation_rate_max: float = 0.001
    always_active_rate_min: float = 0.999
    redundant_abs_correlation_min: float = 0.995
    dominant_ablation_flip_rate_min: float = 0.05

    def validate(self) -> None:
        """Rifiuta configurazioni ambigue prima di produrre etichette diagnostiche."""
        if self.activation_epsilon < 0.0:
            raise ValueError("activation_epsilon deve essere >= 0")
        for name, value in (
            ("dead_activation_rate_max", self.dead_activation_rate_max),
            ("always_active_rate_min", self.always_active_rate_min),
            ("redundant_abs_correlation_min", self.redundant_abs_correlation_min),
            ("dominant_ablation_flip_rate_min", self.dominant_ablation_flip_rate_min),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere in [0, 1]")
        if self.dead_activation_rate_max >= self.always_active_rate_min:
            raise ValueError("La soglia dead deve essere inferiore alla soglia always-active")


def _validate_inputs(model: MLPBCModel, inputs: np.ndarray, action_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalizza un batch e verifica shape, finitezza e presenza di almeno due azioni legali."""
    x = np.asarray(inputs, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=bool)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != model.feature_dim:
        raise ValueError(f"Input shape invalida: {x.shape}; atteso (N, {model.feature_dim}) con N > 0")
    if masks.shape != (x.shape[0], 40):
        raise ValueError(f"Action mask shape invalida: {masks.shape}; atteso {(x.shape[0], 40)}")
    if not bool(np.all(np.isfinite(x))):
        raise ValueError("Input non finiti")
    legal_counts = np.sum(masks, axis=1)
    if bool(np.any(legal_counts < 2)):
        raise ValueError("La diagnostica richiede almeno due azioni legali per osservazione")
    return x, masks


def _masked_actions(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Ritorna l'argmax legale per ogni riga senza alterare i logits originali."""
    return np.argmax(np.where(masks, logits, -np.inf), axis=1).astype(np.int16)


def _baseline_margin(logits: np.ndarray, masks: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcola margine top-1/top-2 e id della seconda azione legale."""
    masked = np.where(masks, logits, -np.inf)
    alternatives = masked.copy()
    alternatives[np.arange(masked.shape[0]), actions] = -np.inf
    runners_up = np.argmax(alternatives, axis=1).astype(np.int16)
    rows = np.arange(masked.shape[0])
    margins = masked[rows, actions] - masked[rows, runners_up]
    return np.asarray(margins, dtype=np.float64), runners_up


def _activation_geometry(hidden: np.ndarray, *, correlation_threshold: float) -> dict[str, Any]:
    """Riassume rango effettivo e coppie quasi duplicate, ignorando colonne costanti."""
    centered = np.asarray(hidden, dtype=np.float64) - np.mean(hidden, axis=0, dtype=np.float64)
    standard_deviation = np.std(centered, axis=0)
    variable_indices = np.flatnonzero(standard_deviation > 1e-12)

    redundant_pairs: list[tuple[float, int, int]] = []
    redundant_units: set[int] = set()
    if variable_indices.size >= 2:
        correlations = np.asarray(np.corrcoef(centered[:, variable_indices], rowvar=False), dtype=np.float64)
        left, right = np.triu_indices(variable_indices.size, k=1)
        values = np.abs(correlations[left, right])
        for pair_index in np.flatnonzero(values >= correlation_threshold):
            unit_left = int(variable_indices[left[pair_index]])
            unit_right = int(variable_indices[right[pair_index]])
            correlation = float(correlations[left[pair_index], right[pair_index]])
            redundant_pairs.append((abs(correlation), unit_left, unit_right))
            redundant_units.update((unit_left, unit_right))
    redundant_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    singular_values = np.linalg.svd(centered, compute_uv=False)
    variances = singular_values * singular_values
    variance_total = float(np.sum(variances))
    if variance_total > 0.0:
        proportions = variances / variance_total
        positive = proportions > 0.0
        effective_rank = float(np.exp(-np.sum(proportions[positive] * np.log(proportions[positive]))))
        cumulative = np.cumsum(proportions)
        components_95 = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
        components_99 = int(np.searchsorted(cumulative, 0.99, side="left") + 1)
        stable_rank = float(variance_total / variances[0]) if variances[0] > 0.0 else 0.0
    else:
        effective_rank = 0.0
        components_95 = 0
        components_99 = 0
        stable_rank = 0.0

    hidden_dim = int(hidden.shape[1])
    return {
        "variable_units": int(variable_indices.size),
        "constant_units": hidden_dim - int(variable_indices.size),
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / hidden_dim,
        "stable_rank": stable_rank,
        "components_for_95pct_variance": components_95,
        "components_for_99pct_variance": components_99,
        "redundant_pair_count": len(redundant_pairs),
        "redundant_unit_count": len(redundant_units),
        "redundant_unit_fraction": len(redundant_units) / hidden_dim,
        "top_redundant_pairs": [
            {"unit_a": left, "unit_b": right, "correlation": correlation}
            for correlation, left, right in redundant_pairs[:20]
        ],
    }


def analyze_hidden_unit_arrays(
    model: MLPBCModel,
    inputs: np.ndarray,
    action_masks: np.ndarray,
    *,
    thresholds: HiddenUnitThresholds | None = None,
) -> dict[str, Any]:
    """Analizza utilizzo, geometria e ablation singola sulle osservazioni fornite.

    I record per unità sono ordinati per indice; le classifiche separate rendono leggibili
    gli elementi più influenti senza perdere il dettaglio completo necessario all'audit.
    """
    configured = thresholds or HiddenUnitThresholds()
    configured.validate()
    x, masks = _validate_inputs(model, inputs, action_masks)

    pre_activation = x @ model.w1 + model.b1
    hidden = np.maximum(pre_activation, 0.0)
    logits = hidden @ model.w2 + model.b2
    baseline_actions = _masked_actions(logits, masks)
    baseline_margins, runners_up = _baseline_margin(logits, masks, baseline_actions)
    rows = np.arange(x.shape[0])

    activation_rates = np.mean(hidden > configured.activation_epsilon, axis=0)
    means = np.asarray(np.mean(hidden, axis=0, dtype=np.float64), dtype=np.float64)
    standard_deviations = np.std(hidden, axis=0, dtype=np.float64)
    p95 = np.quantile(hidden, 0.95, axis=0)
    maximums = np.max(hidden, axis=0)
    incoming_norms = np.linalg.norm(np.asarray(model.w1, dtype=np.float64), axis=0)
    outgoing = np.asarray(model.w2, dtype=np.float64)
    outgoing_norms = np.linalg.norm(outgoing, axis=1)
    outgoing_centered_norms = np.linalg.norm(outgoing - np.mean(outgoing, axis=1, keepdims=True), axis=1)

    unit_rows: list[dict[str, Any]] = []
    for unit_index in range(hidden.shape[1]):
        contribution = hidden[:, unit_index, None] * model.w2[unit_index][None, :]
        ablated_logits = logits - contribution
        ablated_actions = _masked_actions(ablated_logits, masks)

        baseline_scores_after = ablated_logits[rows, baseline_actions]
        alternatives_after = np.where(masks, ablated_logits, -np.inf)
        alternatives_after[rows, baseline_actions] = -np.inf
        best_alternative_after = np.max(alternatives_after, axis=1)
        margin_after = baseline_scores_after - best_alternative_after
        margin_contribution = hidden[:, unit_index] * (
            model.w2[unit_index, baseline_actions] - model.w2[unit_index, runners_up]
        )

        unit_rows.append(
            {
                "unit": unit_index,
                "activation_rate": float(activation_rates[unit_index]),
                "mean_activation": float(means[unit_index]),
                "std_activation": float(standard_deviations[unit_index]),
                "p95_activation": float(p95[unit_index]),
                "max_activation": float(maximums[unit_index]),
                "incoming_l2": float(incoming_norms[unit_index]),
                "outgoing_l2": float(outgoing_norms[unit_index]),
                "outgoing_centered_l2": float(outgoing_centered_norms[unit_index]),
                "mean_abs_baseline_margin_contribution": float(np.mean(np.abs(margin_contribution))),
                "ablation_action_flip_rate": float(np.mean(ablated_actions != baseline_actions)),
                "mean_baseline_margin_loss": float(np.mean(baseline_margins - margin_after)),
            }
        )

    dead_units = [row["unit"] for row in unit_rows if row["activation_rate"] <= configured.dead_activation_rate_max]
    always_active_units = [
        row["unit"] for row in unit_rows if row["activation_rate"] >= configured.always_active_rate_min
    ]
    influential = sorted(
        unit_rows,
        key=lambda row: (
            -row["ablation_action_flip_rate"],
            -row["mean_abs_baseline_margin_contribution"],
            row["unit"],
        ),
    )
    margin_ranked = sorted(
        unit_rows,
        key=lambda row: (-row["mean_abs_baseline_margin_contribution"], row["unit"]),
    )
    dominant_units = [
        row["unit"]
        for row in unit_rows
        if row["ablation_action_flip_rate"] >= configured.dominant_ablation_flip_rate_min
    ]

    return {
        "observations": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "hidden_dim": int(hidden.shape[1]),
        "baseline_top2_margin": {
            "mean": float(np.mean(baseline_margins)),
            "p50": float(np.quantile(baseline_margins, 0.50)),
            "p05": float(np.quantile(baseline_margins, 0.05)),
        },
        "utilization": {
            "dead_units": dead_units,
            "dead_unit_count": len(dead_units),
            "dead_unit_fraction": len(dead_units) / hidden.shape[1],
            "always_active_units": always_active_units,
            "always_active_unit_count": len(always_active_units),
            "always_active_unit_fraction": len(always_active_units) / hidden.shape[1],
            "mean_activation_rate_across_units": float(np.mean(activation_rates)),
            "median_activation_rate_across_units": float(np.median(activation_rates)),
        },
        "geometry": _activation_geometry(
            hidden,
            correlation_threshold=configured.redundant_abs_correlation_min,
        ),
        "influence": {
            "dominant_units": dominant_units,
            "dominant_unit_count": len(dominant_units),
            "max_single_ablation_flip_rate": float(influential[0]["ablation_action_flip_rate"]),
            "mean_single_ablation_flip_rate": float(np.mean([row["ablation_action_flip_rate"] for row in unit_rows])),
            "top_by_action_flip": influential[:20],
            "top_by_margin_contribution": margin_ranked[:20],
        },
        "units": unit_rows,
    }


def analyze_suit_ablation_arrays(
    model: MLPBCModel,
    orbit_inputs: np.ndarray,
    orbit_masks: np.ndarray,
    remap_action_ids: np.ndarray,
) -> dict[str, Any]:
    """Misura come l'ablation di ogni unità cambia i flip sulle 24 rinomine dei semi.

    ``orbit_inputs`` ha shape ``(N, 24, D)``. ``remap_action_ids[p, a]`` riporta
    l'action id ``a`` della permutazione ``p`` nell'orientamento dell'identità.
    """
    inputs = np.asarray(orbit_inputs, dtype=np.float32)
    masks = np.asarray(orbit_masks, dtype=bool)
    remap = np.asarray(remap_action_ids, dtype=np.int16)
    if inputs.ndim != 3 or inputs.shape[1] != 24 or inputs.shape[2] != model.feature_dim:
        raise ValueError(f"Orbit input shape invalida: {inputs.shape}; atteso (N, 24, {model.feature_dim})")
    if masks.shape != (inputs.shape[0], 24, 40):
        raise ValueError(f"Orbit mask shape invalida: {masks.shape}")
    if remap.shape != (24, 40) or bool(np.any(remap < 0)) or bool(np.any(remap >= 40)):
        raise ValueError(f"Mappa action id invalida: {remap.shape}")
    if bool(np.any(np.sum(masks, axis=2) < 2)):
        raise ValueError("Ogni orbita deve avere almeno due azioni legali")

    flat_inputs = inputs.reshape(-1, model.feature_dim)
    flat_masks = masks.reshape(-1, 40)
    hidden = np.maximum(flat_inputs @ model.w1 + model.b1, 0.0)
    logits = hidden @ model.w2 + model.b2
    permutation_indices = np.tile(np.arange(24), inputs.shape[0])

    def remapped_choices(candidate_logits: np.ndarray) -> np.ndarray:
        actions = _masked_actions(candidate_logits, flat_masks)
        original_ids = remap[permutation_indices, actions]
        return original_ids.reshape(inputs.shape[0], 24)

    baseline_choices = remapped_choices(logits)
    baseline_flips = baseline_choices[:, 1:] != baseline_choices[:, [0]]
    baseline_flip_rate = float(np.mean(baseline_flips))

    rows: list[dict[str, Any]] = []
    for unit_index in range(hidden.shape[1]):
        ablated_logits = logits - hidden[:, unit_index, None] * model.w2[unit_index][None, :]
        choices = remapped_choices(ablated_logits)
        flip_rate = float(np.mean(choices[:, 1:] != choices[:, [0]]))
        rows.append(
            {
                "unit": unit_index,
                "ablation_flip_rate": flip_rate,
                "delta_vs_baseline": flip_rate - baseline_flip_rate,
            }
        )

    improved = sorted(rows, key=lambda row: (row["delta_vs_baseline"], row["unit"]))
    worsened = sorted(rows, key=lambda row: (-row["delta_vs_baseline"], row["unit"]))
    return {
        "observations": int(inputs.shape[0]),
        "permutations": 24,
        "nonidentity_comparisons": int(inputs.shape[0] * 23),
        "baseline_flip_rate": baseline_flip_rate,
        "best_single_unit_removal": improved[0],
        "worst_single_unit_removal": worsened[0],
        "top_reductions": improved[:20],
        "top_increases": worsened[:20],
        "units": rows,
    }


def _normalize_hidden_unit_indices(
    model: MLPBCModel,
    unit_indices: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    """Normalizza e valida un insieme non vuoto di indici del livello nascosto."""
    normalized = tuple(int(index) for index in unit_indices)
    if not normalized:
        raise ValueError("Specificare almeno una unità nascosta")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Gli indici delle unità devono essere unici")
    hidden_dim = int(model.b1.shape[0])
    invalid = [index for index in normalized if index < 0 or index >= hidden_dim]
    if invalid:
        raise ValueError(f"Indici unità fuori range 0..{hidden_dim - 1}: {invalid}")
    return normalized


def ablate_mlp_hidden_units(
    model: MLPBCModel,
    unit_indices: list[int] | tuple[int, ...],
    *,
    metadata: dict[str, Any] | None = None,
) -> MLPBCModel:
    """Crea una copia della policy che ignora congiuntamente le unità richieste.

    Azzeriamo le righe corrispondenti di ``w2`` invece di toccare ``w1``/``b1``: il
    forward risultante è esattamente quello ottenuto ponendo a zero quelle attivazioni,
    mentre i pesi di ingresso restano disponibili per audit e confronti.
    """
    normalized = _normalize_hidden_unit_indices(model, unit_indices)

    w2 = np.asarray(model.w2, dtype=np.float32).copy()
    w2[list(normalized), :] = 0.0
    return MLPBCModel(
        w1=np.asarray(model.w1, dtype=np.float32).copy(),
        b1=np.asarray(model.b1, dtype=np.float32).copy(),
        w2=w2,
        b2=np.asarray(model.b2, dtype=np.float32).copy(),
        metadata=dict(metadata if metadata is not None else model.metadata),
        belief_w1=None if model.belief_w1 is None else np.asarray(model.belief_w1, dtype=np.float32).copy(),
        belief_b1=None if model.belief_b1 is None else np.asarray(model.belief_b1, dtype=np.float32).copy(),
        belief_w2=None if model.belief_w2 is None else np.asarray(model.belief_w2, dtype=np.float32).copy(),
        belief_b2=None if model.belief_b2 is None else np.asarray(model.belief_b2, dtype=np.float32).copy(),
    )


def reinitialize_mlp_hidden_units(
    model: MLPBCModel,
    unit_indices: list[int] | tuple[int, ...],
    *,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> MLPBCModel:
    """Riattiva capacità nascosta con ingressi He e uscite inizialmente nulle.

    Ogni unità usa un RNG derivato da ``seed`` e dal proprio indice: una stessa unità
    riceve quindi pesi identici anche in esperimenti annidati, per esempio reset 8 e
    reset 16. Azzerare la riga ``w2`` rende i logits iniziali uguali all'ablation della
    stessa unità; non garantisce uguaglianza col modello originale fuori dalle suite in
    cui l'unità era davvero inattiva.
    """
    normalized = _normalize_hidden_unit_indices(model, unit_indices)
    w1 = np.asarray(model.w1, dtype=np.float32).copy()
    b1 = np.asarray(model.b1, dtype=np.float32).copy()
    w2 = np.asarray(model.w2, dtype=np.float32).copy()
    scale = float(np.sqrt(2.0 / model.feature_dim))
    for unit_index in normalized:
        material = f"briscola.dormant_reinit.v1:{int(seed)}:{unit_index}".encode("ascii")
        unit_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        rng = np.random.default_rng(unit_seed)
        w1[:, unit_index] = rng.normal(0.0, scale, size=model.feature_dim).astype(np.float32)
        b1[unit_index] = 0.0
        w2[unit_index, :] = 0.0
    return MLPBCModel(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=np.asarray(model.b2, dtype=np.float32).copy(),
        metadata=dict(metadata if metadata is not None else model.metadata),
        belief_w1=None if model.belief_w1 is None else np.asarray(model.belief_w1, dtype=np.float32).copy(),
        belief_b1=None if model.belief_b1 is None else np.asarray(model.belief_b1, dtype=np.float32).copy(),
        belief_w2=None if model.belief_w2 is None else np.asarray(model.belief_w2, dtype=np.float32).copy(),
        belief_b2=None if model.belief_b2 is None else np.asarray(model.belief_b2, dtype=np.float32).copy(),
    )


__all__ = [
    "HiddenUnitThresholds",
    "ablate_mlp_hidden_units",
    "analyze_hidden_unit_arrays",
    "analyze_suit_ablation_arrays",
    "reinitialize_mlp_hidden_units",
]
