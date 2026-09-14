"""Metriche passive per diagnosticare la salute di un update A2C.

Questo modulo non implementa correzioni del trainer. Osserva soltanto tensori gia'
calcolati e copie dei parametri prima/dopo Adam. Separare la sonda dall'algoritmo rende
testabile l'invariante piu' importante: attivare la diagnostica non deve cambiare RNG,
gradienti o pesi finali.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class A2CSignalSnapshot:
    """Distribuzioni di return, valore, advantage e attivazioni in un update."""

    steps: int
    return_mean: float
    return_std: float
    value_mean: float
    value_std: float
    advantage_mean: float
    advantage_std: float
    advantage_rms: float
    critic_mean_squared_error: float
    advantage_abs_mean: float
    advantage_min: float
    advantage_max: float
    advantage_positive_fraction: float
    critic_explained_variance: float | None
    hidden_activation_rate_mean: float
    hidden_activation_rate_p10: float
    hidden_activation_rate_p50: float
    hidden_activation_rate_p90: float
    hidden_units_never_active: int
    hidden_mean_activation: float


class A2CSignalAccumulator:
    """Accumula momenti sufficienti senza conservare traiettorie o osservazioni."""

    def __init__(self, hidden_dim: int) -> None:
        if hidden_dim <= 0:
            raise ValueError("hidden_dim deve essere > 0")
        self.hidden_dim = int(hidden_dim)
        self._unit_positive_counts = np.zeros(self.hidden_dim, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Azzera lo stato per il prossimo optimizer update."""
        self.steps = 0
        self.return_sum = 0.0
        self.return_sq_sum = 0.0
        self.value_sum = 0.0
        self.value_sq_sum = 0.0
        self.advantage_sum = 0.0
        self.advantage_sq_sum = 0.0
        self.advantage_abs_sum = 0.0
        self.advantage_min = float("inf")
        self.advantage_max = float("-inf")
        self.advantage_positive_count = 0
        self.hidden_sum = 0.0
        self.hidden_element_count = 0
        self._unit_positive_counts.fill(0)

    def observe(self, *, returns_to_go: np.ndarray, value_preds: np.ndarray, hidden: np.ndarray) -> None:
        """Aggiunge un batch di step on-policy gia' usato dal backprop."""
        returns = np.asarray(returns_to_go, dtype=np.float64)
        values = np.asarray(value_preds, dtype=np.float64)
        hidden_values = np.asarray(hidden, dtype=np.float64)
        if returns.ndim != 1 or values.shape != returns.shape:
            raise ValueError(f"Shape return/value invalide: {returns.shape}, {values.shape}")
        if hidden_values.shape != (returns.shape[0], self.hidden_dim):
            raise ValueError(
                f"Shape hidden invalida: {hidden_values.shape}, attesa {(returns.shape[0], self.hidden_dim)}"
            )
        if returns.size == 0:
            return
        if not (
            bool(np.all(np.isfinite(returns)))
            and bool(np.all(np.isfinite(values)))
            and bool(np.all(np.isfinite(hidden_values)))
        ):
            raise ValueError("La diagnostica A2C ha ricevuto valori non finiti")

        advantages = returns - values
        self.steps += int(returns.size)
        self.return_sum += float(np.sum(returns, dtype=np.float64))
        self.return_sq_sum += float(np.sum(returns * returns, dtype=np.float64))
        self.value_sum += float(np.sum(values, dtype=np.float64))
        self.value_sq_sum += float(np.sum(values * values, dtype=np.float64))
        self.advantage_sum += float(np.sum(advantages, dtype=np.float64))
        self.advantage_sq_sum += float(np.sum(advantages * advantages, dtype=np.float64))
        self.advantage_abs_sum += float(np.sum(np.abs(advantages), dtype=np.float64))
        self.advantage_min = min(self.advantage_min, float(np.min(advantages)))
        self.advantage_max = max(self.advantage_max, float(np.max(advantages)))
        self.advantage_positive_count += int(np.count_nonzero(advantages > 0.0))
        self.hidden_sum += float(np.sum(hidden_values, dtype=np.float64))
        self.hidden_element_count += int(hidden_values.size)
        self._unit_positive_counts += np.count_nonzero(hidden_values > 0.0, axis=0)

    @staticmethod
    def _mean_and_std(total: float, square_total: float, count: int) -> tuple[float, float]:
        """Calcola momenti di popolazione proteggendo dagli errori di arrotondamento."""
        mean = total / count
        variance = max(0.0, square_total / count - mean * mean)
        return float(mean), float(np.sqrt(variance))

    def snapshot(self) -> A2CSignalSnapshot:
        """Materializza metriche JSON-safe per l'update appena osservato."""
        if self.steps <= 0:
            raise ValueError("Nessuno step osservato dalla diagnostica A2C")
        return_mean, return_std = self._mean_and_std(self.return_sum, self.return_sq_sum, self.steps)
        value_mean, value_std = self._mean_and_std(self.value_sum, self.value_sq_sum, self.steps)
        advantage_mean, advantage_std = self._mean_and_std(
            self.advantage_sum,
            self.advantage_sq_sum,
            self.steps,
        )
        advantage_rms = float(np.sqrt(max(0.0, self.advantage_sq_sum / self.steps)))
        return_variance = return_std * return_std
        advantage_variance = advantage_std * advantage_std
        explained_variance = None
        if return_variance > 1e-12:
            explained_variance = float(1.0 - advantage_variance / return_variance)

        unit_rates = self._unit_positive_counts.astype(np.float64) / float(self.steps)
        return A2CSignalSnapshot(
            steps=self.steps,
            return_mean=return_mean,
            return_std=return_std,
            value_mean=value_mean,
            value_std=value_std,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            advantage_rms=advantage_rms,
            critic_mean_squared_error=float(self.advantage_sq_sum / self.steps),
            advantage_abs_mean=float(self.advantage_abs_sum / self.steps),
            advantage_min=float(self.advantage_min),
            advantage_max=float(self.advantage_max),
            advantage_positive_fraction=float(self.advantage_positive_count / self.steps),
            critic_explained_variance=explained_variance,
            hidden_activation_rate_mean=float(np.mean(unit_rates)),
            hidden_activation_rate_p10=float(np.quantile(unit_rates, 0.10)),
            hidden_activation_rate_p50=float(np.quantile(unit_rates, 0.50)),
            hidden_activation_rate_p90=float(np.quantile(unit_rates, 0.90)),
            hidden_units_never_active=int(np.count_nonzero(unit_rates == 0.0)),
            hidden_mean_activation=float(self.hidden_sum / max(1, self.hidden_element_count)),
        )


ArrayGroup = tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class A2CArrayGroups:
    """Raggruppa parametri o gradienti secondo le responsabilita' A2C."""

    trunk: ArrayGroup
    actor_head: ArrayGroup
    critic_head: ArrayGroup

    def all_arrays(self) -> ArrayGroup:
        """Restituisce ogni array una volta, per norme globali."""
        return self.trunk + self.actor_head + self.critic_head

    def copied(self) -> A2CArrayGroups:
        """Copia i parametri per misurare il passo Adam senza influire sul trainer."""
        return A2CArrayGroups(
            trunk=tuple(array.copy() for array in self.trunk),
            actor_head=tuple(array.copy() for array in self.actor_head),
            critic_head=tuple(array.copy() for array in self.critic_head),
        )


def array_group_l2(arrays: ArrayGroup) -> float:
    """Norma L2 con accumulo float64 di un gruppo eterogeneo di array."""
    total = sum(float(np.sum(np.asarray(array, dtype=np.float64) ** 2, dtype=np.float64)) for array in arrays)
    return float(np.sqrt(total))


def array_group_max_abs(arrays: ArrayGroup) -> float:
    """Massimo valore assoluto, utile per individuare un singolo gradiente esploso."""
    if not arrays:
        return 0.0
    return max(float(np.max(np.abs(np.asarray(array, dtype=np.float64)))) for array in arrays)


def array_group_delta_l2(before: ArrayGroup, after: ArrayGroup) -> float:
    """Norma L2 della differenza tra due gruppi con shape identiche."""
    if len(before) != len(after):
        raise ValueError("Gruppi parametro incompatibili")
    total = 0.0
    for left, right in zip(before, after, strict=True):
        if left.shape != right.shape:
            raise ValueError(f"Shape parametro incompatibili: {left.shape}, {right.shape}")
        delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
        total += float(np.sum(delta * delta, dtype=np.float64))
    return float(np.sqrt(total))


def _relative_update(delta_l2: float, parameter_l2: float) -> float | None:
    """Rapporto update/parametro; non inventa un rapporto per un critic ancora nullo."""
    if parameter_l2 <= 1e-12:
        return None
    return float(delta_l2 / parameter_l2)


@dataclass(frozen=True, slots=True)
class A2CUpdateDiagnostics:
    """Fotografia completa ma aggregata di un optimizer update."""

    iteration: int
    games: int
    signals: A2CSignalSnapshot
    trunk_gradient_l2: float
    actor_head_gradient_l2: float
    critic_head_gradient_l2: float
    global_gradient_l2: float
    global_gradient_max_abs: float
    trunk_parameter_l2: float
    actor_head_parameter_l2: float
    critic_head_parameter_l2: float
    trunk_update_l2: float
    actor_head_update_l2: float
    critic_head_update_l2: float
    global_update_l2: float
    trunk_relative_update: float | None
    actor_head_relative_update: float | None
    critic_head_relative_update: float | None

    def to_json(self) -> dict[str, Any]:
        """Converte dataclass annidate in un oggetto JSON standard."""
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> A2CUpdateDiagnostics:
        """Ricostruisce una riga campionata conservata in un checkpoint di resume."""
        raw_signals = payload.get("signals")
        if not isinstance(raw_signals, dict):
            raise ValueError("Riga diagnostica senza signals validi")
        values = dict(payload)
        values["signals"] = A2CSignalSnapshot(**raw_signals)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Riga diagnostica del resume incompatibile") from exc


def build_update_diagnostics(
    *,
    iteration: int,
    games: int,
    signals: A2CSignalSnapshot,
    gradients: A2CArrayGroups,
    parameters_before: A2CArrayGroups,
    parameters_after: A2CArrayGroups,
) -> A2CUpdateDiagnostics:
    """Combina segnali, gradienti normalizzati e passo Adam osservato."""
    trunk_gradient = array_group_l2(gradients.trunk)
    actor_gradient = array_group_l2(gradients.actor_head)
    critic_gradient = array_group_l2(gradients.critic_head)
    global_gradient = array_group_l2(gradients.all_arrays())
    trunk_parameter = array_group_l2(parameters_before.trunk)
    actor_parameter = array_group_l2(parameters_before.actor_head)
    critic_parameter = array_group_l2(parameters_before.critic_head)
    trunk_update = array_group_delta_l2(parameters_before.trunk, parameters_after.trunk)
    actor_update = array_group_delta_l2(parameters_before.actor_head, parameters_after.actor_head)
    critic_update = array_group_delta_l2(parameters_before.critic_head, parameters_after.critic_head)
    global_update = array_group_delta_l2(parameters_before.all_arrays(), parameters_after.all_arrays())
    return A2CUpdateDiagnostics(
        iteration=int(iteration),
        games=int(games),
        signals=signals,
        trunk_gradient_l2=trunk_gradient,
        actor_head_gradient_l2=actor_gradient,
        critic_head_gradient_l2=critic_gradient,
        global_gradient_l2=global_gradient,
        global_gradient_max_abs=array_group_max_abs(gradients.all_arrays()),
        trunk_parameter_l2=trunk_parameter,
        actor_head_parameter_l2=actor_parameter,
        critic_head_parameter_l2=critic_parameter,
        trunk_update_l2=trunk_update,
        actor_head_update_l2=actor_update,
        critic_head_update_l2=critic_update,
        global_update_l2=global_update,
        trunk_relative_update=_relative_update(trunk_update, trunk_parameter),
        actor_head_relative_update=_relative_update(actor_update, actor_parameter),
        critic_head_relative_update=_relative_update(critic_update, critic_parameter),
    )


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    """Riassunto robusto di una metrica attraverso gli update."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("Diagnostica con valori non finiti")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def summarize_update_diagnostics(updates: list[A2CUpdateDiagnostics]) -> dict[str, Any]:
    """Crea un summary compatto senza perdere prima e ultima fotografia."""
    if not updates:
        return {"count": 0, "first": None, "last": None, "distributions": {}}

    explained = [
        float(row.signals.critic_explained_variance)
        for row in updates
        if row.signals.critic_explained_variance is not None
    ]
    distributions = {
        "critic_explained_variance": _distribution(explained),
        "advantage_mean": _distribution([row.signals.advantage_mean for row in updates]),
        "advantage_std": _distribution([row.signals.advantage_std for row in updates]),
        "critic_mean_squared_error": _distribution([row.signals.critic_mean_squared_error for row in updates]),
        "global_gradient_l2": _distribution([row.global_gradient_l2 for row in updates]),
        "trunk_relative_update": _distribution(
            [float(row.trunk_relative_update) for row in updates if row.trunk_relative_update is not None]
        ),
        "actor_head_relative_update": _distribution(
            [float(row.actor_head_relative_update) for row in updates if row.actor_head_relative_update is not None]
        ),
        "critic_head_relative_update": _distribution(
            [float(row.critic_head_relative_update) for row in updates if row.critic_head_relative_update is not None]
        ),
        "hidden_activation_rate_mean": _distribution([row.signals.hidden_activation_rate_mean for row in updates]),
        "hidden_units_never_active": _distribution([float(row.signals.hidden_units_never_active) for row in updates]),
    }
    return {
        "count": len(updates),
        "first": updates[0].to_json(),
        "last": updates[-1].to_json(),
        "critic_negative_explained_variance_fraction": (
            float(np.mean(np.asarray(explained, dtype=np.float64) < 0.0)) if explained else None
        ),
        "distributions": distributions,
    }
