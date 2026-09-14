"""
Modello di valore scalare salvato in `.npz`.

Il modello predice il valore finale atteso di una `PlayerObservation` 2-player dal punto
di vista del giocatore osservante. Per lo Stage 0 dell'ipotesi V-lookahead il target e'
tipicamente il residuo normalizzato `(final_score_delta - current_score_delta) / 120`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..encoding.observation_encoder import EncoderVersion, feature_dim_for_encoder_version


def _parse_metadata_json(raw: Any) -> dict[str, Any]:
    """Parsa `metadata_json` salvato in npz (best effort)."""
    try:
        text = str(raw.item())
    except Exception:
        text = str(raw)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_metadata_json": text}
    return parsed if isinstance(parsed, dict) else {"metadata": parsed}


@dataclass(frozen=True, slots=True)
class MLPValueModel:
    """
    MLP 1-hidden-layer per regressione scalare.

    Convenzione `.npz`:
    - `w1`: (D, H)
    - `b1`: (H,)
    - `w2`: (H, 1) oppure (H,)
    - `b2`: (1,) oppure scalare
    - `metadata_json`: JSON informativo
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    metadata: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        """Dimensione feature attesa."""
        return int(self.w1.shape[0])

    @property
    def hidden_dim(self) -> int:
        """Dimensione hidden layer."""
        return int(self.w1.shape[1])

    def predict_scaled(self, x: np.ndarray) -> float:
        """Predice il target scalato salvato dal training."""
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        return float(h @ self.w2 + float(self.b2))

    def predict_points(self, x: np.ndarray, *, current_score_delta: float = 0.0) -> float:
        """
        Predice il delta finale in punti.

        Se il modello e' addestrato su target residuo, somma il delta corrente passato dal chiamante.
        """
        scale = float(self.metadata.get("target_scale", 120.0) or 120.0)
        raw = self.predict_scaled(x) * scale
        if self.metadata.get("target") == "residual":
            return float(current_score_delta) + raw
        return raw


_VALUE_MODEL_NPZ_CACHE: dict[tuple[str, int, int], MLPValueModel] = {}


def load_value_model_npz(path: str | Path) -> MLPValueModel:
    """Carica un value model `.npz`, con cache basata su path/mtime/size."""
    model_path = Path(path)
    try:
        st = os.stat(model_path)
    except OSError:
        return _load_value_model_npz_uncached(model_path)
    key = (os.path.abspath(str(model_path)), st.st_mtime_ns, int(st.st_size))
    cached = _VALUE_MODEL_NPZ_CACHE.get(key)
    if cached is not None:
        return cached
    model = _load_value_model_npz_uncached(model_path)
    _VALUE_MODEL_NPZ_CACHE[key] = model
    return model


def _load_value_model_npz_uncached(path: Path) -> MLPValueModel:
    """Carica e valida un value model `.npz` senza usare cache."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Formato non supportato: {path} (atteso .npz)")

    with np.load(path) as data:
        keys = set(data.keys())
        missing = {"w1", "b1", "w2", "b2"} - keys
        if missing:
            raise ValueError(f"File value model invalido: mancano chiavi {sorted(missing)}")

        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = _parse_metadata_json(data["metadata_json"])

        fmt = metadata.get("format")
        if fmt != "value_mlp_v1":
            raise ValueError(f"Formato value model non supportato: {fmt!r}")

        # `target` cambia la SEMANTICA di `predict_points` (residuo da sommare al delta
        # corrente vs delta assoluto): un modello senza questo metadato verrebbe interpretato
        # silenziosamente come assoluto, corrompendo il value-lookahead senza alcun errore.
        # Lo richiediamo esplicito, come già facciamo per `encoder_version`.
        target = metadata.get("target")
        if target not in {"residual", "absolute"}:
            raise ValueError(f"Value model senza metadata.target valido: {target!r} (atteso 'residual' o 'absolute')")

        w1 = np.asarray(data["w1"], dtype=np.float32)
        b1 = np.asarray(data["b1"], dtype=np.float32)
        w2_raw = np.asarray(data["w2"], dtype=np.float32)
        b2_raw = np.asarray(data["b2"], dtype=np.float32)

        if w1.ndim != 2 or b1.ndim != 1:
            raise ValueError("Shape invalide per value model: w1 deve essere 2D e b1 1D")
        if w1.shape[1] != b1.shape[0]:
            raise ValueError(f"Hidden dim mismatch: w1={w1.shape} b1={b1.shape}")
        if w2_raw.ndim == 2:
            if w2_raw.shape != (w1.shape[1], 1):
                raise ValueError(f"Shape w2 invalida: {w2_raw.shape}, attesa {(w1.shape[1], 1)}")
            w2 = w2_raw[:, 0]
        elif w2_raw.ndim == 1:
            if w2_raw.shape[0] != w1.shape[1]:
                raise ValueError(f"Shape w2 invalida: {w2_raw.shape}, attesa {(w1.shape[1],)}")
            w2 = w2_raw
        else:
            raise ValueError(f"Shape w2 invalida: ndim={w2_raw.ndim}")

        if b2_raw.size != 1:
            raise ValueError(f"Shape b2 invalida: {b2_raw.shape}, attesa scalare")
        b2 = float(b2_raw.reshape(-1)[0])

        declared_dim = metadata.get("feature_dim")
        if isinstance(declared_dim, int) and int(declared_dim) != int(w1.shape[0]):
            raise ValueError(f"Feature dim mismatch: metadata={declared_dim} actual={int(w1.shape[0])}")

        encoder_version = metadata.get("encoder_version")
        if isinstance(encoder_version, str) and encoder_version in {"v1", "v2", "v3", "v4"}:
            expected = feature_dim_for_encoder_version(cast(EncoderVersion, encoder_version))
            if int(expected) != int(w1.shape[0]):
                raise ValueError(
                    f"Encoder/version mismatch: encoder={encoder_version} expected={expected} actual={w1.shape[0]}"
                )

        return MLPValueModel(w1=w1, b1=b1, w2=w2, b2=b2, metadata=metadata)


def infer_value_encoder_version(model: MLPValueModel) -> EncoderVersion:
    """Ricava la versione encoder dai metadati del value model."""
    raw = model.metadata.get("encoder_version")
    if isinstance(raw, str) and raw in {"v1", "v2", "v3", "v4"}:
        return cast(EncoderVersion, raw)
    raise ValueError("Value model senza metadata.encoder_version valido")
