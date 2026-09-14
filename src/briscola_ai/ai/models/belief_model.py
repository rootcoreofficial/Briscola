"""
Belief network: P(carta in mano avversaria | osservazione lecita), salvata in `.npz`.

Ruolo (Fase 2 del piano belief/expert-iteration):
il PIMC storicamente campiona le mani avversarie UNIFORMEMENTE tra le carte ignote.
Questa rete impara da self-play a stimare quanto è probabile che ogni carta ignota sia
in mano all'avversario, osservando solo informazione pubblica (in particolare la storia
delle prese dell'encoder v4). I pesi risultanti guidano le determinizzazioni
(`determinize_observation(card_weights=...)`).

Anti-cheat: la mano avversaria vera è usata SOLO come label a training; a inference la
rete riceve la stessa `PlayerObservation` lecita di qualunque agente.
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
class MLPBeliefModel:
    """
    MLP 1-hidden-layer con 40 uscite sigmoid: probabilità per-carta di essere in mano avversaria.

    Convenzione `.npz` (stessa famiglia di policy/value):
    - `w1`: (D, H), `b1`: (H,)
    - `w2`: (H, 40), `b2`: (40,)
    - `metadata_json`: JSON con `format="belief_mlp_v1"`, `encoder_version`, `feature_dim`.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    metadata: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        """Dimensione feature attesa dall'encoder."""
        return int(self.w1.shape[0])

    @property
    def hidden_dim(self) -> int:
        """Dimensione hidden layer."""
        return int(self.w1.shape[1])

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Logits (40,) prima della sigmoid."""
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        return h @ self.w2 + self.b2

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        """
        Probabilità (40,) che ciascuna carta sia in mano avversaria.

        Nota d'uso: il chiamante deve considerare SOLO le carte ignote (le altre hanno
        verità nota dall'osservazione); la rete è allenata con loss mascherata sulle ignote,
        quindi fuori maschera i valori non sono significativi.
        """
        logits = self.predict_logits(x)
        return 1.0 / (1.0 + np.exp(-logits))


_BELIEF_MODEL_NPZ_CACHE: dict[tuple[str, int, int], MLPBeliefModel] = {}


def load_belief_model_npz(path: str | Path) -> MLPBeliefModel:
    """Carica un belief model `.npz`, con cache basata su path/mtime/size."""
    model_path = Path(path)
    try:
        st = os.stat(model_path)
    except OSError:
        return _load_belief_model_npz_uncached(model_path)
    key = (os.path.abspath(str(model_path)), st.st_mtime_ns, int(st.st_size))
    cached = _BELIEF_MODEL_NPZ_CACHE.get(key)
    if cached is not None:
        return cached
    model = _load_belief_model_npz_uncached(model_path)
    _BELIEF_MODEL_NPZ_CACHE[key] = model
    return model


def _load_belief_model_npz_uncached(path: Path) -> MLPBeliefModel:
    """Carica e valida un belief model `.npz` senza usare cache."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Formato non supportato: {path} (atteso .npz)")

    with np.load(path) as data:
        keys = set(data.keys())
        missing = {"w1", "b1", "w2", "b2"} - keys
        if missing:
            raise ValueError(f"File belief model invalido: mancano chiavi {sorted(missing)}")

        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = _parse_metadata_json(data["metadata_json"])

        fmt = metadata.get("format")
        if fmt != "belief_mlp_v1":
            raise ValueError(f"Formato belief model non supportato: {fmt!r}")

        w1 = np.asarray(data["w1"], dtype=np.float32)
        b1 = np.asarray(data["b1"], dtype=np.float32)
        w2 = np.asarray(data["w2"], dtype=np.float32)
        b2 = np.asarray(data["b2"], dtype=np.float32)

        if w1.ndim != 2 or b1.ndim != 1 or w1.shape[1] != b1.shape[0]:
            raise ValueError(f"Shape invalide: w1={w1.shape} b1={b1.shape}")
        if w2.shape != (w1.shape[1], 40):
            raise ValueError(f"Shape w2 invalida: {w2.shape}, attesa {(w1.shape[1], 40)}")
        if b2.shape != (40,):
            raise ValueError(f"Shape b2 invalida: {b2.shape}, attesa (40,)")

        declared_dim = metadata.get("feature_dim")
        if isinstance(declared_dim, int) and int(declared_dim) != int(w1.shape[0]):
            raise ValueError(f"Feature dim mismatch: metadata={declared_dim} actual={int(w1.shape[0])}")

        encoder_version = metadata.get("encoder_version")
        if not (isinstance(encoder_version, str) and encoder_version in {"v1", "v2", "v3", "v4"}):
            raise ValueError(f"Belief model senza metadata.encoder_version valido: {encoder_version!r}")
        expected = feature_dim_for_encoder_version(cast(EncoderVersion, encoder_version))
        if int(expected) != int(w1.shape[0]):
            raise ValueError(
                f"Encoder/version mismatch: encoder={encoder_version} expected={expected} actual={w1.shape[0]}"
            )

        return MLPBeliefModel(w1=w1, b1=b1, w2=w2, b2=b2, metadata=metadata)


def infer_belief_encoder_version(model: MLPBeliefModel) -> EncoderVersion:
    """Ricava la versione encoder dai metadati del belief model (validata al load)."""
    return cast(EncoderVersion, model.metadata["encoder_version"])
