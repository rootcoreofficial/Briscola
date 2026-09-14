"""Primitive sicure per checkpoint e resume esatto del trainer A2C.

Il file modello resta un normale ``.npz`` caricabile dal runtime, ma i checkpoint
contengono array aggiuntivi per Adam e uno stato JSON senza pickle. La scrittura avviene
nella stessa directory e viene pubblicata con ``os.replace`` soltanto dopo che NumPy ha
chiuso correttamente l'archivio.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

A2C_RESUME_SCHEMA = "briscola.a2c_resume.v1"


def json_compatible(value: Any) -> Any:
    """Converte stati RNG NumPy/Python in tipi JSON rigorosi, senza pickle."""
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Valore non serializzabile nel checkpoint A2C: {type(value).__name__}")


def tuple_tree(value: Any) -> Any:
    """Ricostruisce le tuple annidate richieste da ``random.Random.setstate``."""
    if isinstance(value, list):
        return tuple(tuple_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: tuple_tree(item) for key, item in value.items()}
    return value


def canonical_json(payload: dict[str, object]) -> str:
    """JSON stabile usato sia nel file sia nel fingerprint di configurazione."""
    return json.dumps(
        json_compatible(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_fingerprint(payload: dict[str, object]) -> str:
    """SHA-256 dei soli parametri che possono cambiare la traiettoria del training."""
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def atomic_savez(path: Path, /, **arrays: Any) -> None:
    """Salva un archivio NPZ atomico nella stessa directory della destinazione."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(raw_tmp_path)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_resume_json(raw: object) -> dict[str, object]:
    """Parsa e valida superficialmente lo stato incorporato nel checkpoint."""
    try:
        payload = json.loads(str(np.asarray(raw).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("resume_state_json del checkpoint non è JSON valido") from exc
    if not isinstance(payload, dict) or payload.get("schema") != A2C_RESUME_SCHEMA:
        raise ValueError(f"Checkpoint senza schema resume {A2C_RESUME_SCHEMA!r}")
    return payload
