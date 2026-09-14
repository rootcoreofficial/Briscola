"""
Catalogo locale di modelli `.npz` (per giocare dalla UI contro una policy addestrata).

Contesto didattico
------------------
Nel repo addestriamo policy (BC / PG / A2C) e le salviamo come file `.npz` (NumPy archive),
con una chiave `metadata_json` che contiene informazioni utili (algoritmo, seed, num_games, ecc.).

Quando vogliamo *giocare* contro questi modelli nella UI, il browser non deve passare
path arbitrari al server (sarebbe un problema di sicurezza). Invece:
- il server mantiene una directory "whitelist" che contiene i modelli selezionabili;
- la UI sceglie un `model_id` (path relativo dentro quella directory);
- il backend risolve l'id in un path in modo sicuro (no `..`, no assoluti, no escape dalla root).

Questo modulo incapsula:
- la scansione dei `.npz` disponibili in una directory;
- il parsing best-effort di `metadata_json`;
- un riassunto (label + descrizione in italiano) per la UI;
- la risoluzione sicura `model_id` -> `Path`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4
from .bc_model import load_bc_model_npz


@dataclass(frozen=True, slots=True)
class LocalModelSpec:
    """
    Metadati di un modello locale selezionabile.

    Campi:
        id: identificatore stabile usato dalla UI (path relativo POSIX dentro la directory modelli).
        filename: solo il nome file (debug/UX).
        label: testo corto per il select.
        description_it: descrizione breve in italiano.
        metadata: dict con i metadati raw (best effort) letti da `metadata_json`.
        last_modified_utc: timestamp ISO (UTC) dell'ultima modifica del file.
    """

    id: str
    filename: str
    label: str
    description_it: str
    metadata: dict[str, Any]
    last_modified_utc: str
    is_compatible: bool
    compatibility_reason_it: str | None


def get_models_dir_from_env() -> Path:
    """
    Directory root che contiene i modelli selezionabili.

    Regola:
    - se `BRISCOLA_MODELS_DIR` è impostata, usiamo quel path;
    - altrimenti usiamo `./data/models` se esiste, come convenzione più pulita;
    - fallback: `./data` (compatibilità con workflow precedenti).
    """
    raw = os.getenv("BRISCOLA_MODELS_DIR", "").strip()
    if raw:
        return Path(raw)
    default_models = Path("./data/models")
    if default_models.exists():
        return default_models
    return Path("./data")


def resolve_model_path(models_dir: Path, model_id: str) -> Path:
    """
    Risolve un `model_id` (path relativo) in un path su disco, in modo sicuro.

    Security:
    - rifiuta path assoluti;
    - rifiuta path che "escono" dalla root (`..`/symlink escape);
    - accetta solo `.npz`.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("ai_model_id mancante o vuoto")

    rel = Path(model_id)
    if rel.is_absolute():
        raise ValueError("ai_model_id non può essere un path assoluto")

    root = models_dir.resolve()
    candidate = (root / rel).resolve()

    try:
        candidate.relative_to(root)
    except Exception as exc:
        raise ValueError("ai_model_id non valido (path traversal)") from exc

    if candidate.suffix.lower() != ".npz":
        raise ValueError("ai_model_id deve puntare a un file .npz")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    return candidate


def validate_model_compatible_for_ui(path: Path) -> None:
    """
    Verifica che un modello `.npz` sia compatibile con l'agent `bc_model` (UI).

    Criteri:
    - file caricato correttamente (chiavi/shape coerenti);
    - `feature_dim` coerente con uno degli encoder 2-player supportati:
      - v1 (248): observation "istantanea" (mano+tavolo+briscola+scalari)
      - v2 (288): v1 + `seen_cards_onehot[40]` (storia pubblica, card counting lecito)
      - v3 (310): v2 + feature strategiche aggregate (briscole/carichi ignoti, fase, presa corrente)
      - v4 (369): v3 + memoria delle prese (aggregati comportamento avversario + ultime 4 prese)
      - v4+belief (409): v4 + 40 probabilità della belief network embedded nell'artefatto

    Nota:
    se in futuro introduciamo encoder diversi (o 4-player), questa funzione dovrà considerare
    anche `metadata.encoder` e/o `metadata.num_players`.
    """
    model = load_bc_model_npz(path)
    supported = {
        int(FEATURE_DIM_2P_V1),
        int(FEATURE_DIM_2P_V2),
        int(FEATURE_DIM_2P_V3),
        int(FEATURE_DIM_2P_V4),
        int(FEATURE_DIM_2P_V4) + 40,  # policy con belief embedded (il loader valida le chiavi belief_*)
    }
    if int(model.feature_dim) not in supported:
        raise ValueError(
            "Model feature_dim mismatch: "
            f"model={int(model.feature_dim)} expected={int(FEATURE_DIM_2P_V1)} (v1), "
            f"{int(FEATURE_DIM_2P_V2)} (v2), {int(FEATURE_DIM_2P_V3)} (v3), "
            f"{int(FEATURE_DIM_2P_V4)} (v4) o {int(FEATURE_DIM_2P_V4) + 40} (v4+belief)."
        )


def _parse_metadata_json(raw: Any) -> dict[str, Any]:
    """Parsa `metadata_json` salvato in `.npz` (best effort)."""
    if raw is None:
        return {}
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


def _has_supported_weights(npz_keys: Iterable[str]) -> bool:
    """
    Heuristica minima per capire se un `.npz` *sembra* una policy compatibile.

    Nota:
    non carichiamo qui il modello completo (costa e può fallire per shape); per il catalogo
    ci basta filtrare i file "chiaramente non modello".
    """
    keys = set(npz_keys)
    if {"w", "b"}.issubset(keys):
        return True
    return {"w1", "b1", "w2", "b2"}.issubset(keys)


def _format_short_algorithm(metadata: dict[str, Any]) -> str:
    train = metadata.get("train")
    if isinstance(train, dict):
        algo = train.get("algorithm")
        if isinstance(algo, str) and algo.strip():
            return algo.strip().upper()

    fmt = metadata.get("format")
    if isinstance(fmt, str):
        f = fmt.lower()
        if "a2c" in f:
            return "A2C"
        if "pg" in f:
            return "PG"
        if "bc" in f:
            return "BC"
    return ""


def _format_short_training_size(metadata: dict[str, Any]) -> str:
    train = metadata.get("train")
    if not isinstance(train, dict):
        return ""
    num_games = train.get("num_games")
    if isinstance(num_games, int) and num_games > 0:
        if num_games >= 1_000_000:
            return f"{num_games / 1_000_000:.1f}M game"
        if num_games >= 1_000:
            return f"{num_games / 1_000:.0f}k game"
        return f"{num_games} game"
    return ""


def _format_short_opponent(metadata: dict[str, Any]) -> str:
    mix = metadata.get("opponent_mix")
    if isinstance(mix, dict):
        parts: list[str] = []
        for name, weight in mix.items():
            if not isinstance(name, str):
                continue
            if isinstance(weight, (int, float)):
                parts.append(f"{name} {weight:.2f}")
        if parts:
            return "mix(" + ", ".join(parts) + ")"

    opp = metadata.get("opponent")
    if isinstance(opp, str) and opp.strip():
        return opp.strip()
    return ""


def _summarize_model_it(filename: str, metadata: dict[str, Any]) -> tuple[str, str]:
    """
    Costruisce `label` e `description_it` per la UI.

    Regola:
    - se `metadata.label`/`metadata.description_it` esistono, le usiamo;
    - altrimenti costruiamo una descrizione best-effort dai campi noti.
    """
    label = ""
    if isinstance(metadata.get("label"), str):
        label = metadata["label"].strip()
    if not label:
        algo = _format_short_algorithm(metadata)
        size = _format_short_training_size(metadata)
        bits = " ".join(b for b in [algo, size] if b)
        label = bits if bits else filename

    description_it = ""
    if isinstance(metadata.get("description_it"), str):
        description_it = metadata["description_it"].strip()

    if not description_it:
        algo = _format_short_algorithm(metadata)
        size = _format_short_training_size(metadata)
        opp = _format_short_opponent(metadata)
        shaping = metadata.get("reward_shaping")
        shaping_txt = ""
        if isinstance(shaping, str) and shaping.strip():
            shaping_txt = "reward shaping" if "shaping" in shaping.lower() else shaping.strip()

        parts: list[str] = []
        if algo:
            parts.append(f"Policy addestrata con {algo}")
        if size:
            parts.append(f"su {size}")
        if opp:
            parts.append(f"contro {opp}")
        if shaping_txt:
            parts.append(f"con {shaping_txt}")

        description_it = ". ".join(parts).strip()
        if description_it:
            description_it += "."
        else:
            description_it = "Modello locale salvato in formato .npz (metadati non disponibili)."

    return label, description_it


def list_local_models(models_dir: Path, *, recursive: bool = False) -> list[LocalModelSpec]:
    """
    Scansiona una directory e ritorna i modelli `.npz` selezionabili.

    Note:
    - ordine stabile (per UX): sort per `id`;
    - best effort: file corrotti o non compatibili vengono ignorati.
    """
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    root = models_dir.resolve()
    candidates = root.rglob("*.npz") if recursive else root.glob("*.npz")

    out: list[LocalModelSpec] = []
    for path in candidates:
        metadata: dict[str, Any] = {}
        is_compatible = False
        compatibility_reason_it: str | None = None

        try:
            with np.load(path) as data:
                if "metadata_json" in data:
                    metadata = _parse_metadata_json(data["metadata_json"])
                if metadata.get("format") in ("value_mlp_v1", "belief_mlp_v1"):
                    # Value e belief model sono asset interni degli agenti search (lookahead/PIMC):
                    # hanno le stesse shape di una policy MLP ma NON sanno giocare — mai in catalogo.
                    continue
                if not _has_supported_weights(data.keys()):
                    compatibility_reason_it = (
                        "File `.npz` non riconosciuto come policy: mancano le chiavi dei pesi "
                        "attese (w/b oppure w1/b1/w2/b2)."
                    )
                else:
                    try:
                        validate_model_compatible_for_ui(path)
                        is_compatible = True
                    except Exception as exc:
                        compatibility_reason_it = str(exc)
        except Exception as exc:
            compatibility_reason_it = f"Impossibile leggere il file `.npz`: {exc}"

        try:
            rel_id = path.resolve().relative_to(root).as_posix()
        except Exception:
            continue

        label, description_it = _summarize_model_it(path.name, metadata)
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        out.append(
            LocalModelSpec(
                id=rel_id,
                filename=path.name,
                label=label,
                description_it=description_it,
                metadata=metadata,
                last_modified_utc=mtime,
                is_compatible=is_compatible,
                compatibility_reason_it=compatibility_reason_it,
            )
        )

    out.sort(key=lambda m: m.id)
    return out
