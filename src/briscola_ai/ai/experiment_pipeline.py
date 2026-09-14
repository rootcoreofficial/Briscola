"""
Utilities per una pipeline “training + evaluation” riproducibile.

Scopo
-----
Quando iteriamo rapidamente su modelli RL (A2C/PG) è facile:
- dimenticare un benchmark (es. holdout);
- salvare risultati con nomi incoerenti;
- perdere traccia della configurazione che ha generato un certo `.npz`.

Questo modulo contiene helper “core” (importabili e testabili) usati da uno script CLI
che esegue l'intera pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

AlgoName = Literal["a2c", "pg", "bc"]


def utc_now_iso() -> str:
    """Ritorna un timestamp ISO-8601 in UTC (utile per manifest)."""
    return datetime.now(tz=UTC).isoformat()


def _slugify(text: str) -> str:
    """
    Trasforma una stringa in uno slug safe per filename.

    Regola:
    - lowercase
    - spazi e caratteri non alfanumerici -> underscore
    - compressione underscore ripetuti
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "x"


def _format_num_games_short(n: int) -> str:
    """Formato compatto: 200000 -> 200k, 1500000 -> 1.5m."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m".replace(".0m", "m")
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def build_experiment_name(
    *,
    algo: AlgoName,
    num_games: int,
    seed: int,
    opponent: str | None,
    opponent_mix: str | None,
    tag: str | None = None,
) -> str:
    """
    Costruisce un nome deterministico per l'esperimento (usato nei path di output).

    Idea:
    il nome deve essere abbastanza descrittivo, ma soprattutto stabile e “greppable”.
    """
    algo_bit = _slugify(algo)
    games_bit = _format_num_games_short(int(num_games))
    seed_bit = f"seed{int(seed)}"

    opp_bit = ""
    if opponent_mix and opponent_mix.strip():
        opp_bit = "mix_" + _slugify(opponent_mix)
    elif opponent and opponent.strip():
        opp_bit = _slugify(opponent)
    else:
        opp_bit = "opp"

    bits = [algo_bit, opp_bit, f"{games_bit}g", seed_bit]
    if tag and tag.strip():
        bits.append(_slugify(tag))
    return "_".join(bits)


@dataclass(frozen=True, slots=True)
class BestMetric:
    """
    Score “singolo numero” estratto dalla evaluation matrix.

    Convenzione:
    - `avg_diff`: differenza punti media (model - opponent) in seat-fair
    """

    opponent: str
    suite: str
    benchmark: str
    avg_diff: float


def extract_best_metric_from_matrix_json(
    matrix_json: dict[str, Any],
    *,
    benchmark: str,
    opponent: str = "heuristic_v1",
    suite: str = "holdout",
) -> BestMetric:
    """
    Estrae una metrica dal JSON prodotto da `EvaluationMatrix.to_json_dict()`.

    Usiamo come default:
    - opponent = `heuristic_v1` (baseline “forte” del progetto)
    - suite = `holdout` (robustezza)
    - metrica = `avg_point_diff_agent_a_minus_agent_b`
    """
    rows = matrix_json.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Matrix JSON invalido: `rows` non è una lista")

    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("opponent") != opponent:
            continue
        suite_obj = row.get("suite")
        if not isinstance(suite_obj, dict) or suite_obj.get("name") != suite:
            continue
        stats = row.get("stats")
        if not isinstance(stats, dict):
            continue
        raw = stats.get("avg_point_diff_agent_a_minus_agent_b")
        if isinstance(raw, (int, float)):
            return BestMetric(opponent=opponent, suite=suite, benchmark=benchmark, avg_diff=float(raw))

    raise ValueError(f"Metrica non trovata: opponent={opponent!r} suite={suite!r}")


def read_json(path: Path) -> dict[str, Any]:
    """Legge un file JSON e ritorna un dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Scrive JSON pretty-printed (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
