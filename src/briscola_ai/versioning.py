"""
Versioni “stabili” per dataset e riproducibilità.

Obiettivo
---------
Quando esportiamo dataset (o salviamo eventi su SQLite), vogliamo poter rispondere
a domande come:
- con quale versione del codice è stata generata questa partita?
- con quale versione delle regole (dominio) è stata giocata?

Questo modulo fornisce helper minimali e best-effort:
- `get_code_version()`: versione del pacchetto (SemVer) con override via env
- `get_rules_version()`: versione regole del dominio (stringa)
"""

from __future__ import annotations

import os
import tomllib
from importlib import metadata
from pathlib import Path

from .domain.version import RULES_VERSION


def get_rules_version() -> str:
    """
    Versione delle regole del dominio.

    È una stringa per semplicità (facile da serializzare e confrontare).
    """
    return RULES_VERSION


def _version_from_pyproject() -> str | None:
    """
    Legge `[project].version` dal `pyproject.toml` del checkout sorgente, se presente.

    Motivo: in un'installazione *editable* i metadata del pacchetto restano fermi all'ultimo
    `pip install` e non seguono i bump di `pyproject.toml` finché non reinstalli. Preferire
    `pyproject` quando esiste rende la versione accurata in sviluppo (footer UI, `/version`,
    cache-busting) senza reinstallare. In un wheel installato (prod) non c'è `pyproject.toml`
    accanto al package → si ricade su `metadata`.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as fp:
            data = tomllib.load(fp)
    except OSError, tomllib.TOMLDecodeError:
        return None
    version = data.get("project", {}).get("version")
    return str(version).strip() if version else None


def get_code_version() -> str:
    """
    Versione del codice (best-effort).

    Ordine di precedenza:
    1) `BRISCOLA_CODE_VERSION` (env) → utile in deploy/CI per iniettare un commit hash o build id.
    2) `pyproject.toml` del checkout sorgente (accurato anche in editable senza reinstallare).
    3) versione del pacchetto installato (metadata del wheel, in prod).
    4) fallback `"unknown"` se non disponibile.
    """
    override = os.getenv("BRISCOLA_CODE_VERSION")
    if override:
        return override.strip()

    from_pyproject = _version_from_pyproject()
    if from_pyproject:
        return from_pyproject

    try:
        return metadata.version("briscola-ai")
    except metadata.PackageNotFoundError:
        return "unknown"
