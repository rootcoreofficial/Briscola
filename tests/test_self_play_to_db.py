"""
Test per self-play → SQLite (event log).

Scopo:
- verificare che lo script self-play scriva eventi coerenti nel DB
- verificare che i metadati stabili (code_version/rules_version) finiscano nella tabella `games`

Questo test è volutamente piccolo (1 partita) per restare veloce.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from briscola_ai.versioning import get_rules_version

# Marker per cicli rapidi locali: `pytest -m "not slow"` / `-m "not numba"`.
pytestmark = pytest.mark.slow


def test_self_play_writes_games_metadata(tmp_path: Path) -> None:
    """
    Esegue self-play su un DB temporaneo e verifica:
    - una riga in `games`
    - colonne `code_version` e `rules_version` valorizzate (best-effort)
    """
    db_path = tmp_path / "self_play.sqlite3"

    # Eseguiamo lo script come processo separato (come faremmo da CLI):
    # i file in `scripts/` non sono installati come package Python.
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "self_play_to_db.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--db",
            str(db_path),
            "--num-games",
            "1",
            "--seed",
            "123",
            "--num-players",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Self-play completato" in proc.stdout

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT num_players, seed, code_version, rules_version FROM games LIMIT 1;").fetchone()
    finally:
        conn.close()

    assert row is not None
    num_players, seed, code_version, rules_version = row
    assert num_players == 2
    assert isinstance(seed, int)
    assert isinstance(code_version, str)
    assert rules_version == get_rules_version()

    # Verifichiamo anche che l'evento `game_created` includa metadati sugli agenti usati.
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT payload_json FROM events WHERE event_type = 'game_created' ORDER BY id LIMIT 1;"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    payload = json.loads(row[0])
    assert payload["num_players"] == 2
    assert payload["agents"] == {"0": "random", "1": "random"}
