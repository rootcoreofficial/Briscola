"""
Test di integrazione minimale per l'event log SQLite.

Perché esiste questo test?
--------------------------
L'event log è una feature "da laboratorio": serve a rendere riproducibili e
osservabili le partite quando inizieremo a costruire dataset per ML.

Qui non testiamo la UI né il training:
verifichiamo solo che, quando il DB è configurato via env, il backend scriva
almeno alcuni eventi base (creazione partita e azione giocata).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from briscola_ai.backend import server
from briscola_ai.backend.game_store import InMemoryGameSessionStore
from briscola_ai.main import app as main_app


def _wait_for_human_turn(client: TestClient, game_id: str, *, prefix: str = "") -> dict:
    """Legge lo stato finché il player 0 è di mano: con avvio casuale può partire l'IA."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        obs = client.get(f"{prefix}/games/{game_id}", params={"player_index": 0}).json()
        if obs.get("my_turn"):
            return obs
        time.sleep(0.05)
    raise AssertionError("il turno umano non è arrivato dopo l'avvio automatico IA")


def test_event_log_writes_basic_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Abilita l'event log via env e verifica che il DB contenga eventi base.

    Nota:
    - Usiamo `TestClient` in un context manager per garantire startup/shutdown (lifespan).
    - Il path è sotto tempdir per evitare side effects nel repository.
    """
    # `tmp_path` è un path unico per test, già isolato: non serve creare sottocartelle.
    db_path = tmp_path / "events.sqlite3"
    monkeypatch.setenv("BRISCOLA_EVENT_DB_PATH", str(db_path))

    # Puliamo lo stato globale (come in `tests/test_api_integration.py`) per evitare
    # interferenze con altri test che usano `briscola_ai.backend.server`.
    server.game_store = InMemoryGameSessionStore()
    server.game_timestamps.clear()
    server.game_data.clear()

    with TestClient(server.app) as client:
        create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
        assert create.status_code == 200
        game_id = create.json()["game_id"]

        obs = _wait_for_human_turn(client, game_id)
        action = client.post(
            f"/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
        )
        assert action.status_code == 200

    # Verifica contenuto DB dopo lo shutdown (connessione chiusa e flush su disco).
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT event_type FROM events WHERE game_id = ? ORDER BY id ASC;",
            (game_id,),
        ).fetchall()
    finally:
        conn.close()

    event_types = [r[0] for r in rows]
    assert "game_created" in event_types
    assert "action_play_card" in event_types


def test_event_log_marks_user_abandoned_games(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L'abbandono volontario deve restare distinguibile dai timeout automatici nei dati."""
    db_path = tmp_path / "events.sqlite3"
    monkeypatch.setenv("BRISCOLA_EVENT_DB_PATH", str(db_path))

    server.game_store = InMemoryGameSessionStore()
    server.game_timestamps.clear()
    server.game_data.clear()

    with TestClient(server.app) as client:
        create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
        assert create.status_code == 200
        game_id = create.json()["game_id"]

        abandon = client.post(f"/games/{game_id}/abandon", json={"player_index": 0})
        assert abandon.status_code == 200

    conn = sqlite3.connect(db_path)
    try:
        game_row = conn.execute(
            "SELECT aborted_reason FROM games WHERE game_id = ?;",
            (game_id,),
        ).fetchone()
        event_rows = conn.execute(
            "SELECT event_type FROM events WHERE game_id = ? ORDER BY id ASC;",
            (game_id,),
        ).fetchall()
    finally:
        conn.close()

    assert game_row == ("user_abandoned",)
    assert "game_aborted" in [row[0] for row in event_rows]


def test_event_log_works_when_api_is_mounted_under_main_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regressione: quando l'API è montata sotto `briscola_ai.main:app`, vogliamo che il lifespan
    inizializzi comunque event log e cleanup.

    Questo test replica più da vicino il modo in cui si avvia il server con `briscola-server`.
    """
    db_path = tmp_path / "events_mounted.sqlite3"
    monkeypatch.setenv("BRISCOLA_EVENT_DB_PATH", str(db_path))

    # Pulizia stato globale per evitare interferenze.
    server.game_store = InMemoryGameSessionStore()
    server.game_timestamps.clear()
    server.game_data.clear()

    with TestClient(main_app) as client:
        create = client.post("/api/games", json={"num_players": 2, "player_names": ["A", "B"]})
        assert create.status_code == 200
        game_id = create.json()["game_id"]

        obs = _wait_for_human_turn(client, game_id, prefix="/api")
        action = client.post(
            f"/api/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
        )
        assert action.status_code == 200

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT event_type FROM events WHERE game_id = ? ORDER BY id ASC;",
            (game_id,),
        ).fetchall()
    finally:
        conn.close()

    event_types = [r[0] for r in rows]
    assert "game_created" in event_types
    assert "action_play_card" in event_types


def test_count_games_tracks_registered_games() -> None:
    """
    `count_games` alimenta la diagnostica `/api/meta` (event_log_games_recorded):
    parte da 0, conta le partite registrate ed e' idempotente rispetto a ensure_game
    ripetute sulla stessa partita.
    """
    from briscola_ai.backend.event_log import EventLog, EventLogConfig

    log = EventLog(EventLogConfig(path=":memory:"))
    assert log.count_games() == 0
    log.ensure_game("g1", num_players=2, seed=1)
    log.ensure_game("g2", num_players=2, seed=2)
    log.ensure_game("g1", num_players=2, seed=1)  # duplicato: non deve contare
    assert log.count_games() == 2


def test_count_games_by_model_reads_game_created_payload() -> None:
    """
    `count_games_by_model` alimenta GET /api/stats/games: raggruppa per ai_model_id
    (fallback ai_agent) letto dal payload di `game_created`, e conta le completate
    via finished_at. Copre anche lo storico perche' legge gli eventi, non uno schema nuovo.
    """
    from briscola_ai.backend.event_log import EventLog, EventLogConfig

    log = EventLog(EventLogConfig(path=":memory:"))

    def created(game_id, model=None, agent="bc_model"):
        log.ensure_game(game_id, num_players=2, seed=1)
        log.log_event(
            game_id,
            "game_created",
            {"ai_agent": agent, "ai_model_id": model},
            server_version=0,
        )

    created("g1", model="best_a2c_v10.npz")
    created("g2", model="best_a2c_v10.npz")
    created("g3", model=None, agent="heuristic_v1")
    log.try_mark_game_finished("g1")

    rows = log.count_games_by_model()
    assert rows is not None
    by = {r["model"]: r for r in rows}
    assert by["best_a2c_v10.npz"]["total"] == 2
    assert by["best_a2c_v10.npz"]["completed"] == 1
    assert by["heuristic_v1"]["total"] == 1
    assert by["heuristic_v1"]["completed"] == 0
