"""
Test di integrazione per l'API HTTP/WebSocket.

Questi test usano `fastapi.testclient.TestClient` e, di conseguenza, lavorano
contro lo stato globale mantenuto in `briscola_ai.backend.server`.

Nota: puliamo sempre lo stato globale con una fixture `autouse` per evitare
interferenze tra casi di test.
"""

import asyncio
import json
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4
from briscola_ai.backend import server
from briscola_ai.backend.event_log import EventLog, EventLogConfig
from briscola_ai.backend.game_store import GameSession, InMemoryGameSessionStore
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.state import GameState, PlayerState
from briscola_ai.main import app as main_app


def _reset_server_state() -> None:
    """Resetta lo store sessioni (in-memory) e i buffer per-replica in `server`."""
    server.game_store = InMemoryGameSessionStore()
    server.game_timestamps.clear()
    server.game_data.clear()


def _get_session(game_id: str) -> GameSession | None:
    """Helper sincrono per leggere una sessione dallo store (per i test)."""
    return asyncio.run(server.game_store.get(game_id))


def _put_state(game_id: str, state: GameState, *, version: int = 0) -> None:
    """
    Helper per i test che forzano uno stato finale: legge la sessione esistente (per riusarne
    `ai_seats`/`action_seed`) e la riscrive con lo stato/versione forniti.
    """

    async def _do() -> None:
        existing = await server.game_store.get(game_id)
        if existing is not None:
            existing.state = state
            existing.version = version
            await server.game_store.set(existing)
        else:
            now = datetime.now().isoformat()
            await server.game_store.set(
                GameSession(
                    game_id=game_id,
                    state=state,
                    version=version,
                    ai_seats={},
                    action_seed=0,
                    created_at=now,
                    updated_at=now,
                )
            )

    asyncio.run(_do())


def _force_starting_player(monkeypatch: pytest.MonkeyPatch, player_index: int) -> None:
    """Rende deterministico il sorteggio del primo giocatore nei test API."""
    monkeypatch.setattr(server, "_choose_starting_player", lambda _seed, _num_players: player_index)


@pytest.fixture(autouse=True)
def _clean_server_state() -> Generator[None]:
    """
    I test d'integrazione usano stato globale in `briscola_ai.backend.server`.

    Per evitare interferenze tra test, resettiamo lo store sessioni e i buffer in memoria
    prima/dopo ogni test.
    """
    _reset_server_state()
    yield
    _reset_server_state()


def _write_dummy_bc_model_npz(path: Path) -> None:
    """
    Crea un file `.npz` minimo compatibile con `BCModelAgent`.

    Nota:
    Usiamo `D=248` che è la dimensione feature attuale di `encode_player_observation_2p` (v1).
    """
    d = 248
    h = 8
    rng = np.random.default_rng(0)
    w1 = rng.normal(size=(d, h)).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = rng.normal(size=(h, 40)).astype(np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "action_dim": 40,
        "train": {"algorithm": "bc", "num_games": 1234},
        "description_it": "Modello dummy per test (non usare in produzione).",
    }
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(metadata, ensure_ascii=False))


def _write_dummy_bc_model_npz_with_feature_dim(
    path: Path,
    *,
    feature_dim: int,
    metrics: list[dict[str, float]] | None = None,
) -> None:
    """Come `_write_dummy_bc_model_npz`, ma con feature_dim configurabile (per test compatibilità)."""
    d = int(feature_dim)
    h = 8
    rng = np.random.default_rng(0)
    w1 = rng.normal(size=(d, h)).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = rng.normal(size=(h, 40)).astype(np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "action_dim": 40,
        "train": {"algorithm": "bc", "num_games": 1},
        "label": "Dummy",
        "description_it": "Modello dummy per test compatibilità.",
    }
    if metrics is not None:
        metadata["metrics"] = metrics
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(metadata, ensure_ascii=False))


def _write_dummy_value_model_npz(path: Path) -> None:
    """Crea un value model minimo: asset interno per `bc_model_value_lookahead_8x8`, non policy UI."""
    d = int(FEATURE_DIM_2P_V3)
    h = 4
    metadata = {
        "format": "value_mlp_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "encoder_version": "v3",
        "target": "residual",
        "target_scale": 120.0,
    }
    np.savez(
        path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h,), dtype=np.float32),
        b2=np.asarray([0.0], dtype=np.float32),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )


def _write_dummy_belief_model_npz(path: Path) -> None:
    """Crea la belief minima richiesta dalle varianti PIMC belief dell'API."""
    d = int(FEATURE_DIM_2P_V4)
    h = 4
    np.savez(
        path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "belief_mlp_v1",
                "feature_dim": d,
                "hidden_dim": h,
                "encoder_version": "v4",
            },
            ensure_ascii=False,
        ),
    )


def test_backend_root_healthcheck() -> None:
    """Smoke test: l'endpoint root del backend risponde e contiene un messaggio."""
    client = TestClient(server.app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"]


def test_meta_exposes_event_log_runtime_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`GET /meta` e `/version` distinguono configurazione logging e logger realmente montato."""
    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "dataset")
    server.app.state.event_log = None
    client = TestClient(server.app)
    r = client.get("/meta")
    assert r.status_code == 200
    payload = r.json()
    assert payload["event_log_mode"] == "dataset"
    assert payload["dataset_requires_consent"] is True
    assert payload["event_log_available"] is False
    assert payload["event_log_healthy"] is False
    assert payload["event_log_backend"] is None
    assert payload["event_log_database_name"] is None
    assert payload["event_log_database_host"] is None
    assert payload["debug_state_endpoint_enabled"] is False

    log = EventLog(EventLogConfig(path=str(tmp_path / "events.sqlite3")))
    server.app.state.event_log = log
    try:
        r_available = client.get("/meta")
        assert r_available.status_code == 200
        payload_available = r_available.json()
        assert payload_available["event_log_available"] is True
        assert payload_available["event_log_healthy"] is True
        assert payload_available["event_log_backend"] == "sqlite"
        assert payload_available["event_log_database_name"] == "events.sqlite3"
        assert payload_available["event_log_database_host"] is None

        version = TestClient(main_app).get("/version")
        assert version.status_code == 200
        version_payload = version.json()
        assert version_payload["event_log_available"] is True
        assert version_payload["event_log_healthy"] is True
        assert version_payload["event_log_backend"] == "sqlite"
        assert version_payload["event_log_database_name"] == "events.sqlite3"
        assert version_payload["event_log_database_host"] is None
    finally:
        log.close()
        server.app.state.event_log = None

    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "debug")
    r2 = client.get("/meta")
    assert r2.status_code == 200
    payload2 = r2.json()
    assert payload2["event_log_mode"] == "debug"
    assert payload2["dataset_requires_consent"] is False

    monkeypatch.setenv("BRISCOLA_DEBUG_STATE_ENDPOINT", "unsafe-full-state")
    r3 = client.get("/meta")
    assert r3.status_code == 200
    assert r3.json()["debug_state_endpoint_enabled"] is True


def test_create_game_requires_consent_in_dataset_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """In `event_log_mode=dataset` il backend deve richiedere consenso esplicito."""
    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "dataset")
    client = TestClient(server.app)

    missing = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "Bob"]},
    )
    assert missing.status_code == 400
    assert "Consenso" in missing.json()["detail"]

    ok = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "Bob"], "consent_to_data_collection": True},
    )
    assert ok.status_code == 200


def test_abandon_game_deletes_open_session() -> None:
    """`POST /games/{id}/abandon` deve chiudere una partita aperta senza assegnare un risultato."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["Alice", "IA"]})
    assert create.status_code == 200
    game_id = create.json()["game_id"]
    assert _get_session(game_id) is not None

    abandon = client.post(f"/games/{game_id}/abandon", json={"player_index": 0})
    assert abandon.status_code == 200
    assert abandon.json()["status"] == "abandoned"
    assert _get_session(game_id) is None

    missing = client.get(f"/games/{game_id}", params={"player_index": 0})
    assert missing.status_code == 404


def test_create_game_sets_random_first_player(monkeypatch: pytest.MonkeyPatch) -> None:
    """Il primo giocatore deve derivare dal seed della partita e comparire nei payload pubblici."""
    _force_starting_player(monkeypatch, 1)
    client = TestClient(server.app)

    create = client.post("/games", json={"num_players": 2, "player_names": ["Alice", "IA"]})
    assert create.status_code == 200
    payload = create.json()
    assert payload["first_player"] == 1
    assert payload["current_turn"] == 1

    session = _get_session(payload["game_id"])
    assert session is not None
    assert session.state.first_player == 1
    assert session.state.current_turn == 1
    assert server._initial_ai_start_delay_seconds(session, human_player_index=0) == pytest.approx(
        server._AI_STARTS_PRESENTATION_DELAY_SECONDS
    )

    session.version = 1
    assert server._initial_ai_start_delay_seconds(session, human_player_index=0) == 0.0
    session.version = 0

    obs = client.get(f"/games/{payload['game_id']}", params={"player_index": 0}).json()
    assert obs["first_player"] == 1


def test_list_ai_agents_exposes_metadata_in_italian() -> None:
    """`GET /ai/agents` deve esporre nomi e descrizioni (in italiano) per la UI."""
    client = TestClient(server.app)
    r = client.get("/ai/agents")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    assert isinstance(payload.get("common_note_it"), str)
    assert payload["common_note_it"]

    agents = payload.get("agents")
    assert isinstance(agents, list)
    assert agents

    by_name = {a["name"]: a for a in agents}
    assert "random" in by_name
    assert "greedy_points" in by_name
    assert "heuristic_v1" in by_name
    assert "heuristic_v2" in by_name
    assert "bc_model" in by_name
    assert "bc_model_hybrid_endgame" in by_name
    assert "bc_model_value_lookahead_8x8" in by_name
    assert "bc_model_pimc_16x8" in by_name
    assert "bc_model_pimc_belief_12x8" in by_name

    assert isinstance(by_name["heuristic_v1"].get("description_it"), str)
    assert by_name["heuristic_v1"]["description_it"]
    assert isinstance(by_name["heuristic_v2"].get("description_it"), str)
    assert by_name["heuristic_v2"]["description_it"]
    assert by_name["bc_model_hybrid_endgame"]["requires_model_selection"] is True
    assert by_name["bc_model_value_lookahead_8x8"]["requires_model_selection"] is True
    assert by_name["bc_model_value_lookahead_8x8"]["requires_model_id"] == "value_v0_h128_clean50k_seed20260701.npz"
    assert by_name["bc_model_pimc_16x8"]["requires_model_selection"] is True


def test_recommended_12x8_requires_belief_and_can_create_a_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Il 12x8 ufficiale attraversa catalogo e API quando la belief richiesta è presente."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "student.npz", feature_dim=int(FEATURE_DIM_2P_V4))
    client = TestClient(server.app)

    recommended_without_belief = {a["name"]: a for a in client.get("/ai/agents").json()["agents"]}[
        "bc_model_pimc_belief_12x8"
    ]
    assert recommended_without_belief["available"] is False
    assert recommended_without_belief["requires_model_present"] is False

    _write_dummy_belief_model_npz(tmp_path / "belief_v0_h128_50k_seed20260702.npz")
    by_name = {a["name"]: a for a in client.get("/ai/agents").json()["agents"]}
    recommended = by_name["bc_model_pimc_belief_12x8"]
    assert recommended["available"] is True
    assert recommended["requires_model_selection"] is True
    assert recommended["requires_model_id"] == "belief_v0_h128_50k_seed20260702.npz"

    created = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_pimc_belief_12x8",
            "ai_model_id": "student.npz",
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["ai_agent"] == "bc_model_pimc_belief_12x8"
    assert payload["ai_model_id"] == "student.npz"
    session = _get_session(payload["game_id"])
    assert session is not None
    assert session.ai_seats[1].agent_name == "bc_model_pimc_belief_12x8"
    assert session.ai_seats[1].model_id == "student.npz"


def test_list_ai_agents_reports_availability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`available` deve riflettere la presenza del modello richiesto (no opzioni rotte nella UI)."""
    # Directory modelli vuota: nessun modello compatibile, niente best_a2c.npz.
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    client = TestClient(server.app)
    by_name = {a["name"]: a for a in client.get("/ai/agents").json()["agents"]}

    # Agenti senza dipendenze da modello: sempre disponibili.
    assert by_name["random"]["available"] is True
    assert by_name["heuristic_v1"]["available"] is True
    assert by_name["hybrid_endgame"]["available"] is True

    # Dipendono da best_a2c.npz (assente) → non disponibili, con requires_model_id dichiarato.
    assert by_name["hybrid_endgame_best_a2c"]["available"] is False
    assert by_name["hybrid_endgame_best_a2c"]["requires_model_id"] == "best_a2c.npz"

    # bc_model: nessun modello compatibile in dir → non disponibile.
    assert by_name["bc_model"]["available"] is False
    assert by_name["bc_model_hybrid_endgame"]["available"] is False
    assert by_name["bc_model_hybrid_endgame"]["requires_model_selection"] is True
    assert by_name["bc_model_value_lookahead_8x8"]["available"] is False
    assert by_name["bc_model_value_lookahead_8x8"]["requires_model_selection"] is True
    assert by_name["bc_model_value_lookahead_8x8"]["requires_model_present"] is False
    assert by_name["bc_model_pimc_16x8"]["available"] is False
    assert by_name["bc_model_pimc_16x8"]["requires_model_selection"] is True

    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "best_a2c_v6.npz", feature_dim=310)
    by_name_with_model = {a["name"]: a for a in client.get("/ai/agents").json()["agents"]}
    assert by_name_with_model["bc_model"]["available"] is True
    assert by_name_with_model["bc_model_hybrid_endgame"]["available"] is True
    assert by_name_with_model["bc_model_value_lookahead_8x8"]["available"] is False
    assert by_name_with_model["bc_model_pimc_16x8"]["available"] is True

    _write_dummy_value_model_npz(tmp_path / "value_v0_h128_clean50k_seed20260701.npz")
    by_name_with_value = {a["name"]: a for a in client.get("/ai/agents").json()["agents"]}
    assert by_name_with_value["bc_model_value_lookahead_8x8"]["available"] is True
    assert by_name_with_value["bc_model_value_lookahead_8x8"]["requires_model_present"] is True


def test_list_ai_models_returns_model_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`GET /ai/models` deve elencare i modelli `.npz` disponibili (senza path assoluti)."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))

    _write_dummy_bc_model_npz_with_feature_dim(
        tmp_path / "compatible_v1.npz",
        feature_dim=248,
        metrics=[{"episode": 1.0, "avg_score_diff": 2.0}, {"episode": 2.0, "avg_score_diff": 3.0}],
    )
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "compatible_v2.npz", feature_dim=288)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "incompatible.npz", feature_dim=10)
    _write_dummy_value_model_npz(tmp_path / "value_v0_h128_clean50k_seed20260701.npz")

    client = TestClient(server.app)
    r = client.get("/ai/models")
    assert r.status_code == 200

    payload = r.json()
    models = payload.get("models")
    assert isinstance(models, list)
    assert models

    assert "models_dir" not in payload  # non vogliamo esporre path server-side
    assert payload["recommended_model"] == "best_a2c_v15.npz"
    by_id = {m["id"]: m for m in models}
    assert "compatible_v1.npz" in by_id
    assert "compatible_v2.npz" in by_id
    assert "incompatible.npz" in by_id
    assert "value_v0_h128_clean50k_seed20260701.npz" not in by_id

    ok_v1 = by_id["compatible_v1.npz"]
    assert ok_v1["is_compatible"] is True
    assert ok_v1.get("compatibility_reason_it") is None
    assert ok_v1["metadata"]["metrics_count"] == 2
    assert "metrics" not in ok_v1["metadata"]

    ok_v2 = by_id["compatible_v2.npz"]
    assert ok_v2["is_compatible"] is True
    assert ok_v2.get("compatibility_reason_it") is None

    bad = by_id["incompatible.npz"]
    assert bad["is_compatible"] is False
    assert isinstance(bad.get("compatibility_reason_it"), str)
    assert bad["compatibility_reason_it"]


def test_list_ai_models_reports_recommended_model_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Il catalogo modelli espone lo stesso recommended model usato dal provisioning."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("BRISCOLA_DEFAULT_MODEL_ID", "best_a2c_v3.npz")
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "best_a2c_v3.npz", feature_dim=310)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "best_a2c_v4.npz", feature_dim=310)

    payload = TestClient(server.app).get("/ai/models").json()

    assert payload["recommended_model"] == "best_a2c_v3.npz"
    assert {m["id"] for m in payload["models"]} == {"best_a2c_v3.npz", "best_a2c_v4.npz"}


def test_create_game_supports_bc_model_with_ai_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`POST /games` deve supportare `ai_agent=bc_model` + `ai_model_id` (whitelisted)."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "dummy_model.npz", feature_dim=248)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "dummy_model_v2.npz", feature_dim=288)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "bad_model.npz", feature_dim=10)

    client = TestClient(server.app)

    missing = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["A", "B"], "ai_agent": "bc_model"},
    )
    assert missing.status_code == 400
    assert "ai_model_id" in missing.json()["detail"]

    traversal = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "../dummy_model.npz",
        },
    )
    assert traversal.status_code == 400
    assert "path traversal" in traversal.json()["detail"].lower()

    ok = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "dummy_model.npz",
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["ai_agent"] == "bc_model"
    assert payload["ai_model_id"] == "dummy_model.npz"

    ok_v2 = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "dummy_model_v2.npz",
        },
    )
    assert ok_v2.status_code == 200
    payload_v2 = ok_v2.json()
    assert payload_v2["ai_agent"] == "bc_model"
    assert payload_v2["ai_model_id"] == "dummy_model_v2.npz"

    bad = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "bad_model.npz",
        },
    )
    assert bad.status_code == 400
    assert "feature_dim" in bad.json()["detail"]


def test_create_game_supports_bc_model_hybrid_endgame_with_ai_model_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`bc_model_hybrid_endgame` deve usare lo stesso path sicuro di `bc_model`."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "best_a2c_v6.npz", feature_dim=310)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "bad_model.npz", feature_dim=10)

    client = TestClient(server.app)

    missing = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["A", "B"], "ai_agent": "bc_model_hybrid_endgame"},
    )
    assert missing.status_code == 400
    assert "ai_model_id" in missing.json()["detail"]

    traversal = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_hybrid_endgame",
            "ai_model_id": "../best_a2c_v6.npz",
        },
    )
    assert traversal.status_code == 400
    assert "path traversal" in traversal.json()["detail"].lower()

    ok = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_hybrid_endgame",
            "ai_model_id": "best_a2c_v6.npz",
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["ai_agent"] == "bc_model_hybrid_endgame"
    assert payload["ai_model_id"] == "best_a2c_v6.npz"
    session = _get_session(payload["game_id"])
    assert session is not None
    assert session.ai_seats[1].agent_name == "bc_model_hybrid_endgame"
    assert session.ai_seats[1].model_id == "best_a2c_v6.npz"

    bad = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_hybrid_endgame",
            "ai_model_id": "bad_model.npz",
        },
    )
    assert bad.status_code == 400
    assert "feature_dim" in bad.json()["detail"]


def test_create_game_supports_bc_model_value_lookahead_with_ai_model_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`bc_model_value_lookahead_8x8` resta una scelta avanzata ma validata prima della partita."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "best_a2c_v6.npz", feature_dim=310)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "bad_model.npz", feature_dim=10)

    client = TestClient(server.app)

    missing_model_id = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["A", "B"], "ai_agent": "bc_model_value_lookahead_8x8"},
    )
    assert missing_model_id.status_code == 400
    assert "ai_model_id" in missing_model_id.json()["detail"]

    missing_value_model = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_value_lookahead_8x8",
            "ai_model_id": "best_a2c_v6.npz",
        },
    )
    assert missing_value_model.status_code == 400
    assert "value_v0_h128_clean50k_seed20260701" in missing_value_model.json()["detail"]

    _write_dummy_value_model_npz(tmp_path / "value_v0_h128_clean50k_seed20260701.npz")

    ok = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_value_lookahead_8x8",
            "ai_model_id": "best_a2c_v6.npz",
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["ai_agent"] == "bc_model_value_lookahead_8x8"
    assert payload["ai_model_id"] == "best_a2c_v6.npz"
    session = _get_session(payload["game_id"])
    assert session is not None
    assert session.ai_seats[1].agent_name == "bc_model_value_lookahead_8x8"
    assert session.ai_seats[1].model_id == "best_a2c_v6.npz"

    bad = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_value_lookahead_8x8",
            "ai_model_id": "bad_model.npz",
        },
    )
    assert bad.status_code == 400
    assert "feature_dim" in bad.json()["detail"]


def test_create_game_supports_bc_model_pimc_16x8_with_ai_model_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`bc_model_pimc_16x8` deve usare lo stesso path sicuro degli altri agenti `.npz`."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "best_a2c_v6.npz", feature_dim=310)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "bad_model.npz", feature_dim=10)

    client = TestClient(server.app)

    missing = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["A", "B"], "ai_agent": "bc_model_pimc_16x8"},
    )
    assert missing.status_code == 400
    assert "ai_model_id" in missing.json()["detail"]

    traversal = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_pimc_16x8",
            "ai_model_id": "../best_a2c_v6.npz",
        },
    )
    assert traversal.status_code == 400
    assert "path traversal" in traversal.json()["detail"].lower()

    ok = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_pimc_16x8",
            "ai_model_id": "best_a2c_v6.npz",
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["ai_agent"] == "bc_model_pimc_16x8"
    assert payload["ai_model_id"] == "best_a2c_v6.npz"
    session = _get_session(payload["game_id"])
    assert session is not None
    assert session.ai_seats[1].agent_name == "bc_model_pimc_16x8"
    assert session.ai_seats[1].model_id == "best_a2c_v6.npz"

    bad = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model_pimc_16x8",
            "ai_model_id": "bad_model.npz",
        },
    )
    assert bad.status_code == 400
    assert "feature_dim" in bad.json()["detail"]


def test_create_game_get_state_and_play_action_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: crea partita, legge observation e gioca una carta valida."""
    _force_starting_player(monkeypatch, 0)
    client = TestClient(server.app)

    create = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "Bob"]},
    )
    assert create.status_code == 200
    payload = create.json()
    game_id = payload["game_id"]
    assert payload["status"] == "created"
    assert payload["num_players"] == 2
    assert payload["player_names"] == ["Alice", "Bob"]

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state_p0 = client.get(f"/games/{game_id}", params={"player_index": 0})
        assert state_p0.status_code == 200
        obs = state_p0.json()
        if obs["my_turn"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("il turno umano non è arrivato dopo l'avvio automatico IA")

    assert obs["my_index"] == 0
    assert obs["my_turn"] is True
    assert obs["first_player"] in (0, 1)
    assert obs["valid_actions"]
    initial_version = obs.get("server_version", 0)

    action = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
    )
    assert action.status_code == 200
    result = action.json()
    assert "played_card" in result or "error" in result
    assert "error" not in result
    played_card = result.get("played_card")

    state_p0_after = client.get(f"/games/{game_id}", params={"player_index": 0})
    assert state_p0_after.status_code == 200
    obs_after = state_p0_after.json()

    # Con modello server-driven l'IA può giocare "subito" (in un task asincrono) e quindi,
    # tra la POST e questa GET, lo stato potrebbe essere già avanzato oltre il semplice
    # "dopo la carta umana" (es. mano completa + nuova carta IA come prima di mano).
    #
    # Invece di assumere un `my_turn` specifico, verifichiamo invarianti più robuste:
    # - la `server_version` è avanzata almeno di 1 (abbiamo giocato un'azione umana)
    # - la carta giocata non è più nella mano del giocatore 0
    assert obs_after.get("server_version", 0) >= initial_version + 1

    if played_card:
        played_suit = played_card.get("suit")
        played_number = played_card.get("number")
        assert played_suit is not None
        assert played_number is not None

        assert not any(
            (card.get("suit") == played_suit and card.get("number") == played_number) for card in obs_after["my_hand"]
        )


def test_play_action_rejects_wrong_turn() -> None:
    """Regola di turnazione: un giocatore non può giocare quando non è il suo turno."""
    client = TestClient(server.app)

    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]
    session = _get_session(game_id)
    assert session is not None
    wrong_player = 1 - session.state.current_turn

    r = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": wrong_player, "card_index": 0},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "Non è il tuo turno"


def test_play_action_rejects_ai_controlled_player(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un client HTTP non deve poter pilotare manualmente il player controllato dall'IA."""

    async def _no_ai(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(server, "_maybe_ai_turn", _no_ai)
    _force_starting_player(monkeypatch, 0)

    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "IA"]})
    game_id = create.json()["game_id"]

    obs = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
    first = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
    )
    assert first.status_code == 200

    blocked = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 1, "card_index": 0},
    )
    assert blocked.status_code == 400
    assert "controllato dall'IA" in blocked.json()["detail"]


def test_dataset_mode_logs_ai_action_for_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """In modalità dataset salviamo anche una mossa IA minimale e sanificata per audit PIMC."""

    async def _no_auto_ai(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "dataset")
    monkeypatch.setattr(server, "_maybe_ai_turn", _no_auto_ai)
    _force_starting_player(monkeypatch, 0)

    db_path = tmp_path / "events.sqlite3"
    log = EventLog(EventLogConfig(path=str(db_path)))
    server.app.state.event_log = log
    try:
        client = TestClient(server.app)
        create = client.post(
            "/games",
            json={
                "num_players": 2,
                "player_names": ["Nome Umano", "Nome IA"],
                "ai_agent": "random",
                "consent_to_data_collection": True,
            },
        )
        assert create.status_code == 200
        game_id = create.json()["game_id"]

        obs = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
        action = client.post(
            f"/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
        )
        assert action.status_code == 200

        async def _run_one_ai_turn() -> None:
            async with server.game_store.lock(game_id):
                session = await server.game_store.get(game_id)
                assert session is not None
                await server._execute_ai_turn_locked(session, human_player_index=0)

        asyncio.run(_run_one_ai_turn())
    finally:
        log.close()
        server.app.state.event_log = None

    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT event_type, payload_json FROM events WHERE game_id = ? ORDER BY id;",
            (game_id,),
        ).fetchall()
    finally:
        conn.close()

    by_type = {event_type: json.loads(payload_json) for event_type, payload_json in rows}
    assert "human_action" in by_type
    assert "ai_action" in by_type
    assert "ai_card_reveal" not in by_type  # dataset mode resta minimale: niente evento realtime nel DB.
    assert "action_play_card" not in by_type

    human_payload = by_type["human_action"]
    assert isinstance(human_payload["result"], dict)
    assert isinstance(human_payload["result"]["played_card"], dict)

    ai_payload = by_type["ai_action"]
    assert ai_payload["is_ai"] is True
    assert ai_payload["ai_agent"] == "random"
    assert ai_payload["ai_model_id"] is None
    assert isinstance(ai_payload["card_index"], int)
    assert isinstance(ai_payload["observation"], dict)
    assert isinstance(ai_payload["next_observation"], dict)
    assert isinstance(ai_payload["result"], dict)
    assert ai_payload["decision_trace"] is None
    assert ai_payload["observation"]["players"][0]["name"] == "player_0"
    assert ai_payload["observation"]["players"][1]["name"] == "player_1"


def test_main_app_serves_ui_and_mounts_api() -> None:
    """La FastAPI principale deve servire UI statica e montare `/api/`."""
    client = TestClient(main_app)

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers.get("content-type", "")
    assert root.headers.get("cache-control") == "no-cache"
    assert "__BRISCOLA_ASSET_VERSION__" not in root.text
    assert "/static/css/style.css?v=" in root.text
    assert "/static/js/game.js?v=" in root.text

    card_asset = client.get("/static/assets/cards/clubs_1.png")
    assert card_asset.status_code == 200
    assert "image" in card_asset.headers.get("content-type", "")

    api_root = client.get("/api/")
    assert api_root.status_code == 200
    assert api_root.json()["message"]


def test_get_game_state_returns_404_for_unknown_game() -> None:
    """Errore corretto: stato partita inesistente => 404."""
    client = TestClient(server.app)
    r = client.get("/games/not-a-real-game-id")
    assert r.status_code == 404
    assert r.json()["detail"] == "Partita non trovata"


def test_get_game_state_without_player_index_is_forbidden_by_default() -> None:
    """
    Anti-cheat: la vista full-state (mani di tutti + `next_deck_card`) NON deve essere
    raggiungibile da un client qualunque. Senza `BRISCOLA_DEBUG_STATE_ENDPOINT` => 403.
    """
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    r = client.get(f"/games/{game_id}")
    assert r.status_code == 403
    assert "full-state" in r.json()["detail"]

    # La vista fair per-giocatore resta disponibile.
    observation = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
    assert observation["type"] == "observation"
    assert "next_deck_card" not in observation


def test_get_game_state_without_player_index_returns_game_state_dto(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contratto debug (opt-in): `GET /games/{id}` senza player_index ritorna `type: \"game_state\"`."""
    monkeypatch.setenv("BRISCOLA_DEBUG_STATE_ENDPOINT", "unsafe-full-state")
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    r = client.get(f"/games/{game_id}")
    assert r.status_code == 200
    payload = r.json()

    assert payload["type"] == "game_state"
    assert payload["num_players"] == 2
    assert payload["is_team_game"] is False
    assert payload["next_deck_card"] is not None
    assert isinstance(payload.get("players"), list)
    assert len(payload["players"]) == 2

    # Deve includere mani complete (debug/spectator), quindi `hand` è presente.
    assert "hand" in payload["players"][0]
    assert isinstance(payload["players"][0]["hand"], list)


def test_get_game_state_rejects_legacy_boolean_debug_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hardening: un vecchio `=1` non deve riaprire per errore le mani sul deploy pubblico."""
    monkeypatch.setenv("BRISCOLA_DEBUG_STATE_ENDPOINT", "1")
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    assert client.get(f"/games/{game_id}").status_code == 403
    assert client.get("/meta").json()["debug_state_endpoint_enabled"] is False


def test_get_game_state_rejects_invalid_player_index() -> None:
    """Errore corretto: player_index fuori range => 400."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    r = client.get(f"/games/{game_id}", params={"player_index": 999})
    assert r.status_code == 400
    assert "indice giocatore" in r.json()["detail"].lower()


def test_get_game_result_returns_404_for_unknown_game() -> None:
    """Errore corretto: result di partita inesistente => 404."""
    client = TestClient(server.app)
    r = client.get("/games/not-a-real-game-id/result")
    assert r.status_code == 404
    assert r.json()["detail"] == "Partita non trovata"


def test_get_game_result_returns_in_progress_when_game_not_finished() -> None:
    """Se la partita non è terminata, `/result` deve indicare che è in progress."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    # Contratto: anche quando la partita è in corso, il risultato ha shape stabile (DTO).
    assert payload["type"] == "game_result"
    assert payload["game_in_progress"] is True
    assert payload["game_over"] is False
    assert payload["is_team_game"] is False
    assert payload["points"] == {}


def test_get_game_result_2p_finished_returns_stable_dto() -> None:
    """
    `/result` deve avere shape stabile anche a partita terminata (2-player).

    Nota didattica:
    qui usiamo lo stato in memoria del server per creare un end-game "deterministico"
    (evitando di dover giocare 40 mosse via HTTP in un test d'integrazione).
    """
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    # Forziamo uno stato finale coerente.
    _put_state(
        game_id,
        GameState(
            num_players=2,
            is_team_game=False,
            teams=None,
            players=(
                PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=70),
                PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=50),
            ),
            deck=tuple(),
            trump_card=Card(Suit.CUPS, Rank.TWO),
            table_cards=tuple(),
            current_turn=0,
            first_player=0,
            game_over=True,
            winner_index=0,
            winning_team=None,
        ),
        version=123,
    )

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["type"] == "game_result"
    assert payload["server_version"] == 123
    assert payload["game_in_progress"] is False
    assert payload["game_over"] is True
    assert payload["is_team_game"] is False
    assert payload["winner"] == "A"
    assert payload["winner_index"] == 0
    assert payload["points"] == {"A": 70, "B": 50}
    assert payload["point_difference"] == 20


def test_get_game_result_2p_tie_omits_winner_index() -> None:
    """In pareggio 2-player, `winner_index` deve essere None (e quindi non presente nel JSON)."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    _put_state(
        game_id,
        GameState(
            num_players=2,
            is_team_game=False,
            teams=None,
            players=(
                PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=60),
                PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=60),
            ),
            deck=tuple(),
            trump_card=Card(Suit.CUPS, Rank.TWO),
            table_cards=tuple(),
            current_turn=0,
            first_player=0,
            game_over=True,
            winner_index=None,
            winning_team=None,
        ),
    )

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["winner"] == "Pareggio"
    assert "winner_index" not in payload


def test_get_game_result_4p_finished_returns_team_fields() -> None:
    """`/result` deve esporre `team_points` e `winning_team` quando la partita è a squadre."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 4, "player_names": ["A", "B", "C", "D"]})
    game_id = create.json()["game_id"]

    _put_state(
        game_id,
        GameState(
            num_players=4,
            is_team_game=True,
            teams=((0, 2), (1, 3)),
            players=(
                PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=40),
                PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=20),
                PlayerState(name="C", hand=tuple(), captured_cards=tuple(), points=30),
                PlayerState(name="D", hand=tuple(), captured_cards=tuple(), points=30),
            ),
            deck=tuple(),
            trump_card=Card(Suit.CUPS, Rank.TWO),
            table_cards=tuple(),
            current_turn=0,
            first_player=0,
            game_over=True,
            winner_index=None,
            winning_team=0,
        ),
    )

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["is_team_game"] is True
    assert payload["winner"] == "Squadra 0 (A e C)"
    assert payload["winning_team"] == 0
    assert payload["team_points"] == {"Team 0": 70, "Team 1": 50}
    assert payload["points"] == {"A": 40, "B": 20, "C": 30, "D": 30}
    assert payload["point_difference"] == 20


def test_get_game_result_4p_tie_omits_winning_team() -> None:
    """In pareggio 4-player, `winning_team` deve essere None (e quindi non presente nel JSON)."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 4, "player_names": ["A", "B", "C", "D"]})
    game_id = create.json()["game_id"]

    _put_state(
        game_id,
        GameState(
            num_players=4,
            is_team_game=True,
            teams=((0, 2), (1, 3)),
            players=(
                PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=30),
                PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=30),
                PlayerState(name="C", hand=tuple(), captured_cards=tuple(), points=30),
                PlayerState(name="D", hand=tuple(), captured_cards=tuple(), points=30),
            ),
            deck=tuple(),
            trump_card=Card(Suit.CUPS, Rank.TWO),
            table_cards=tuple(),
            current_turn=0,
            first_player=0,
            game_over=True,
            winner_index=None,
            winning_team=None,
        ),
    )

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["winner"] == "Pareggio"
    assert "winning_team" not in payload


def test_server_version_is_monotone_on_actions_when_ai_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifica che `server_version` sia monotona sugli endpoint HTTP, senza rumore da task IA.

    Nota:
    - disabilitiamo il task IA per rendere il test deterministico.
    - giochiamo 3 azioni seguendo `current_turn` del GameStateDTO (debug).
    """

    async def _no_ai(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(server, "_maybe_ai_turn", _no_ai)
    # Il loop usa la vista full-state (debug) per seguire `current_turn`: va abilitata esplicitamente.
    monkeypatch.setenv("BRISCOLA_DEBUG_STATE_ENDPOINT", "unsafe-full-state")

    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    # Questo test esercita volutamente il loop HTTP manuale per entrambi i player:
    # rimuoviamo la config IA dal seat 1 così `_is_ai_controlled_player` lo lascia giocabile.
    async def _clear_ai_seats() -> None:
        session = await server.game_store.get(game_id)
        assert session is not None
        session.ai_seats = {}
        await server.game_store.set(session)

    asyncio.run(_clear_ai_seats())

    # Versione iniziale: 0
    obs0 = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
    assert obs0["server_version"] == 0

    versions = []
    for _ in range(3):
        full = client.get(f"/games/{game_id}").json()
        current_turn = full["current_turn"]
        card_index = full["valid_actions"][0]

        out = client.post(
            f"/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": current_turn, "card_index": card_index},
        ).json()
        versions.append(out["server_version"])

    assert versions == sorted(versions)
    assert versions == [1, 2, 3]


def test_websocket_rejects_unknown_game() -> None:
    """WebSocket su partita inesistente: il server chiude subito la connessione."""
    client = TestClient(server.app)
    with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/ws/not-a-real-game-id/0"):
        pass
    assert excinfo.value.code == 1000


def test_websocket_ping_pong_and_receives_update_after_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """WS: ping/pong funziona e, dopo una giocata HTTP, arriva uno snapshot aggiornato."""
    _force_starting_player(monkeypatch, 0)
    client = TestClient(server.app)

    create = client.post("/games", json={"num_players": 2, "player_names": ["Alice", "Bob"]})
    game_id = create.json()["game_id"]

    with client.websocket_connect(f"/ws/{game_id}/0") as ws:
        initial = ws.receive_json()
        assert initial["type"] == "observation"
        assert initial["my_index"] == 0
        assert initial["my_turn"] is True
        assert initial["valid_actions"]

        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong == {"type": "pong"}

        play = client.post(
            f"/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": 0, "card_index": initial["valid_actions"][0]},
        )
        assert play.status_code == 200
        play_version = play.json()["server_version"]

        # Primo snapshot dopo la giocata. Nel modello pub/sub il "refresh" include lo stato
        # point-in-time (vedi notify_clients) ed è pubblicato PRIMA dello scheduling del task IA;
        # il pub/sub è FIFO, quindi il primo messaggio riflette lo stato subito dopo la mossa
        # umana: la versione coincide con la risposta HTTP e ora tocca all'IA (my_turn False).
        updated = ws.receive_json()
        assert updated["type"] == "observation"
        assert updated["my_index"] == 0
        assert updated["server_version"] == play_version
        assert updated["my_turn"] is False
        assert updated["valid_actions"] == []


def test_http_observation_matches_ws_observation_format() -> None:
    """
    Contratto: `GET /games/{id}?player_index=X` deve restituire lo stesso formato
    di uno snapshot WS (`type: "observation"`).

    Nota:
    - non confrontiamo l'intero payload per uguaglianza byte-per-byte (ordine chiavi),
      ma fissiamo campi e shape principali: `type`, `players`, `table_cards`, `my_hand`.
    """
    client = TestClient(server.app)

    create = client.post("/games", json={"num_players": 2, "player_names": ["Alice", "Bob"]})
    assert create.status_code == 200
    game_id = create.json()["game_id"]

    with client.websocket_connect(f"/ws/{game_id}/0") as ws:
        ws_obs = ws.receive_json()

    http_obs = client.get(f"/games/{game_id}", params={"player_index": 0}).json()

    assert ws_obs["type"] == "observation"
    assert http_obs["type"] == "observation"

    # Shape principale: players e table_cards sono strutture "esplicite" (DTO), non tuple/chiavi dinamiche.
    assert isinstance(ws_obs.get("players"), list)
    assert isinstance(http_obs.get("players"), list)
    assert isinstance(ws_obs.get("table_cards"), list)
    assert isinstance(http_obs.get("table_cards"), list)
    assert isinstance(ws_obs.get("my_hand"), list)
    assert isinstance(http_obs.get("my_hand"), list)
    assert isinstance(ws_obs.get("seen_cards_onehot"), list)
    assert isinstance(http_obs.get("seen_cards_onehot"), list)
    assert len(ws_obs["seen_cards_onehot"]) == 40
    assert len(http_obs["seen_cards_onehot"]) == 40

    # Controllo minimo su un elemento card: deve avere i campi DTO attesi.
    if http_obs["my_hand"]:
        card = http_obs["my_hand"][0]
        assert set(card.keys()) >= {"suit", "rank", "number", "points"}


def test_server_lifespan_cancels_cleanup_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allo shutdown dell'app, il task di cleanup periodico deve essere cancellato."""
    cancelled = {"value": False}

    async def fake_cleanup_inactive_games() -> None:
        # Implementazione fake: serve solo a verificare che il lifespan cancelli il task.
        try:
            while True:
                await server.asyncio.sleep(3600)
        except server.asyncio.CancelledError:
            cancelled["value"] = True
            raise

    monkeypatch.setattr(server, "cleanup_inactive_games", fake_cleanup_inactive_games)

    with TestClient(server.app):
        pass

    assert cancelled["value"] is True


def test_create_game_allows_selecting_ai_agent_and_ai_turn_uses_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regressione anti-cheat:
    - il client può scegliere l'agente IA all'avvio (`ai_agent`)
    - quando l'IA gioca, la policy riceve una `PlayerObservation` (non `GameState`)
    """
    _force_starting_player(monkeypatch, 0)

    async def _no_ai(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(server, "_maybe_ai_turn", _no_ai)

    class CheatDetectingAgent:
        name = "cheat_detector"

        def choose_card_index(self, observation, *, rng):  # noqa: ANN001
            assert observation.player_index == 1
            assert not hasattr(observation, "deck")
            assert not hasattr(observation, "players")
            assert observation.deck_size >= 0
            assert len(observation.hand) > 0
            return 0

    client = TestClient(server.app)
    create = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "IA"], "ai_agent": "heuristic_v1"},
    )
    assert create.status_code == 200
    game_id = create.json()["game_id"]

    # Verifica che la selezione sia stata applicata (per la UI: player 1 è l'IA).
    session = _get_session(game_id)
    assert session is not None
    assert session.ai_seats[1].agent_name == "heuristic_v1"

    # Sostituiamo l'agente con uno che fallisce se vede informazione nascosta.
    # Gli agenti sono ricostruiti per-mossa via `_agent_for_seat`: lo monkeypatchiamo.
    cheat_agent = CheatDetectingAgent()
    monkeypatch.setattr(server, "_agent_for_seat", lambda _cfg: cheat_agent)

    # Giochiamo una mossa umana per passare il turno all'IA.
    obs0 = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
    assert obs0["my_turn"] is True
    play = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": obs0["valid_actions"][0]},
    )
    assert play.status_code == 200

    # Eseguiamo una mossa IA (sincrona per test) sotto lock.
    async def _run_ai_once() -> None:
        async with server.game_store.lock(game_id):
            current = await server.game_store.get(game_id)
            assert current is not None
            await server._execute_ai_turn_locked(current, human_player_index=0)

    asyncio.run(_run_ai_once())


def test_health_endpoint() -> None:
    """`/health` deve rispondere 200 con stato ok (liveness per cloud/load balancer)."""
    with TestClient(main_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_version_endpoint_reports_versions_and_model_presence() -> None:
    """`/version` espone code/rules version e diagnostica presenza del modello consigliato."""
    with TestClient(main_app) as client:
        resp = client.get("/version")
        assert resp.status_code == 200
        body = resp.json()
        assert "code_version" in body
        assert "rules_version" in body
        assert body["recommended_model"] == "best_a2c_v15.npz"
        assert isinstance(body["recommended_model_present"], bool)
        assert body["value_lookahead_model"] == "value_v0_h128_clean50k_seed20260701.npz"
        assert isinstance(body["value_lookahead_model_present"], bool)
        assert "models_dir" in body


def test_version_recommended_model_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/version` deve riflettere `BRISCOLA_DEFAULT_MODEL_ID` (coerente col provisioning)."""
    monkeypatch.setenv("BRISCOLA_DEFAULT_MODEL_ID", "custom_model.npz")
    with TestClient(main_app) as client:
        body = client.get("/version").json()
        assert body["recommended_model"] == "custom_model.npz"


def test_play_action_when_local_game_data_buffer_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simula una replica diversa: lo stato vive nello store ma il buffer `game_data` locale non c'è.

    play_action deve funzionare (setdefault), senza KeyError/500: è il caso multi-replica
    in cui l'azione arriva su una replica che non ha creato la partita.
    """
    _force_starting_player(monkeypatch, 0)
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]
    obs = client.get(f"/games/{game_id}", params={"player_index": 0}).json()

    server.game_data.clear()  # nessun buffer locale per questa partita (altra "replica")

    action = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
    )
    assert action.status_code == 200
    assert "error" not in action.json()


def test_create_game_rejects_invalid_ai_agent() -> None:
    """Un `ai_agent` non valido deve dare 400 alla creazione (validazione, non crash nel task IA)."""
    client = TestClient(server.app)
    r = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["A", "B"], "ai_agent": "agente_inesistente"},
    )
    assert r.status_code == 400


def test_root_injects_realtime_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/` deve iniettare la modalità realtime risolta (placeholder sostituito; default ws senza Redis)."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("BRISCOLA_REDIS_URL", raising=False)
    monkeypatch.delenv("REDISCLOUD_URL", raising=False)
    monkeypatch.delenv("BRISCOLA_REALTIME_MODE", raising=False)
    with TestClient(main_app) as client:
        html = client.get("/").text
    assert "__BRISCOLA_REALTIME_MODE_VALUE__" not in html  # placeholder valore sostituito
    assert 'window.__BRISCOLA_REALTIME_MODE__ = "ws"' in html  # nome variabile intatto + valore risolto


def test_create_game_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Sopra il tetto per-IP nella finestra, `POST /games` deve rispondere 429.

    Il conftest disattiva il limite per l'intera suite; qui lo riabilitiamo con un valore
    piccolo per esercitare il percorso reale.
    """
    monkeypatch.setenv("BRISCOLA_CREATE_GAME_RATE_LIMIT", "3")
    server._create_game_requests.clear()
    client = TestClient(server.app)

    payload = {"num_players": 2, "player_names": ["A", "B"]}
    for _ in range(3):
        assert client.post("/games", json=payload).status_code == 200

    r = client.post("/games", json=payload)
    assert r.status_code == 429
    assert "Troppe partite" in r.json()["detail"]

    # Ripulisce lo stato condiviso del modulo per i test successivi.
    server._create_game_requests.clear()


def test_create_game_rejects_oversized_or_invalid_input() -> None:
    """I vincoli Pydantic devono bloccare input fuori misura PRIMA della logica di dominio."""
    client = TestClient(server.app)

    # num_players fuori range (il vincolo Field scatta prima di new_game_state).
    assert client.post("/games", json={"num_players": 7}).status_code == 422

    # Nome giocatore chilometrico: finirebbe nello stato, nei log e nei messaggi WS.
    r = client.post("/games", json={"num_players": 2, "player_names": ["x" * 100, "B"]})
    assert r.status_code == 422

    # Troppi nomi.
    r = client.post("/games", json={"num_players": 2, "player_names": ["A", "B", "C", "D", "E"]})
    assert r.status_code == 422

    # card_index fuori dallo spazio azioni [0,39].
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]
    r = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": 99},
    )
    assert r.status_code == 422


def test_event_log_sanitizes_player_names_in_debug_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Privacy: i nomi liberi dei giocatori non devono finire nell'event log nemmeno in
    modalità `debug` (in cloud scrive su un DB condiviso senza consenso esplicito).
    """
    import sqlite3

    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "debug")
    db_path = tmp_path / "events.sqlite3"
    log = EventLog(EventLogConfig(path=str(db_path)))
    previous_log = getattr(server.app.state, "event_log", None)
    server.app.state.event_log = log
    try:
        client = TestClient(server.app)
        r = client.post("/games", json={"num_players": 2, "player_names": ["NomePrivato", "B"]})
        assert r.status_code == 200
    finally:
        server.app.state.event_log = previous_log
        log.close()

    conn = sqlite3.connect(db_path)
    try:
        payloads = [row[0] for row in conn.execute("SELECT payload_json FROM events")]
    finally:
        conn.close()

    assert payloads, "atteso almeno un evento loggato in modalità debug"
    for payload in payloads:
        assert "NomePrivato" not in payload
