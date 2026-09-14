"""Contratti del laboratorio post-partita: osservazione lecita e validazione del replay."""

import random

from fastapi.testclient import TestClient

from briscola_ai.backend import server
from briscola_ai.domain.observation import PlayerObservation
from briscola_ai.main import app as main_app


class _SecondCardAgent:
    """Doppio leggero del PIMC per testare il contratto HTTP senza caricare gli asset reali."""

    name = "test_replay_agent"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        """Restituisce la seconda carta per rendere verificabile la conversione della risposta."""
        del observation, rng
        return 1


def _initial_replay_payload() -> dict:
    """Istante iniziale coerente: 3+3 carte in mano e 34 nel mazzo."""
    return {
        "hand": [
            {"suit": "clubs", "number": 1},
            {"suit": "cups", "number": 2},
            {"suit": "coins", "number": 3},
        ],
        "trump_card": {"suit": "swords", "number": 4},
        "deck_size": 34,
        "opponent_hand_size": 3,
        "my_points": 0,
        "opponent_points": 0,
        "first_player": 0,
    }


def test_replay_advice_builds_a_partial_observation(monkeypatch) -> None:
    """Il replay restituisce una carta della mano senza richiedere carte nascoste dell'altro posto."""
    monkeypatch.setattr(server, "_replay_agent", _SecondCardAgent())

    response = TestClient(server.app).post("/replay/advice", json=_initial_replay_payload())

    assert response.status_code == 200
    assert response.json() == {
        "card": {"suit": "cups", "rank": "TWO", "number": 2, "points": 0},
        "agent": "test_replay_agent",
        "model_id": "best_a2c_v15.npz",
    }


def test_replay_advice_rejects_inconsistent_card_counters(monkeypatch) -> None:
    """Un replay incompleto non deve produrre una raccomandazione apparentemente attendibile."""
    monkeypatch.setattr(server, "_replay_agent", _SecondCardAgent())
    payload = _initial_replay_payload()
    payload["deck_size"] = 33

    response = TestClient(server.app).post("/replay/advice", json=payload)

    assert response.status_code == 422
    assert "sommare 40" in response.json()["detail"]


def test_replay_page_is_served_by_the_main_application() -> None:
    """La pagina touch è una route dedicata e non interferisce con la home di gioco normale."""
    response = TestClient(main_app).get("/replay")

    assert response.status_code == 200
    assert "Assistente Briscola" in response.text
    assert "Carta consigliata" in response.text
    assert "/static/css/replay.css" in response.text
    assert '="./css/replay.css' not in response.text
    assert "replay.js" in response.text
