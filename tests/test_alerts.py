"""
Notifiche email per gli errori (backend/alerts.py): config, dedup, tetto orario, handler.

Il sender Mailgun è iniettabile: nei test viene sostituito da una lista, così si verifica
il comportamento (accodata/deduplicata) senza rete.
"""

from __future__ import annotations

import time

import pytest

import briscola_ai.backend.alerts as alerts


@pytest.fixture(autouse=True)
def _clean_alert_state(monkeypatch):
    """Ogni test parte con throttle azzerato e config spenta (a meno di override)."""
    alerts.reset_throttle_state_for_tests()
    for env in ("MAILGUN_API_KEY", "MAILGUN_DOMAIN", "BRISCOLA_ALERT_EMAIL_TO", "BRISCOLA_ALERT_EMAIL_FROM"):
        monkeypatch.delenv(env, raising=False)
    yield
    alerts.reset_throttle_state_for_tests()


def _configure(monkeypatch):
    monkeypatch.setenv("MAILGUN_API_KEY", "key-test")
    monkeypatch.setenv("MAILGUN_DOMAIN", "sandbox.example.org")
    monkeypatch.setenv("BRISCOLA_ALERT_EMAIL_TO", "dev@example.org")


def _boom() -> RuntimeError:
    try:
        raise RuntimeError("errore di prova")
    except RuntimeError as exc:
        return exc


def test_disabled_without_configuration() -> None:
    """Senza env Mailgun le notifiche sono spente: nessun invio, nessun errore."""
    assert alerts.get_alert_config().enabled is False
    assert alerts.notify_exception(_boom()) is False


def test_notify_sends_email_with_traceback_and_context(monkeypatch) -> None:
    _configure(monkeypatch)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(alerts, "_sender", lambda cfg, subject, body: sent.append((subject, body)))

    assert alerts.notify_exception(_boom(), context={"path": "/api/test"}) is True
    for _ in range(50):  # l'invio gira in un thread
        if sent:
            break
        time.sleep(0.05)
    assert sent, "email non inviata"
    subject, body = sent[0]
    assert "RuntimeError" in subject
    assert "errore di prova" in body
    assert "Traceback" in body
    assert "/api/test" in body


def test_same_error_is_deduplicated(monkeypatch) -> None:
    """Stessa firma (tipo + punto del codice) entro la finestra: una sola email."""
    _configure(monkeypatch)
    monkeypatch.setattr(alerts, "_sender", lambda cfg, subject, body: None)

    assert alerts.notify_exception(_boom()) is True
    assert alerts.notify_exception(_boom()) is False  # dedup


def test_hourly_cap_limits_email_storms() -> None:
    """Firme tutte diverse ma oltre il tetto orario: si smette di inviare."""
    now = time.time()
    sent = sum(1 for i in range(alerts.MAX_EMAILS_PER_HOUR + 10) if alerts._should_send(f"sig-{i}", now=now))
    assert sent == alerts.MAX_EMAILS_PER_HOUR


def test_http_exception_handler_returns_500_and_notifies(monkeypatch) -> None:
    """Rotta che esplode: 500 pulito al client + notifica accodata (sender mockato)."""
    _configure(monkeypatch)
    sent: list[str] = []
    monkeypatch.setattr(alerts, "_sender", lambda cfg, subject, body: sent.append(subject))

    from fastapi.testclient import TestClient

    from briscola_ai.backend import server

    route_path = "/__test_boom_alerts"
    if not any(getattr(r, "path", None) == route_path for r in server.app.routes):

        @server.app.get(route_path)
        async def _boom_route():  # pragma: no cover - il corpo esplode subito
            raise ValueError("esplosione controllata")

    client = TestClient(server.app, raise_server_exceptions=False)
    response = client.get(route_path)
    assert response.status_code == 500
    assert response.json() == {"detail": "Errore interno del server"}
    for _ in range(50):
        if sent:
            break
        time.sleep(0.05)
    assert sent and "ValueError" in sent[0]
