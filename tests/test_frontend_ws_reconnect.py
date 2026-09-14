"""
E2E frontend: la riconnessione WebSocket deve essere singola e pulita (no flapping).

Regressione del bug 2026-07-06: i gestori del vecchio socket restavano attivi dopo la
riconnessione e innescavano catene parallele di retry; inoltre il contatore di backoff
veniva azzerato a ogni giro. Il test forza la caduta del socket dal lato client e
verifica: esattamente UNA riconnessione, stato "Connesso" ripristinato, partita ancora
giocabile, e robustezza a cadute ripetute.

Richiede Playwright + Chromium (dev-only): skip pulito se assenti.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytestmark = pytest.mark.slow

playwright = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_server():
    """Avvia briscola-server su una porta libera e attende /health."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "briscola_ai.main", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=1):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.skip("server non partito")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_ws_reconnects_once_and_game_stays_playable(live_server: str) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception:
            pytest.skip("Chromium Playwright non installato (python -m playwright install chromium)")
        page = browser.new_page()
        # Traccia ogni WebSocket creato dalla pagina: il conteggio è l'oracolo anti-flapping.
        page.add_init_script(
            """
            window.__sockets = [];
            const NativeWS = window.WebSocket;
            window.WebSocket = function(url, protocols) {
                const ws = protocols ? new NativeWS(url, protocols) : new NativeWS(url);
                window.__sockets.push(ws);
                return ws;
            };
            window.WebSocket.prototype = NativeWS.prototype;
            Object.assign(window.WebSocket, NativeWS);
            """
        )
        page.goto(live_server)
        page.wait_for_timeout(1000)
        page.click("#advanced-options-toggle")
        page.wait_for_selector("#ai-agent-select")
        page.select_option("#ai-agent-select", "heuristic_v1")
        page.check("#data-consent-checkbox")
        page.click("#start-game")
        page.wait_for_function("document.getElementById('game-status').textContent.includes('Connesso')", timeout=10000)
        n0 = page.evaluate("window.__sockets.length")

        # Caduta forzata -> deve seguire UNA sola riconnessione (niente tempeste).
        page.evaluate("window.__sockets.at(-1).close(4001, 'test')")
        page.wait_for_function("document.getElementById('game-status').textContent.includes('Connesso')", timeout=15000)
        page.wait_for_timeout(3000)  # finestra anti-flapping
        assert page.evaluate("window.__sockets.length") == n0 + 1

        # La partita resta giocabile (se è il turno umano, gioca una carta).
        if page.evaluate("!!document.querySelector('#player-hand .card:not(.disabled)')"):
            page.click("#player-hand .card:not(.disabled)")
            page.wait_for_timeout(1000)

        # Seconda caduta: la gestione deve reggere ripetutamente.
        page.evaluate("window.__sockets.at(-1).close(4001, 'test2')")
        page.wait_for_function("document.getElementById('game-status').textContent.includes('Connesso')", timeout=15000)
        page.wait_for_timeout(2000)
        assert page.evaluate("window.__sockets.length") == n0 + 2
        browser.close()


def test_action_network_blip_recovers_without_raw_error(live_server: str) -> None:
    """
    Robustezza azioni HTTP: se il POST della mossa fallisce per un blip di rete,
    il client risincronizza dallo stato server e mostra un invito a riprovare —
    MAI un errore crudo. Il retry successivo deve andare a buon fine.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception:
            pytest.skip("Chromium Playwright non installato (python -m playwright install chromium)")
        page = browser.new_page()
        page.goto(live_server)
        page.wait_for_timeout(1000)
        page.click("#advanced-options-toggle")
        page.wait_for_selector("#ai-agent-select")
        page.select_option("#ai-agent-select", "heuristic_v1")
        page.check("#data-consent-checkbox")
        page.click("#start-game")
        page.wait_for_selector("#player-hand .card:not(.disabled)", timeout=15000)

        state = {"aborted": False}

        def handle(route):
            # Abortisce SOLO il primo POST dell'azione: simula un singhiozzo di rete.
            if route.request.method == "POST" and "/actions" in route.request.url and not state["aborted"]:
                state["aborted"] = True
                route.abort("connectionfailed")
            else:
                route.continue_()

        page.route("**/api/games/**", handle)
        page.click("#player-hand .card:not(.disabled)")
        page.wait_for_timeout(3500)  # tempo per la riconciliazione (resync 400ms + margine)

        msg = page.text_content("#turn-message") or ""
        assert state["aborted"], "blip non simulato"
        assert "Errore:" not in msg, f"errore crudo mostrato al giocatore: {msg}"

        # Il retry manuale deve funzionare e la partita proseguire.
        page.wait_for_selector("#player-hand .card:not(.disabled)", timeout=5000)
        page.click("#player-hand .card:not(.disabled)")
        page.wait_for_timeout(3000)
        assert "Errore:" not in (page.text_content("#turn-message") or "")
        browser.close()
