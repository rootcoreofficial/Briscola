"""
E2E frontend: avviso non bloccante "il server si sta svegliando" sulle richieste lente.

Contesto: il deploy pubblico (FastAPI Cloud) usa lo scale-to-zero, quindi la prima
richiesta dopo un periodo di idle paga il risveglio della replica (~10-15s). Il layer
API (`api.js`) fa partire un timer per ogni fetch REST: se la risposta non arriva entro
~2.5s la UI mostra un messaggio di cortesia riusando i canali di stato esistenti
(badge `#game-status` nell'header + dettaglio dell'overlay di avvio partita), e lo
rimuove appena la richiesta si conclude.

Tecnica di test: le richieste lente sono simulate intercettando la rete con
`page.route` e ritardando la risposta. Poiché il route handler sincrono blocca il
thread Python durante il ritardo, NON possiamo osservare il DOM "dal vivo" mentre la
richiesta è in volo: registriamo invece le transizioni del DOM lato browser con un
MutationObserver (init script) e le verifichiamo a posteriori.

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

WAKE_TOKEN = "svegliando"  # sottostringa del messaggio mostrato dalla UI

# Init script: logga ogni cambiamento di testo del badge di stato e del dettaglio
# dell'overlay di startup, così possiamo asserire che l'avviso è comparso E sparito
# anche se Python era bloccato nel route handler mentre succedeva.
STATUS_LOGGER_INIT_SCRIPT = """
window.__statusLog = [];
document.addEventListener('DOMContentLoaded', () => {
    const track = (id) => {
        const el = document.getElementById(id);
        if (!el) return;
        const push = () => window.__statusLog.push({ id, text: el.textContent });
        push();
        new MutationObserver(push).observe(el, { childList: true, characterData: true, subtree: true });
    };
    track('game-status');
    track('startup-loading-detail');
});
"""


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


def test_wake_notice_appears_on_slow_create_and_disappears(live_server: str) -> None:
    """
    Caso reale principale: la creazione partita colpisce una replica "addormentata".

    Verifica che, con un POST /api/games rallentato oltre la soglia:
    - l'avviso compaia sia nel badge header sia nel dettaglio dell'overlay di avvio;
    - sparisca appena la risposta arriva (badge tornato allo stato connessione,
      overlay tornato al testo di default);
    - NON compaia per le richieste veloci (i fetch di metadati della home in locale).
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception:
            pytest.skip("Chromium Playwright non installato (python -m playwright install chromium)")
        page = browser.new_page()
        page.add_init_script(STATUS_LOGGER_INIT_SCRIPT)
        page.goto(live_server)
        page.wait_for_timeout(1000)

        # Sanity check: in locale i fetch di metadati sono rapidi, quindi l'avviso
        # non deve essere mai comparso prima di avviare la partita.
        early_log = page.evaluate("window.__statusLog")
        assert not any(WAKE_TOKEN in entry["text"] for entry in early_log), early_log

        # Rallenta SOLO la creazione partita: 4s > soglia di 2.5s del layer API.
        # Nota: time.sleep blocca il dispatcher sync di Playwright, ma il JS della
        # pagina continua a girare nel browser (il timer dell'avviso scatta comunque).
        def handle(route):
            if route.request.method == "POST":
                time.sleep(4.0)
            route.continue_()

        page.route("**/api/games", handle)
        page.click("#advanced-options-toggle")
        page.wait_for_selector("#ai-agent-select")
        page.select_option("#ai-agent-select", "heuristic_v1")
        page.check("#data-consent-checkbox")
        page.click("#start-game")
        page.wait_for_function("document.getElementById('game-status').textContent.includes('Connesso')", timeout=20000)

        log = page.evaluate("window.__statusLog")
        # L'avviso è comparso nel badge header...
        assert any(e["id"] == "game-status" and WAKE_TOKEN in e["text"] for e in log), log
        # ...e nell'overlay di avvio (che copre l'header durante la creazione).
        assert any(e["id"] == "startup-loading-detail" and WAKE_TOKEN in e["text"] for e in log), log

        # A risposta arrivata l'avviso è sparito: badge sullo stato connessione,
        # dettaglio overlay ripristinato al testo di default.
        assert WAKE_TOKEN not in (page.text_content("#game-status") or "")
        detail = page.evaluate("document.getElementById('startup-loading-detail').textContent")
        assert detail == "Caricamento tavolo e IA"
        browser.close()


def test_wake_notice_uses_counter_for_concurrent_requests(live_server: str) -> None:
    """
    Richieste concorrenti: l'avviso deve accendersi UNA volta alla prima richiesta
    lenta e spegnersi solo quando l'ULTIMA si conclude (contatore, non booleano).

    Pilotiamo direttamente il layer API della pagina: registriamo un listener che
    logga le notifiche e lanciamo due GET /api/meta rallentati in parallelo.
    I due route handler sincroni girano in serie (~4s e ~8s), quindi le richieste
    si concludono in momenti diversi: la sequenza attesa resta [true, false].
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

        def handle(route):
            time.sleep(4.0)
            route.continue_()

        page.route("**/api/meta", handle)
        notices = page.evaluate(
            """
            async () => {
                window.__notices = [];
                API.setSlowRequestListener((active) => window.__notices.push(active));
                await Promise.all([API.getServerMeta(), API.getServerMeta()]);
                return window.__notices;
            }
            """
        )
        assert notices == [True, False], notices
        browser.close()
