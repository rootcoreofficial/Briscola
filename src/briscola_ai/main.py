"""
Entry point dell'applicazione web.

Questa app FastAPI:
- monta l'API sotto `/api` (vedi `briscola_ai.backend.server`)
- serve gli asset statici sotto `/static`
- serve la UI principale su `/`

Per avviare in locale:
  - `briscola-server --reload`
  - oppure `python -m briscola_ai.main --reload`
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager, suppress
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .backend import server as backend_server
from .versioning import get_code_version


def _provision_startup_models() -> list[str]:
    """
    Provisioning best-effort degli asset `.npz` necessari in cloud.

    La policy consigliata (`best_a2c_v15.npz`, o override `BRISCOLA_DEFAULT_MODEL_ID`) resta il modello principale.
    Il value model e' opzionale ma necessario per rendere disponibile `bc_model_value_lookahead_8x8`: lo scarichiamo
    solo se l'operatore imposta `BRISCOLA_VALUE_MODEL_URL` o un pin `BRISCOLA_VALUE_MODEL_SHA256`.
    """
    from .ai.models import (
        DEFAULT_MODEL_ID,
        PIMC_BELIEF_MODEL_ID,
        VALUE_LOOKAHEAD_MODEL_ID,
        ensure_model_available,
        get_models_dir_from_env,
    )

    models_dir = get_models_dir_from_env()
    messages: list[str] = []

    _, provisioning_msg = ensure_model_available(
        models_dir=models_dir,
        model_id=os.getenv("BRISCOLA_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID),
        url=os.getenv("BRISCOLA_MODEL_URL"),
        sha256=os.getenv("BRISCOLA_MODEL_SHA256"),
    )
    messages.append(f"Model provisioning: {provisioning_msg}")

    value_url = os.getenv("BRISCOLA_VALUE_MODEL_URL")
    value_sha256 = os.getenv("BRISCOLA_VALUE_MODEL_SHA256")
    if value_url or value_sha256:
        _, value_msg = ensure_model_available(
            models_dir=models_dir,
            model_id=VALUE_LOOKAHEAD_MODEL_ID,
            url=value_url,
            sha256=value_sha256,
            url_env_name="BRISCOLA_VALUE_MODEL_URL",
        )
        messages.append(f"Value model provisioning: {value_msg}")

    belief_url = os.getenv("BRISCOLA_BELIEF_MODEL_URL")
    belief_sha256 = os.getenv("BRISCOLA_BELIEF_MODEL_SHA256")
    if belief_url or belief_sha256:
        _, belief_msg = ensure_model_available(
            models_dir=models_dir,
            model_id=PIMC_BELIEF_MODEL_ID,
            url=belief_url,
            sha256=belief_sha256,
            url_env_name="BRISCOLA_BELIEF_MODEL_URL",
        )
        messages.append(f"Belief model provisioning: {belief_msg}")

    return messages


async def _run_startup_background_work() -> None:
    """
    Provisioning modelli DOPO che l'app ha iniziato a servire.

    Perché in background (misurato in produzione il 2026-07-07, log FastAPI Cloud):
    su scale-to-zero l'idle timeout è ~90s, quindi il cold start è frequentissimo; tutto
    ciò che sta nel lifespan prima dello yield lo paga il primo visitatore. Con i tre
    asset committati nell'immagine il provisioning degrada a una verifica SHA locale
    (millisecondi) e resta come fallback/pin di versione.

    Storia utile per chi legge: qui viveva anche il warm-up dei kernel Numba (10.2s per
    la search JIT, poi 2s per il solo solver). Dal 2026-07-07 il runtime web è
    ZERO-NUMBA — search PIMC python e solver endgame python con playout della principal
    variation nei rollout — quindi non c'è più nulla da compilare. Il `print` col tempo
    resta come telemetria nei log della piattaforma.
    """
    started = time.perf_counter()
    try:
        for provisioning_msg in await asyncio.to_thread(_provision_startup_models):
            print(provisioning_msg)
    except Exception as exc:  # difesa extra: il provisioning non deve abbattere il task
        print(f"Model provisioning: errore inatteso, ignorato ({exc!r}).")
    provisioned = time.perf_counter()
    print(f"Startup background: provisioning modelli completato in {provisioned - started:.1f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown dell'app principale.

    Nota importante:
    l'API backend è montata come sub-app (`/api`). In alcuni setup i mounted sub-app
    non ricevono eventi lifespan. Per evitare che features come cleanup e event log
    restino disabilitate, le inizializziamo esplicitamente qui.
    """
    # Event log: logica condivisa col lifespan del backend (una sola implementazione,
    # vedi `backend_server.initialize_event_log_from_env`). Lo stato vive sul sub-app
    # backend, che è quello che i suoi endpoint interrogano.
    event_log, event_log_created_here = backend_server.initialize_event_log_from_env(backend_server.app)

    # Provisioning modelli in BACKGROUND: l'app inizia a servire subito
    # (cold start ~19s → ~7s misurati) e paga compilazione/download mentre il visitatore
    # carica la home. Dettagli e numeri nel docstring di `_run_startup_background_work`.
    startup_work_task = asyncio.create_task(_run_startup_background_work())

    cleanup_task = asyncio.create_task(backend_server.cleanup_inactive_games())
    try:
        yield
    finally:
        cleanup_task.cancel()
        startup_work_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        with suppress(asyncio.CancelledError):
            await startup_work_task
        if event_log is not None and event_log_created_here:
            event_log.close()
            backend_server.app.state.event_log = None


# Crea l'applicazione FastAPI principale
#
# Nota:
# usiamo `get_code_version()` per allineare la versione OpenAPI alla versione del pacchetto
# (o all'override via env `BRISCOLA_CODE_VERSION`).
app = FastAPI(title="Briscola AI", version=get_code_version(), lifespan=lifespan)

# Ottiene la directory del file corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Monta l'app API sotto /api
app.mount("/api", backend_server.app)

# Monta i file statici
static_dir = os.path.join(current_dir, "frontend", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _asset_version() -> str:
    """
    Versione usata per il cache busting degli asset statici.

    Di default segue la versione del pacchetto; `BRISCOLA_ASSET_VERSION` permette di forzare
    un valore diverso in deploy senza dover fare necessariamente un bump applicativo.
    """
    # Override esplicito (utile in deploy se si vuole forzare un valore preciso).
    override = os.getenv("BRISCOLA_ASSET_VERSION", "").strip()
    if override:
        return quote(override, safe="")

    # Altrimenti deriviamo la versione dal CONTENUTO statico: il `mtime` massimo tra i file CSS/JS.
    # Così ogni modifica a CSS/JS invalida automaticamente la cache del browser, sia in locale
    # (senza dover reinstallare il pacchetto o bumpare la versione) sia tra un deploy e l'altro.
    # Manteniamo `get_code_version()` come prefisso leggibile.
    try:
        latest_mtime_ns = 0
        for root, _dirs, files in os.walk(static_dir):
            for filename in files:
                if filename.endswith((".css", ".js")):
                    mtime_ns = os.stat(os.path.join(root, filename)).st_mtime_ns
                    latest_mtime_ns = max(latest_mtime_ns, mtime_ns)
        if latest_mtime_ns:
            return quote(f"{get_code_version()}-{latest_mtime_ns:x}", safe="-")
    except OSError:
        pass
    return quote(get_code_version(), safe="")


def _realtime_mode() -> str:
    """
    Modalità realtime suggerita al frontend. Default: **WebSocket**.

    Il WebSocket funziona anche in cloud multi-replica perché il fan-out degli eventi passa per
    Redis pub/sub (vedi `backend/game_store.py`): un client su una qualsiasi replica riceve gli
    eventi della partita. Override via `?polling=1` / `?ws=1` nell'URL (il polling resta un
    fallback di debug); forzabile con `BRISCOLA_REALTIME_MODE`.
    """
    forced = os.getenv("BRISCOLA_REALTIME_MODE", "").strip().lower()
    if forced in {"polling", "ws"}:
        return forced
    return "ws"


# Serve il file HTML principale
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve la pagina HTML principale (single-page UI)."""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__BRISCOLA_ASSET_VERSION__", _asset_version())
    html = html.replace("__BRISCOLA_REALTIME_MODE_VALUE__", _realtime_mode())
    # Versione "software" (SemVer) mostrata nel footer — distinta dall'asset version (cache-busting).
    html = html.replace("__BRISCOLA_APP_VERSION__", get_code_version())
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})


@app.get("/diario", response_class=HTMLResponse)
async def read_diario():
    """
    Serve il diario di bordo: la storia del progetto raccontata in tono divulgativo.

    Fonte unica: `static/diario.md` (markdown, facile da aggiornare); il rendering HTML
    avviene qui a richiesta e viene vestito dal template `static/diario_template.html`.
    """
    import markdown as md_lib

    with open(os.path.join(static_dir, "diario.md"), encoding="utf-8") as f:
        md_source = f.read()
    content = md_lib.markdown(md_source, extensions=["smarty"])
    with open(os.path.join(static_dir, "diario_template.html"), encoding="utf-8") as f:
        template = f.read()
    html = template.replace("__DIARIO_CONTENT__", content)
    html = html.replace("__BRISCOLA_ASSET_VERSION__", _asset_version())
    html = html.replace("__BRISCOLA_APP_VERSION__", get_code_version())
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})


@app.get("/replay", response_class=HTMLResponse)
async def read_replay_lab():
    """
    Serve l'assistente gestuale per registrare una partita fisica in corso.

    È una pagina separata dalla partita normale: non apre WebSocket né crea sessioni nel
    game store. Le carte registrate restano nel browser e ogni richiesta di consiglio al
    backend è una singola istantanea anonima e senza persistenza.
    """
    replay_path = os.path.join(static_dir, "replay.html")
    with open(replay_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__BRISCOLA_ASSET_VERSION__", _asset_version())
    html = html.replace("__BRISCOLA_APP_VERSION__", get_code_version())
    # Nel sito gli asset vivono sotto `/static`; nell'APK Capacitor la stessa pagina
    # viene copiata nella root del WebView e conserva invece il prefisso relativo `./`.
    html = html.replace('="./', '="/static/')
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    """SEO di base: consenti l'indicizzazione delle pagine pubbliche e dichiara la sitemap."""
    body = "User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: https://ai.briscola.dev/sitemap.xml\n"
    return Response(content=body, media_type="text/plain")


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap_xml():
    """Sitemap minimale: le due pagine indicizzabili (home e diario di bordo)."""
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "  <url><loc>https://ai.briscola.dev/</loc><changefreq>weekly</changefreq></url>\n"
        "  <url><loc>https://ai.briscola.dev/diario</loc><changefreq>monthly</changefreq></url>\n"
        "</urlset>\n"
    )
    return Response(content=body, media_type="application/xml")


# Serve la favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve la favicon (esclusa dallo schema OpenAPI)."""
    return FileResponse(os.path.join(static_dir, "favicon.ico"))


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness check minimale (per piattaforme cloud / load balancer)."""
    return {"status": "ok"}


@app.get("/version")
async def version_info():
    """
    Diagnostica deploy: versioni e presenza del modello consigliato.

    Utile in cloud per verificare che il modello consigliato sia risolvibile nella directory modelli
    effettiva (che dipende da `BRISCOLA_MODELS_DIR` o dalla working directory).
    """
    from .ai.models import DEFAULT_MODEL_ID, PIMC_BELIEF_MODEL_ID, VALUE_LOOKAHEAD_MODEL_ID, get_models_dir_from_env
    from .versioning import get_rules_version

    models_dir = get_models_dir_from_env()
    # Coerente col provisioning: stesso `BRISCOLA_DEFAULT_MODEL_ID` usato allo startup.
    recommended_model = os.getenv("BRISCOLA_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID)
    return {
        "code_version": get_code_version(),
        "rules_version": get_rules_version(),
        "models_dir": str(models_dir),
        "recommended_model": recommended_model,
        "recommended_model_present": (models_dir / recommended_model).exists(),
        "value_lookahead_model": VALUE_LOOKAHEAD_MODEL_ID,
        "value_lookahead_model_present": (models_dir / VALUE_LOOKAHEAD_MODEL_ID).exists(),
        "pimc_belief_model": PIMC_BELIEF_MODEL_ID,
        "pimc_belief_model_present": (models_dir / PIMC_BELIEF_MODEL_ID).exists(),
        **backend_server.event_log_runtime_metadata(),
    }


def run_server(host="0.0.0.0", port=8000, reload=False):
    """Avvia il server con uvicorn"""
    # Parsing argomenti CLI:
    # - usiamo i parametri della funzione come default (così `run_server(host=..., ...)` resta possibile)
    # - se l'utente passa argomenti, li rispettiamo.
    import argparse

    parser = argparse.ArgumentParser(description="Avvia il server di Briscola AI")
    parser.add_argument("--host", default=host, help="Host su cui esporre il server")
    parser.add_argument("--port", type=int, default=port, help="Porta su cui esporre il server")
    parser.add_argument("--reload", action="store_true", default=reload, help="Abilita auto-reload per lo sviluppo")
    parser.add_argument(
        "--event-db",
        default=os.getenv("BRISCOLA_EVENT_DB_PATH", "./data/briscola_events.sqlite3"),
        help=(
            "Percorso del DB SQLite per l'event log (Phase 4). "
            "Default: ./data/briscola_events.sqlite3. "
            "Usa stringa vuota per disabilitare (es. --event-db '')."
        ),
    )
    parser.add_argument(
        "--event-log-mode",
        default=os.getenv("BRISCOLA_EVENT_LOG_MODE", "debug"),
        choices=["debug", "dataset", "off"],
        help=(
            "Modalità event log: "
            "`debug` (completa), `dataset` (riduce dimensione DB per raccolta umani), `off` (disabilita logging). "
            "Default: debug."
        ),
    )

    args = parser.parse_args()

    print(f"Avvio server Briscola AI su {args.host}:{args.port}")
    print("Premi Ctrl+C per fermare il server")

    host = args.host
    port = args.port
    reload = args.reload

    # Configurazione event log:
    # - CLI è la fonte più esplicita
    # - l'app (main/backend) legge la variabile d'ambiente nel lifespan
    if args.event_db.strip() == "":
        os.environ.pop("BRISCOLA_EVENT_DB_PATH", None)
    else:
        os.environ["BRISCOLA_EVENT_DB_PATH"] = args.event_db
    os.environ["BRISCOLA_EVENT_LOG_MODE"] = args.event_log_mode

    uvicorn.run("briscola_ai.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server(reload=True)
