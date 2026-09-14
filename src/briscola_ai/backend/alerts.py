"""
Notifiche email best-effort per gli errori del backend (Mailgun).

Perché esiste: in produzione un'eccezione non gestita finisce solo nei log della
piattaforma, che nessuno guarda in tempo reale. Questo modulo manda una email al
maintainer con i dettagli (tipo, messaggio, traceback, endpoint) appena succede.

Principi di design:
- **best-effort e non bloccante**: l'invio avviene in un thread separato; qualunque
  errore nell'invio viene solo loggato. Le notifiche non devono MAI rompere o
  rallentare una risposta al giocatore.
- **anti-tempesta**: lo stesso errore ripetuto (stessa "firma": tipo + punto del
  traceback) viene notificato al più una volta per finestra; c'è anche un tetto
  globale di email per ora. Un bug in un loop non deve produrre 10.000 email.
- **niente dati sensibili**: la mail contiene percorso e metodo della richiesta,
  mai il body (che può contenere il nome del giocatore).

Configurazione (tutte env, tutte opzionali — senza config il modulo è spento):
- `MAILGUN_API_KEY`: API key del dominio Mailgun.
- `MAILGUN_DOMAIN`: dominio mittente (es. `sandboxXXX.mailgun.org` per il piano free).
- `BRISCOLA_ALERT_EMAIL_TO`: destinatario delle notifiche.
- `BRISCOLA_ALERT_EMAIL_FROM` (opzionale): mittente; default `briscola-ai@<dominio>`.
- `MAILGUN_EU` (opzionale, `1`): usa l'endpoint EU (`api.eu.mailgun.net`).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger("briscola.alerts")

# Anti-tempesta: una notifica per firma-errore ogni 15 minuti, massimo 20 email/ora.
DEDUP_WINDOW_SECONDS = 15 * 60
MAX_EMAILS_PER_HOUR = 20
_SEND_TIMEOUT_SECONDS = 10

_lock = threading.Lock()
_last_sent_by_signature: dict[str, float] = {}
_sent_timestamps: list[float] = []


@dataclass(frozen=True)
class AlertConfig:
    """Configurazione risolta dalle env (None nei campi mancanti = notifiche spente)."""

    api_key: str | None
    domain: str | None
    to_email: str | None
    from_email: str | None
    eu_endpoint: bool

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.domain and self.to_email)


def get_alert_config() -> AlertConfig:
    """Legge la configurazione dalle variabili d'ambiente (a ogni chiamata: testabile)."""
    domain = os.getenv("MAILGUN_DOMAIN", "").strip() or None
    return AlertConfig(
        api_key=os.getenv("MAILGUN_API_KEY", "").strip() or None,
        domain=domain,
        to_email=os.getenv("BRISCOLA_ALERT_EMAIL_TO", "").strip() or None,
        from_email=os.getenv("BRISCOLA_ALERT_EMAIL_FROM", "").strip() or (f"briscola-ai@{domain}" if domain else None),
        eu_endpoint=os.getenv("MAILGUN_EU", "").strip() in ("1", "true", "yes"),
    )


def _exception_signature(exc: BaseException) -> str:
    """
    Firma stabile dell'errore per la dedup: tipo + file/riga dell'ultimo frame "nostro".

    Due giocatori che inciampano nello stesso bug producono UNA notifica, non due.
    """
    tb = exc.__traceback__
    last = None
    while tb is not None:
        frame_file = tb.tb_frame.f_code.co_filename
        if "briscola_ai" in frame_file:
            last = f"{frame_file}:{tb.tb_lineno}"
        tb = tb.tb_next
    return f"{type(exc).__name__}@{last or 'unknown'}"


def _should_send(signature: str, *, now: float | None = None) -> bool:
    """Applica dedup per firma e tetto orario. Thread-safe."""
    ts = now if now is not None else time.time()
    with _lock:
        last = _last_sent_by_signature.get(signature)
        if last is not None and ts - last < DEDUP_WINDOW_SECONDS:
            return False
        # tetto globale nell'ultima ora
        cutoff = ts - 3600
        _sent_timestamps[:] = [t for t in _sent_timestamps if t > cutoff]
        if len(_sent_timestamps) >= MAX_EMAILS_PER_HOUR:
            return False
        _last_sent_by_signature[signature] = ts
        _sent_timestamps.append(ts)
        return True


def _mailgun_send(config: AlertConfig, subject: str, body: str) -> None:
    """POST all'API Mailgun (urllib: nessuna dipendenza nuova). Solleva su errore HTTP."""
    host = "api.eu.mailgun.net" if config.eu_endpoint else "api.mailgun.net"
    url = f"https://{host}/v3/{config.domain}/messages"
    data = urllib.parse.urlencode(
        {
            "from": config.from_email,
            "to": config.to_email,
            "subject": subject,
            "text": body,
        }
    ).encode()
    auth = base64.b64encode(f"api:{config.api_key}".encode()).decode()
    request = urllib.request.Request(url, data=data, headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(request, timeout=_SEND_TIMEOUT_SECONDS) as response:
        response.read()


# Iniettabile nei test (e sostituibile con altri provider in futuro).
_sender = _mailgun_send


def notify_exception(exc: BaseException, *, context: dict | None = None) -> bool:
    """
    Invia (in un thread, best-effort) la notifica per un'eccezione non gestita.

    Ritorna True se la notifica è stata ACCODATA (config presente + non deduplicata):
    utile nei test; il successo dell'invio effettivo è solo loggato.
    """
    config = get_alert_config()
    if not config.enabled:
        return False

    signature = _exception_signature(exc)
    if not _should_send(signature):
        return False

    from ..versioning import get_code_version

    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    context_text = json.dumps(context or {}, ensure_ascii=False, indent=2, default=str)
    subject = f"[briscola.ai] {type(exc).__name__}: {str(exc)[:120]}"
    body = (
        f"Eccezione non gestita su briscola.ai (versione {get_code_version()})\n"
        f"Firma: {signature}\n"
        f"Quando: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
        f"Contesto richiesta:\n{context_text}\n\n"
        f"Traceback:\n{tb_text}\n"
        f"(dedup: al più una notifica per firma ogni {DEDUP_WINDOW_SECONDS // 60} minuti; "
        f"tetto {MAX_EMAILS_PER_HOUR}/ora)"
    )

    def _send() -> None:
        try:
            _sender(config, subject, body)
            logger.info("Alert email inviata (%s)", signature)
        except Exception:
            logger.exception("Invio alert email fallito (best-effort, nessun retry)")

    threading.Thread(target=_send, name="briscola-alert-email", daemon=True).start()
    return True


def reset_throttle_state_for_tests() -> None:
    """Azzera dedup/tetto orario (solo per i test)."""
    with _lock:
        _last_sent_by_signature.clear()
        _sent_timestamps.clear()
