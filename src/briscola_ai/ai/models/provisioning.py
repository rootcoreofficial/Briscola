"""
Provisioning dei modelli al deploy: verifica gli asset ufficiali e li scarica se assenti.

Gli asset ufficiali piccoli (policy consigliata, value e belief) sono tracciati nel repository per
evitare download e compilazioni durante i frequenti cold start cloud. Il provisioning resta come
fallback e come meccanismo di override: scarica da un URL configurato (per esempio un GitHub Release
asset) **solo se il file richiesto non è già presente**.

Principio anti-fragilità: il provisioning non deve MAI impedire l'avvio dell'app. Se l'URL non è
impostato o il download fallisce, l'app parte lo stesso (il default agente è `random` e `/version`
segnala l'assenza del modello).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# Id del modello consigliato (coerente con la baseline ufficiale e con `/version`).
DEFAULT_MODEL_ID = "best_a2c_v15.npz"

# Id del value model usato dall'agente selezionabile `bc_model_value_lookahead_8x8`.
VALUE_LOOKAHEAD_MODEL_ID = "value_v0_h128_clean50k_seed20260701.npz"

# Id della belief network usata dalle varianti `bc_model_pimc_belief_*` per pesare le determinizzazioni.
PIMC_BELIEF_MODEL_ID = "belief_v0_h128_50k_seed20260702.npz"

# Timeout (s) per il download: evita che un endpoint lento/appeso blocchi lo startup dell'app.
_DEFAULT_DOWNLOAD_TIMEOUT_S = 30.0

# Schemi URL ammessi: `https`/`http` per i Release asset, `file` per test/uso locale.
_ALLOWED_URL_SCHEMES = {"https", "http", "file"}


def _sha256_matches(data: bytes, expected: str) -> bool:
    return hashlib.sha256(data).hexdigest().lower() == expected.strip().lower()


def ensure_model_available(
    *,
    models_dir: Path,
    model_id: str,
    url: str | None,
    sha256: str | None = None,
    timeout: float = _DEFAULT_DOWNLOAD_TIMEOUT_S,
    url_env_name: str = "BRISCOLA_MODEL_URL",
) -> tuple[bool, str]:
    """
    Garantisce che `models_dir/model_id` esista (e, se `sha256` è dato, che corrisponda),
    scaricandolo da `url` quando assente o non più valido.

    Ritorna `(disponibile, messaggio)`. Non solleva mai: ogni errore è catturato e riportato nel
    messaggio, così l'avvio dell'app non viene interrotto da un problema di provisioning.

    Comportamento:
    - se il file esiste e non è dato `sha256` → ok (fiducia sul file locale);
    - se esiste ma `sha256` non corrisponde → si riscarica (se c'è `url`), così l'hash funge da
      "pin di versione"; senza `url` si segnala l'incoerenza;
    - download con `timeout`, verifica `sha256`, scrittura atomica (tmp nella stessa dir + `os.replace`).

    Limite noto: se un file con sha errato è già presente e il re-download non è possibile (nessun
    `url` o download fallito), il file stale NON viene rimosso e ritorniamo `(False, ...)`. I layer
    che si basano sulla sola esistenza del file (catalogo UI, `/version`) potrebbero quindi ancora
    esporlo: l'integrità a serve-time non è garantita da questa funzione.
    """
    target = models_dir / model_id

    if target.exists():
        if not sha256:
            return True, f"modello già presente: {target}"
        try:
            if _sha256_matches(target.read_bytes(), sha256):
                return True, f"modello già presente e verificato: {target}"
        except OSError as exc:
            return False, f"modello presente ma illeggibile: {exc!r}"
        if not url:
            return False, f"modello presente ma sha256 non corrisponde e nessun URL per riscaricare: {target}"
        # hash diverso + url disponibile → procediamo a riscaricare (sovrascrittura atomica).
    elif not url:
        return False, f"modello assente e nessun {url_env_name} impostato"

    scheme = urlparse(url or "").scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        return False, f"schema URL non ammesso: {scheme!r} (ammessi: {sorted(_ALLOWED_URL_SCHEMES)})"

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - URL operatore-trusted
            data = resp.read()
    except Exception as exc:  # provisioning best-effort: non deve crashare l'avvio
        return False, f"download fallito: {exc!r}"

    if sha256 and not _sha256_matches(data, sha256):
        digest = hashlib.sha256(data).hexdigest()
        return False, f"sha256 non corrispondente (atteso {sha256.strip()}, ottenuto {digest})"

    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(models_dir), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    except Exception as exc:
        return False, f"scrittura modello fallita: {exc!r}"

    return True, f"modello scaricato in {target} ({len(data)} byte)"
