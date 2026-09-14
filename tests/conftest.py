"""
Configurazione condivisa dei test.

Isolamento dall'ambiente (ermeticità)
-------------------------------------
Alcuni componenti scelgono un backend in base a variabili d'ambiente:
- `REDIS_URL`/`BRISCOLA_REDIS_URL` → game store su Redis;
- `DATABASE_URL`/`BRISCOLA_DATABASE_URL` → event log su Postgres.

Se queste env fossero impostate nella shell di chi lancia i test (o in CI), la suite proverebbe a
contattare servizi reali (Redis/Postgres) invece di usare i backend in-memory/SQLite previsti dai
test → test non deterministici o che falliscono per motivi ambientali. La fixture autouse qui sotto
rimuove quelle variabili per OGNI test, così la suite resta ermetica. I test che vogliono testare
esplicitamente la selezione per-env le re-impostano da soli via `monkeypatch`.
"""

from __future__ import annotations

import pytest

_ENV_TO_CLEAR = (
    "REDIS_URL",
    "BRISCOLA_REDIS_URL",
    "REDISCLOUD_URL",
    "DATABASE_URL",
    "BRISCOLA_DATABASE_URL",
    "BRISCOLA_MODEL_URL",
    "BRISCOLA_MODEL_SHA256",
    "BRISCOLA_VALUE_MODEL_URL",
    "BRISCOLA_VALUE_MODEL_SHA256",
    # Vista full-state di debug: deve restare opt-in anche se abilitata nella shell di chi lancia i test.
    "BRISCOLA_DEBUG_STATE_ENDPOINT",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rimuove le env che selezionano backend esterni, così i test non dipendono dall'ambiente."""
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    # Rate limit sulla creazione partite disattivato di default: la suite crea decine di
    # partite dallo stesso "IP" (TestClient) in pochi secondi e scatterebbero 429 spuri.
    # I test del rate limit lo riabilitano esplicitamente con un valore piccolo.
    monkeypatch.setenv("BRISCOLA_CREATE_GAME_RATE_LIMIT", "0")
