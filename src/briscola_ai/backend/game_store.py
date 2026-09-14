"""
Game store condiviso (Fase 1): sposta lo stato partita da memoria di processo a uno store
accessibile da tutte le repliche (Redis in cloud), così azioni e WebSocket che finiscono su
repliche diverse non danno più "partita non trovata".

Design:
- `GameSession` = stato + metadati minimi necessari a riprendere la partita da qualsiasi replica:
  `GameState`, `version` (server_version monotono), config IA per posto (nome agente + model_id,
  NON l'oggetto Agent, che viene ricostruito con `build_agent`), `action_seed` per RNG deterministico.
- `GameSessionStore` astratto con due implementazioni:
  - `InMemoryGameSessionStore` (default, dev/test): nessuna dipendenza esterna;
  - `RedisGameSessionStore` (prod): attivo solo se è configurato un URL Redis. Import di `redis` lazy.
- Serializzazione JSON (carte come id via `domain.serialization`), quindi entrambe le implementazioni
  hanno la **stessa semantica**: le mutazioni non persistono finché non si chiama `set`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..domain.serialization import game_state_from_dict, game_state_to_dict
from ..domain.state import GameState

# TTL default delle sessioni (rinnovato a ogni `set`). Una partita di Briscola è breve;
# qualche ora copre abbondantemente una sessione attiva senza accumulare stato all'infinito.
DEFAULT_SESSION_TTL_SECONDS = 6 * 60 * 60

# Nomi env candidati per l'URL Redis (override esplicito prima, poi convenzioni comuni).
_REDIS_URL_ENV_CANDIDATES = ("BRISCOLA_REDIS_URL", "REDIS_URL", "REDISCLOUD_URL")


@dataclass
class AiSeatConfig:
    """Config IA di un posto: come ricostruire l'agente (no oggetto Agent, non serializzabile)."""

    agent_name: str
    model_id: str | None = None


@dataclass
class GameSession:
    """Sessione partita serializzabile, sufficiente a riprendere il gioco da qualsiasi replica."""

    game_id: str
    state: GameState
    version: int
    ai_seats: dict[int, AiSeatConfig]
    action_seed: int
    created_at: str
    updated_at: str


def session_to_json(session: GameSession) -> str:
    """Serializza una `GameSession` in JSON (stato via `game_state_to_dict`)."""
    payload: dict[str, Any] = {
        "game_id": session.game_id,
        "state": game_state_to_dict(session.state),
        "version": int(session.version),
        "ai_seats": {
            str(seat): {"agent_name": cfg.agent_name, "model_id": cfg.model_id}
            for seat, cfg in session.ai_seats.items()
        },
        "action_seed": int(session.action_seed),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }
    return json.dumps(payload)


def session_from_json(raw: str) -> GameSession:
    """Deserializza una `GameSession` prodotta da `session_to_json`."""
    data = json.loads(raw)
    return GameSession(
        game_id=str(data["game_id"]),
        state=game_state_from_dict(data["state"]),
        version=int(data["version"]),
        ai_seats={
            int(seat): AiSeatConfig(agent_name=str(cfg["agent_name"]), model_id=cfg.get("model_id"))
            for seat, cfg in data["ai_seats"].items()
        },
        action_seed=int(data["action_seed"]),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
    )


@runtime_checkable
class GameSessionStore(Protocol):
    """Interfaccia minima dello store sessioni partita."""

    async def get(self, game_id: str) -> GameSession | None: ...

    async def set(self, session: GameSession, *, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> None: ...

    async def delete(self, game_id: str) -> None: ...

    def lock(self, game_id: str) -> contextlib.AbstractAsyncContextManager[None]: ...

    async def publish(self, game_id: str, message: str) -> None:
        """Pubblica un messaggio sul canale eventi della partita (per fan-out WebSocket multi-replica)."""
        ...

    def subscribe(self, game_id: str) -> AsyncGenerator[str]:
        """Async generator che produce i messaggi pubblicati sul canale eventi della partita."""
        ...


class InMemoryGameSessionStore:
    """Store in-memory (default dev/test). Serializza come Redis per avere identica semantica."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Pub/sub locale: per ogni partita, l'insieme delle code dei subscriber attivi.
        self._subscribers: dict[str, set[asyncio.Queue[str]]] = {}

    async def get(self, game_id: str) -> GameSession | None:
        raw = self._data.get(game_id)
        return session_from_json(raw) if raw is not None else None

    async def set(self, session: GameSession, *, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> None:
        # ttl_seconds ignorato in-memory (niente scadenza): coerente per dev/test.
        self._data[session.game_id] = session_to_json(session)

    async def delete(self, game_id: str) -> None:
        self._data.pop(game_id, None)
        # Rimuoviamo anche il lock per non accumulare un entry per ogni partita mai creata
        # (leak lento ma illimitato in processi long-running). Solo se non è detenuto:
        # rimuovere un lock mentre qualcuno lo tiene creerebbe due lock per la stessa partita.
        lk = self._locks.get(game_id)
        if lk is not None and not lk.locked():
            self._locks.pop(game_id, None)

    @contextlib.asynccontextmanager
    async def lock(self, game_id: str) -> AsyncIterator[None]:
        lk = self._locks.setdefault(game_id, asyncio.Lock())
        async with lk:
            yield

    async def publish(self, game_id: str, message: str) -> None:
        for queue in list(self._subscribers.get(game_id, ())):
            queue.put_nowait(message)

    async def subscribe(self, game_id: str) -> AsyncGenerator[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.setdefault(game_id, set()).add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            subs = self._subscribers.get(game_id)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    self._subscribers.pop(game_id, None)


class RedisGameSessionStore:
    """Store su Redis (prod). `redis` è importato lazy; il client può essere iniettato per i test."""

    def __init__(self, url: str | None = None, *, client: Any = None) -> None:
        if client is not None:
            self._redis = client
        elif url is not None:
            import redis.asyncio as redis_asyncio  # import lazy: solo se si usa Redis
            from redis.asyncio.retry import Retry
            from redis.backoff import ExponentialBackoff
            from redis.exceptions import ConnectionError as RedisConnectionError
            from redis.exceptions import TimeoutError as RedisTimeoutError

            # Robustezza ai blip (incidente 2026-07-06: "Connection refused" transitori):
            # - retry con backoff esponenziale DENTRO il comando (3 tentativi, ~50-500ms):
            #   un singhiozzo di rete si auto-ripara senza mai diventare un errore utente;
            # - health_check_interval: una connessione rimasta inattiva viene verificata
            #   (PING) prima dell'uso, cosi' le connessioni morte nel pool non esplodono
            #   in faccia alla prima richiesta dopo una pausa.
            self._redis = redis_asyncio.from_url(
                url,
                decode_responses=True,
                retry=Retry(ExponentialBackoff(cap=0.5, base=0.05), retries=3),
                retry_on_error=[RedisConnectionError, RedisTimeoutError],
                health_check_interval=30,
            )
        else:
            raise ValueError("RedisGameSessionStore richiede `url` oppure `client`.")

    @staticmethod
    def _key(game_id: str) -> str:
        return f"game:{game_id}"

    @staticmethod
    def _channel(game_id: str) -> str:
        return f"game:{game_id}:events"

    async def get(self, game_id: str) -> GameSession | None:
        raw = await self._redis.get(self._key(game_id))
        return session_from_json(raw) if raw is not None else None

    async def set(self, session: GameSession, *, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> None:
        await self._redis.set(self._key(session.game_id), session_to_json(session), ex=ttl_seconds)

    async def delete(self, game_id: str) -> None:
        await self._redis.delete(self._key(game_id))

    @contextlib.asynccontextmanager
    async def lock(self, game_id: str) -> AsyncIterator[None]:
        # Lock distribuito Redis: serializza le azioni concorrenti sulla stessa partita,
        # anche tra repliche diverse.
        #
        # `timeout` è il TTL di sicurezza del lock (rilascio automatico se il detentore muore).
        # Deve essere ampiamente superiore alla mossa IA più lenta (PIMC/value-lookahead
        # possono richiedere secondi sotto carico): se scadesse mentre è logicamente detenuto,
        # un'operazione concorrente potrebbe mutare lo stato in parallelo. 120s è un compromesso:
        # ordini di grandezza sopra una mossa normale, ma recupera comunque un processo morto.
        redis_lock = self._redis.lock(f"game:{game_id}:lock", timeout=120, blocking_timeout=10)
        acquired = await redis_lock.acquire()
        if not acquired:
            # Dopo `blocking_timeout` non abbiamo ottenuto il lock: NON cediamo il contesto
            # (altrimenti muteremmo lo stato senza esclusione). Il chiamante gestirà l'errore.
            raise TimeoutError(f"Lock Redis non acquisito per la partita {game_id}")
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await redis_lock.release()

    async def publish(self, game_id: str, message: str) -> None:
        await self._redis.publish(self._channel(game_id), message)

    async def subscribe(self, game_id: str) -> AsyncGenerator[str]:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel(game_id))
        try:
            async for msg in pubsub.listen():
                # `listen()` emette anche messaggi di servizio (subscribe/unsubscribe): filtriamo.
                if msg.get("type") == "message":
                    yield msg["data"]
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(self._channel(game_id))
            with contextlib.suppress(Exception):
                await pubsub.aclose()


def resolve_redis_url() -> str | None:
    """Ritorna l'URL Redis dalle env candidate (override esplicito prima), o None."""
    for name in _REDIS_URL_ENV_CANDIDATES:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def build_game_session_store() -> GameSessionStore:
    """Crea lo store: Redis se è configurato un URL, altrimenti in-memory (default dev/test)."""
    url = resolve_redis_url()
    if url:
        return RedisGameSessionStore(url)
    return InMemoryGameSessionStore()
