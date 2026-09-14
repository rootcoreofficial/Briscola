"""
Test del game store condiviso (`GameSessionStore`): in-memory e Redis (via fakeredis).

Il caso chiave e' "due repliche che condividono lo store": dimostra che una partita creata da una
replica e' trovabile dall'altra (la causa di "partita non trovata" sotto autoscaling).
"""

from __future__ import annotations

import asyncio

import pytest

from briscola_ai.backend.game_store import (
    AiSeatConfig,
    GameSession,
    InMemoryGameSessionStore,
    RedisGameSessionStore,
    build_game_session_store,
    resolve_redis_url,
    session_from_json,
    session_to_json,
)
from briscola_ai.domain.state import new_game_state


def _make_session(game_id: str = "g1", version: int = 1) -> GameSession:
    return GameSession(
        game_id=game_id,
        state=new_game_state(num_players=2, seed=11),
        version=version,
        ai_seats={1: AiSeatConfig(agent_name="bc_model", model_id="best_a2c_v3.npz")},
        action_seed=42,
        created_at="2026-06-24T00:00:00Z",
        updated_at="2026-06-24T00:00:00Z",
    )


def test_session_json_roundtrip() -> None:
    """Serializzare e rideserializzare una GameSession (stato, versione, seat IA, seed)
    deve restituire un oggetto uguale campo per campo."""
    s = _make_session()
    assert session_from_json(session_to_json(s)) == s


def test_inmemory_set_get_delete_and_no_leak() -> None:
    """Lo store in-memory deve supportare set/get/delete e isolare le copie: mutare l'oggetto
    restituito non deve persistere senza un nuovo set() (stessa semantica di Redis)."""
    store = InMemoryGameSessionStore()

    async def scenario() -> None:
        await store.set(_make_session(version=1))
        got = await store.get("g1")
        assert got is not None and got.version == 1

        # Mutare l'oggetto restituito NON deve persistere senza set() (semantica come Redis).
        got.version = 999
        again = await store.get("g1")
        assert again is not None and again.version == 1

        await store.delete("g1")
        assert await store.get("g1") is None

    asyncio.run(scenario())


def test_inmemory_lock_is_usable() -> None:
    """Il lock in-memory deve funzionare come context manager async, permettendo di scrivere
    e poi rileggere la sessione all'interno della sezione protetta."""
    store = InMemoryGameSessionStore()

    async def scenario() -> None:
        async with store.lock("g1"):
            await store.set(_make_session())
        assert await store.get("g1") is not None

    asyncio.run(scenario())


def _fake_redis_pair():
    """Due client fakeredis che condividono lo stesso server (simula due repliche)."""
    from fakeredis import FakeServer, aioredis

    server = FakeServer()
    c1 = aioredis.FakeRedis(server=server, decode_responses=True)
    c2 = aioredis.FakeRedis(server=server, decode_responses=True)
    return c1, c2


def test_redis_store_set_get_delete() -> None:
    """Lo store Redis (via fakeredis) deve supportare set/get/delete preservando i campi
    della sessione (es. version)."""
    from fakeredis import aioredis

    store = RedisGameSessionStore(client=aioredis.FakeRedis(decode_responses=True))

    async def scenario() -> None:
        await store.set(_make_session(version=3))
        got = await store.get("g1")
        assert got is not None and got.version == 3
        await store.delete("g1")
        assert await store.get("g1") is None

    asyncio.run(scenario())


def test_redis_two_replicas_share_session() -> None:
    """Il caso che risolve "partita non trovata": replica A crea, replica B trova."""
    c1, c2 = _fake_redis_pair()
    replica_a = RedisGameSessionStore(client=c1)
    replica_b = RedisGameSessionStore(client=c2)

    async def scenario() -> None:
        await replica_a.set(_make_session(game_id="shared", version=5))
        got = await replica_b.get("shared")
        assert got is not None
        assert got.game_id == "shared"
        assert got.version == 5
        assert got.state == new_game_state(num_players=2, seed=11)

    asyncio.run(scenario())


def test_redis_lock_is_usable() -> None:
    """Il lock Redis deve funzionare come context manager async, permettendo di scrivere
    e poi rileggere la sessione all'interno della sezione protetta."""
    from fakeredis import aioredis

    store = RedisGameSessionStore(client=aioredis.FakeRedis(decode_responses=True))

    async def scenario() -> None:
        async with store.lock("g1"):
            await store.set(_make_session())
        assert await store.get("g1") is not None

    asyncio.run(scenario())


def test_factory_selects_store_by_env(monkeypatch) -> None:
    """La factory deve scegliere lo store in base all'ambiente: in-memory senza URL Redis,
    RedisGameSessionStore quando `BRISCOLA_REDIS_URL` è impostata."""
    monkeypatch.delenv("BRISCOLA_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDISCLOUD_URL", raising=False)
    assert resolve_redis_url() is None
    assert isinstance(build_game_session_store(), InMemoryGameSessionStore)

    monkeypatch.setenv("BRISCOLA_REDIS_URL", "redis://localhost:6379/0")
    assert resolve_redis_url() == "redis://localhost:6379/0"
    assert isinstance(build_game_session_store(), RedisGameSessionStore)


def test_redis_lock_raises_if_not_acquired() -> None:
    """Se `acquire()` ritorna False (timeout), il context manager NON deve cedere il contesto."""

    class _FakeLock:
        async def acquire(self) -> bool:
            return False

        async def release(self) -> None:
            return None

    class _FakeRedis:
        def lock(self, *args: object, **kwargs: object) -> _FakeLock:
            return _FakeLock()

    store = RedisGameSessionStore(client=_FakeRedis())

    async def scenario() -> None:
        entered = False
        with pytest.raises(TimeoutError):
            async with store.lock("g1"):
                entered = True
        assert entered is False  # il blocco protetto NON deve essere eseguito

    asyncio.run(scenario())


def test_inmemory_pubsub_delivers_messages() -> None:
    """Il pub/sub in-memory deve recapitare al subscriber, nell'ordine, i messaggi pubblicati su un canale."""
    store = InMemoryGameSessionStore()

    async def scenario() -> None:
        received: list[str] = []

        async def reader() -> None:
            async for m in store.subscribe("g1"):
                received.append(m)
                if len(received) >= 2:
                    break

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.05)  # lascia registrare il subscriber
        await store.publish("g1", "a")
        await store.publish("g1", "b")
        await asyncio.wait_for(task, timeout=2)
        assert received == ["a", "b"]

    asyncio.run(scenario())


def test_redis_pubsub_across_replicas() -> None:
    """Pub/sub Redis: un evento pubblicato da una replica raggiunge il subscriber su un'altra."""
    c1, c2 = _fake_redis_pair()
    publisher = RedisGameSessionStore(client=c1)
    subscriber = RedisGameSessionStore(client=c2)

    async def scenario() -> None:
        received: list[str] = []

        async def reader() -> None:
            async for m in subscriber.subscribe("g1"):
                received.append(m)
                break

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.1)  # lascia completare il subscribe
        await publisher.publish("g1", "hello")
        await asyncio.wait_for(task, timeout=3)
        assert received == ["hello"]

    asyncio.run(scenario())
