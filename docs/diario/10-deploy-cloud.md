# Approfondimento — Il deploy cloud

**Capitolo del diario:** [Intermezzo — La porta si apre](https://ai.briscola.dev/diario) · **Periodo:** 23–25 giugno 2026

## Dalla demo locale al sito pubblico in tre giorni

| Problema | Soluzione | Commit |
|---|---|---|
| Il server cloud gira su PIÙ repliche: lo stato in RAM di una replica è invisibile alle altre | `GameSessionStore` con backend Redis + lock distribuito per partita | `1003602`, `6b6b20a` |
| Gli eventi realtime (WebSocket) devono raggiungere il client su QUALSIASI replica | fan-out via pub/sub Redis; snapshot per-giocatore ricostruiti dal subscriber (anti-cheat) | `461ba0f`, `624c0e1` |
| I modelli `.npz` sono gitignored: il cloud non li ha | provisioning allo startup da URL di release GitHub con verifica SHA256 | `0541e36` |
| Event log locale SQLite non adatto a repliche | backend Postgres (Neon) selezionato da `DATABASE_URL` | `f2ebb2d` |
| Versioni/asset da verificare al volo | `/health`, `/version` con presenza modelli e stato event log | `2dd1773` |

## Note

- Il polling era stato promosso a default per il cloud (`7b217ce`) e ridimensionato a
  fallback di debug quando il pub/sub ha funzionato.
- In locale tutto continua a girare in-memory + SQLite: Redis/Postgres si attivano solo
  se le env `REDIS_URL`/`DATABASE_URL` sono presenti (import lazy).
- Il sito vive su <https://ai.briscola.dev> (FastAPI Cloud, entrypoint `main:app`).
