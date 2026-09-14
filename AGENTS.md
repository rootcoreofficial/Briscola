# Repository Guidelines

> Questo file è il documento di riferimento condiviso per gli agenti (Claude Code, Codex, …).
> Progetto **didattico**: la chiarezza (docstring, commenti, spiegazioni del "perché") conta quanto la correttezza.

## Big picture (architettura)

Briscola end-to-end: motore di regole puro → backend HTTP/WS → UI → pipeline dati → training/valutazione di IA. Il punto chiave da capire prima di toccare il codice è che **esistono due implementazioni parallele dello stesso gioco**, tenute volutamente in sincronia:

1. **Dominio canonico** (`src/briscola_ai/domain/`) — fonte di verità, puro e testabile.
   - `models.py`: `Card`, `Suit`, `Rank` (carte canoniche).
   - `state.py`: `GameState`/`PlayerState` immutabili (`@dataclass(frozen=True, slots=True)`), serializzabili → abilitano replay deterministico.
   - `engine.py`: transizione pura `step(state, PlayCardAction) -> StepResult`, deterministica dato seme/stato. Supporta **2 e 4 giocatori** (4p a squadre).
   - `rules.py`: regole isolate (es. `who_wins_trick`).
   - `observation.py` + `card_id.py`: `PlayerObservation` (vista parziale lecita) e la mappatura carta ↔ id canonico `[0,39]`.

2. **Fast path** (`src/briscola_ai/ai/fast/` in puro Python su interi/array NumPy, con kernel `@njit` Numba in `src/briscola_ai/ai/numba/`) — reimplementazione 2-player ad alto throughput. Serve per self-play e training massivi. Copre solo gli agenti tradotti nel path fast (`random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`). **Deve restare coerente col dominio**: i test (`tests/test_fast_*`) verificano la parità. Se cambi una regola nel dominio, aggiorna anche il fast path e i suoi test.

**Anti-cheat (invariante centrale):** gli agenti non ricevono mai `GameState` completo, solo `PlayerObservation`. Vale per le baseline (`ai/agents/`), i modelli (`ai/models/bc_model.py`) e il reward shaping (`ai/training/reward_shaping.py` usa solo informazione pubblica + mano del giocatore). Non introdurre scorciatoie che leggano carte nascoste o l'ordine del mazzo.

**Backend** (`src/briscola_ai/backend/`): FastAPI montato sotto `/api` da `main.py` (che serve anche la UI e gli statici ed espone `/health` e `/version`). `dto.py` = contratto Pydantic v2; `server.py` = REST + WebSocket (architettura ibrida: REST per le azioni, WS per gli aggiornamenti). Il backend **avanza automaticamente** quando è il turno dell'IA e **non** introduce delay di presentazione (quello è compito del frontend). Punti chiave del runtime (pensati per il deploy cloud multi-replica):

- **Stato partita**: vive in un `GameSessionStore` (`game_store.py`) — `InMemoryGameSessionStore` in locale, `RedisGameSessionStore` se è impostata `REDIS_URL`. Salva `GameState` (serializzato via `domain/serialization.py`) + la config IA per posto; l'oggetto `Agent` **non** è serializzato (si ricostruisce con `build_agent`, con cache). Lock per partita (asyncio in-memory / lock Redis distribuito) per serializzare le azioni concorrenti. Evita di reintrodurre dict globali di stato (`active_games` & co. sono stati rimossi).
- **Realtime**: il fan-out degli eventi WebSocket passa per il **pub/sub** dello store (Redis in prod, fan-out asyncio in dev), così `ai_card_reveal`/`trick_result`/snapshot raggiungono il client su QUALSIASI replica. Gli snapshot per-giocatore sono ricostruiti dal subscriber (anti-cheat). `?polling=1` resta come fallback di debug.
- **Event log** (`event_log.py`): append-only per dataset. `EventLog` su SQLite (default locale) **oppure** `PostgresEventLog` se è impostata `DATABASE_URL` (deploy persistente/condiviso); il backend è scelto da `build_event_log`. Stesso schema `games`/`events`. Attivo solo se configurato (`BRISCOLA_EVENT_DB_PATH` o `DATABASE_URL`); con modalità `dataset` richiede il consenso utente.
- **Provisioning modello** (`ai/models/provisioning.py`): gli asset runtime ufficiali piccoli sono tracciati in Git;
  allo startup il provisioning funge da fallback/override best-effort. Se un asset manca, può scaricare la policy da
  `BRISCOLA_MODEL_URL`, il value model da `BRISCOLA_VALUE_MODEL_URL` e la belief network da
  `BRISCOLA_BELIEF_MODEL_URL`, verificando i rispettivi SHA-256. Un errore non blocca l'avvio.

**Modelli locali (`.npz`)**: la UI seleziona un avversario `bc_model` da un catalogo server-side (`ai/models/catalog.py`). Il browser invia solo un `ai_model_id` (path relativo) tra quelli di `GET /api/ai/models`; il backend rifiuta path traversal e carica solo da `BRISCOLA_MODELS_DIR` (default `./data/models/`). I trainer salvano nei `.npz` i metadati (`label`, `description_it`, `feature_dim`).

> **Nota peso `.npz`:** i pesi della MLP sono piccoli (~0.18 MB per v6). Non salvare l'intera storia `metrics`
> nel `metadata_json` dei modelli da pubblicare come release asset o usare in cloud: `np.savez` non comprime e una
> stringa JSON NumPy (`<U...`) può occupare circa 4 byte per carattere. I vecchi `best_a2c_v3/4/5.npz` pesano
> 41–52 MB per questo motivo, mentre `best_a2c_v6.npz` pesa ~0.2 MB con metadata sintetici. Per run lunghi usare
> metadata essenziali o `--metrics-mode summary`. La dimensione del file non indica capacità del modello né quantità
> di training.

**Pipeline ML** (vedi `README.md` e `PLAN.md` per il dettaglio): dominio testabile → backend/UI → event log (SQLite in locale, **Postgres** in cloud) → export JSONL versionato (`export_dataset.py` / `export_live_actions.py`, SQLite o Postgres) → self-play → valutazione offline riproducibile → training (BC/PG/A2C). `PLAN.md` è la **fonte di verità su cosa fare dopo**: è volutamente breve (stato corrente + prossime azioni), leggilo per intero prima di pianificare.

**Regola di manutenzione di `PLAN.md`**: se una riga non cambia cosa facciamo domani e ha già
un riferimento stabile nel diario o in `docs/plans/`, toglila dal piano e lascia al massimo
un rimando breve. Il piano deve restare operativo: stato reale, prossima decisione, vincoli
che impediscono errori futuri e comandi utili. La cronaca degli esperimenti, una volta
raccontata nel diario/approfondimento, non va duplicata lì.

## Setup, Build, and Run

Richiede **Python 3.14** e [`uv`](https://github.com/astral-sh/uv).

- Crea env: `uv venv -p python3.14`
- Installa (editable): `uv pip install -e .`
- Dev deps: `uv pip install -e ".[dev]"`
- Avvia server: `briscola-server --reload` (UI su `http://localhost:8000`; `--host`/`--port` supportati)
- Simulazioni headless: `python scripts/simulate_games.py --num-games 100 --seed 42`

Deps di runtime per il cloud: `redis` (game store) e `psycopg` (event log Postgres) sono già in `dependencies`, ma importate **lazy** — usate solo se `REDIS_URL`/`DATABASE_URL` sono impostate (in locale tutto gira in-memory + SQLite). Dev-only: `fakeredis` (test dello store Redis) e `playwright` (ispezione UI/layout; richiede `python -m playwright install chromium`).

### Runtime & deploy (variabili d'ambiente)

Tutte opzionali; in locale i default vanno bene. In cloud multi-replica serve Redis per stato/realtime condivisi e,
se si abilita l'event log persistente, Postgres. Gli asset runtime ufficiali sono già tracciati in Git; il
provisioning è un fallback/override opzionale. Il sito è live su `https://ai.briscola.dev`; l'entrypoint per
`fastapi run` è lo shim `main:app` nella root.

> **Deploy automatico al push**: FastAPI Cloud è collegato al repo GitHub e ridistribuisce **da solo** a ogni push su `master` (il build gira `uv sync --locked` → ricordarsi `uv lock` a ogni bump di versione). NON serve — e non va lanciato — `fastapi deploy` manualmente dall'agente. Dopo il push, verificare l'esito da `GET /version` (versione live) e dai log della piattaforma; il build impiega ~1-3 min. Nota **scale-to-zero**: idle ~90s, il cold start è frequente (mitigato da un keep-alive esterno se configurato).

- `REDIS_URL` (o `BRISCOLA_REDIS_URL`): attiva `RedisGameSessionStore` + pub/sub realtime. Se assente → in-memory + WebSocket diretto.
- `DATABASE_URL` (o `BRISCOLA_DATABASE_URL`): attiva l'event log su Postgres. Se assente → SQLite (`BRISCOLA_EVENT_DB_PATH`) o disabilitato.
- `BRISCOLA_EVENT_LOG_MODE`: `debug` (default) | `dataset` (minimale, richiede consenso) | `off`.
- `BRISCOLA_MODEL_URL` + `BRISCOLA_MODEL_SHA256` (+ `BRISCOLA_DEFAULT_MODEL_ID`): provisioning del modello consigliato allo startup.
- `BRISCOLA_VALUE_MODEL_URL` + `BRISCOLA_VALUE_MODEL_SHA256`: provisioning del value model richiesto da `bc_model_value_lookahead_8x8`.
- `BRISCOLA_BELIEF_MODEL_URL` + `BRISCOLA_BELIEF_MODEL_SHA256`: provisioning della belief network richiesta dagli
  agenti `bc_model_pimc_belief_*`.
- `BRISCOLA_MODELS_DIR` (default `./data/models/`); `BRISCOLA_CORS_ALLOW_ORIGINS` (default `*`, restringere in prod).
- `BRISCOLA_REALTIME_MODE` (`ws`|`polling`, override; default `ws`); `BRISCOLA_ASSET_VERSION` (override del cache-busting, che di default deriva da versione + mtime degli static).
- `BRISCOLA_DEBUG_STATE_ENDPOINT=unsafe-full-state`: abilita la vista full-state di `GET /api/games/{id}` senza
  `player_index` (mani di tutti + `next_deck_card`, per debug/spectator). Default **disabilitata** (403); i valori
  booleani come `1` sono rifiutati deliberatamente. Non impostarla nella produzione pubblica.
- `BRISCOLA_CREATE_GAME_RATE_LIMIT`: tetto di partite create per IP al minuto su `POST /api/games` (default `30`; `0` disabilita — la suite test lo azzera via conftest).

### Pipeline AI (script in `scripts/`)

- Self-play → DB: `python scripts/self_play_to_db.py` (verso SQLite, no HTTP)
- Self-play veloce (summary-only, no DB): `python scripts/fast_self_play.py`
- Export dataset: `python scripts/export_dataset.py` (SQLite → JSONL versionato)
- Training BC (imitation): `python scripts/train_bc.py` (legge JSONL, target = action_id `[0,39]` + action mask;
  split train/validation/test per `game_id`, default 80/10/10)
- Training value: `python scripts/train_value.py` / `python scripts/train_value_pairwise.py` (split per partita;
  gli NPZ compatti devono contenere `game_ids`, senza fallback allo split per record)
- Training RL: `python scripts/train_a2c.py` / `python scripts/train_pg.py` (salvano `.npz` in `data/models/`)
  - A2C long-run: `--num-games` resta l'orizzonte totale, `--stop-after-games` chiude il segmento e
    `--resume checkpoint.npz` ripristina policy, critic, Adam, RNG, schedule e metriche. Usare
    `--metrics-mode summary --diagnostics-every N`; il resume rifiuta configurazioni, asset o commit diversi.
    Lo scouting v14 50M usa il launcher congelato `scripts/run_a2c_super_training_50m.sh start 10|20|30|40|50`;
    ogni target va autorizzato separatamente dopo lo screen precedente.
- Pipeline riproducibile (train + eval matrix + manifest): `python scripts/run_experiment.py`
- Valutazione offline: `python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 heuristic_v1 --agent1 random`
  - engine selezionabile: `--engine domain|fast|numba`; modi seat-fair `--seat-fair --seed-suite small|medium`; benchmark `--benchmark medium|big`; carica modello con `--agent0 bc_model --agent0-model ./data/models/best_a2c.npz`
- Matrix / decision quality: `scripts/evaluate_matrix.py`, `scripts/evaluate_decision_quality.py`
- Audit appaiato su observation live: `scripts/audit_live_policy_replay.py`
- Gate belief multi-stile riprendibile: `python scripts/run_belief_v1_gate.py --resume` (dataset da roster, fold
  leave-one-opponent-out, stop automatico prima del candidato se le metriche offline falliscono)
- Diagnostica salute A2C: `python scripts/run_a2c_health_probe.py --resume` (tre seed brevi, telemetria passiva e
  instradamento preregistrato verso una sola ablation; non promuove modelli)
- Confronto schedule A2C: `python scripts/run_a2c_paired_schedule_probe.py --resume` (seriale vs paired su tre seed,
  sia a pari partite sia a pari mazzi; job lungo da affidare al maintainer)
- Diagnostica simmetria semi: `scripts/probe_suit_symmetry.py` (24 rinomine complete di `PlayerObservation`,
  mosse forzate escluse, report JSON riproducibile)
- Distillazione simmetrica sharded: `generate_suit_distillation_shards.py` crea manifest + NPZ riprendibili con
  split globale per `game_id`; `train_suit_distillation_shards.py` apre un solo shard alla volta. Il run teacher
  20M da 250k usa `scripts/run_suit_distillation_20m_250k.sh` e separa obbligatoriamente raccolta e training.
- Diagnostica errori policy: `scripts/probe_policy_regret.py` (controfattuali PIMC cross-fitted, endgame esatto,
  contesto pubblico e routing automatico per cluster)
- Benchmark throughput: `python scripts/benchmark_perf.py`

Dataset, run e benchmark in `data/` e `benchmarks/experiments/` sono locali (gitignored), salvo i piccoli asset
runtime ufficiali esplicitamente tracciati in `data/models/`.

## Quality gate (prima di ogni commit)

Se le modifiche toccano Python (codice/test), esegui e correggi:

- `ruff format src tests scripts` e `ruff check --fix src tests scripts` (import sorting incluso via regole `I`; **niente** `black`/`isort`)
- `mypy src`
- `make docs-check` (JSON rigoroso, link locali, versione/modello, capitoli diario e report Excel aggiornato)
- `pytest`
  - singolo file: `pytest tests/test_trick_rules.py`
  - singolo test: `pytest tests/test_trick_rules.py::test_name` (o `pytest -k "pattern"`)
  - ciclo rapido (salta subprocess e compilazioni JIT): `pytest -m "not slow and not numba"`
  - coverage: `pytest --cov=briscola_ai`

La CI GitHub Actions (`.github/workflows/ci.yml`) replica lo stesso gate (ruff format/check, mypy, docs-check,
pytest+coverage) su ogni push/PR: deve restare allineata a questa sezione.

Aggiorna `PLAN.md` se necessario (deve riflettere lo stato reale del repo). Il badge coverage in `README.md` è aggiornato automaticamente dalla CI sui push a `master`: non toccarlo a mano. Tieni aggiornato questo file quando cambiano tooling/convenzioni.

## Coding Style & Naming

- Indentazione: 4 spazi per Python; JS/CSS coerente coi file esistenti. Line length 120 (ruff, `py314`).
- Naming: `snake_case` per funzioni/variabili, `PascalCase` per le classi; moduli piccoli e dentro il loro layer (`domain/`, `backend/`, `frontend/`, `ai/`).
- **Documentation-first**: docstring chiare e dettagliate per moduli/classi/funzioni pubbliche (intento, input/output, invarianti).
- **Commenting-first**: spiega logica non ovvia, edge case, e cosa asserisce ogni scenario di test.
- **Agent communication**: per ogni iterazione spiega cosa è cambiato e perché in modo didattico (assunzioni, ragionamento, come verificare in locale).
- **Comandi lunghi al maintainer**: i job che superano ~5 minuti (training, valutazioni grosse) NON vanno lanciati dall'agente restando in attesa. Prepara il comando ESATTO pronto da incollare, con output su file di log (`nohup ... > log 2>&1 &`) e i path degli artefatti attesi; indica cosa guardare nel log per capire l'esito e qual è il passo successivo. Il maintainer lo lancia quando vuole; l'agente riprende dai log/artefatti (anche in una sessione successiva) e registra i risultati in `PLAN.md`/`docs/plans/`.

## Testing

`pytest` (dev dep); test in `tests/` come `test_*.py`. Coprono invarianti di dominio, casi limite endgame, parità fast/numba vs dominio, encoder osservazioni, reward shaping, integrazione API/WS, game store (in-memory/Redis via `fakeredis`, incl. pub/sub) e backend event log (SQLite + Postgres con connessione fake). Un fixture autouse in `tests/conftest.py` azzera `REDIS_URL`/`DATABASE_URL` per ogni test, così la suite resta **ermetica** (non contatta servizi reali anche se quelle env sono presenti nell'ambiente).

## Commits & Pull Requests

- Commit (formato): `type(scope): summary` — es. `feat(domain): aggiungi helper punteggio`, `fix(ui): correggi animazione carte`.
  - **IMPORTANTE**: le descrizioni dei commit devono essere SEMPRE in italiano (finalità didattiche).
- Versioning (SemVer, pre-1.0): versione canonica in `pyproject.toml` (`[project].version`).
  - `0.1.x` (patch): fix/refactor/test/doc/performance senza cambiare contratti pubblici.
  - `0.(n+1).0` (minor): nuove feature visibili o cambi rilevanti a componenti/pipeline; anche per cambi **breaking** ai contratti pubblici (API/WS/DTO o schema dataset/export), con nota esplicita.
  - L'agente segnala quando un change merita un bump e propone la versione; la decisione finale resta al maintainer.
  - **`uv lock` per ogni bump**: il lockfile registra anche la versione di `briscola-ai`; senza
    rigenerarlo il build di FastAPI Cloud fallisce (`uv sync --locked` rifiuta il mismatch —
    successo il 2026-07-07 con la 0.30.0). Quindi: bump in `pyproject.toml` → `uv lock` →
    committa ENTRAMBI i file insieme.
  - **Tag git per ogni bump**: dopo aver bumpato `pyproject.toml`, crea il tag annotato corrispondente e pushalo, così la serie `vX.Y.Z` su GitHub resta completa e allineata alla history:
    ```bash
    git tag -a vX.Y.Z -m "Versione X.Y.Z"
    git push <remote> vX.Y.Z
    ```
    Nota: `get_code_version()` (footer UI, `/version`, cache-busting) legge la versione da `pyproject.toml` anche in editable, quindi non serve reinstallare per vederla aggiornata.
  - **Verifica catalogo/UI a ogni cambio di asset**: dopo promozioni o nuovi asset nella directory
    modelli, controlla ENTRAMBI i versi del catalogo (`GET /api/ai/models`): il nuovo modello deve
    apparire `is_compatible: true`, e gli asset interni (value/belief, format `value_mlp_v1`/`belief_mlp_v1`)
    NON devono apparire come selezionabili. Le due regressioni di v0.22.x/v0.23.x nascevano qui.
  - **Report modelli per ogni release**: a ogni nuova release rigenera `docs/reports/model_progress.xlsx` con
    `uv run python scripts/build_model_report.py`. Controlla esplicitamente il foglio Dashboard e in particolare il
    grafico di progressione: deve includere il nuovo best/versione promossa e il range del grafico deve arrivare
    all'ultima riga dei modelli ufficiali.
- PR: descrivi il change, includi passi di riproduzione per i bug, e screenshot per cambi UI (`src/briscola_ai/frontend/static/`).
