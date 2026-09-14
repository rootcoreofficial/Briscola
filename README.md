# Briscola AI

![Coverage](https://img.shields.io/badge/coverage-70%25-yellow)

Un progetto didattico “end‑to‑end” nato da un’esigenza concreta: **studiare le reti neurali con un progetto reale**, non con esempi astratti.

La Briscola è un ottimo “laboratorio” perché obbliga a mettere insieme tutti i pezzi:
- un **motore di regole** corretto e testabile;
- un **backend** che espone un contratto stabile (API/WS);
- una **UI** che rende leggibile la sequenza delle mani;
- una **pipeline dati** per arrivare a dataset, baseline e training.

Obiettivo: arrivare a un’IA (rete neurale) che impari a giocare in modo riproducibile, misurabile e spiegabile.

> Questo README spiega *come* usare e *perché* è fatto così (parte didattica). Per lo stato corrente e le prossime
> azioni vedi `PLAN.md`; il dettaglio operativo dei comandi è in `--help` di ogni script.

## Funzionalità

- Regole complete della Briscola; motore (`domain/`) con **2 giocatori** e **4 giocatori** (a squadre).
- Interfaccia web con aggiornamenti in tempo reale via WebSocket.
- IA selezionabile, server‑driven (il backend avanza la partita quando tocca all’IA):
  - baseline: `random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`;
  - ibridi endgame e search PIMC/value-lookahead, incluso il default agile `bc_model_pimc_belief_12x8`;
  - policy `.npz` scelta dalla UI da un catalogo server‑side (nessun path arbitrario dal browser).
- Encoder osservazione **v1 / v2 / v3 / v4** (vedi sotto) e fast path numerico Python/Numba per training ed
  evaluation veloci.
- Pipeline dati completa: event log configurabile SQLite/Postgres → export JSONL → self‑play → valutazione offline →
  training BC/RL.

Nota didattica:
- la UI è pensata e testata soprattutto in **2 giocatori** (modalità principale);
- il 4‑player è supportato dal motore (usato per regressione) ma **non è ancora pienamente coperto dal frontend**.

## Quick start

Questo progetto usa [uv](https://github.com/astral-sh/uv). Requisiti: Python **3.14** e `uv`.

```bash
uv venv -p python3.14
uv pip install -e ".[dev]"   # runtime + strumenti dev
briscola-server --reload     # UI su http://localhost:8000
```

## Come giocare

1. Inserisci il tuo nome, scegli l’avversario (IA) e premi “Avvia partita” (la UI mostra una descrizione dell’IA dai metadati del backend).
2. Clicca una carta in mano per giocarla.
3. L’IA risponde automaticamente al suo turno.

La UI avvia partite **2‑player**. Per flussi 4‑player (senza UI) usa gli script headless o le API.

### Giocare contro un modello locale (`.npz`)

Se hai addestrato un modello (BC / PG / A2C) salvato in `.npz`, puoi usarlo come avversario:

1. Metti il file in una directory whitelist lato server: consigliato `./data/models/` (oppure imposta `BRISCOLA_MODELS_DIR`).
2. (Ri)avvia il server.
3. Nella UI scegli **“Modello locale (.npz)”** e seleziona il file dal dropdown.

Note:
- il dropdown mostra `metadata_json.label`/`description_it` del `.npz` (i trainer del progetto li salvano in automatico);
- un `.npz` incompatibile (chiavi mancanti o `feature_dim` non supportata) viene segnalato dal catalogo e disabilitato nella UI;
- **sicurezza**: il browser invia solo un `ai_model_id` (path relativo) tra quelli esposti da `GET /api/ai/models`; il backend rifiuta path traversal e carica solo dentro `BRISCOLA_MODELS_DIR`.

## Approccio step‑by‑step (didattico)

L’idea è costruire una pipeline ML “dal basso”, in modo verificabile:

1. **Dominio testabile**: regole e transizioni pure in `domain/` + test su invarianti e casi limite.
2. **Backend/UI**: FastAPI + WS per far giocare umani e rendere osservabile lo stato.
3. **Raccolta dati**: event log append‑only configurabile su SQLite o Postgres.
4. **Export dataset**: event log → JSONL con schema versionato.
5. **Self‑play**: generazione rapida di partite dal dominio.
6. **Valutazione**: match offline riproducibili (win‑rate/punti medi).
7. **Training**: imitation/RL quando contratti e pipeline sono stabili.

## Sviluppo assistito da agenti AI

Il codice e il brainstorming del progetto sono stati sviluppati a più riprese in
collaborazione con **agenti AI** (Claude Code, Codex, Gemini): il repository è anche un
**banco di prova** per valutarli su un progetto reale — con convenzioni rigide
(`AGENTS.md` è il documento di riferimento condiviso che tutti gli agenti seguono),
quality gate automatici e decisioni finali sempre del maintainer. La storia di questo
processo, errori inclusi, è nel [diario di bordo](https://ai.briscola.dev/diario).

## Struttura del progetto

- `src/briscola_ai/domain/` – dominio canonico, puro e testabile
  - `models.py` (`Card`/`Suit`/`Rank`), `state.py` (`GameState`), `engine.py` (`step(state, action)`), `rules.py`, `observation.py`, `card_id.py` (mappa carta ↔ id 0–39), `serialization.py` (`GameState` ↔ dict JSON)
- `src/briscola_ai/backend/` – adattatore HTTP/WS (FastAPI): `dto.py` (Pydantic v2), `server.py`, `game_store.py` (stato partita in‑memory/Redis + pub/sub), `event_log.py` (SQLite/Postgres), `observation_builder.py`
- `src/briscola_ai/ai/`
  - `agents/` – baseline, ibridi endgame, factory e catalogo agenti
  - `endgame/` – solver esatto del finale 2‑player
  - `encoding/` – encoder v1/v2/v3/v4 e spazio azioni
  - `models/` – agente modello `.npz`, catalogo per la UI e provisioning modello
  - `fast/` – motore "fast" 2‑player (interi/array NumPy)
  - `numba/` – kernel JIT Numba (vedi nota sotto)
  - `evaluation/` / `training/` – valutazione offline e componenti training condivisi
- `src/briscola_ai/frontend/static/` – UI (HTML/CSS/JS), immagini carte in `assets/cards/`
- `tests/` – unit + integrazione API/WS (pytest)
- `scripts/` – simulazione, self‑play, export, training, evaluation, benchmark
- `docs/reports/model_progress.xlsx` – report Excel curato dei modelli significativi e delle milestone di promozione
- `PLAN.md` – stato corrente e prossime azioni (fonte di verità). Dataset/benchmark sono locali e gitignored; fanno
  eccezione i piccoli asset `.npz` di runtime esplicitamente tracciati in `data/models/`.

### I tre motori dello stesso gioco (dominio · fast · numba)

Lo **stesso** gioco è implementato a tre livelli, tenuti **in parità dai test** (`tests/test_fast_*`):

- **dominio** (`domain/engine.py`) — il motore "standard": puro, immutabile, leggibile. È la **fonte di verità**, usato da backend, UI e test. Ottimizzato per chiarezza, non per velocità.
- **fast** (`ai/fast/`) — riscrittura 2‑player su **interi/array NumPy** (niente oggetti `Card`/`GameState`): stessa logica, molto più veloce. Serve a self‑play, training ed evaluation massivi.
- **numba** (`ai/numba/`) — gli stessi kernel del fast path compilati **JIT con Numba**: ancora più rapidi. Include
  anche entrypoint training-first per componenti neurali avanzati, come il core depth‑1 di V-lookahead su stati
  numerici già determinizzati e il collector A2C che può usarlo come avversario fast.

Negli script si scelgono con `--engine domain|fast|numba` (es. `evaluate_agents.py`, `--rollout-engine`/`--fast-rollout` in `train_a2c.py`). Regola d'oro: il dominio decide la correttezza; fast/numba devono dare **risultati identici** (se cambi una regola nel dominio, aggiorna anche fast/numba e i test di parità). I numeri di throughput sono nella sezione [Performance](#performance-fast-path-pythonnumba).

## Backend (FastAPI + WebSocket)

Architettura ibrida HTTP + WebSocket.

**Perché ibrida?** Il polling REST è inefficiente e ad alta latenza per un gioco real‑time; il solo WebSocket complica azioni puntuali, testing e gestione errori. Quindi:
- **REST** per le *azioni* del client (semantica chiara, stateless, debug facile);
- **WebSocket** per gli *aggiornamenti* dal server (push in tempo reale).

### Endpoint HTTP (REST)

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check minimale per cloud/load balancer |
| `GET` | `/version` | Diagnostica deploy: versione, modelli presenti, event log realmente montato |
| `GET` | `/api/meta` | Metadati runtime per UI: modalità event log, consenso dataset, diagnostica DB |
| `GET` | `/api/ai/agents` | Catalogo agenti IA disponibili per la UI |
| `GET` | `/api/ai/models` | Catalogo modelli `.npz` selezionabili |
| `POST` | `/api/games` | Crea una nuova partita |
| `GET` | `/api/games/{id}` | Stato completo di debug; `403` di default, richiede l'opt-in esplicito descritto sotto |
| `GET` | `/api/games/{id}?player_index={i}` | Vista del giocatore `i` (`type: "observation"`) |
| `POST` | `/api/games/{id}/actions` | Il giocatore gioca una carta |
| `GET` | `/api/games/{id}/result` | Risultato finale |

### Endpoint Diagnostici

`GET /version` è pensato per controllare rapidamente un deploy cloud senza aprire la UI. Espone almeno:
`code_version`, `rules_version`, directory modelli, modello consigliato e presenza degli asset
(`recommended_model_present`, `value_lookahead_model_present`, `pimc_belief_model_present`). Quando l'event log è
configurato include anche `event_log_available`,
`event_log_healthy`, `event_log_backend`, `event_log_database_name` e `event_log_database_host`, così puoi verificare
che il processo live stia usando il Postgres/Neon atteso.

`GET /api/meta` espone gli stessi campi runtime utili alla UI, più `event_log_mode`, `dataset_requires_consent` e
`cors_allow_origins`. In produzione dataset il controllo minimo atteso è:
`event_log_mode=dataset`, `event_log_available=true`, `event_log_healthy=true`, `event_log_backend=postgres` e host/nome
DB coerenti col pannello Neon.

`GET /api/games/{id}` senza `player_index` mostra le mani complete e `next_deck_card`, quindi è **disabilitato per
default** e risponde `403`. Può essere abilitato solo per debug/spectator controllato con
`BRISCOLA_DEBUG_STATE_ENDPOINT=unsafe-full-state`; i vecchi valori booleani come `1` sono rifiutati per evitare
attivazioni accidentali e la variabile non va impostata nella produzione pubblica. La UI, gli agenti e il WebSocket
player-facing usano sempre `GET /api/games/{id}?player_index={i}` o snapshot `observation`, che contengono soltanto
l'informazione lecita del giocatore.

### WebSocket (tempo reale)

Connessione: `ws://host/api/ws/{game_id}/{player_index}`. **Tutti i messaggi includono un campo `type`**:

| Messaggio | `type` | Descrizione |
|----------|---------|-------------|
| Snapshot (observation) | `observation` | Stato della partita per il giocatore del WS (`my_hand`, `my_turn`, `table_cards`, …) |
| Reveal IA | `ai_card_reveal` | L’IA mostra la carta che sta per giocare |
| Risultato mano | `trick_result` | Carte, vincitore, punti della mano |
| Keepalive | `pong` | Risposta ai ping del client |

Gli snapshot includono `server_version` (intero monotono, incrementato a ogni azione) per diagnosticare ordering/reconnect. Regola client: `type === "observation"` → snapshot, altrimenti evento.

### Flusso di gioco e modello server‑driven

```
POST /api/games            -> game_id
WS connect /api/ws/{id}/0  -> snapshot
POST .../actions (gioca)   -> snapshot
                           -> ai_card_reveal -> trick_result -> snapshot
```

Il backend è “server‑authoritative”: dopo la mossa umana, se tocca all’IA gioca da solo ed emette `ai_card_reveal` → `trick_result` → snapshot. Il frontend gestisce solo la **presentazione** (mette in coda gli eventi e applica gli snapshot dopo gli hold), così la **logica di gioco** (backend) resta separata dalla **logica di presentazione** (frontend). Il backend non introduce delay di animazione (niente `asyncio.sleep()`).

**Stato e scalabilità (multi‑replica).** Lo stato delle partite vive in un `GameSessionStore` (`backend/game_store.py`): **in memoria** in locale, **Redis** in cloud quando è impostata `REDIS_URL`. In deploy con più repliche questo evita il “partita non trovata” (azioni/WS che colpiscono repliche diverse). L'architettura REST+WebSocket resta invariata: il fan‑out degli eventi WS passa per il **pub/sub** dello store, così ogni client riceve `ai_card_reveal`/`trick_result`/`observation` da qualsiasi replica. Gli `observation` per‑giocatore sono ricostruiti dal subscriber (anti‑cheat: mai lo stato completo).

## Frontend (UI web)

**Smoke test manuale** (2–3 min): avvia il server, apri `http://localhost:8000` con la Console DevTools aperta, gioca 3 mani complete e verifica che la sequenza sia sempre **reveal → seconda carta → risultato**, senza errori in console, freeze o carte duplicate sul tavolo.

**Debug** quando “si blocca”: controlla la Console (warning su `observation`/`server_version`) e la tab Network → filtro **WS** (connessione attiva, messaggi `observation`/`ai_card_reveal`/`trick_result`). Modalità senza WebSocket (polling) per isolare i bug: apri `http://localhost:8000/?polling=1`.

## Sviluppo (test, lint, typecheck)

Con le dev deps installate (`uv pip install -e ".[dev]"`):

```bash
ruff format src tests scripts
ruff check src tests scripts
mypy src
pytest                       # test
pytest --cov=briscola_ai --cov-report=term-missing   # coverage
```

Il badge coverage in cima si aggiorna automaticamente: la CI, sui push a `master`, ricalcola la percentuale con
`pytest --cov` e committa il README solo se è cambiata (colore in base alla soglia). Non serve toccarlo a mano.

## AI & ML

### Anti‑cheat: osservazione parziale (information set)

Nel dominio `GameState` contiene informazione **completa** (ordine del mazzo, mani di tutti). Se un agente la ricevesse, potrebbe “barare” leggendo informazione nascosta, rendendo i benchmark non significativi. Per questo:

- gli agenti ricevono una `PlayerObservation`, vista **parziale e lecita**, costruita con `make_player_observation(state, player_index)`;
- contiene: mano del giocatore, carte sul tavolo, briscola scoperta, `deck_size`, punteggi e dimensioni mani, e due one‑hot pubbliche (sotto);
- **non** contiene: il mazzo come sequenza (`state.deck`) né le carte specifiche degli avversari.

Riferimenti: `domain/observation.py`, test `tests/test_domain_observation.py`.

Due one‑hot pubbliche (entrambe lecite, derivate solo da informazione visibile):
- `seen_cards_onehot[40]`: carte **viste** = briscola scoperta + tavolo + carte uscite nelle prese;
- `out_of_play_cards_onehot[40]`: carte **non più disponibili** = prese + tavolo (la briscola scoperta NON è qui finché è pescabile/in mano). Invariante: `out_of_play ⊆ seen`.

### Encoder: v1, v2, v3, v4

Lo stesso stato lecito può essere codificato a livelli crescenti di “memoria/strategia”:

- **v1** (`feature_dim=248`): vista istantanea (mano, tavolo, briscola, scalari di stato).
- **v2** (`288`): v1 + `seen_cards_onehot[40]` → card counting lecito (storia pubblica).
- **v3** (`310`): v2 + 22 feature **strategiche aggregate**, leggibili: briscole/carichi ignoti, assi/tre usciti per seme, fase partita (`deck_size`, carte in mano, endgame flag), e info sulla presa corrente. Usa `out_of_play_cards_onehot` per distinguere “visto” da “fuori gioco”.
- **v4** (`369`): v3 + 59 feature di **memoria delle prese** (`trick_history`): aggregati sul comportamento avversario
  (semi giocati, tagli, risposte, uscite con briscola/carichi) + dettaglio delle ultime 4 prese. È l'encoder della linea
  promossa da v8 fino all'attuale `best_a2c_v15`; quando fu introdotto, il suo contributo isolato fu +0.27
  punti/partita (CI +0.12..+0.42).
- **v4+belief** (`409`, solo policy sperimentali): v4 + 40 probabilità della belief network embedded. Come input
  diretto della policy è risultato negativo (−0.56: informazione ridondante); la belief resta invece utile per pesare
  le determinizzazioni della search del default `bc_model_pimc_belief_12x8` e della variante massima 64×10.

L’encoder canonico vive in `ai/encoding/observation_encoder.py`; esiste in versione **domain** (oggetto), **fast**
(Python) e **Numba**, con test di **parità** che garantiscono lo stesso vettore. In partita (`ai_agent=bc_model`) il
backend sceglie l’encoder dai metadati del modello (`encoder_version`) o, per i vecchi artefatti, dalla `feature_dim`
(248/288/310/369; 409 identifica una policy sperimentale v4+belief con i relativi pesi embedded).

### Agenti disponibili

Il catalogo della UI espone:

- `random`, `greedy_points`, `heuristic_v1`, `heuristic_v2` – baseline dal caso puro a euristiche leggibili;
- `hybrid_endgame` – `heuristic_v2` nel mid-game + **solver esatto** a mazzo vuoto;
- `hybrid_endgame_best_a2c` – alias di compatibilità che richiede il file locale legacy `best_a2c.npz`;
- `bc_model` – policy `.npz` scelta dal catalogo, con encoder dedotto e validato dai metadati;
- `bc_model_hybrid_endgame` – policy scelta + solver esatto nel finale;
- `bc_model_value_lookahead_8x8` – policy scelta + solver + lookahead depth-1 guidata dal value model interno;
- `bc_model_pimc_16x8` – policy scelta + PIMC uniforme 16×8 + solver;
- `bc_model_pimc_belief_12x8` – **default consigliato** con v15: PIMC con 12 determinizzazioni pesate dalla belief
  network, finestra 8 e solver; nel gate appaiato da 20.000 partite è non inferiore a v14 16×8 e riduce di circa il
  25% la latenza media e p95 della search;
- `bc_model_pimc_belief_16x8` – precedente default, ancora selezionabile per confronti storici e per usare quattro
  determinizzazioni in più;
- `bc_model_pimc_belief_64x10` – variante **massima** con 64 determinizzazioni e finestra 10: è la più forte nei
  benchmark storici a parità di policy, ma è sensibilmente più costosa, resta una scelta avanzata e non è stata
  nuovamente confrontata con 16×8 sulla policy v14.

Factory e strumenti offline supportano inoltre nomi diagnostici non mostrati nella UI: `heuristic_trump_saver`
(sonda di exploitability), l'alias locale `best_a2c` e `bc_model_pimc_belief_16x10` (ablation eval-only della
finestra). Gli agenti belief richiedono
`belief_v0_h128_50k_seed20260702.npz`; il value-lookahead richiede
`value_v0_h128_clean50k_seed20260701.npz`. Sono asset interni e non policy selezionabili nel catalogo modelli.

Il **solver endgame** calcola la mossa ottima esatta con minimax a mazzo vuoto. `ai/endgame/solver.py` resta
l'oracolo didattico sul dominio canonico; `ai/endgame/fast_solver.py` è il solver completo numerico/Python;
`ai/endgame/numba_solver.py` è il path choose-only JIT per loop offline ad alto throughput e training Numba.
Per la V-lookahead esiste anche `ai/numba/value_lookahead.py`: è un kernel depth‑1 da array numerici per training ed
evaluation su stati già determinizzati. Non sostituisce il runtime UI anti-cheat, che continua a partire da
`PlayerObservation` e a determinizzare esplicitamente l'information set.
L’agente ibrido lo usa in modo **anti‑cheat** ricostruendo lo stato di finale dalla sola `PlayerObservation`.

### Raccolta dati ed export

L'event log viene inizializzato **solo quando è configurato**. Il comando `briscola-server` configura per comodità
SQLite su `./data/briscola_events.sqlite3` (disabilitabile con `--event-db ''`); avviando invece l'app ASGI
direttamente servono `BRISCOLA_EVENT_DB_PATH` oppure `DATABASE_URL`/`BRISCOLA_DATABASE_URL`. Postgres ha precedenza
su SQLite ed è il backend corretto in cloud multi-replica, dove un file locale sarebbe effimero e per-replica.
Entrambi implementano lo stesso schema append-only `games`/`events` e registrano, tra gli altri, `seed`,
`code_version` e `rules_version`.

Tre modalità di logging (`BRISCOLA_EVENT_LOG_MODE` o `--event-log-mode`):
- `debug` (default): completa, utile per troubleshooting;
- `dataset`: pensata per raccogliere partite **umane** tenendo il DB piccolo — salva eventi self-contained
  `human_action` e `ai_action` (observation → action → reward/done → next_observation) + marker `game_finished`,
  **non** salva `player_names` (privacy), usa un `client_id` pseudonimo per split per‑giocatore, ed esige il
  **consenso** (checkbox UI; il backend rifiuta `POST /api/games` senza consenso). `ai_action` include una traccia
  minimale `decision_trace` per distinguere fallback/solver/search PIMC senza salvare payload realtime;
- `off`: non registra eventi anche se un backend DB è configurato.

Export in JSONL (schema v1, pensato per il 2‑player):

```bash
python scripts/export_dataset.py --db ./data/briscola_events.sqlite3 --out ./data/dataset.jsonl
```

Con `DATABASE_URL`/`BRISCOLA_DATABASE_URL` presente, l'exporter legge da Postgres; passa `--db` per forzare
SQLite locale anche in un ambiente che ha la variabile Postgres:

```bash
DATABASE_URL=... python scripts/export_dataset.py --out ./data/dataset.jsonl
```

Default: solo azioni del `player_index=0`, escluse quelle IA, solo partite complete. Opzioni utili: `--all-players`, `--include-ai`, `--no-next-state` (supervised), `--include-incomplete`. In modalità `dataset` l’exporter usa preferibilmente `human_action` e anonimizza i nomi giocatore negli snapshot (`player_0`, `player_1`, ...), mantenendo `client_id` come pseudonimo per split train/val.

Report aggregato dell'event log (nessun payload/client_id stampato):

```bash
python scripts/report_event_log.py --db ./data/briscola_events.sqlite3
DATABASE_URL=... python scripts/report_event_log.py --json
```

Audit aggregato delle partite per versione/agente/modello, utile per capire se le partite PIMC sono nel DB e se il
log contiene anche eventi IA auditabili:

```bash
DATABASE_URL=... python scripts/audit_event_log_games.py --code-version 0.38.0 --show-games
DATABASE_URL=... python scripts/audit_event_log_games.py --ai-agent bc_model_pimc_belief_12x8 --json
```

Export dettagliato delle singole mosse IA/PIMC auditabili, con `decision_trace`, observation lecita e
`next_observation` sanificate:

```bash
DATABASE_URL=... python scripts/export_ai_actions.py \
  --ai-agent bc_model_pimc_belief_12x8 \
  --out ./data/prod_pimc_ai_actions.jsonl
```

Export unico action-by-action per audit di campo (mosse umane + IA nello stesso JSONL, partite complete di default,
`client_id` escluso dall'output salvo opt-in):

```bash
DATABASE_URL=... python scripts/export_live_actions.py \
  --ai-agent bc_model_pimc_belief_12x8 \
  --ai-model-id best_a2c_v15.npz \
  --exclude-client-id loadtest-bot \
  --out ./data/prod_live_actions_v15.jsonl
```

`scripts/audit_live_policy_replay.py` mostra poi due policy e i rispettivi stack PIMC sulle **stesse** observation
live. Il report contiene solo aggregati e hash, verifica la fedeltà al ramo registrato e distingue fallback, search
campionaria e solver; non simula il seguito alternativo della partita e quindi misura differenze di scelta, non forza
causale. Protocollo e primo audit v13-v14: `docs/plans/live-policy-replay-v13-v14-2026-07-14.md`.

**Deploy (cloud multi‑replica)**: imposta `REDIS_URL` (stato partita condiviso + realtime via pub/sub) e, per la
raccolta dati persistente, `DATABASE_URL` (event log Postgres). Sul sito live entrambi sono configurati e l'event log
usa la modalità `dataset`. Restringi le origin con `BRISCOLA_CORS_ALLOW_ORIGINS=https://tuodominio` (default `*`, solo
per sviluppo). L'elenco completo delle variabili d'ambiente è in `AGENTS.md`.

### Simulazioni e self‑play (headless)

```bash
# Simulazione semplice (dominio)
python scripts/simulate_games.py --num-games 100 --seed 42 --num-players 2

# Self-play verso SQLite (scegli gli agenti per seat con --agents)
python scripts/self_play_to_db.py --db ./data/briscola_events.sqlite3 --num-games 100 --seed 42 --agents heuristic_v1,random

# Fast self-play "summary-only" (no DB/DTO): seed/agenti/punti/vincitore, una riga per partita
python scripts/fast_self_play.py --num-games 100000 --seed 0 --agents greedy_points,random --out-jsonl /tmp/fast.jsonl
```

### Valutazione

Confronti riproducibili senza UI/server:

```bash
python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model --agent0-model ./data/models/best_a2c_v15.npz --agent1 heuristic_v1

# Diagnostica: la policy cambia se rinominiamo coerentemente i quattro semi?
python scripts/probe_suit_symmetry.py \
  --model ./data/models/best_a2c_v15.npz \
  --out-json ./data/suit_symmetry_v15.json
```

Concetti chiave:
- **engine**: `domain` (canonico, supporta tutti gli agenti), `fast`/`numba` (più veloci; `numba` supporta modelli MLP vs baseline fast‑compatible).
- **seat‑fair** (`--seat-fair`): per ogni seed gioca due partite scambiando i posti (rimuove il bias “il player 0 inizia sempre”). Richiede `--num-games` pari.
- **seed suite** riproducibili: `--seed-suite small|medium`, `--seed-suite-file`, oppure `--seed-suite-range-start N` (utile per **holdout**, es. `--seed-suite-range-start 1000000`).
- **preset**: `--benchmark small|medium|big` (= 2000/10000/100000 game, sempre seat‑fair) e `--out-json` per salvare.

Strumenti aggiuntivi:
- `scripts/evaluate_matrix.py` – valuta un modello contro una lista di avversari su due suite (`standard` e `holdout`).
- `scripts/evaluate_decision_quality.py` – metriche di **stile**, non solo forza:
  - `trump_waste_rate`: gioca briscola pur avendo una risposta vincente **non‑briscola**;
  - `trump_overkill_rate`: quando vince con briscola, usa una briscola più costosa del necessario;
  - a parità di seed, `--workers` cambia solo la velocità anche con agenti stocastici. Contratto e riproduzione del
    difetto corretto: `docs/plans/decision-quality-rng-2026-07-14.md`.
- `scripts/evaluate_pimc.py` – harness offline PIMC/determinizzazione sopra policy `.npz`: confronta search, modello
  puro, solver-control o un'altra configurazione PIMC, anche con policy distinte sui due lati. Registra CI su
  score/avg diff, integrita' e latenza search media/p50/p95. È lo strumento per ablation della search; la
  distillazione PIMC→policy è invece un ramo storico chiuso.
- `scripts/run_belief_v1_gate.py` – pipeline riprendibile per belief multi-stile: genera il dataset da roster
  versionato, esegue i fold leave-one-opponent-out e crea il candidato all-styles solo dopo un GO offline. Il pilot
  non può autorizzare una promozione; soglie e comando lungo sono in `docs/plans/belief-v1-multistile-2026-07-14.md`.
- `scripts/probe_suit_symmetry.py` – rinomina semanticamente mano, briscola, tavolo, storia e one-hot pubbliche sotto
  tutte le 24 permutazioni dei semi, poi rimappa le 40 azioni prima del confronto. Esclude le mosse forzate e usa solo
  `PlayerObservation`: sulla v13 la sonda trovava **18,19% di flip dell'argmax**; la distillazione v14 li porta al
  **6,04%**. La JS usa il softmax dei logits grezzi a temperatura 1 e non è una probabilità calibrata di vittoria.
  Metodo e limiti sono in `docs/plans/sonda-simmetria-semi-2026-07-11.md` e nel report della distillazione v0.
- `scripts/generate_suit_distillation_shards.py` / `scripts/train_suit_distillation_shards.py` – estensione per
  corpus di teacher simmetrici grandi: split globale per partita, manifest JSON rigoroso, shard NPZ atomici con
  SHA-256, resume della raccolta e training con un solo shard in RAM. Il launcher congelato del teacher 20M da
  250k e' `scripts/run_suit_distillation_20m_250k.sh`; protocollo e gate sono in
  `docs/plans/suit-distillation-20m-250k-2026-07-17.md`.
- `scripts/diagnose_hidden_units.py` – confronta v14 e v13 sugli stessi 4.096 stati e misura attivazioni, rango
  effettivo, correlazioni e ablation esatta di ogni unità ReLU. Il primo audit trova 123/256 unità v14 quasi inattive
  e non giustifica il widening 320/384; protocollo e limiti in `docs/plans/hidden-unit-diagnostic-v0-2026-07-12.md`.
- **Guard anti‑overkill** (`inference_overkill_guard`): post-processing diagnostico che, da secondo di mano, gioca la
  briscola vincente **minima**. È attivabile dai metadati o con `BRISCOLA_BC_OVERKILL_GUARD=1` per A/B; il default
  v14 non lo usa: la conservazione delle briscole resta una scelta appresa dalla policy, non una correzione runtime.

### Performance (fast path Python/Numba)

Il dominio canonico è la fonte di verità; il fast path 2‑player (`ai/fast/`) replica la stessa logica su interi/array per alzare il throughput, con kernel JIT in `ai/numba/`. È tenuto coerente dai test di parità. Misure con `scripts/benchmark_perf.py` (modi `*-random`, `fast-eval`, `numba-eval`, `numba-mlp`).

Esempio dell’ordine di grandezza: il **training A2C v3** su 20k partite passa da ~419 games/sec (`--rollout-engine domain`) a ~5900 games/sec (`--rollout-engine fast --fast-rollout numba`), ~14×; questo rende fattibili run da 1M partite in pochi minuti.

### Pipeline di training (BC → RL)

Idea didattica: prima un modello supervisionato che **imita** un teacher (Behavior Cloning), poi RL per **superarlo**.

**Spazio azioni**: “40 carte + action mask” (il modello sceglie tra 40 classi; la mask abilita solo le carte in mano).

**Behavior Cloning** (`scripts/train_bc.py`): allena su un JSONL esportato un modello (lineare o MLP) che riproduce le scelte del teacher. Encoder selezionabile con `--encoder-version v1|v2|v3|v4` (v3/v4 richiedono dataset con `out_of_play` popolato; v4 aggiunge la memoria delle prese). Per fine-tuning controllato supporta `--init` da un MLP `.npz` compatibile e `--bc-anchor ... --bc-anchor-beta ...` per restare vicino a un modello congelato. Per esperimenti di distillazione può filtrare il dataset con `--filter-disagree-with-model`: tiene solo gli esempi in cui il teacher sceglie una carta diversa dal modello base. Train, validation e test sono separati per `game_id` (default 80/10/10); un record allenabile senza identità di partita viene rifiutato per evitare leakage fra mosse della stessa partita.

**Distillazione PIMC — nota storica (ramo chiuso).** L'idea di comprimere le mosse della search
PIMC in una policy reattiva via BC è stata tentata due volte (giugno su v6, luglio come iterazione-0
di Expert Iteration su v7) e chiusa entrambe le volte con esito NEGATIVO e diagnosi: la
cross-entropy su mosse-argmax è un operatore lossy (una MLP hidden=512 memorizza il train al ~100%
e resta al 56–57% in validation). Gli script (`generate_pimc_teacher_dataset.py`, i flag
`--filter-disagree-with-model`, `--soft-labels`) restano utilizzabili per esperimenti, ma la strada
consigliata per sfruttare la search è usarla come **avversario di training** (è così che sono nati
v7 e v8) o direttamente a runtime (`bc_model_pimc_belief_12x8` come default, 64×10 come massimo). Dettagli e numeri:
`docs/plans/belief-expert-iteration.md` e `docs/diario/06-search-endgame.md`.

**Value learning / V-lookahead — ramo storico chiuso.** Questo ramo ha provato ad apprendere un valore scalare
`V(observation)` per ordinare le carte in una lookahead corta. Il `value_v0_h128_clean50k_seed20260701.npz` promosso
resta un asset runtime valido per l'agente selezionabile `bc_model_value_lookahead_8x8`; i successivi tentativi
value-v1, leaf-level e pairwise non lo hanno però superato materialmente, quindi non sono la direzione attiva.

Il codice rimane per riproducibilità e didattica:

- `generate_value_dataset.py` / `generate_value_dataset_numba.py` raccolgono target JSONL o `.npz` compatto; il
  formato compatto `value_dataset_npz_v2` conserva `game_ids` e `game_seeds`;
- `train_value.py` allena la regressione scalare e `train_value_pairwise.py` aggiunge ranking intra-root; entrambi
  separano train/validation/test per partita e salvano conteggi + digest dello split nei metadati;
- `evaluate_value_ranking.py`, `evaluate_value_lookahead.py`, `evaluate_value_lookahead_pair.py` e
  `evaluate_value_lookahead_quality.py` costituiscono i gate offline e seat-fair;
- `ai/numba/value_lookahead.py` conserva il core JIT training-first utilizzabile come avversario nei rollout A2C.

Le ricette v6-v8 e i relativi risultati sono deliberatamente storici e vivono in
`docs/plans/belief-expert-iteration.md`, nel report modelli e in `--help` degli script; non vanno interpretati come
comandi per rigenerare gli esperimenti storici; l'attuale best v14 deriva invece dalla distillazione simmetrica.
Gli NPZ value/leaf precedenti ai campi `game_ids` vanno rigenerati prima di un nuovo training: il trainer non puo'
ricostruire in modo affidabile i confini fra partite. Protocollo e motivazione sono in
`docs/plans/dataset-split-per-partita-2026-07-14.md`.

**Reinforcement Learning**: BC tende a *eguagliare* il teacher, non a superarlo. Per superarlo:
- **REINFORCE** (`scripts/train_pg.py`): policy gradient sul return finale. È corretto ma rumoroso.
- **A2C** (`scripts/train_a2c.py`, consigliato): aggiunge un *critic* `V(s)` e usa l’**advantage** `A = G − V(s)` come baseline appresa, con un reward più denso (delta `punti_policy − punti_opp` per turno della policy, senza barare).

La salute numerica di A2C si può osservare senza cambiare il training con
`--diagnostics-json report.json`: il file separato riassume critic, advantage,
attivazioni, gradienti e passi Adam per update, senza salvare osservazioni. La parità
bit-per-bit dei pesi con sonda accesa/spenta è coperta dai test. Il protocollo multi-seed
`scripts/run_a2c_health_probe.py` usa questi dati soltanto per scegliere una singola
ablation successiva; soglie e limiti sono in
`docs/plans/a2c-health-diagnostic-v0-2026-07-14.md`.

I run A2C multi-milione possono dichiarare l'orizzonte totale con `--num-games` e fermare
il processo a un confine intermedio con `--stop-after-games`. I checkpoint allineati agli
update incorporano policy, critic, momenti Adam, RNG, cursore e digest della schedule e
storico delle metriche; `--resume checkpoint.npz` continua quindi lo stesso training,
non un nuovo warm-start. Il resume rifiuta cambi di flag, asset o codice. Per mantenere
la memoria limitata usare `--metrics-mode summary` e campionare la telemetria con
`--diagnostics-every N`. Schedule, metriche e test bit-identico sono descritti nel
protocollo [`a2c-super-training-50m-2026-07-14.md`](docs/plans/a2c-super-training-50m-2026-07-14.md).
Il launcher congelato di quell'esperimento è `scripts/run_a2c_super_training_50m.sh`:
richiede esplicitamente `start 10|20|30|40|50` e non concatena i blocchi senza uno screen.

Gli errori residui di una policy si possono cercare con
`scripts/probe_policy_regret.py`: la sonda prova tutte le carte su determinizzazioni
compatibili, sceglie l'alternativa su meta' campioni e la giudica sull'altra meta', usa
il solver esatto a mazzo vuoto e produce categorie automatiche. Il report separa le fasi
policy-only dalle decisioni gia' coperte dal PIMC/solver del prodotto. Protocollo e
limiti: `docs/plans/policy-regret-audit-v14-2026-07-14.md`. Sul run formale v14 192x64
non trova errori affidabili nelle 96 decisioni early/mid; i 14 casi confermati sono tutti
nelle finestre gia' gestite da PIMC (9) o solver (5), quindi non autorizza un retraining.

Tecniche utili (tutte come flag, vedi `--help`):
- **opponent mix** (`--opponent-mix name:peso,...`) per robustezza (evita overfitting su un avversario);
- **warm‑start** da un BC (`--init`) e **BC‑anchor** (`--bc-anchor ... --bc-anchor-beta`) per restare vicino allo stile del teacher;
- **training schedule paired** (`--training-schedule paired`): ripete seed di mazzo e avversario su seat `{0,1}`
  dentro lo stesso update. Il confronto su tre seed non ha ridotto la variabilita' ne' superato il seriale a pari
  partite; `serial` resta il default e non e' autorizzato un run paired piu' lungo. Il runner riprendibile e'
  `scripts/run_a2c_paired_schedule_probe.py`; protocollo e risultato vivono in
  `docs/plans/a2c-paired-schedule-v0-2026-07-14.md`;
- **reward shaping anti‑overkill** (`--overkill-penalty-mode flat|gap --overkill-penalty-beta`);
- **suit augmentation paired sperimentale** (`--suit-augmentation paired`): duplica ogni traiettoria con una
  rinomina coerente dei semi e media il loss sui `2N` step. Il default è `off`; lo screening v0 su v13 non ha ridotto
  l'asimmetria e non giustifica un run lungo. Contratto, risultato e limiti sono in
  `docs/plans/suit-augmentation-paired-v0-2026-07-11.md`;
- **suit consistency sperimentale** (`--suit-consistency-beta`): lascia l'A2C on-policy sull'esperienza reale e
  aggiunge una forward-KL dalla distribuzione originale, fermata come target, alla copia rinominata. Nello screening
  v0 beta `0.1` riduce il flip medio `18,32% -> 15,64%` senza perdita policy-only misurabile, ma il follow-up 50k
  risale a `16,47%` e perde `-0,77` punti/partita: ramo chiuso, nessun modello promosso. Protocollo:
  `docs/plans/suit-consistency-v0-2026-07-11.md`;
- **suit margin consistency sperimentale** (`--suit-margin-beta`, `--suit-margin-cap`): una hinge preserva carta
  argmax e margine teacher nella copia rinominata, escludendo mosse forzate. Beta `0.3` conserva il gap ma si ferma a
  `14,42%` di flip, sopra il gate `<12%`; ramo chiuso senza run lungo. Report:
  `docs/plans/suit-margin-v0-2026-07-11.md`;
- **league**: allenare contro un campione congelato. L'alias `best_a2c` carica il file locale **legacy**
  `best_a2c.npz`, non il campione ufficiale corrente. Per usare v15 indica `bc_model` nel mix e passa esplicitamente
  `--opponent-model ./data/models/best_a2c_v15.npz` (il fast rollout Numba supporta al più un tipo di
  opponent-modello per mix);
- **value-lookahead opponent**: nel fast rollout Numba puoi allenare contro `bc_model_value_lookahead_8x8` passando sia
  `--opponent-model ./data/models/best_a2c_v15.npz` sia
  `--opponent-value-model ./data/models/value_v0_h128_clean50k_seed20260701.npz`. Questo path usa lo stato numerico
  già determinizzato del rollout come singola determinizzazione: è un avversario di training forte, non una replica
  bit-a-bit dell'agente UI che campiona information set da `PlayerObservation`.
- **curriculum** (`--curriculum easy_standard_hard`) per stage easy→standard→hard.

**Pipeline riproducibile** (`scripts/run_experiment.py`): un comando unico fa training → evaluation matrix → manifest → aggiorna il best locale. Supporta `--rollout-engine fast --fast-rollout numba` e `--eval-engine numba` per i run lunghi. Output in `data/models/` e `benchmarks/experiments/<name>/`.

> I dettagli operativi vivono in `--help` degli script. La cronologia e i numeri completi sono nel diario,
> in `docs/plans/` e nel report modelli; `PLAN.md` contiene soltanto stato reale e prossime decisioni.

### Baseline AI ufficiale

La policy ufficiale è **`data/models/best_a2c_v15.npz`** (encoder v4, hidden 256). È una singola MLP distillata
dalla media esatta delle risposte del checkpoint 20M sulle 24 rinomine coerenti dei semi, usando 250.000 partite e
9,5 milioni di decisioni. Il flip dell'argmax sotto rinomina scende al `2,9456%` e l'overkill sui piatti poveri al
`1,0637%`. Come policy diretta batte v14 di `+0,18046` punti su 100.000 partite (CI `+0,08..+0,28`) e ottiene
`+21,86688` nel controllo omogeneo big contro `heuristic_v1`.

Il default effettivo della UI è **`bc_model_pimc_belief_12x8` su `best_a2c_v15.npz`**, senza guard runtime. Nel gate
da 20.000 partite contro v14 16×8 il delta è `+0,1052` (CI `-0,1126..+0,3230`): non dimostra forza superiore, ma
supera il margine preregistrato di non inferiorità. La latenza media e p95 della search scendono entrambe a circa
`0,75x`. V14 16×8 resta selezionabile; anche `bc_model_pimc_belief_64x10` resta la variante a search massima e
`bc_model_value_lookahead_8x8` il ramo storico guidato dal value model.

Gli asset runtime necessari al sito sono **tracciati in Git**: policy v10, v11, v13, v14 e v15, value model e belief network.
Gli altri `.npz` in `data/models/`, i dataset e gli artefatti in `benchmarks/experiments/` restano locali e
gitignored. Questa distinzione è intenzionale: il repository contiene ciò che serve a riprodurre il runtime e le
milestone ufficiali, non l'intera storia dei run.

Gli agenti search richiedono asset ausiliari in `BRISCOLA_MODELS_DIR`: il value model per
`bc_model_value_lookahead_8x8` e la belief network per entrambe le varianti `bc_model_pimc_belief_*`. Il catalogo
modelli UI **non** li mostra come policy selezionabili (sono asset interni, filtrati per formato).
Gli asset versionati sono già inclusi nell'immagine di deploy. Il provisioning best-effort ne verifica gli SHA e,
se un file manca, può riscaricarlo dai Release asset configurati tramite env:

```text
BRISCOLA_DEFAULT_MODEL_ID=best_a2c_v15.npz
BRISCOLA_MODEL_URL=https://raw.githubusercontent.com/ilCapo77/briscola.ai/v0.38.0/data/models/best_a2c_v15.npz
BRISCOLA_MODEL_SHA256=2f2dca3d4e77a363783124feeb30f482a85a740077222936b025b37b865f2eb6
BRISCOLA_VALUE_MODEL_URL=https://github.com/ilCapo77/briscola.ai/releases/download/v0.16.0/value_v0_h128_clean50k_seed20260701.npz
BRISCOLA_VALUE_MODEL_SHA256=5f93f1c5f2bf2869a575abf91ceba8a3e9aeb4ada48ba4ffac8d0f5507fb34f0
BRISCOLA_BELIEF_MODEL_URL=https://github.com/ilCapo77/briscola.ai/releases/download/v0.23.0/belief_v0_h128_50k_seed20260702.npz
BRISCOLA_BELIEF_MODEL_SHA256=4100b23b65a5566e047230ced665b91eef1942ea31e4a4cbe201b64545e7d035
```

Il provisioning è best-effort: se un download fallisce l'app parte comunque, ma `/version` espone
`recommended_model_present`, `value_lookahead_model_present` e `pimc_belief_model_present` per
verificare il deploy. Su ambienti con disco limitato evita di rendere disponibili troppi `.npz`
contemporaneamente.

### Report progressione modelli

Il report Excel curato vive in `docs/reports/model_progress.xlsx` ed è generato da:

```bash
uv run python scripts/build_model_report.py
```

Serve a tracciare solo i modelli **significativi**: best ufficiali, teacher/anchor importanti e candidati scartati
che spiegano una decisione. La Dashboard traccia la serie confrontabile v8-v11, v14 e v15 (big 100k, stessi seed e
configurazione promossa); v13 resta fuori dal grafico perché dispone soltanto dei gate medium, riportati nelle tabelle
di evidenza. Le altre tab contengono milestone, dettagli modello, prove di promozione, decision quality e fonti.

Il build normale legge il manifest canonico versionato
`docs/reports/evidence/model_progress.v1.json`, quindi funziona anche senza gli artefatti storici locali. Solo quando
si aggiornano le prove a partire dai `.npz` e benchmark originali si usa
`uv run python scripts/build_model_report.py --refresh-evidence`, revisionando insieme manifest e foglio generato.

Il gate documentale completo è ermetico e non controlla URL esterni:

```bash
make docs-check
```

Valida JSON rigoroso, link locali e link GitHub interni, coerenza fra versione/modello/documenti, sequenza dei
capitoli e uguaglianza byte-per-byte del report Excel rigenerato in una directory temporanea. Il comando canonico
sottostante è `uv run python scripts/check_docs.py` ed è eseguito anche dalla CI.

Aggiornalo quando promuovi un nuovo best o quando un esperimento importante cambia una decisione. Non usarlo come dump
di tutti i run: gli esperimenti locali restano in `benchmarks/experiments/` (gitignored), mentre il report conserva la
narrazione sintetica e verificabile.

Esempio di confronto testa-a-testa tra policy ufficiale e anchor precedente:

```bash
python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model --agent0-model ./data/models/best_a2c_v15.npz \
  --agent1 bc_model --agent1-model ./data/models/best_a2c_v14.npz
```

## Stato e roadmap

La release corrente del repository è `0.39.0`; ogni push di `master` viene distribuito automaticamente su
<https://ai.briscola.dev> tramite FastAPI Cloud, con stato partita su Redis, realtime via pub/sub ed event log
Postgres in modalità `dataset`. Il deploy effettivo va controllato tramite `/version`. Stato corrente, invarianti da
non rompere e prossime azioni sono in **`PLAN.md`**.

La storia del progetto — scelte, errori e svolte, raccontati in tono divulgativo — è il **diario di
bordo**: <https://ai.briscola.dev/diario> (fonte: `src/briscola_ai/frontend/static/diario.md`). Il racconto arriva al
**capitolo 21, “Dodici mondi bastano”**; gli approfondimenti tecnici delle singole tappe, incluso il percorso verso
v15 in `docs/diario/23-dodici-mondi-bastano.md`, vivono in `docs/diario/`.

## Licenza

Progetto rilasciato con licenza MIT – vedi `LICENSE`.
