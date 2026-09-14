"""
API backend (FastAPI) per Briscola AI.

Questo modulo espone:
- endpoint HTTP per creare una partita, ottenere lo stato e giocare una carta
- endpoint WebSocket per inviare aggiornamenti in tempo reale ai client

Scelte implementative:
- Lo stato delle partite vive in un `GameSessionStore` (vedi `game_store.py`): in-memory in dev,
  Redis in cloud (multi-replica). Questo evita "partita non trovata" quando azioni/WS finiscono su
  repliche diverse. `game_data`/`game_timestamps` restano per-replica (best-effort: buffer ML, cleanup).
- Gli eventi realtime (reveal carta IA, risultato mano, refresh snapshot) viaggiano TUTTI sul
  pub/sub dello store (`publish`/`subscribe`): così raggiungono i client su QUALSIASI replica
  (Redis in prod, fan-out asyncio in dev). Ogni connessione WebSocket avvia un task subscriber
  che inoltra gli eventi al proprio socket; i "refresh" vengono tradotti nell'osservazione
  per-giocatore (anti-cheat: mai lo stato completo).
- L'agente IA non è serializzato: la sessione salva la sua config (nome + model_id) e l'agente viene
  ricostruito per mossa (con cache modello).

Contratto WebSocket (riferimento unico)
---------------------------------------
Esistono DUE livelli di messaggi, da non confondere:

1. Messaggi INTERNI sul pub/sub dello store (mai inviati verbatim al client):
   - `{"type": "refresh", "server_version": int, "state": <GameState serializzato>}`
     Pubblicato da `notify_clients` dopo ogni avanzamento. Lo `state` è embeddato
     point-in-time perché il subscriber NON deve rileggere dallo store (lo stato potrebbe
     essere già avanzato dal task IA, invertendo l'ordine percepito degli eventi).

2. Messaggi VERSO IL CLIENT (quello che il frontend riceve davvero):
   - `ObservationDTO` (`type: "observation"`): il subscriber traduce ogni `refresh` nella
     vista per-giocatore (anti-cheat: mai lo stato completo sul socket).
   - `AiCardRevealDTO` (`type: "ai_card_reveal"`): carta giocata dall'IA (inoltrato verbatim).
   - `TrickResultDTO` (`type: "trick_result"`): esito della presa (inoltrato verbatim).
   - `{"type": "ping"|"pong"}`: keepalive, filtrati dal client senza toccare lo stato UI.

L'ordinamento è garantito dal publisher (pubblicazioni sequenziali nello stesso task) e dal
guard `server_version` lato client, che scarta snapshot più vecchi dell'ultimo applicato.
"""

import asyncio
import json
import os
import random
import threading
import time
import uuid
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from ..ai.agents import AI_AGENTS_COMMON_NOTE_IT, Agent, agent_uses_selected_model, build_agent, list_agent_specs
from ..ai.models import (
    DEFAULT_MODEL_ID,
    get_models_dir_from_env,
    list_local_models,
    resolve_model_path,
    validate_model_compatible_for_ui,
)
from ..domain.card_id import card_to_id
from ..domain.engine import PlayCardAction, step
from ..domain.models import Card, Rank, Suit
from ..domain.observation import PlayerObservation, make_player_observation
from ..domain.serialization import game_state_from_dict, game_state_to_dict
from ..domain.state import GameState as DomainGameState
from ..domain.state import new_game_state
from ..versioning import get_code_version, get_rules_version
from .dto import (
    AiCardRevealDTO,
    CardDTO,
    GameResultDTO,
    PlayActionResultDTO,
    TableCardDTO,
    TrickResultDTO,
)
from .event_log import EventLogProtocol, build_event_log, parse_event_db_path, resolve_database_url
from .event_log_privacy import sanitize_dataset_payload
from .game_store import (
    AiSeatConfig,
    GameSession,
    InMemoryGameSessionStore,
    build_game_session_store,
)
from .observation_builder import build_game_state_dto, build_observation_dto

# `InMemoryGameSessionStore` è ri-esportato qui per comodità dei test, che resettano lo store
# con `server.game_store = InMemoryGameSessionStore()`.
__all__ = ["app", "game_store", "InMemoryGameSessionStore"]


# Modelli per richieste e risposte API
#
# I vincoli Field non sostituiscono la validazione di dominio (es. `new_game_state` rifiuta
# comunque num_players diversi da 2/4): mettono un tetto a input arbitrariamente grandi che
# finirebbero nello stato, nei log e nei messaggi WS (nomi chilometrici, liste enormi, ...).
class GameConfig(BaseModel):
    """Payload per creare una partita."""

    num_players: int = Field(ge=2, le=4)
    player_names: list[str] | None = Field(default=None, max_length=4)
    ai_agent: str | None = Field(default=None, max_length=100)
    ai_model_id: str | None = Field(default=None, max_length=200)
    client_id: str | None = Field(default=None, max_length=100)
    consent_to_data_collection: bool | None = None

    @field_validator("player_names")
    @classmethod
    def _limit_player_name_length(cls, value: list[str] | None) -> list[str] | None:
        """I nomi sono input libero mostrato in UI e nei log: 40 caratteri bastano."""
        if value is None:
            return None
        max_len = 40
        for name in value:
            if len(name) > max_len:
                raise ValueError(f"Nome giocatore troppo lungo (max {max_len} caratteri)")
        return value


class GameAction(BaseModel):
    """Payload per giocare una carta."""

    game_id: str = Field(max_length=100)
    player_index: int = Field(ge=0, le=3)
    card_index: int = Field(ge=0, le=39)
    # Metadati client-side (opzionali): utili per analisi qualità dati umani.
    client_observed_server_version: int | None = None
    client_decision_time_ms: int | None = Field(default=None, ge=0)


class ReplayCardInput(BaseModel):
    """Carta inserita nel laboratorio di replay, nel formato minimo adatto ai gesti."""

    suit: str = Field(pattern="^(clubs|cups|coins|swords)$")
    number: int = Field(ge=1, le=10)

    def to_domain(self) -> Card:
        """Converte il payload validato nella carta canonica del dominio."""
        rank = next(rank for rank in Rank if rank.number == self.number)
        return Card(suit=Suit(self.suit), rank=rank)


class ReplayTableCardInput(BaseModel):
    """Carta pubblica sul tavolo e giocatore che l'ha calata nel replay a due posti."""

    card: ReplayCardInput
    player_index: int = Field(ge=0, le=1)


class ReplayAdviceRequest(BaseModel):
    """
    Stato pubblico di un singolo istante di replay a due giocatori.

    Il browser invia soltanto dati che un giocatore potrebbe ricostruire dopo una partita:
    la propria mano, le carte già mostrate, la briscola, punteggio e contatori pubblici.
    Non accettiamo né ordine del mazzo né mano dell'altro giocatore.
    """

    hand: list[ReplayCardInput] = Field(min_length=1, max_length=3)
    trump_card: ReplayCardInput
    completed_cards: list[ReplayCardInput] = Field(default_factory=list, max_length=40)
    table_cards: list[ReplayTableCardInput] = Field(default_factory=list, max_length=1)
    deck_size: int = Field(ge=0, le=34)
    opponent_hand_size: int = Field(ge=0, le=3)
    my_points: int = Field(ge=0, le=120)
    opponent_points: int = Field(ge=0, le=120)
    first_player: int = Field(ge=0, le=1)


class ReplayAdviceResponse(BaseModel):
    """Carta consigliata dal profilo massimo per l'istantanea di replay ricevuta."""

    card: CardDTO
    agent: str
    model_id: str


class GameAbandonRequest(BaseModel):
    """Payload opzionale per abbandonare una partita in corso."""

    player_index: int | None = Field(default=None, ge=0, le=3)


class GameState(BaseModel):
    """Payload per richiedere lo stato di una partita (opzionale: vista per giocatore)."""

    game_id: str
    player_index: int | None = None


def _get_event_log() -> EventLogProtocol | None:
    """
    Helper per accedere al logger dalla app FastAPI.

    Per semplicità, il riferimento vive in `app.state.event_log` e viene inizializzato
    nel lifespan. Se la feature non è configurata, ritorniamo `None`.
    """
    return getattr(app.state, "event_log", None)


def _get_event_log_mode() -> str:
    """
    Modalità di logging eventi.

    - `debug` (default): log completo, utile per debug (azioni, reveal/trick IA e lifecycle WS).
    - `dataset`: log minimale, orientato a dataset umano (riduce molto la dimensione del DB).
    - `off`: non loggare nulla (senza cambiare DB path).
    """
    raw = os.getenv("BRISCOLA_EVENT_LOG_MODE", "debug").strip().lower()
    if raw in {"debug", "dataset", "off"}:
        return raw
    return "debug"


def event_log_runtime_metadata() -> dict[str, str | bool | None]:
    """
    Diagnostica dell'event log effettivamente montato nel processo.

    `event_log_mode` descrive la configurazione desiderata; questi campi dicono
    invece se il logger e' davvero inizializzato. In cloud e' il controllo rapido
    per distinguere "dataset mode attivo" da "Postgres non raggiungibile allo startup".
    """
    log = _get_event_log()
    if log is None:
        return {
            "event_log_available": False,
            "event_log_healthy": False,
            "event_log_backend": None,
            "event_log_database_name": None,
            "event_log_database_host": None,
            "event_log_games_recorded": None,
        }
    health_check = getattr(log, "health_check", None)
    healthy = bool(health_check()) if callable(health_check) else None
    count_games = getattr(log, "count_games", None)
    games_recorded = count_games() if callable(count_games) else None
    return {
        "event_log_available": True,
        "event_log_healthy": healthy,
        "event_log_backend": getattr(log, "backend_name", "unknown"),
        "event_log_database_name": getattr(log, "database_name", None),
        "event_log_database_host": getattr(log, "database_host", None),
        # Diagnostica di curiosita'/monitoraggio: quante partite ha registrato l'event log.
        "event_log_games_recorded": games_recorded,
    }


def _error_alerts_configured() -> bool:
    """True se le notifiche email per gli errori sono configurate (env Mailgun presenti)."""
    from .alerts import get_alert_config

    return get_alert_config().enabled


def _games_stats_payload() -> dict:
    """
    Statistiche aggregate delle partite registrate, per modello/agente avversario.

    Legge l'event log (best-effort): il modello viene dal payload di `game_created`,
    quindi copre anche lo storico. Aggregati anonimi: nessun dato personale.
    """
    log = _get_event_log()
    if log is None:
        return {"available": False, "total": None, "by_model": None}
    count_games = getattr(log, "count_games", None)
    by_model_fn = getattr(log, "count_games_by_model", None)
    total = count_games() if callable(count_games) else None
    by_model = by_model_fn() if callable(by_model_fn) else None
    completed = sum(row["completed"] for row in by_model) if by_model else None
    return {"available": True, "total": total, "completed": completed, "by_model": by_model}


def _safe_log_event(
    game_id: str,
    event_type: str,
    payload: dict,
    *,
    server_version: int | None = None,
    player_index: int | None = None,
    state: DomainGameState | None = None,
) -> None:
    """
    Wrapper “best-effort” per loggare eventi.

    Il logging è un optional feature: se il DB non è configurato o se una scrittura fallisce
    non vogliamo interrompere la partita.

    `state` (opzionale) consente al chiamante, che ha già la sessione in scope, di fornire lo
    stato di dominio per popolare la tabella `games` (num_players/seed) senza una lettura extra
    dallo store.
    """
    log = _get_event_log()
    if log is None:
        return

    mode = _get_event_log_mode()
    if mode == "off":
        return
    if mode == "dataset":
        # In modalità dataset riduciamo il DB tenendo solo eventi utili al dataset/debug privacy-safe:
        # mosse umane self-contained e, per audit IA, mosse IA self-contained con ObservationDTO sanificata.
        allowed = {"game_created", "human_action", "ai_action", "game_finished", "game_aborted"}
        if event_type not in allowed:
            return

    try:
        # Garantiamo che la partita esista nella tabella `games` (idempotente).
        if state is not None:
            # Compatibilità: se il dominio non espone `seed`, proviamo a prenderlo dal payload.
            seed = getattr(state, "seed", None)
            if seed is None:
                seed_from_payload = payload.get("seed")
                seed = seed_from_payload if isinstance(seed_from_payload, int) else None

            log.ensure_game(
                game_id,
                num_players=state.num_players,
                seed=seed,
                code_version=get_code_version(),
                rules_version=get_rules_version(),
            )
        # Privacy: i nomi giocatore sono input libero dell'utente. Non li persistiamo mai in
        # chiaro, in NESSUNA modalità: anche `debug` in cloud scrive su un DB condiviso, e la
        # sanificazione (nomi -> `player_N`) non toglie nulla di utile al debugging.
        # La funzione è idempotente: i payload già sanificati dai chiamanti restano invariati.
        log.log_event(
            game_id,
            event_type,
            sanitize_dataset_payload(payload),
            server_version=server_version,
            player_index=player_index,
        )
    except Exception as exc:
        # Best-effort: non propaghiamo eccezioni lato API/WS.
        print(f"Event log: errore scrittura evento {event_type!r} (game_id={game_id}, error={exc!r}).")


def _safe_set_client_id(game_id: str, client_id: str | None) -> None:
    """Best-effort: salva `client_id` nella tabella `games` (se event log abilitato)."""
    if not client_id:
        return
    log = _get_event_log()
    if log is None or _get_event_log_mode() == "off":
        return
    with suppress(Exception):
        log.set_client_id(game_id, client_id=str(client_id))


def _maybe_log_game_finished(game_id: str, *, state: DomainGameState, server_version: int) -> None:
    """
    Se la partita è finita (`game_over=true`), logga un evento `game_finished` (best-effort).

    Nota didattica:
    questo evento serve soprattutto per filtrare dataset: esportiamo solo partite complete.
    """
    if not state.game_over:
        return

    log = _get_event_log()
    if log is None or _get_event_log_mode() == "off":
        return

    try:
        # Garantiamo l'anchor in tabella `games` (idempotente).
        seed = getattr(state, "seed", None)
        log.ensure_game(
            game_id,
            num_players=state.num_players,
            seed=seed if isinstance(seed, int) else None,
            code_version=get_code_version(),
            rules_version=get_rules_version(),
        )
        updated = log.try_mark_game_finished(game_id)
    except Exception:
        updated = False

    if not updated:
        return

    final_points = [p.points for p in state.players]
    winning_index: int | None = None
    if not state.is_team_game:
        best = max(final_points) if final_points else 0
        winners = [i for i, pts in enumerate(final_points) if pts == best]
        winning_index = winners[0] if len(winners) == 1 else None

    _safe_log_event(
        game_id,
        "game_finished",
        {
            "game_over": True,
            "num_players": state.num_players,
            "is_team_game": state.is_team_game,
            "final_points_by_player_index": final_points,
            "winning_player_index": winning_index,
        },
        server_version=server_version,
        state=state,
    )


def initialize_event_log_from_env(target_app: FastAPI) -> tuple[EventLogProtocol | None, bool]:
    """
    Inizializza (o riusa) l'event log su `target_app.state.event_log` in base alle env.

    Perché è un helper condiviso: il backend può girare direttamente (`server:app`) oppure
    montato dentro l'app principale (`main:app`), e in alcuni setup i sub-app montati non
    ricevono eventi lifespan. Entrambi i lifespan usano quindi questa stessa logica —
    tenerne due copie aveva già iniziato a farle divergere.

    Regole:
    - Postgres se `DATABASE_URL` è impostata, altrimenti SQLite se `BRISCOLA_EVENT_DB_PATH`
      è dato, altrimenti feature disabilitata;
    - se un logger esiste già con la stessa identità (`path`), viene riusato senza toccarlo;
    - se la config è cambiata tra due startup (tipico nei test), il vecchio viene chiuso;
    - un fallimento di init NON blocca il server (feature opzionale).

    Ritorna `(event_log, created_here)`: `created_here` dice al chiamante se è lui il
    proprietario (e quindi chi deve chiuderlo allo shutdown).
    """
    database_url = resolve_database_url()
    sqlite_path = parse_event_db_path(os.getenv("BRISCOLA_EVENT_DB_PATH"))
    desired = database_url or sqlite_path

    event_log: EventLogProtocol | None = getattr(target_app.state, "event_log", None)
    created_here = False

    if event_log is not None and (desired is None or event_log.path != desired):
        with suppress(Exception):
            event_log.close()
        event_log = None
        target_app.state.event_log = None

    if event_log is None and desired is not None:
        try:
            event_log = build_event_log(sqlite_path=sqlite_path, database_url=database_url)
            created_here = event_log is not None
        except Exception as exc:
            # Il logger è un "optional feature": se fallisce non vogliamo bloccare il server.
            print(f"Event log: inizializzazione fallita, feature disabilitata ({exc!r}).")
            event_log = None
        target_app.state.event_log = event_log

    return event_log, created_here


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestisce startup/shutdown dell'app FastAPI.

    Usiamo un task in background per fare periodicamente cleanup delle partite inattive.
    """
    event_log, event_log_created_here = initialize_event_log_from_env(app)

    cleanup_task = asyncio.create_task(cleanup_inactive_games())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        if event_log is not None and event_log_created_here:
            event_log.close()
            app.state.event_log = None


# Crea l'app FastAPI
#
# Nota:
# usiamo `get_code_version()` per tenere allineata la versione OpenAPI con la versione del pacchetto
# (o con l'override via env `BRISCOLA_CODE_VERSION`).
app = FastAPI(title="Briscola AI API", version=get_code_version(), lifespan=lifespan)


def _parse_cors_allow_origins() -> list[str]:
    """
    Parsea `BRISCOLA_CORS_ALLOW_ORIGINS` (CSV) per limitare le origin ammesse.

    Esempi:
    - `BRISCOLA_CORS_ALLOW_ORIGINS=https://example.com`
    - `BRISCOLA_CORS_ALLOW_ORIGINS=https://a.com,https://b.com`

    Default:
    - se la variabile non è impostata, usiamo `*` (comportamento “dev-friendly”).
    """
    raw = os.getenv("BRISCOLA_CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


# Aggiunge middleware CORS per consentire richieste cross-origin.
#
# Nota:
# In produzione è consigliato impostare `BRISCOLA_CORS_ALLOW_ORIGINS` con il tuo dominio.
cors_allow_origins = _parse_cors_allow_origins()
cors_allow_credentials = "*" not in cors_allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store condiviso delle sessioni partita (stato + versioning + config IA + seed azioni).
#
# In cloud (Redis configurato) lo store è accessibile da tutte le repliche: azioni e WebSocket
# che finiscono su repliche diverse non danno più "partita non trovata". In dev/test è in-memory.
game_store = build_game_session_store()

# Stato best-effort, per-replica (non critico per la correttezza cross-replica):
# - `game_timestamps`: per il cleanup periodico delle partite inattive su questa replica.
# - `game_data`: buffer in memoria delle azioni (base per pipeline ML futura).
#
# Le connessioni WebSocket NON sono più tracciate qui: ogni connessione si iscrive al pub/sub
# dello store (`game_store.subscribe`) e inoltra gli eventi al proprio socket. Questo rende la
# consegna funzionante anche cross-replica (Redis in prod).
game_timestamps: dict[str, datetime] = {}
game_data: dict[str, list[dict]] = {}  # Memorizza le azioni per il training ML

# Il laboratorio di replay non conserva partite né dati degli utenti. Manteniamo solo un
# agente già caricato per evitare di rileggere tre `.npz` a ogni istantanea analizzata.
_replay_agent: Agent | None = None
_replay_agent_lock = threading.RLock()
_REPLAY_AGENT_NAME = "bc_model_pimc_belief_64x10"

# Cap sui buffer per-replica: senza limiti, un flusso continuo di partite (o un client abusivo)
# farebbe crescere la RAM senza vincolo fino al cleanup orario. I valori sono larghi rispetto
# all'uso legittimo (una partita 2p ha 40 giocate).
_MAX_BUFFERED_GAMES = 2000
_MAX_ACTIONS_PER_GAME = 400

_DEFAULT_AI_AGENT_NAME = "random"
_AI_PLAYER_DISPLAY_NAME = "Giocatore AI"
_STARTING_PLAYER_SEED_SALT = 0xB415C01A
_AI_STARTS_PRESENTATION_DELAY_SECONDS = 3.0


def _choose_starting_player(seed: int, num_players: int) -> int:
    """
    Sceglie in modo riproducibile chi comincia una partita creata via API.

    `new_game_state` resta volutamente stabile: il seed decide shuffle e pescate iniziali.
    La UI pubblica invece vuole un primo giocatore casuale, quindi deriviamo un secondo RNG
    dal seed della partita senza alterare l'ordine del mazzo. Helper separato anche perché
    i test API possono fissare esplicitamente il primo giocatore senza dipendere dal salt.
    """
    return random.Random(seed ^ _STARTING_PLAYER_SEED_SALT).randrange(num_players)


def _utcnow() -> datetime:
    """
    Ora corrente timezone-aware in UTC.

    Perché non `datetime.now()` naive: i timestamp finiscono in `updated_at` sullo store
    condiviso e vengono confrontati da repliche diverse; con orologi/timezone non allineati
    il confronto naive sbaglierebbe la staleness. UTC aware è privo di ambiguità.
    """
    return datetime.now(UTC)


def _remember_game_action(game_id: str, entry: dict) -> None:
    """
    Accoda un'azione nel buffer ML per-replica applicando i cap anti-crescita.

    - per partita: teniamo solo le ultime `_MAX_ACTIONS_PER_GAME` azioni;
    - globale: sopra `_MAX_BUFFERED_GAMES` partite bufferizzate, scartiamo la più vecchia
      (best-effort: il buffer è un'ottimizzazione locale, non la fonte dati canonica).
    """
    actions = game_data.setdefault(game_id, [])
    actions.append(entry)
    if len(actions) > _MAX_ACTIONS_PER_GAME:
        del actions[: len(actions) - _MAX_ACTIONS_PER_GAME]
    if len(game_data) > _MAX_BUFFERED_GAMES:
        oldest_id = min(game_timestamps, key=lambda gid: game_timestamps[gid], default=None)
        if oldest_id is not None and oldest_id != game_id:
            game_data.pop(oldest_id, None)
            game_timestamps.pop(oldest_id, None)


def _agent_for_seat(cfg: AiSeatConfig) -> Agent:
    """
    Ricostruisce l'agente IA da `AiSeatConfig` (l'oggetto Agent non è serializzato nello store).

    Nota: il caricamento dei modelli è cached, quindi ricostruire l'agente a ogni mossa è cheap.
    """
    if agent_uses_selected_model(cfg.agent_name):
        path = resolve_model_path(get_models_dir_from_env(), cfg.model_id or "")
        return build_agent(cfg.agent_name, model_path=path)
    return build_agent(cfg.agent_name)


def _is_ai_controlled_player(session: GameSession, player_index: int) -> bool:
    """
    Ritorna True se `player_index` è controllato dal backend per questa partita.

    Nota di sicurezza:
    la UI attuale espone solo l'umano come player 0 e l'IA come player 1. L'endpoint HTTP
    resta però chiamabile manualmente: questa guardia evita che un client giochi le mosse
    del player controllato dall'IA.
    """
    return player_index in session.ai_seats


def _ai_decision_trace(agent: Agent) -> dict[str, Any] | None:
    """
    Restituisce una traccia minimale della decisione IA appena eseguita, se disponibile.

    L'agente viene ricostruito per mossa, quindi i contatori `metrics` appartengono alla singola
    decisione corrente. Non salviamo punteggi interni o determinizzazioni: bastano ramo usato e
    contatori per distinguere fallback, solver e search/lookahead nei log di produzione.
    """
    metrics = getattr(agent, "metrics", None)
    if metrics is None:
        return None

    is_value_lookahead = hasattr(metrics, "lookahead_decisions")
    lookahead_decisions = int(getattr(metrics, "lookahead_decisions", 0))
    search_decisions = int(getattr(metrics, "search_decisions", 0))
    solver_decisions = int(getattr(metrics, "endgame_solver_decisions", 0))
    fallback_decisions = int(getattr(metrics, "fallback_decisions", 0))
    coerced_moves = int(getattr(metrics, "coerced_moves", 0))
    failed_determinizations = int(getattr(metrics, "failed_determinizations", 0))
    successful_determinizations = int(getattr(metrics, "successful_determinizations", 0))
    failed_leaf_evaluations = int(getattr(metrics, "failed_leaf_evaluations", 0))
    completed_leaf_evaluations = int(getattr(metrics, "completed_leaf_evaluations", 0))
    overkill_guard_adjustments = int(getattr(metrics, "overkill_guard_adjustments", 0))
    search_seconds = float(getattr(metrics, "total_search_seconds", getattr(metrics, "search_elapsed_seconds", 0.0)))

    if lookahead_decisions > 0:
        decision_type = "lookahead"
    elif search_decisions > 0:
        decision_type = "search"
    elif solver_decisions > 0:
        decision_type = "solver"
    elif fallback_decisions > 0:
        decision_type = "fallback"
    else:
        decision_type = "unknown"

    return {
        "agent_kind": "value_lookahead" if is_value_lookahead else "pimc",
        "decision_type": decision_type,
        "lookahead_decisions": lookahead_decisions,
        "search_decisions": search_decisions,
        "endgame_solver_decisions": solver_decisions,
        "fallback_decisions": fallback_decisions,
        "coerced_moves": coerced_moves,
        "successful_determinizations": successful_determinizations,
        "failed_determinizations": failed_determinizations,
        "completed_leaf_evaluations": completed_leaf_evaluations,
        "failed_leaf_evaluations": failed_leaf_evaluations,
        "overkill_guard_adjustments": overkill_guard_adjustments,
        "search_seconds": search_seconds,
    }


def _display_name_for_player(session: GameSession, player_index: int) -> str:
    """
    Nome pubblico breve per messaggi UI.

    I nomi dei player sono parte dello stato di dominio e possono contenere dettagli lunghi
    (es. label del modello selezionato in versioni vecchie della UI). Nei messaggi di partita
    mostriamo invece un'etichetta stabile e leggibile per i seat controllati dall'IA.
    """
    state = session.state
    if _is_ai_controlled_player(session, player_index):
        return _AI_PLAYER_DISPLAY_NAME
    if 0 <= player_index < len(state.players):
        return state.players[player_index].name
    return f"Giocatore {player_index + 1}"


def _schedule_ai_turn_if_needed(
    session: GameSession,
    *,
    human_player_index: int = 0,
    initial_delay_seconds: float = 0.0,
) -> None:
    """
    Avvia il task IA quando lo stato corrente è fermo su un seat controllato dal backend.

    Serve soprattutto per l'avvio partita casuale: se il sorteggio assegna la prima mano
    all'IA, non esiste ancora un'azione umana che possa innescare `_maybe_ai_turn`.
    """
    state = session.state
    if state.game_over:
        return
    if state.num_players != 2:
        return
    if state.current_turn == human_player_index:
        return
    if not _is_ai_controlled_player(session, state.current_turn):
        return
    asyncio.create_task(
        _maybe_ai_turn(
            game_id=session.game_id,
            human_player_index=human_player_index,
            initial_delay_seconds=initial_delay_seconds,
        )
    )


def _initial_ai_start_delay_seconds(session: GameSession, *, human_player_index: int) -> float:
    """
    Ritardo di sola presentazione quando il sorteggio assegna la prima mano all'IA.

    Il backend normalmente non introduce delay di animazione: la UI gestisce reveal e prese.
    Qui però il primo turno IA è un caso diverso, perché può partire subito dopo lo snapshot
    iniziale e rendere invisibile il messaggio "Comincia l'IA". Il ritardo resta confinato
    alla versione 0 della partita e avviene prima di acquisire il lock di gioco.
    """
    state = session.state
    if session.version != 0:
        return 0.0
    if state.current_turn != state.first_player:
        return 0.0
    if state.current_turn == human_player_index:
        return 0.0
    if not _is_ai_controlled_player(session, state.current_turn):
        return 0.0
    return _AI_STARTS_PRESENTATION_DELAY_SECONDS


def _metadata_for_model_catalog_ui(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Riduce i metadati del modello alla parte utile per il browser.

    I file `.npz` possono conservare serie lunghe di metriche di training. Quelle sono corrette
    come artefatto locale, ma non devono viaggiare nell'endpoint della UI a ogni caricamento:
    basta un conteggio sintetico e manteniamo invece i campi descrittivi/inferenziali.
    """
    out = dict(metadata)
    metrics = out.pop("metrics", None)
    if isinstance(metrics, list):
        out["metrics_count"] = len(metrics)
    return out


@app.get("/ai/agents", response_model=dict)
async def list_ai_agents():
    """
    Elenca gli agenti IA disponibili (metadati per UI), con un flag `available`.

    `available` dice se l'agente è realmente giocabile nel deploy corrente:
    - agenti che richiedono un modello "bundled" (`spec.requires_model_id`, es. `best_a2c.npz`) sono
      disponibili solo se quel file è presente nella directory modelli;
    - gli agenti che richiedono un modello scelto (`bc_model`, `bc_model_hybrid_endgame`) sono
      disponibili se esiste almeno un modello `.npz` compatibile nel catalogo;
    - gli altri (random/greedy/euristiche/hybrid_endgame) sono sempre disponibili.
    La UI usa il flag per disabilitare le opzioni rotte (evita "manca il modello") e per scegliere
    un default sensato.
    """
    models_dir = get_models_dir_from_env()
    has_compatible_model = any(m.is_compatible for m in list_local_models(models_dir, recursive=False))

    agents = []
    for spec in list_agent_specs():
        requires_model_selection = agent_uses_selected_model(spec.name)
        required_model_present = spec.requires_model_id is None or (models_dir / spec.requires_model_id).exists()
        if requires_model_selection:
            available = has_compatible_model and required_model_present
        elif spec.requires_model_id is not None:
            available = required_model_present
        else:
            available = True
        agents.append(
            {
                "name": spec.name,
                "label": spec.label,
                "description_it": spec.description_it,
                "requires_model_id": spec.requires_model_id,
                "requires_model_present": required_model_present,
                "requires_model_selection": requires_model_selection,
                "available": available,
            }
        )

    return {"common_note_it": AI_AGENTS_COMMON_NOTE_IT, "agents": agents}


@app.get("/ai/models", response_model=dict)
async def list_ai_models():
    """
    Elenca i modelli `.npz` disponibili sul server (per l'agente `bc_model`).

    Nota sicurezza:
    la UI riceve solo `model_id` (path relativo dentro una directory whitelisted).
    Il backend risolverà poi l'id in un path reale con controlli anti-path-traversal.
    """
    models_dir = get_models_dir_from_env()
    models = list_local_models(models_dir, recursive=False)
    recommended_model = os.getenv("BRISCOLA_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID)
    return {
        "recommended_model": recommended_model,
        "models": [
            {
                "id": m.id,
                "filename": m.filename,
                "label": m.label,
                "description_it": m.description_it,
                "metadata": _metadata_for_model_catalog_ui(m.metadata),
                "last_modified_utc": m.last_modified_utc,
                "is_compatible": m.is_compatible,
                "compatibility_reason_it": m.compatibility_reason_it,
            }
            for m in models
        ],
    }


def _get_replay_agent() -> Agent:
    """
    Restituisce il profilo massimo del laboratorio, costruendolo una sola volta.

    Il lock protegge sia la prima inizializzazione sia la scelta: PIMC conserva piccole
    telemetrie nell'istanza e il server può ricevere più replay contemporaneamente.
    """
    global _replay_agent
    with _replay_agent_lock:
        if _replay_agent is None:
            models_dir = get_models_dir_from_env()
            model_path = resolve_model_path(models_dir=models_dir, model_id=_replay_model_id())
            _replay_agent = build_agent(_REPLAY_AGENT_NAME, model_path=model_path)
        return _replay_agent


def _replay_model_id() -> str:
    """Usa lo stesso override del modello consigliato impiegato dal runtime principale."""
    return os.getenv("BRISCOLA_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def _onehot_card_ids(cards: list[Card]) -> tuple[int, ...]:
    """Rappresenta un insieme di carte pubbliche nel formato canonico da 40 bit."""
    flags = [0] * 40
    for card in cards:
        flags[card_to_id(card)] = 1
    return tuple(flags)


def _build_replay_observation(payload: ReplayAdviceRequest) -> PlayerObservation:
    """
    Valida e trasforma un'istantanea di replay nella vista lecita dell'analista.

    L'uguaglianza ``mano + tavolo + prese + carte ignote == 40`` è importante: non
    serve solo a intercettare una registrazione incompleta, ma garantisce che PIMC
    possa campionare esclusivamente mondi compatibili con il replay inserito.
    """
    hand = [item.to_domain() for item in payload.hand]
    trump = payload.trump_card.to_domain()
    completed = [item.to_domain() for item in payload.completed_cards]
    table = [(item.card.to_domain(), item.player_index) for item in payload.table_cards]

    # Una carta non può stare contemporaneamente nella mano, sul tavolo o in una presa.
    # La briscola è l'eccezione naturale: può essere ancora nel mazzo oppure già nelle prese.
    card_ids = [card_to_id(card) for card in hand]
    card_ids.extend(card_to_id(card) for card, _player_index in table)
    card_ids.extend(card_to_id(card) for card in completed)
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("Una carta compare più di una volta nel replay")

    known_count = len(hand) + len(table) + len(completed)
    expected_total = known_count + payload.opponent_hand_size + payload.deck_size
    if expected_total != 40:
        raise ValueError("Contatori incoerenti: mano, tavolo, prese, mazzo e mano avversaria devono sommare 40 carte")

    completed_points = sum(card.rank.points for card in completed)
    if payload.my_points + payload.opponent_points != completed_points:
        raise ValueError("I punti dichiarati non corrispondono alle carte nelle prese registrate")

    # `seen` include la briscola scoperta; `out_of_play` no, finché non entra davvero
    # in una presa. È la stessa distinzione usata dalla partita normale.
    seen = _onehot_card_ids([trump, *completed, *(card for card, _ in table)])
    out_of_play = _onehot_card_ids([*completed, *(card for card, _ in table)])
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="Analista",
        hand=tuple(hand),
        trump_card=trump,
        deck_size=payload.deck_size,
        table_cards=tuple(table),
        current_turn=0,
        first_player=payload.first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(payload.my_points, payload.opponent_points),
        players_hand_sizes=(len(hand), payload.opponent_hand_size),
        seen_cards_onehot=seen,
        out_of_play_cards_onehot=out_of_play,
    )


def _choose_replay_card(observation: PlayerObservation) -> tuple[Card, str]:
    """Esegue la search nel worker thread e normalizza difensivamente l'indice scelto."""
    with _replay_agent_lock:
        agent = _get_replay_agent()
        card_index = agent.choose_card_index(observation, rng=random.Random(0))
        if card_index < 0 or card_index >= len(observation.hand):
            raise ValueError("L'agente ha restituito una carta non presente nella mano")
        return observation.hand[card_index], agent.name


@app.post("/replay/advice", response_model=ReplayAdviceResponse)
async def get_replay_advice(payload: ReplayAdviceRequest) -> ReplayAdviceResponse:
    """
    Analizza un istante di una partita **già registrata**, senza salvarne alcun dato.

    Per evitare di bloccare il loop ASGI durante PIMC 64x10, il calcolo CPU viene eseguito
    in un worker thread. Qualsiasi incoerenza nel replay è restituita come 422, utile alla
    UI per chiedere la carta o il punteggio mancante invece di fornire un consiglio falso.
    """
    try:
        observation = _build_replay_observation(payload)
        card, agent_name = await asyncio.to_thread(_choose_replay_card, observation)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ReplayAdviceResponse(card=CardDTO.from_domain(card), agent=agent_name, model_id=_replay_model_id())


@app.get("/")
async def root():
    """Health-check minimale."""
    return {"message": "Benvenuto nelle API di Briscola AI"}


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc: Exception):
    """
    Rete di sicurezza: eccezione non gestita -> 500 pulito al client + notifica email
    al maintainer (best-effort, con dedup: vedi backend/alerts.py). Il traceback resta
    comunque nei log della piattaforma.
    """
    from fastapi.responses import JSONResponse

    from .alerts import notify_exception

    notify_exception(
        exc,
        context={
            "method": getattr(request, "method", None),
            "path": str(getattr(getattr(request, "url", None), "path", "")),
            "client": getattr(getattr(request, "client", None), "host", None),
        },
    )
    import logging

    logging.getLogger("briscola.server").exception(
        "Eccezione non gestita su %s", getattr(getattr(request, "url", None), "path", "?")
    )
    return JSONResponse(status_code=500, content={"detail": "Errore interno del server"})


@app.get("/meta", response_model=dict)
async def meta() -> dict:
    """
    Metadati “di runtime” per UI/deploy.

    Scopi:
    - mostrare/abilitare UX legata alla raccolta dati (consenso) quando `event_log_mode=dataset`.
    - debug rapido (versioni).
    """
    mode = _get_event_log_mode()
    return {
        "code_version": get_code_version(),
        "rules_version": get_rules_version(),
        "event_log_mode": mode,
        **event_log_runtime_metadata(),
        "dataset_requires_consent": mode == "dataset",
        "debug_state_endpoint_enabled": _debug_state_endpoint_enabled(),
        "cors_allow_origins": cors_allow_origins,
        "error_alerts_configured": _error_alerts_configured(),
    }


@app.get("/stats/games", response_model=dict)
async def stats_games() -> dict:
    """
    Statistiche aggregate e anonime: partite registrate (totali/completate) per
    modello/agente avversario, lette dall'event log. Best-effort: campi null se
    l'event log non e' configurato o non risponde.
    """
    return _games_stats_payload()


# Rate limiting sulla creazione partite: senza tetto, un client può creare partite senza
# limite (RAM/Redis/DB unbounded). Sliding window in-process per IP: in multi-replica il
# limite effettivo è `limite * num_repliche`, accettabile come protezione di primo livello.
_CREATE_GAME_RATE_WINDOW_SECONDS = 60.0
_create_game_requests: dict[str, list[float]] = {}


def _create_game_rate_limit() -> int:
    """
    Tetto di partite create per IP nella finestra (0 o negativo = disabilitato).

    Letto a ogni richiesta (non a import time) così i test possono controllarlo con
    `monkeypatch.setenv` e l'operatore può cambiarlo senza redeploy del codice.
    """
    raw = os.getenv("BRISCOLA_CREATE_GAME_RATE_LIMIT", "30").strip()
    try:
        return int(raw)
    except ValueError:
        return 30


def _check_create_game_rate_limit(client_ip: str) -> None:
    """Solleva 429 se `client_ip` ha superato il tetto di partite create nella finestra."""
    limit = _create_game_rate_limit()
    if limit <= 0:
        return  # disabilitato (utile nei test/dev)
    now = time.monotonic()
    window_start = now - _CREATE_GAME_RATE_WINDOW_SECONDS
    timestamps = [t for t in _create_game_requests.get(client_ip, []) if t > window_start]
    if len(timestamps) >= limit:
        raise HTTPException(
            status_code=429,
            detail="Troppe partite create di recente: riprova tra qualche secondo.",
        )
    timestamps.append(now)
    _create_game_requests[client_ip] = timestamps
    # Pulizia best-effort della mappa (bounded): via gli IP senza richieste recenti.
    if len(_create_game_requests) > 10_000:
        for ip in list(_create_game_requests):
            if not any(t > window_start for t in _create_game_requests[ip]):
                _create_game_requests.pop(ip, None)


@app.post("/games", response_model=dict)
async def create_game(config: GameConfig, request: Request):
    """Crea una nuova partita di Briscola"""
    client_ip = request.client.host if request.client is not None else "unknown"
    _check_create_game_rate_limit(client_ip)
    try:
        if _get_event_log_mode() == "dataset" and config.consent_to_data_collection is not True:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Consenso mancante: per raccogliere dati umani (event_log_mode=dataset) "
                    "devi accettare la registrazione anonima delle mosse."
                ),
            )

        # Seed per rendere riproducibile lo shuffle in fase di debugging/dataset.
        # In produzione useremmo RNG più robusto o un seed esplicito del client.
        seed = random.randrange(0, 2**32)
        state = new_game_state(config.num_players, config.player_names, seed=seed)
        starting_player = _choose_starting_player(seed, config.num_players)
        state = replace(state, current_turn=starting_player, first_player=starting_player)

        # Genera un ID univoco per la partita
        game_id = str(uuid.uuid4())
        ai_agent_name = config.ai_agent or _DEFAULT_AI_AGENT_NAME

        # Config IA (solo 2-player, come la UI attuale).
        ai_seats: dict[int, AiSeatConfig] = {}
        if config.num_players == 2:
            if agent_uses_selected_model(ai_agent_name):
                models_dir = get_models_dir_from_env()
                # Validiamo il path del modello PRIMA di salvare la sessione (come prima).
                model_path = resolve_model_path(models_dir, config.ai_model_id or "")
                validate_model_compatible_for_ui(model_path)
                # Validiamo anche eventuali asset fissi richiesti dall'agente (es. value model).
                build_agent(ai_agent_name, model_path=model_path)
                ai_seats = {1: AiSeatConfig(agent_name=ai_agent_name, model_id=config.ai_model_id)}
            else:
                # Validiamo l'agente PRIMA di salvare la sessione (come prima): un nome non valido
                # o un alias non disponibile (es. best_a2c senza file) deve dare 400 alla creazione,
                # non un crash più tardi nel task IA. L'oggetto costruito viene scartato (la sessione
                # salva solo la config; l'agente è ricostruito per mossa, con cache).
                build_agent(ai_agent_name)
                ai_seats = {1: AiSeatConfig(agent_name=ai_agent_name, model_id=None)}

        now_iso = _utcnow().isoformat()
        session = GameSession(
            game_id=game_id,
            state=state,
            version=0,
            ai_seats=ai_seats,
            action_seed=seed ^ 0x9E3779B9,
            created_at=now_iso,
            updated_at=now_iso,
        )
        await game_store.set(session)

        game_timestamps[game_id] = _utcnow()
        game_data[game_id] = [
            {
                "timestamp": _utcnow().isoformat(),
                "event": "game_created",
                "seed": seed,
                "ai_agent": ai_agent_name if config.num_players == 2 else None,
                "ai_model_id": config.ai_model_id
                if (config.num_players == 2 and agent_uses_selected_model(ai_agent_name))
                else None,
            }
        ]

        # Event log (opzionale): metadati partita.
        _safe_log_event(
            game_id,
            "game_created",
            {
                "seed": seed,
                "code_version": get_code_version(),
                "rules_version": get_rules_version(),
                "num_players": config.num_players,
                "is_team_game": state.is_team_game,
                "first_player": starting_player,
                "ai_agent": ai_agent_name if config.num_players == 2 else None,
                "ai_model_id": config.ai_model_id
                if (config.num_players == 2 and agent_uses_selected_model(ai_agent_name))
                else None,
                "client_id": config.client_id,
                "consent_to_data_collection": bool(config.consent_to_data_collection is True),
            },
            server_version=0,
            state=state,
        )
        _safe_set_client_id(game_id, config.client_id)

        return {
            "game_id": game_id,
            "status": "created",
            "num_players": config.num_players,
            "is_team_game": state.is_team_game,
            "first_player": starting_player,
            "current_turn": state.current_turn,
            "player_names": [p.name for p in state.players],
            "ai_agent": ai_agent_name if config.num_players == 2 else None,
            "ai_model_id": config.ai_model_id
            if (config.num_players == 2 and agent_uses_selected_model(ai_agent_name))
            else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail="Modello .npz non trovato (ai_model_id)") from e


def _debug_state_endpoint_enabled() -> bool:
    """
    Ritorna True se la vista full-state di `GET /games/{id}` (senza `player_index`) è abilitata.

    Perché esiste:
    la vista full-state espone le mani di TUTTI i giocatori e `next_deck_card`, quindi
    contraddice l'invariante anti-cheat se raggiungibile da qualunque client in produzione.
    La teniamo come strumento di debug/spectator locale, ma va abilitata esplicitamente
    dall'operatore con `BRISCOLA_DEBUG_STATE_ENDPOINT=unsafe-full-state`. Il valore
    volutamente esplicito evita che un vecchio flag booleano lasci per errore carte nascoste
    accessibili su un deploy pubblico. Leggiamo l'env a ogni richiesta (non a import time),
    così i test possono attivarla/disattivarla con `monkeypatch`.
    """
    raw = os.getenv("BRISCOLA_DEBUG_STATE_ENDPOINT", "").strip().lower()
    return raw == "unsafe-full-state"


@app.get("/games/{game_id}", response_model=dict)
async def get_game_state(game_id: str, player_index: int | None = None):
    """Ottiene lo stato corrente di una partita"""
    session = await game_store.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    game = session.state

    # Aggiorna il timestamp per mantenere la partita attiva
    game_timestamps[game_id] = _utcnow()

    if player_index is not None:
        # Restituisce una vista specifica per il giocatore (stesso formato dei messaggi WS)
        try:
            observation_dto = build_observation_dto(game, player_index, session.version)
            _schedule_ai_turn_if_needed(
                session,
                human_player_index=player_index,
                initial_delay_seconds=_initial_ai_start_delay_seconds(session, human_player_index=player_index),
            )
            return observation_dto.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        # Vista completa (mani di tutti + `next_deck_card`) per spettatori o debugging.
        # Anti-cheat: disabilitata di default, l'operatore deve abilitarla esplicitamente.
        if not _debug_state_endpoint_enabled():
            raise HTTPException(
                status_code=403,
                detail=(
                    "Vista full-state disabilitata (anti-cheat). Usa `player_index` per la vista "
                    "del giocatore, oppure imposta "
                    "BRISCOLA_DEBUG_STATE_ENDPOINT=unsafe-full-state per il debug locale."
                ),
            )
        game_state_dto = build_game_state_dto(game, session.version)
        return game_state_dto.model_dump()


@app.post("/games/{game_id}/abandon", response_model=dict)
async def abandon_game(game_id: str, payload: GameAbandonRequest) -> dict:
    """
    Abbandona una partita in corso.

    La semantica è intenzionalmente semplice:
    - se la partita è ancora aperta, la marchiamo come abortita nell'event log (best-effort);
    - rimuoviamo la sessione dallo store, così polling/WS/azioni successive non la tengono viva;
    - non assegniamo una vittoria a tavolino: per dataset e report è una partita incompleta.
    """
    async with game_store.lock(game_id):
        session = await game_store.get(game_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Partita non trovata")

        state = session.state
        server_version = session.version
        if state.game_over:
            return {"game_id": game_id, "status": "already_finished"}

        log = _get_event_log()
        aborted_marked = False
        if log is not None and _get_event_log_mode() != "off":
            try:
                seed = getattr(state, "seed", None)
                log.ensure_game(
                    game_id,
                    num_players=state.num_players,
                    seed=seed if isinstance(seed, int) else None,
                    code_version=get_code_version(),
                    rules_version=get_rules_version(),
                )
                aborted_marked = log.try_mark_game_aborted(game_id, aborted_reason="user_abandoned")
            except Exception:
                aborted_marked = False

        if aborted_marked:
            _safe_log_event(
                game_id,
                "game_aborted",
                {
                    "reason": "user_abandoned",
                    "player_index": payload.player_index,
                },
                server_version=server_version,
                player_index=payload.player_index,
                state=state,
            )

        await game_store.delete(game_id)
        game_timestamps.pop(game_id, None)
        game_data.pop(game_id, None)

    return {"game_id": game_id, "status": "abandoned"}


@app.post("/games/{game_id}/actions", response_model=PlayActionResultDTO, response_model_exclude_none=True)
async def play_action(game_id: str, action: GameAction) -> PlayActionResultDTO:
    """Gioca una carta nella partita"""
    if await game_store.get(game_id) is None:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    should_schedule_ai = False

    async with game_store.lock(game_id):
        session = await game_store.get(game_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Partita non trovata")
        state = session.state
        server_version_before = session.version

        # Verifica che sia il turno del giocatore
        if state.current_turn != action.player_index:
            raise HTTPException(status_code=400, detail="Non è il tuo turno")

        if _is_ai_controlled_player(session, action.player_index):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Il giocatore {action.player_index} è controllato dall'IA: "
                    "le sue mosse vengono eseguite automaticamente dal server."
                ),
            )

        observation_before: dict | None = None
        if _get_event_log_mode() == "dataset":
            # Per dataset umano usiamo sempre ObservationDTO (vista parziale anti-cheat).
            try:
                observation_before = build_observation_dto(
                    state, action.player_index, server_version_before
                ).model_dump()
            except Exception:
                observation_before = None

        # Esegue l'azione
        new_state, step_result = step(
            state,
            PlayCardAction(player_index=action.player_index, card_index=action.card_index),
        )
        if step_result.error:
            # Usiamo un errore HTTP standard invece di un payload con `error`.
            # Questo rende l'API più prevedibile per i client e coerente con gli altri endpoint.
            raise HTTPException(status_code=400, detail=step_result.error)

        session.state = new_state
        session.version += 1
        session.updated_at = _utcnow().isoformat()
        await game_store.set(session)
        server_version = session.version

        if step_result.played_card is None or step_result.player is None:
            # Invariante: su successo, `step()` deve restituire sempre played_card e player.
            raise HTTPException(status_code=500, detail="Risposta dominio incompleta (played_card/player)")

        trick_cards_dto: list[TableCardDTO] | None = None
        captured_cards_dto: list[CardDTO] = []
        if step_result.trick_completed:
            trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in step_result.trick_cards]
            captured_cards_dto = [CardDTO.from_domain(card) for card, _ in step_result.trick_cards]

        action_result_dto = PlayActionResultDTO(
            server_version=server_version,
            played_card=CardDTO.from_domain(step_result.played_card),
            player=step_result.player,
            trick_completed=step_result.trick_completed,
            trick_winner=step_result.trick_winner,
            trick_size=len(step_result.trick_cards),
            cards_dealt=step_result.cards_dealt,
            trick_cards=trick_cards_dto,
            captured_cards=captured_cards_dto,
        )

        # Aggiorna timestamp
        game_timestamps[game_id] = _utcnow()

        # Registra l'azione per il training ML.
        # setdefault: game_data e' per-replica; se l'azione arriva su una replica diversa da
        # quella che ha creato la partita, la lista non esiste ancora (stato vive nello store).
        _remember_game_action(
            game_id,
            {
                "timestamp": _utcnow().isoformat(),
                "player_index": action.player_index,
                "card_index": action.card_index,
                # Salviamo il DTO (JSON-friendly) invece di oggetti di dominio.
                "result": action_result_dto.model_dump(exclude_none=True),
            },
        )

        # Event log (opzionale): azione umana + risultato.
        if _get_event_log_mode() == "dataset":
            reward = 0
            if step_result.trick_completed:
                trick_points = sum(card.rank.points for card, _ in step_result.trick_cards)
                winner = step_result.trick_winner
                if isinstance(winner, int):
                    reward = trick_points if winner == action.player_index else -trick_points

            next_observation: dict | None = None
            try:
                next_observation = build_observation_dto(new_state, action.player_index, server_version).model_dump()
            except Exception:
                next_observation = None

            _safe_log_event(
                game_id,
                "human_action",
                sanitize_dataset_payload(
                    {
                        "player_index": action.player_index,
                        "card_index": action.card_index,
                        "observation": observation_before,
                        "reward": reward,
                        "done": bool(new_state.game_over is True),
                        "next_observation": next_observation,
                        "result": action_result_dto.model_dump(exclude_none=True),
                        "client_observed_server_version": action.client_observed_server_version,
                        "client_decision_time_ms": action.client_decision_time_ms,
                    }
                ),
                server_version=server_version,
                player_index=action.player_index,
                state=new_state,
            )
        else:
            _safe_log_event(
                game_id,
                "action_play_card",
                {
                    "is_ai": False,
                    "player_index": action.player_index,
                    "card_index": action.card_index,
                    "result": action_result_dto.model_dump(exclude_none=True),
                },
                server_version=server_version,
                player_index=action.player_index,
                state=new_state,
            )

        # Se la mano è stata completata dall'umano, invia notifica con carte e vincitore.
        #
        # Nota architetturale:
        # evitiamo `asyncio.sleep()` nel backend per ritardi "di presentazione". Il tempo
        # di visualizzazione del risultato è gestito dal frontend (che può “trattenere”
        # lo snapshot successivo finché l'utente ha letto il risultato della mano).
        if step_result.trick_completed:
            trick_cards = list(step_result.trick_cards)
            winner_index = step_result.trick_winner if step_result.trick_winner is not None else 0
            points = sum(card.rank.points for card, _ in step_result.trick_cards)
            await notify_trick_result(session, trick_cards, winner_index, points)
            # Subito dopo inviamo anche lo stato aggiornato (tavolo vuoto, nuove carte pescate).
            # Il frontend decide se applicarlo subito o dopo un delay.
            await notify_clients(game_id, new_state, server_version)
        else:
            # Mano non completata: notifica normale
            await notify_clients(game_id, new_state, server_version)

        # Calcoliamo qui se dobbiamo far giocare l'IA (fuori dal lock scheduliamo solo il task).
        if not new_state.game_over and new_state.num_players == 2 and new_state.current_turn != action.player_index:
            should_schedule_ai = True
        elif new_state.game_over:
            _maybe_log_game_finished(game_id, state=new_state, server_version=server_version)

    # Modello "standard": se dopo la mossa umana tocca all'IA, il backend gioca automaticamente.
    # Nota UX: non inseriamo `asyncio.sleep()` per animazioni; il frontend gestisce i timing
    # trattenendo gli update (reveal/risultato mano) quando li riceve.
    #
    # Nota architetturale (task IA fuori lock):
    # Schediliamo il task IA *dopo* aver rilasciato il lock per evitare deadlock e permettere
    # alla risposta HTTP di tornare subito al client. Il check `game_after.game_over` qui è
    # solo un'ottimizzazione: la vera guardia è dentro `_maybe_ai_turn`, che riacquisisce il
    # lock e verifica nuovamente lo stato prima di giocare. Questo pattern è safe perché:
    # 1. Il task può trovare la partita già terminata/rimossa → ritorna subito.
    # 2. Eventuali azioni concorrenti (es. reconnect) sono serializzate dal lock interno.
    if should_schedule_ai:
        asyncio.create_task(_maybe_ai_turn(game_id=game_id, human_player_index=action.player_index))

    return action_result_dto


async def _maybe_ai_turn(
    game_id: str,
    human_player_index: int,
    *,
    initial_delay_seconds: float = 0.0,
) -> None:
    """
    Esegue automaticamente le mosse dell'IA quando è il suo turno (2-player).

    Nota architetturale:
    - modello standard: il backend avanza la partita senza richiedere un trigger dal client.
    - il frontend controlla quasi tutta la *presentazione* (hold/animazioni) senza influenzare
      il dominio; l'unica eccezione è la pausa iniziale quando il sorteggio fa partire l'IA.
    """
    if initial_delay_seconds > 0:
        await asyncio.sleep(initial_delay_seconds)

    # In 2-player ci aspettiamo al massimo una mossa IA per volta, ma gestiamo anche
    # eventuali casi futuri dove l'IA potrebbe avere turni consecutivi (safety loop).
    safety = 10
    while safety > 0:
        safety -= 1
        async with game_store.lock(game_id):
            session = await game_store.get(game_id)
            if session is None:
                return
            state = session.state
            if state.game_over:
                return
            if state.num_players != 2:
                return
            if state.current_turn == human_player_index:
                return

            await _execute_ai_turn_locked(session, human_player_index)


async def _execute_ai_turn_locked(session: GameSession, human_player_index: int) -> None:
    """
    Esegue UNA singola mossa IA.

    Precondizione:
    - il chiamante ha acquisito `game_store.lock(game_id)` e passa la sessione già caricata.
    """
    game_id = session.game_id
    state = session.state

    # Se la partita è finita o tocca al giocatore umano, non fare nulla
    if state.game_over or state.current_turn == human_player_index:
        return

    # AI gioca una carta usando l'agente configurato.
    #
    # Nota anti-cheat:
    # la policy riceve `PlayerObservation` (vista parziale lecita), non `GameState` completo.
    ai_player_index = state.current_turn
    valid_actions = list(range(len(state.players[ai_player_index].hand))) if not state.game_over else []

    if not valid_actions:
        return

    # RNG deterministico per mossa: dipende dal seed della partita e dalla versione corrente.
    rng = random.Random(session.action_seed ^ session.version)
    seat_cfg = session.ai_seats.get(ai_player_index)
    agent = _agent_for_seat(seat_cfg) if seat_cfg is not None else None

    observation_before: dict | None = None
    if _get_event_log_mode() == "dataset":
        try:
            observation_before = build_observation_dto(state, ai_player_index, session.version).model_dump()
        except Exception:
            observation_before = None

    decision_trace: dict[str, Any] | None = None
    action_coerced = False
    if agent is None:
        card_index = rng.randrange(len(valid_actions))
    else:
        observation = make_player_observation(state, ai_player_index)
        # La scelta dell'agente è CPU-bound e può durare centinaia di ms (PIMC, value-lookahead,
        # solver): eseguirla direttamente bloccherebbe l'intero event loop del worker, congelando
        # TUTTE le altre partite/WS/HTTP della replica. La spostiamo in un thread del pool di
        # default. Il lock di partita resta volutamente acquisito: lo stato non deve cambiare
        # mentre l'agente decide (vedi nota sul TTL del lock Redis in `game_store.py`).
        loop = asyncio.get_running_loop()
        card_index = await loop.run_in_executor(None, lambda: agent.choose_card_index(observation, rng=rng))
        if card_index not in valid_actions:
            # Fallback di sicurezza: se un agente ritorna un indice invalido, non blocchiamo la partita.
            action_coerced = True
            card_index = rng.randrange(len(valid_actions))
        decision_trace = _ai_decision_trace(agent)

    selected_card = state.players[ai_player_index].hand[card_index]

    # Pubblica il messaggio per rivelare la carta nella mano IA (usando DTO).
    # Il fan-out verso i socket avviene tramite i task subscriber (pub/sub dello store).
    reveal_dto = AiCardRevealDTO(
        card_index=card_index,
        card=CardDTO.from_domain(selected_card),
        decision_type=decision_trace.get("decision_type") if decision_trace is not None else None,
    )
    _safe_log_event(
        game_id,
        "ai_card_reveal",
        reveal_dto.model_dump(),
        server_version=session.version,
        player_index=ai_player_index,
        state=state,
    )
    await game_store.publish(game_id, reveal_dto.model_dump_json())

    new_state, step_result = step(state, PlayCardAction(player_index=ai_player_index, card_index=card_index))
    if step_result.error:
        return

    session.state = new_state
    session.version += 1
    session.updated_at = _utcnow().isoformat()
    await game_store.set(session)
    server_version = session.version

    # Event log + game_data: usiamo un DTO JSON-friendly anche per le mosse IA.
    trick_cards_dto: list[TableCardDTO] | None = None
    captured_cards_dto: list[CardDTO] = []
    if step_result.trick_completed:
        trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in step_result.trick_cards]
        captured_cards_dto = [CardDTO.from_domain(card) for card, _ in step_result.trick_cards]

    if step_result.played_card is None or step_result.player is None:
        return

    ai_action_result_dto = PlayActionResultDTO(
        server_version=server_version,
        played_card=CardDTO.from_domain(step_result.played_card),
        player=step_result.player,
        trick_completed=step_result.trick_completed,
        trick_winner=step_result.trick_winner,
        trick_size=len(step_result.trick_cards),
        cards_dealt=step_result.cards_dealt,
        trick_cards=trick_cards_dto,
        captured_cards=captured_cards_dto,
    )

    if _get_event_log_mode() == "dataset":
        reward = 0
        if step_result.trick_completed:
            trick_points = sum(card.rank.points for card, _ in step_result.trick_cards)
            winner = step_result.trick_winner
            if isinstance(winner, int):
                reward = trick_points if winner == ai_player_index else -trick_points

        next_observation: dict | None = None
        try:
            next_observation = build_observation_dto(new_state, ai_player_index, server_version).model_dump()
        except Exception:
            next_observation = None

        _safe_log_event(
            game_id,
            "ai_action",
            sanitize_dataset_payload(
                {
                    "is_ai": True,
                    "player_index": ai_player_index,
                    "ai_agent": seat_cfg.agent_name if seat_cfg is not None else None,
                    "ai_model_id": seat_cfg.model_id if seat_cfg is not None else None,
                    "card_index": card_index,
                    "action_coerced": action_coerced,
                    "observation": observation_before,
                    "reward": reward,
                    "done": bool(new_state.game_over is True),
                    "next_observation": next_observation,
                    "result": ai_action_result_dto.model_dump(exclude_none=True),
                    "decision_trace": decision_trace,
                }
            ),
            server_version=server_version,
            player_index=ai_player_index,
            state=new_state,
        )

    # Se la mano è stata completata, invia notifica speciale
    if step_result.trick_completed:
        trick_cards = list(step_result.trick_cards)
        winner_index = step_result.trick_winner if step_result.trick_winner is not None else 0
        points = sum(card.rank.points for card, _ in step_result.trick_cards)
        await notify_trick_result(session, trick_cards, winner_index, points)
        await notify_clients(game_id, new_state, server_version)
    else:
        await notify_clients(game_id, new_state, server_version)

    # Registra l'azione AI (setdefault: game_data e' per-replica, vedi nota in play_action).
    game_timestamps[game_id] = _utcnow()
    _remember_game_action(
        game_id,
        {
            "timestamp": _utcnow().isoformat(),
            "player_index": ai_player_index,
            "card_index": card_index,
            "result": ai_action_result_dto.model_dump(exclude_none=True),
            "is_ai": True,
        },
    )

    _safe_log_event(
        game_id,
        "action_play_card",
        {
            "is_ai": True,
            "player_index": ai_player_index,
            "card_index": card_index,
            "result": ai_action_result_dto.model_dump(exclude_none=True),
        },
        server_version=server_version,
        player_index=ai_player_index,
        state=new_state,
    )
    if new_state.game_over:
        _maybe_log_game_finished(game_id, state=new_state, server_version=server_version)
    return


@app.get("/games/{game_id}/result", response_model=GameResultDTO, response_model_exclude_none=True)
async def get_game_result(game_id: str) -> GameResultDTO:
    """Ottiene il risultato finale di una partita"""
    session = await game_store.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    state = session.state
    server_version = session.version
    if not state.game_over:
        return GameResultDTO(
            server_version=server_version,
            game_in_progress=True,
            game_over=False,
            is_team_game=state.is_team_game,
            points={},
        )

    points_by_player = {p.name: p.points for p in state.players}
    if state.is_team_game and state.teams is not None:
        team_0_points = sum(state.players[i].points for i in state.teams[0])
        team_1_points = sum(state.players[i].points for i in state.teams[1])

        if team_0_points > team_1_points:
            winning_team = 0
        elif team_1_points > team_0_points:
            winning_team = 1
        else:
            winning_team = None

        if winning_team is None:
            winner_str = "Pareggio"
        else:
            team_players = state.teams[winning_team]
            p0_name = state.players[team_players[0]].name
            p1_name = state.players[team_players[1]].name
            winner_str = f"Squadra {winning_team} ({p0_name} e {p1_name})"

        return GameResultDTO(
            server_version=server_version,
            game_in_progress=False,
            game_over=True,
            is_team_game=True,
            winner=winner_str,
            winning_team=winning_team,
            team_points={"Team 0": team_0_points, "Team 1": team_1_points},
            points=points_by_player,
            point_difference=abs(team_0_points - team_1_points),
        )

    p0 = state.players[0].points
    p1 = state.players[1].points
    if p0 > p1:
        winner_index = 0
    elif p1 > p0:
        winner_index = 1
    else:
        winner_index = None

    return GameResultDTO(
        server_version=server_version,
        game_in_progress=False,
        game_over=True,
        is_team_game=False,
        winner=state.players[winner_index].name if winner_index is not None else "Pareggio",
        winner_index=winner_index,
        points=points_by_player,
        point_difference=abs(p0 - p1),
    )


async def _ws_subscriber(websocket: WebSocket, game_id: str, player_index: int) -> None:
    """
    Task di consegna realtime per UNA connessione WebSocket.

    Si iscrive al pub/sub dello store per la partita e inoltra ogni evento al socket:
    - messaggi `reveal`/`trick_result`: inoltrati verbatim (il `type` è già nel JSON);
    - messaggi `refresh`: NON contengono un'osservazione (sarebbe per-giocatore); il subscriber
      rilegge lo stato dallo store e costruisce l'osservazione PER QUESTO `player_index`
      (anti-cheat: ogni client riceve solo la propria vista parziale).

    Robustezza: un singolo messaggio malformato non deve uccidere il loop; in caso di errore di
    `send` consideriamo la connessione persa e usciamo (il `finally` dell'endpoint farà cleanup).
    """
    # `aclosing` garantisce la chiusura deterministica del generator (unsubscribe/cleanup della
    # coda o del pubsub) anche quando usciamo con `break` o per cancellazione del task.
    async with aclosing(game_store.subscribe(game_id)) as events:
        async for raw in events:
            try:
                msg = json.loads(raw)
                if msg.get("type") == "refresh":
                    # Stato point-in-time incluso nel messaggio (vedi `notify_clients`): lo usiamo
                    # per costruire l'osservazione di QUESTO player, senza rileggere il "latest"
                    # (che potrebbe essere già avanzato dalla mossa IA → ordine eventi sbagliato).
                    state_dict = msg.get("state")
                    if state_dict is None:
                        continue
                    state = game_state_from_dict(state_dict)
                    obs = build_observation_dto(state, player_index, int(msg.get("server_version", 0)))
                    await websocket.send_text(obs.model_dump_json())
                else:
                    # reveal/trick: inoltro verbatim del JSON pubblicato.
                    await websocket.send_text(raw)
            except WebSocketDisconnect, RuntimeError:
                # Il socket è andato (disconnect o "websocket closed"): chiudiamo il loop.
                break
            except Exception:
                # Messaggio malformato o errore non fatale di parsing: ignora e prosegui.
                continue


@app.websocket("/ws/{game_id}/{player_index}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, player_index: int):
    """Endpoint WebSocket per aggiornamenti della partita in tempo reale"""
    try:
        session = await game_store.get(game_id)
    except Exception as exc:
        # Store (Redis) non raggiungibile: incidente reale visto in produzione il 2026-07-06.
        # Degrada con grazia: 1013 = "try again later", il backoff del client riprova; e
        # il maintainer riceve la notifica (con dedup) invece di scoprirlo dai log.
        from .alerts import notify_exception

        notify_exception(exc, context={"path": f"/ws/{game_id}/{player_index}", "phase": "store.get (open)"})
        with suppress(Exception):
            await websocket.close(code=1013, reason="Store non disponibile, riprova")
        return
    if session is None:
        await websocket.close(code=1000, reason="Partita non trovata")
        return

    state = session.state

    if player_index < 0 or player_index >= state.num_players:
        await websocket.close(code=1000, reason="Indice giocatore non valido")
        return

    await websocket.accept()

    # Avvia il task subscriber che inoltra a questo socket gli eventi pubblicati sullo store.
    sub_task = asyncio.create_task(_ws_subscriber(websocket, game_id, player_index))

    try:
        # Invia lo stato iniziale della partita (usando DTO)
        dto = build_observation_dto(state, player_index, session.version)
        _safe_log_event(
            game_id,
            "ws_connected",
            {"player_index": player_index},
            server_version=session.version,
            player_index=player_index,
            state=state,
        )
        await websocket.send_text(dto.model_dump_json())
        _schedule_ai_turn_if_needed(
            session,
            human_player_index=player_index,
            initial_delay_seconds=_initial_ai_start_delay_seconds(session, human_player_index=player_index),
        )

        # Mantiene la connessione aperta e gestisce i messaggi
        while True:
            # Attende messaggi (le azioni verranno inviate via HTTP)
            data = await websocket.receive_text()

            # Elabora eventuali comandi inviati via WebSocket
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        # Best-effort: rileggiamo la sessione per loggare una server_version aggiornata.
        # La rilettura passa dallo store (Redis): se e' giu' NON deve trasformare una
        # normale disconnessione in un'eccezione ASGI (incidente del 2026-07-06).
        try:
            current = await game_store.get(game_id)
        except Exception:
            current = None
        _safe_log_event(
            game_id,
            "ws_disconnected",
            {"player_index": player_index},
            server_version=current.version if current is not None else 0,
            player_index=player_index,
            state=current.state if current is not None else None,
        )
    except Exception as exc:
        # Errore inatteso nel loop WS (fuori dal contratto disconnect): notifica + chiusura pulita.
        from .alerts import notify_exception

        notify_exception(exc, context={"path": f"/ws/{game_id}/{player_index}", "phase": "ws loop"})
        with suppress(Exception):
            await websocket.close(code=1011, reason="Errore interno")
    finally:
        # Ferma il task subscriber: la connessione non esiste più.
        sub_task.cancel()
        with suppress(asyncio.CancelledError):
            await sub_task


async def notify_clients(game_id: str, state: DomainGameState, server_version: int) -> None:
    """
    Notifica i client connessi che lo snapshot della partita è cambiato.

    Pubblichiamo un messaggio "refresh" sul pub/sub dello store, includendo lo stato
    **point-in-time** (`game_state_to_dict(state)`): ogni task subscriber ne ricava l'osservazione
    per il proprio `player_index` (anti-cheat). È importante includere lo stato di QUESTO preciso
    momento e non rileggere il "latest" dallo store: altrimenti, se l'IA ha già mosso, il client
    riceverebbe lo stato post-IA prima del relativo `ai_card_reveal`, perdendo lo stato intermedio.
    """
    await game_store.publish(
        game_id,
        json.dumps({"type": "refresh", "server_version": server_version, "state": game_state_to_dict(state)}),
    )


async def notify_trick_result(session: GameSession, trick_cards: list, winner_index: int, points: int):
    """
    Notifica i client del risultato della mano con le carte visibili.

    Questo messaggio speciale permette al frontend di mostrare entrambe le carte
    e indicare chiaramente chi ha vinto la mano. È pubblicato sul pub/sub dello store, così
    raggiunge i client su qualsiasi replica.
    """
    game_id = session.game_id
    winner_name = _display_name_for_player(session, winner_index)

    # Costruisci DTO per il risultato della mano
    trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in trick_cards]
    trick_result_dto = TrickResultDTO(
        trick_cards=trick_cards_dto,
        winner_index=winner_index,
        winner_name=winner_name,
        points=points,
        server_version=session.version,
    )
    _safe_log_event(
        game_id,
        "trick_result",
        trick_result_dto.model_dump(),
        server_version=session.version,
        state=session.state,
    )
    await game_store.publish(game_id, trick_result_dto.model_dump_json())


async def cleanup_inactive_games():
    """Rimuove le partite inattive da più di 1 ora"""
    while True:
        await asyncio.sleep(3600)  # Check every hour
        now = _utcnow()

        # Trova le partite da rimuovere
        games_to_remove = []
        for game_id, timestamp in game_timestamps.items():
            if (now - timestamp).total_seconds() > 3600:  # 1 hour
                games_to_remove.append(game_id)

        # Rimuove le partite inattive
        for game_id in games_to_remove:
            session = await game_store.get(game_id)

            # Staleness AUTORITATIVA: decidere dal solo timestamp locale è sbagliato su store
            # condiviso (questa replica potrebbe aver creato la partita ma non servire più le
            # azioni, mentre un'altra replica la sta ancora giocando). Usiamo `session.updated_at`.
            truly_stale = session is None
            if session is not None:
                try:
                    updated = datetime.fromisoformat(session.updated_at)
                    if updated.tzinfo is None:
                        # Sessioni legacy con timestamp naive (pre-UTC): le trattiamo come UTC.
                        # Nel transitorio la staleness può sbagliare di qualche ora, in favore
                        # del non-cancellare (il TTL dello store resta la rete di sicurezza).
                        updated = updated.replace(tzinfo=UTC)
                    truly_stale = (now - updated).total_seconds() > 3600
                except Exception:
                    truly_stale = False

            if not truly_stale:
                # Attiva altrove: NON toccare lo store condiviso; smetti solo di tracciarla
                # localmente (i buffer per-replica). Le connessioni WS locali restano valide.
                game_timestamps.pop(game_id, None)
                game_data.pop(game_id, None)
                continue

            # Event log: partita rimossa per inattività (non completa).
            # Logghiamo prima di eliminare lo stato, così possiamo salvare anche `server_version`.
            log = _get_event_log()
            if log is not None and _get_event_log_mode() != "off":
                try:
                    state = session.state if session is not None else None
                    if state is not None:
                        seed = getattr(state, "seed", None)
                        log.ensure_game(
                            game_id,
                            num_players=state.num_players,
                            seed=seed if isinstance(seed, int) else None,
                            code_version=get_code_version(),
                            rules_version=get_rules_version(),
                        )
                    aborted_marked = log.try_mark_game_aborted(game_id, aborted_reason="inactive_timeout")
                except Exception:
                    aborted_marked = False
                if aborted_marked:
                    _safe_log_event(
                        game_id,
                        "game_aborted",
                        {"reason": "inactive_timeout"},
                        server_version=session.version if session is not None else 0,
                        state=session.state if session is not None else None,
                    )

            await game_store.delete(game_id)
            if game_id in game_timestamps:
                del game_timestamps[game_id]

            # Salva i dati della partita prima di rimuoverla
            if game_id in game_data:
                # In un'app reale, salva su database
                # Per ora, logga soltanto che li salveremmo
                print(f"Salverei i dati della partita {game_id} ({len(game_data[game_id])} azioni)")
                del game_data[game_id]
