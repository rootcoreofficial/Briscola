"""
Bot Locust: giocatori virtuali che giocano partite COMPLETE di briscola via REST.

Perché partite vere e non semplici GET: il costo del backend sta nei punti caldi
reali — lock per partita, store (Redis in cloud), inferenza del modello e search
PIMC quando l'IA muove. Un bot che gioca esercita esattamente quel percorso
(lo stesso della UI in modalità polling: POST partita → GET stato → POST carta).

Uso (dalla root del repo):

    # 1) Contro un server LOCALE (prima di toccare il cloud):
    uv run briscola-server --port 8000            # in un altro terminale
    uv run locust -f scripts/loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 2 -t 2m

    # 2) Contro la produzione (con prudenza: vedi README.md accanto a questo file):
    BOT_AGENT=bc_model uv run locust -f scripts/loadtest/locustfile.py \
        --host https://ai.briscola.dev --headless -u 5 -r 1 -t 3m

Configurazione via variabili d'ambiente:
- `BOT_AGENT`: avversario IA (default `bc_model` = solo inferenza, economico;
  `bc_model_pimc_belief_64x10` = search pesante, worst case realistico;
  `heuristic_v2` = quasi gratis, stressa solo motore/store).
- `BOT_MODEL_ID`: ai_model_id esplicito (default: il consigliato dal catalogo).
- `BOT_THINK_MIN` / `BOT_THINK_MAX`: secondi di "pensiero" tra le mosse
  (default 0.5–2.0; abbassare per stressare di più a parità di utenti).

Note sui limiti che INCONTRERAI (sono parte del test):
- `POST /api/games` è rate-limitato per IP (default 30/min): il bot gestisce il
  429 aspettando e riprovando — dal grafico Locust il 429 appare come failure
  "rate_limited" così resta visibile ma distinguibile dai veri errori.
- il piano free di Redis ha un tetto di connessioni: oltre un certo numero di
  repliche/partite simultanee è LUI il collo di bottiglia, non l'app.
"""

from __future__ import annotations

import os
import random
import time

from locust import HttpUser, between, task

BOT_AGENT = os.getenv("BOT_AGENT", "bc_model")
BOT_MODEL_ID = os.getenv("BOT_MODEL_ID", "")
THINK_MIN = float(os.getenv("BOT_THINK_MIN", "0.5"))
THINK_MAX = float(os.getenv("BOT_THINK_MAX", "2.0"))


class BriscolaPlayer(HttpUser):
    """
    Un giocatore virtuale: crea una partita e la gioca fino in fondo, poi ricomincia.

    Ogni "tick" del task esegue UN passo del ciclo di gioco (crea / osserva / gioca),
    con una pausa di pensiero tra i tick: cosi' il profilo di carico somiglia a un
    umano veloce, e il numero di utenti Locust ~= giocatori simultanei reali.
    """

    wait_time = between(THINK_MIN, THINK_MAX)

    def on_start(self) -> None:
        self.game_id: str | None = None
        self.model_id: str | None = None
        # Dopo un 429 aspettiamo la prossima finestra del rate limit invece di
        # martellare: senza questo, un bot 'impaziente' produce centinaia di 429
        # al minuto e sporca le statistiche (visto nel primo smoke test locale).
        self.create_backoff_until: float = 0.0
        # Risolve una volta il modello consigliato dal catalogo (come fa la UI).
        if BOT_MODEL_ID:
            self.model_id = BOT_MODEL_ID
        else:
            with self.client.get("/api/ai/models", name="/api/ai/models", catch_response=True) as res:
                if res.status_code == 200:
                    data = res.json()
                    self.model_id = data.get("recommended_model") or next(
                        (m["id"] for m in data.get("models", []) if m.get("is_compatible")), None
                    )

    def _create_game(self) -> None:
        if time.monotonic() < self.create_backoff_until:
            return  # in attesa della prossima finestra del rate limit
        payload = {
            "num_players": 2,
            "player_names": ["locust", "AI"],
            "ai_agent": BOT_AGENT,
            # In prod l'event log e' in modalita' dataset e RICHIEDE il consenso
            # (senza -> 400). Il bot acconsente ma si firma con un client_id
            # riconoscibile: le sue partite si escludono dal dataset con
            #   WHERE client_id != 'loadtest-bot'  (o dal nome giocatore 'locust').
            "consent_to_data_collection": True,
            "client_id": os.getenv("BOT_CLIENT_ID", "loadtest-bot"),
        }
        if BOT_AGENT.startswith("bc_model") and self.model_id:
            payload["ai_model_id"] = self.model_id
        with self.client.post("/api/games", json=payload, name="/api/games [create]", catch_response=True) as res:
            if res.status_code == 429:
                # Rate limit per IP: e' un limite ATTESO dell'infrastruttura, non un bug.
                # Backoff con jitter: i bot si ripresentano sparpagliati nella finestra.
                self.create_backoff_until = time.monotonic() + random.uniform(10.0, 30.0)
                res.failure("rate_limited")
                return
            if res.status_code != 200:
                res.failure(f"create fallita: {res.status_code}")
                return
            self.game_id = res.json().get("game_id")
            res.success()

    @task
    def play_step(self) -> None:
        """Un passo del ciclo: crea partita, oppure osserva e (se tocca a noi) gioca."""
        if self.game_id is None:
            self._create_game()
            return

        with self.client.get(
            f"/api/games/{self.game_id}",
            params={"player_index": 0},
            name="/api/games/[id] [observe]",
            catch_response=True,
        ) as res:
            if res.status_code == 404:
                # Partita scaduta/persa (TTL dello store): ricomincia senza contare errore.
                res.success()
                self.game_id = None
                return
            if res.status_code != 200:
                res.failure(f"observe fallita: {res.status_code}")
                return
            obs = res.json()
            res.success()

        if obs.get("game_over"):
            self.client.get(
                f"/api/games/{self.game_id}/result",
                name="/api/games/[id]/result",
            )
            self.game_id = None
            return

        if not obs.get("my_turn"):
            return  # l'IA sta muovendo: il prossimo tick ripassera' a osservare

        hand_size = len(obs.get("my_hand") or [])
        if hand_size == 0:
            return
        payload = {
            "game_id": self.game_id,
            "player_index": 0,
            # Nella briscola a 2 ogni carta in mano e' giocabile: indice casuale.
            "card_index": random.randrange(hand_size),
            "client_observed_server_version": obs.get("server_version"),
        }
        with self.client.post(
            f"/api/games/{self.game_id}/actions",
            json=payload,
            name="/api/games/[id]/actions [play]",
            catch_response=True,
        ) as res:
            if res.status_code == 400 and "turno" in res.text:
                # Race benigna: lo snapshot era vecchio e l'IA aveva gia' mosso.
                res.success()
                return
            if res.status_code == 404:
                res.success()
                self.game_id = None
                return
            if res.status_code != 200:
                res.failure(f"play fallita: {res.status_code}")
                return
            res.success()
