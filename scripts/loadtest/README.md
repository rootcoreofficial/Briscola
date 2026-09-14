# Load test dell'infrastruttura (Locust)

Bot che simulano giocatori veri: creano partite e le giocano fino in fondo via REST
(lo stesso percorso della UI in modalità polling). Servono a rispondere a UNA domanda:
**quanti giocatori simultanei regge l'infrastruttura attuale, e cosa cede per primo?**

Candidati al collo di bottiglia, in ordine di probabilità:

1. **CPU delle repliche** quando l'IA muove (soprattutto con PIMC belief: ~64×10 rollout a mossa);
2. **connessioni Redis** (piano free: tetto basso; ogni partita usa store + pub/sub);
3. **rate limit** su `POST /api/games` (default 30/min per IP — il bot lo mostra come failure `rate_limited`, attesa e non bug);
4. Postgres/Neon per l'event log (già best-effort: degrada, non rompe).

## Passo 1 — In locale (sempre prima del cloud)

```bash
# Terminale 1: il server
uv run briscola-server --port 8000

# Terminale 2: 10 giocatori, rampa 2/s, 2 minuti, avversario economico
uv run locust -f scripts/loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 10 -r 2 -t 2m --only-summary
```

Con la UI web di Locust (grafici live): togli `--headless` e apri http://localhost:8089.

## Passo 2 — Trova il limite locale con l'avversario pesante

```bash
BOT_AGENT=bc_model_pimc_belief_64x10 BOT_THINK_MIN=0.2 BOT_THINK_MAX=0.8 \
uv run locust -f scripts/loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 30 -r 2 -t 5m --only-summary
```

Guarda: p95 di `[play]` e `[observe]` (sopra ~2s l'esperienza degrada), failure diverse
da `rate_limited`, e la CPU del processo server.

## Passo 3 — Produzione (con prudenza)

È il TUO sito e puoi testarlo, ma: repliche condivise coi giocatori veri, piani free
con quote. Regole di buon senso:

- orari di scarico, partire piccoli: `-u 5 -r 1 -t 3m`, raddoppiare solo se p95 e failure restano sani;
- tenere aperta la dashboard FastAPI Cloud (metriche/repliche) durante il test;
- il rate limit di creazione (30/min/IP) limita naturalmente un singolo bot: per superarlo
  serve alzare `BRISCOLA_CREATE_GAME_RATE_LIMIT` in env — NON farlo in prod, è la protezione;
- fermarsi al primo segnale di degrado per gli utenti reali.

```bash
BOT_AGENT=bc_model uv run locust -f scripts/loadtest/locustfile.py \
    --host https://ai.briscola.dev --headless -u 5 -r 1 -t 3m --only-summary
```

## Come leggere i risultati

- **RPS e p50/p95/p99 per endpoint**: `[play]` include l'inferenza/search dell'IA (il costo vero);
- **failure `rate_limited`**: hai saturato il tetto di creazione partite, non l'app;
- **failure 5xx o timeout**: quello è il limite reale — annota utenti simultanei e p95 al momento del cedimento;
- il numero di utenti Locust ≈ giocatori umani simultanei (il bot "pensa" 0.5–2s tra le mosse, come un umano veloce).

Risultati e osservazioni vanno annotati in `PLAN.md` (sezione hardening) con data,
commit del server e configurazione usata: i numeri senza contesto invecchiano male.
