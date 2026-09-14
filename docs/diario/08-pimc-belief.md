# Approfondimento — Il sussurro trova casa: PIMC belief in produzione

**Capitolo del diario:** [Capitolo 9](https://ai.briscola.dev/diario) · **Periodo:** 3 luglio 2026

## La belief network

MLP 369→128→40 con sigmoid: dalle feature v4 (incluso il comportamento avversario nelle
prese passate) stima P(carta in mano avversaria) per ognuna delle 40 carte. Offline:
top-k recall 0.593 contro 0.399 dell'uniforme. Anti-cheat invariato: è inferenza da
informazione pubblica, non sbirciatina.

## Dove NON funzionava e dove funziona

| Uso della belief | Esito |
|---|---|
| Pesare determinizzazioni PIMC, finestra deploy 16×8 | +0.15 n.s. (finestra troppo piccola) |
| Input aggiuntivo della policy (409 feature) | −0.56: dannosa (ridondante) |
| **Pesare 64 determinizzazioni, finestra 10, search a rollout** | **+3.83 → confermato +3.66** |

## I numeri di release (4.000 partite seat-fair, stessi semi, vs v8+solver)

| Agente | Punti/partita | CI95 | Costo |
|---|---|---|---|
| `bc_model_pimc_belief_64x10` | **+3.66** | +3.32..+4.00 | 0.267 s/partita (~75 ms/mossa pensata) |
| `bc_model_value_lookahead_8x8` (precedente) | +2.12 | +1.83..+2.41 | 0.021 s/partita |

Per ogni mossa nella finestra, l'agente simula 64 mani avversarie pesate sul comportamento
osservato e gioca ogni continuazione fino in fondo: ~670 rollout per partita reale.
Release: v0.23.0 (asset belief con provisioning `BRISCOLA_BELIEF_MODEL_URL/SHA256`).

## Il porting su Numba (v0.25.0)

Il ciclo caldo (determinizzazione anti-cheat, campionamento pesato senza rimpiazzo,
rollout a terminale con policy+guard+solver) è stato portato in kernel JIT
(`ai/numba/pimc.py`): forza equivalente al Python (+3.38 vs +3.83 sullo stesso protocollo,
CI sovrapposte), costo per mossa pensata **37 ms contro 73** (~2×, non i 20-50× sperati:
il costo Python era già dominato dalle matmul BLAS di numpy). L'agente di produzione usa
il kernel, con fallback trasparente al Python sugli stati non determinizzabili; dal
v0.27.1 il kernel è compilato allo startup (warm-up), non alla prima mossa del giocatore.

> **Nota retrospettiva (v0.31.x).** Questo paragrafo descrive il runtime delle release
> v0.25-v0.30, non quello attuale. Con v0.31.0 la search di produzione è tornata al path
> Python con la configurazione 16×8; con v0.31.1 anche il solver web è diventato Python
> puro tramite `solve_endgame_fast` e principal variation. Il processo web oggi non
> importa Numba; i kernel JIT restano disponibili per training, self-play, valutazioni e
> benchmark offline. Misure e motivazione sono in
> [Perché il sito era lento al risveglio](15-zero-numba.md).

## La composizione col modello più forte (misura del 2026-07-05)

Il margine della search NON si riduce al crescere della policy — si compone:

| Base | PIMC belief 64×10 vs base+solver (400 partite, seed 42) |
|---|---|
| v8 | +3.66 (conferma 4k) |
| **v10** | **+4.22** (CI +3.02..+5.43) |

Spiegazione: la search usa la policy stessa come motore dei rollout — una policy più
forte produce simulazioni più realistiche, quindi valutazioni migliori. I due
miglioramenti si sommano invece di cannibalizzarsi.
