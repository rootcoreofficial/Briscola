# Approfondimento — I due allievi (esperimento a due bracci)

**Capitolo del diario:** [Capitolo 11](https://ai.briscola.dev/diario) · **Periodo:** 3–4 luglio 2026

## Il disegno sperimentale

Stesso allievo di partenza (best_a2c_v8), due regimi di allenamento, poi esami incrociati.

| | Braccio A — volume+varietà | Braccio B — maestro d'élite |
|---|---|---|
| Partite | 20.000.000 | 5.000.000 |
| Avversari | VL(v8) 65%, v8 specchio 15%, h2 10%, h1 6%, random 4% | PIMC belief 32 det × finestra 10 (kernel JIT, mode 3) |
| Durata | ~10 h | ~26 h |
| Ricetta | proposta dal maintainer | proposta dell'agente |

## I risultati (big holdout 100k appaiate, CI su coppie)

| Esame | Braccio A | Braccio B |
|---|---|---|
| vs best_a2c_v8 | **+0.97** (CI +0.82..+1.12) | +0.40 (CI +0.25..+0.55) |
| vs heuristic_v1 | **+18.78** (record assoluto) | 16.71 (regressione: v8 faceva 17.61) |
| Testa a testa (A vs B) | **+0.63** (CI +0.48..+0.78) | — |

## Le lezioni

1. **Volume+varietà > maestro d'élite** in questo regime: A vince su ogni asse, scontro
   diretto incluso.
2. **Il maestro insegna più in fretta per partita** (+0.08/M contro +0.05/M di A) ma il suo
   costo orario lo condanna (26h per un quarto del progresso di A in 10h).
3. **La dieta monotona erode lo stile**: B, senza il "bar" tra gli avversari, peggiora contro
   l'euristica. La quota bar del mix di A (20%) recupera e supera il record (18.78 > 18.73
   di v7): prima promozione che migliora ENTRAMBI i metri della progressione.

Il braccio A è stato promosso **best_a2c_v9** (release v0.26.0).
