# Approfondimento — Dominio canonico e patto anti-cheat

**Capitolo del diario:** [Capitolo 2](https://ai.briscola.dev/diario) · **Periodo:** 17–22 gennaio 2026

## Il refactor del 17 gennaio

Nasce il motore puro: `GameState` immutabile (`@dataclass(frozen=True)`) + transizione
`step(state, action) -> StepResult` deterministica dato il seme (commit `c846272`).
Metodo di migrazione: doppia implementazione in parallelo + test di parità + rimozione del
legacy (`940f274`) — un pattern che il progetto riuserà per il fast path e i kernel Numba.

## L'invariante anti-cheat (commit `2568721`)

Gli agenti non ricevono mai `GameState` completo, solo `PlayerObservation`:

| L'agente vede | L'agente NON vede |
|---|---|
| la propria mano | la mano avversaria |
| il tavolo e la briscola scoperta | l'ordine del mazzo |
| le carte già uscite (memoria pubblica) | la prossima pescata |
| la storia delle prese (chi ha giocato cosa) | — |

L'invariante è protetto da test automatici e vale per baseline, modelli, reward shaping e
per la ricostruzione di stato del solver endgame (dettaglio fine in `f57c891`: i punti dello
stato ricostruito vanno azzerati perché la base è una costante che non cambia argmax/argmin).
