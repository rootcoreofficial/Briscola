# Approfondimento — I due giorni di luglio: belief, ExIt e le leve misurate

**Capitolo del diario:** [Capitolo 8](https://ai.briscola.dev/diario) · **Periodo:** 1–3 luglio 2026

Piano completo con gate e kill criteria: [`docs/plans/belief-expert-iteration.md`](../plans/belief-expert-iteration.md).

## La diagnosi (Fase 0, sonde economiche)

| Sonda | Risultato |
|---|---|
| Exploitability di v7 (best response dedicata) | +0.70 — quasi inesploittabile nella sua classe |
| Oracle PIMC 64×10 vs v7 | **+3.76** — headroom dimostrato |
| From scratch (stesso budget) | −5.42 — il warm-start è un patrimonio |

## Le leve dell'allievo, tutte misurate (CI su coppie, big holdout 100k)

| Leva | Esito vs controllo |
|---|---|
| Feature v4 (memoria delle prese) | **+0.27** (CI +0.12..+0.42) — prima evidenza positiva del programma encoder |
| Capacità (Net2Net 128→256) | +0.18 (confuso con un giro di sparring in più): marginale |
| Belief come input della policy | **−0.56** (CI −1.05..−0.06): dannosa — è funzione delle stesse feature v4, ridondante |

## Il giudice che non vede il futuro (expert full-game)

Value model v4 su tutte le fasi (MAE 14.1 punti — rumore irriducibile del mazzo) usato per
guidare la lookahead: curva dose-risposta pulita.

| Finestra (carte ignote) | Esito vs v8+solver |
|---|---|
| 8 | +1.80 |
| 12 | +0.57 n.s. |
| 16 | −0.88 n.s. |
| 22 | −2.56 |
| 34 (tutta partita) | **−5.16** |

Morale dei due giorni: **"il vincolo è il maestro"** — e il maestro migliore non è una stima,
è una simulazione (capitolo 9).
