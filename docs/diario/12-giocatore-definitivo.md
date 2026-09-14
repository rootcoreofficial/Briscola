# Approfondimento — La rosa completa di avversari (v10)

**Capitolo del diario:** [Capitolo 12](https://ai.briscola.dev/diario) · **Periodo:** 4–5 luglio 2026

## La ricetta (dosi del maintainer)

30.000.000 di partite, init = best_a2c_v9, ~27 ore. Mix di avversari:

| Avversario | Quota | Ruolo |
|---|---|---|
| PIMC belief 16 det × finestra 10 (kernel JIT, mode 3) | 25% | il maestro d'élite, alla dose per-partita più efficiente |
| value-lookahead(v9) | 35% | lo sparring quotidiano |
| v9 allo specchio | 15% | scoprire le proprie abitudini |
| heuristic_v2 / heuristic_v1 / random | 12% / 8% / 5% | il "bar": mantiene gli exploit anti-semplici |

Prerequisito tecnico: l'aggancio del kernel PIMC come opponent di training nel mix
(v0.25.0 + `bc_model_pimc_belief` nel mix, commit `1467a3b`).

## I risultati (big holdout 100k appaiate, CI su coppie)

| Esame | Esito |
|---|---|
| vs best_a2c_v9 | **+0.66** (CI +0.51..+0.81) |
| vs heuristic_v1 | **+20.52** — record assoluto (v9: 18.78, v7: 18.73) |

## Le note oneste

- Rendimenti decrescenti confermati: +0.97 (v9) → +0.66 (v10) nonostante più partite e il
  maestro migliore nel mix. L'asintoto stimato del gioco a due si avvicina.
- La quota bar alzata da 20% a 25% ha spinto il metro anti-semplici da 18.78 a 20.52:
  la voce del mix col miglior rapporto costo/beneficio del progetto.

- Il margine della search sopra la policy si COMPONE invece di ridursi: PIMC belief
  64×10 su base v10 rende **+4.22** (CI +3.02..+5.43, 400 partite) contro il +3.66 su v8 —
  la policy più forte migliora i rollout della search stessa. Il default del sito
  (v10 + PIMC belief) beneficia di entrambi i progressi.

Promosso **best_a2c_v10** (release v0.27.0).
