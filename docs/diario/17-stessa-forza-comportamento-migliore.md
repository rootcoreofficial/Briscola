# Approfondimento — Stessa forza, comportamento migliore

**Capitolo del diario:** [Capitolo 17](https://ai.briscola.dev/diario) · **Periodo:** 9 luglio 2026

## Domanda

Dopo v12 restavano due famiglie di sospetti:

- comportamenti visibili nelle partite vinte dagli umani, soprattutto carichi guidati e tagliati;
- sprechi di briscola da secondi di mano, quando il modello vinceva una presa povera con una briscola più costosa del necessario.

La prima famiglia è stata chiusa con sonde descrittive, base-rate di campo e ablation
controfattuali. La seconda è diventata v13: reward shaping morbido anti-overkill.

## Piste chiuse prima di v13

### Finestra PIMC 8→10

Il candidato `bc_model_pimc_belief_16x10` è stato testato come variante eval-only del default
`16x8`.

| Confronto | Esito |
|---|---:|
| 16x10 vs 16x8, 10k seat-fair | -0.25 punti |
| CI95 coppie | -0.45..-0.05 |

Conclusione: allargare la finestra non migliora il default; in quel regime aggiunge rumore
più che valore.

### Carichi guidati contro umani

Evento contato sui dati live esportati: IA leader, carico non-briscola Asso/Tre, `deck<=8`,
taglio umano con briscola.

| Gruppo | Partite | Con almeno un evento |
|---|---:|---:|
| IA perse | 19 | 12 (63.2%) |
| IA non perse | 47 | 20 (42.6%) |

Fisher two-sided `p=0.176`; lift `1.48x`; nelle partite dove l'evento capita l'IA non perde
comunque il 62.5% delle volte. I punti concessi per taglio sono quasi identici nei due gruppi
(`15.57` vs `15.50`).

L'ablation eval-only del lead-load guard riduce un po' i tagli ma perde punti contro
mirror/trump_saver. Lettura: evento reale, non difetto causale isolabile.

### Asso di briscola e cavata

`scripts/trump_play_probe.py` ha controllato tre ipotesi:

- asso di briscola giocato troppo presto;
- cavata insufficiente;
- cavata eccessiva.

Risultato sintetico:

- l'asso di briscola guidato presto è raro, circa 4-9 casi su 1000 partite;
- l'asso viene valorizzato, con presa media ~16-18 punti;
- `pull_more` è neutro;
- `pull_less` crolla, per esempio -6.7 punti contro `heuristic_trump_saver`.

Conclusione: niente guard e niente shaping su asso/cavata.

## v13: ricetta

Artefatto promosso:

```text
data/models/best_a2c_v13.npz
```

Pesi identici al run:

```text
data/models/a2c_v13_overkill_gap_b0300_5M_seed20260709.npz
```

Ricetta:

```text
init: best_a2c_v11.npz
encoder: v4
hidden: 256
num_games: 5,000,000
seed: 20260709
opponent_mix:
  bc_model: 0.15
  bc_model_pimc_belief: 0.40
  bc_model_value_lookahead_8x8: 0.20
  heuristic_trump_saver: 0.12
  heuristic_v1: 0.04
  heuristic_v2: 0.06
  random: 0.03
overkill_penalty_mode: gap
overkill_penalty_beta: 0.3
overkill_low_lead_points_max: 2
bc_anchor: best_a2c_v11.npz
bc_anchor_beta: 0.01
inference_overkill_guard: false
```

Nota: il guard runtime anti-overkill non viene reintrodotto. v13 deve scegliere meglio da sola.

## Gate di forza

### Policy-only, 10k seat-fair

| Confronto | v13 | v11 | Lettura |
|---|---:|---:|---|
| v13 vs v11 | -0.03, CI -0.38..+0.32 | — | neutro |
| vs heuristic_trump_saver | +14.80 | +14.59 | pari |
| vs heuristic_v1 | +21.52 | +21.25 | pari |

### Default reale PIMC belief 16x8, 10k seat-fair

| Confronto | v13 | v11 | Lettura |
|---|---:|---:|---|
| v13 vs v11 | +0.14, CI -0.20..+0.47 | — | neutro |
| vs heuristic_trump_saver | +16.30 | +16.06 | pari |
| vs heuristic_v1 | +22.40 | +22.48 | pari |

Conclusione: nessun salto di forza dimostrato, ma nessuna regressione materiale nei gate.

## Gate comportamentale

Metrica target: overkill di briscola su piatto povero da secondi di mano.

| Avversario | v11 | v13 | Delta |
|---|---:|---:|---:|
| v11 / mirror | 27.8% | 6.1% | -21.7 pp |
| heuristic_v1 | 31.4% | 7.8% | -23.6 pp |

Overkill totale:

| Avversario | v11 | v13 |
|---|---:|---:|
| v11 / mirror | 23.6% | 15.7% |
| heuristic_v1 | 30.0% | 22.1% |

Il segnale forte è sul target: piatti poveri, non tutti gli overkill indistintamente.

## Decisione

Promuovere v13 come default più pulito:

- stesso agente runtime: `bc_model_pimc_belief_16x8`;
- nuovo modello: `best_a2c_v13.npz`;
- nessun inference guard;
- messaggio pubblico: **stessa forza, comportamento migliore**.

La promozione non va letta come una nuova soglia competitiva. È una correzione di stile
misurabile che non costa forza nei gate disponibili.
