# Ablation suit augmentation paired v0 (2026-07-11)

> Nota interna di decisione. Evidenza sintetica versionata:
> [suit_augmentation_paired_v0.v1.json](../reports/evidence/suit_augmentation_paired_v0.v1.json).
> Gli artefatti completi restano locali e gitignored in
> `benchmarks/experiments/suit_augmentation_paired_v0/`.

## Domanda e verdetto

La policy v13 cambia carta nel 18,19% dei confronti in cui i quattro semi vengono soltanto
rinominati. Abbiamo quindi provato ad affiancare a ogni traiettoria A2C una copia equivalente
con i semi rinominati, per verificare se l'esposizione esplicita alle due versioni riducesse
questa dipendenza accidentale.

**Verdetto: STOP.** Su tre seed brevi il paired v0 porta il flip medio da **18,32% a
18,84%** (`+0,53` punti percentuali), aumenta anche la divergenza JS e perde in media
**0,15 punti/partita** contro il controllo corrispondente. Non si promuove alcun modello e
non si giustifica un run lungo. Il codice resta disponibile, spento per default, come
ablation riproducibile e come base verificata per la prossima consistency loss.

## Implementazione e invarianti

Il nuovo modulo `ai/training/suit_augmentation.py` applica una permutazione unica all'intera
traiettoria numerica:

- rinomina tutti i blocchi carta delle feature v1-v4;
- rinomina one-hot e contatori indicizzati per seme;
- riallinea action mask e action id sulle stesse 40 carte;
- lascia invariati reward e return, perché il valore strategico della partita non cambia;
- usa un RNG dedicato, così attivare il flag non altera mazzi, azioni o avversari del ramo
  originale;
- media il gradiente su originale+copia (`2N` step), senza raddoppiare la scala del loss.

La parità numerica non è verificata contro una seconda tabella di indici: per tutte le 24
permutazioni, gli encoder v1-v4 e più stati reali, i test confrontano il fast transform con
`PlayerObservation -> permute semanticamente -> encode`. Un test CLI dimostra inoltre che
il default storico e `--suit-augmentation off` producono array e metadata identici; un test
di gradiente mostra che originale+copia identità, mediati su `2N`, equivalgono all'originale
su `N`.

Quando è presente il BC anchor, il target della copia viene ottenuto trasformando la
distribuzione del teacher sull'osservazione originale. Interrogare direttamente il teacher
asimmetrico sulla copia avrebbe reintrodotto proprio la preferenza che l'ablation cerca di
rimuovere. L'attuale input `v4+belief` da 409 feature è rifiutato esplicitamente: la
trasformazione delle 40 probabilità belief embedded non ha ancora un contratto verificato.

## Protocollo

Entrambi i rami partono da `best_a2c_v13.npz` e usano la sua ricetta: encoder v4, hidden
256, BC anchor v11 con beta `0.01`, reward shaping overkill `gap` beta `0.3`, seat-fair e lo
stesso mix di sette avversari. L'unica differenza causale è il flag paired. Per ogni seed:

- controllo: default storico (`--suit-augmentation off` implicito);
- trattamento: `--suit-augmentation paired`;
- budget: 10.000 partite, update ogni 20;
- seed: `20260711`, `20260712`, `20260713`;
- screening simmetria: 4.096 osservazioni x 24 rinomine per modello;
- forza: 10.000 partite seat-fair sulla suite `medium`, paired contro controllo e ciascun
  candidato contro v13.

Ricetta di training, con `SEED`, `MODE` e `OUT` sostituiti per i sei run:

```bash
uv run python scripts/train_a2c.py \
  --init data/models/best_a2c_v13.npz --out "$OUT" \
  --encoder-version v4 --rollout-engine fast --fast-rollout numba \
  --opponent-mix bc_model:0.15,bc_model_pimc_belief:0.40,bc_model_value_lookahead_8x8:0.20,heuristic_trump_saver:0.12,heuristic_v1:0.04,heuristic_v2:0.06,random:0.03 \
  --opponent-model data/models/best_a2c_v11.npz \
  --opponent-belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --opponent-value-model data/models/value_v1_v4_fullgame_h128_seed20260718.npz \
  --opponent-value-max-unknown-cards 8 \
  --bc-anchor data/models/best_a2c_v11.npz --bc-anchor-beta 0.01 \
  --overkill-penalty-mode gap --overkill-penalty-beta 0.3 \
  --overkill-low-lead-points-max 2 --lr 0.0003 --entropy-beta 0.0005 \
  --value-coef 0.5 --gamma 1.0 --update-every 20 --seat-fair \
  --metrics-mode summary --num-games 10000 --seed "$SEED" $MODE
```

Il benchmark preliminare da 2.000 partite ha impiegato `9,24s` nel controllo e `9,90s` nel
paired sullo stesso host. Il sovraccosto wall-clock osservato è quindi circa il 7%, perché
il rollout PIMC domina il costo e la copia aggiunge soprattutto algebra NumPy nel backward.

## Risultati di simmetria

Ogni cella usa la stessa sonda canonica della v13. `flip` è la quota di rinomine non
identità che cambia l'argmax; `JS` misura quanto cambia l'intera distribuzione di
probabilità; `stati con flip` conta osservazioni con almeno una scelta diversa.

| ramo | seed | flip | JS media (bit) | stati con flip |
|---|---:|---:|---:|---:|
| controllo | 20260711 | 18,20% | 0,14078 | 51,54% |
| controllo | 20260712 | 18,63% | 0,14281 | 51,22% |
| controllo | 20260713 | 18,12% | 0,14013 | 50,85% |
| **controllo medio** | - | **18,32%** | **0,14124** | **51,20%** |
| paired | 20260711 | 18,96% | 0,14691 | 52,61% |
| paired | 20260712 | 18,97% | 0,14675 | 52,83% |
| paired | 20260713 | 18,59% | 0,14359 | 51,39% |
| **paired medio** | - | **18,84%** | **0,14575** | **52,28%** |
| **paired - controllo** | - | **+0,53 pp** | **+0,00451** | **+1,07 pp** |

La direzione è coerente sui tre seed per flip e JS: il paired non mostra nemmeno un segnale
preliminare favorevole. La v13 non allenata misura 18,19%, quindi i controlli brevi restano
vicini al punto di partenza e confermano che il protocollo non degrada da solo la metrica.

## Risultati di forza

I numeri sono differenze medie di punti per partita del primo agente. Ogni riga contiene
10.000 partite seat-fair; la media sui seed è descrittiva, non una nuova CI aggregata.

| confronto | seed 1 | seed 2 | seed 3 | media |
|---|---:|---:|---:|---:|
| paired - controllo corrispondente | -0,08 | -0,35 | -0,02 | **-0,15** |
| controllo - v13 | -0,11 | -0,03 | +0,10 | **-0,01** |
| paired - v13 | -0,30 | -0,47 | -0,22 | **-0,33** |

Nei direct match paired-controllo le CI per singolo seed sono rispettivamente
`[-0,35; +0,20]`, `[-0,65; -0,05]` e `[-0,30; +0,27]`. Due run sono inconcludenti da
soli; il secondo è negativo. Nel complesso non emerge alcun vantaggio che compensi il
peggioramento netto della metrica primaria.

## Perché il risultato è plausibile

Mostrare una copia rinominata è una tecnica naturale nel supervised learning, dove
l'etichetta corretta è indipendente dalla policy. In policy gradient, invece, l'azione
della copia è stata campionata dalla policy sull'osservazione originale, non dalla policy
attuale sull'osservazione rinominata. Finché la rete è asimmetrica, le due distribuzioni
non coincidono: trattare entrambe come campioni on-policy introduce un'approssimazione che
può aggiungere rumore o bias. Il test non dimostra che questa sia l'unica causa, ma spiega
perché raddoppiare esempi equivalenti non garantisce una policy più coerente.

Un budget di 10k è deliberatamente uno screening: non prova che un run molto più lungo non
possa invertire il segno. Il gate serve però proprio a non spendere milioni di partite su
un intervento che peggiora entrambe le metriche iniziali. Allungare lo stesso paired v0 non
è quindi il prossimo passo corretto.

## Prossimo esperimento

Mantenere intatto il loss A2C originale e aggiungere una **consistency loss esplicita**:
la distribuzione della policy sull'osservazione originale diventa un target con
stop-gradient; la policy sulla copia rinominata deve riprodurla dopo il riallineamento delle
40 azioni. Questo chiede direttamente la proprietà desiderata senza fingere che l'azione
duplicata sia stata campionata dalla seconda policy.

Prima del training servono due controlli: su un batch congelato un singolo update deve
ridurre la divergenza senza cambiare la action mask, e coefficiente zero deve essere
identico al trainer storico. Poi basta una piccola griglia di coefficienti su tre seed,
con lo stesso screening usato qui. Nessun gate PIMC è giustificato finché flip e forza
policy-only non migliorano entrambi.

**Follow-up completato:** la consistency loss separata supera lo screening con beta `0.1`
(flip medio `15,64%`, forza diretta neutra contro v13). Non è ancora una promozione; il
prossimo gate è il run intermedio da 50k descritto in
[`suit-consistency-v0-2026-07-11.md`](suit-consistency-v0-2026-07-11.md).
