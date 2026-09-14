# Audit automatico del regret di v14 (2026-07-14)

> Verdetto: **nessun cluster di errore policy-only; non autorizzare un nuovo
> training**. I 14 errori affidabili sono tutti nelle finestre gia' gestite da PIMC o
> solver. Evidenza:
> [policy_regret_v14.v1.json](../reports/evidence/policy_regret_v14.v1.json).

## Domanda

Le piste su simmetria, capacita', belief, dose PIMC e schedule A2C hanno escluso diverse
correzioni generiche. Prima di scegliere un'altra architettura o un nuovo training,
vogliamo quindi misurare in quali decisioni v14 lasci davvero punti a un'alternativa.

Perdere una partita non rende sbagliata una singola carta: puo' essere soltanto sfortuna.
La sonda congela invece l'informazione disponibile al giocatore, prova tutte le carte
legali negli stessi mondi plausibili e misura il vantaggio della migliore alternativa.

## Contratto anti-cheat

`estimate_policy_regret` accetta una `PlayerObservation`, mai il `GameState` reale. Usa:

- mano del decisore, tavolo, punteggi, dimensione del mazzo e storia pubblica;
- belief v0 per pesare le carte ignote nelle determinizzazioni;
- seed della partita soltanto come provenance del collector, mai come input dello
  stimatore;
- solver esatto a mazzo vuoto, dove la mano avversaria e' deducibile dall'osservazione.

Il report conserva il contesto pubblico separato dalla provenance. Non serializza mano
avversaria o ordine reale del mazzo.

## Suite bilanciata

La raccolta usa coppie seat-fair con stesso mazzo e posti scambiati. La policy v14 gioca
contro:

- mirror v14;
- `heuristic_trump_saver`;
- `heuristic_v1`.

Le decisioni forzate sono escluse. Le 192 osservazioni riempiono esattamente 24 celle:

`3 avversari x 4 fasi x 2 posizioni x 8 osservazioni`.

Le fasi sono `early`, `mid`, `pimc_window` (al massimo 8 carte vive ignote, mazzo non
vuoto) ed `endgame`; le posizioni sono apertura e risposta. Questo impedisce che un
avversario o una fase facile dominino il risultato aggregato.

## Stima controfattuale

Fuori dall'endgame, per ogni osservazione:

1. si campionano 64 stati nascosti compatibili con la sola osservazione, pesati da
   belief v0 con mix uniforme `0,10`;
2. in ogni stato si provano tutte le carte della mano;
3. tutte le alternative condividono stato determinizzato e seed di rollout;
4. v14 completa la partita per entrambi i lati, con solver esatto nel finale;
5. le prime 32 righe scelgono una sola alternativa candidata;
6. le ultime 32, mai usate per sceglierla, stimano il delta paired
   `alternativa - scelta v14`.

Lo split selection/evaluation evita di scegliere la carta fortunata e certificarla sugli
stessi campioni. A mazzo vuoto non servono campioni: tutte le azioni sono valutate con
minimax esatto.

## Definizione di errore affidabile

Una decisione campionaria e' affidabile soltanto se:

- l'alternativa scelta sul primo split e' diversa dalla carta v14;
- il regret medio sul secondo split e' almeno `1,0` punto;
- il limite inferiore dell'intervallo normale con `z=2,576` (circa 99%) e' maggiore di
  zero.

Il 99%, piu' severo del 95% usato nelle valutazioni aggregate, riduce i falsi allarmi
perche' qui controlliamo molte decisioni separatamente. Nell'endgame esatto basta un
regret di almeno `1,0` punto.

## Tassonomia automatica

Le etichette descrivono soltanto il cambio di carta, senza pretendere di leggere la
causa interna della rete:

- mancata risposta vincente o mancata cattura di una presa ricca;
- briscola anticipata, passaggio da briscola a non-briscola o overkill;
- esposizione/scarto di un carico;
- rinuncia intenzionale alla presa corrente, anche quando vale pochi punti;
- `other` quando nessuna regola pubblica descrive il caso.

Ogni decisione e' inoltre distinta per layer del prodotto:

- `policy_only`: fasi early/mid, dove la scelta v14 arriva direttamente al gioco;
- `pimc_search_window`: il default live esegue gia' la search;
- `exact_solver`: il default live sostituisce gia' la policy col solver.

Gli ultimi due gruppi restano diagnostici, ma non possono da soli giustificare un nuovo
training per migliorare il comportamento live.

## Gate di instradamento

La sonda non promuove modelli. Autorizza la progettazione di **un solo intervento
mirato** soltanto se una stessa etichetta diversa da `other`, nel gruppo `policy_only`:

1. compare in almeno 3 errori affidabili;
2. attraversa almeno 2 avversari;
3. attraversa almeno 2 coppie di partite.

Il report assegna automaticamente uno dei verdetti:

- `actionable_policy_error_cluster`: cluster completo, si puo' progettare un intervento;
- `unclassified_policy_error_cluster`: almeno tre casi `other`, migliorare prima la
  tassonomia;
- `sparse_policy_error_signal`: casi affidabili ma isolati, nessun training;
- `no_policy_error_signal`: nessun errore affidabile nelle fasi esposte.

Nessun tuning post-hoc di soglie, roster o seed e nessuna promozione v15 sono consentiti.

## Pilot pre-formale

Un pilot da 72 osservazioni e 16 determinizzazioni ha validato tempi, otto bucket di
fase/posizione e serializzazione. Non entra nell'evidenza finale: ha fatto emergere prima
del congelamento due correzioni conservative, cioe' il bilanciamento esplicito anche per
avversario e la separazione degli errori gia' coperti da PIMC/solver. Il run formale usa
un seed invariato (`20260720`) ma piu' campioni e intervalli al 99%.

## Risultati formali

Il run ha raccolto 192 decisioni in 34 partite (17 coppie), riempiendo esattamente
ognuna delle 24 celle. Ogni avversario contribuisce 64 osservazioni; ogni combinazione
fase/posizione ne contribuisce 24. Sono state escluse 34 mosse forzate.

La sonda seleziona un'alternativa diversa da v14 in 81 casi su 192 (`42,19%`). Questo
numero **non e' un tasso di errore**: sullo split indipendente molte alternative perdono
il vantaggio o hanno incertezza troppo larga.

| layer effettivo | decisioni | disaccordi candidati | errori affidabili | regret medio affidabile |
|---|---:|---:|---:|---:|
| policy-only (early + mid) | 96 | 36 | **0** | n/a |
| finestra PIMC | 48 | 20 | 9 | 19,24 |
| solver esatto | 48 | 25 | 5 | 3,20 |

Nelle 96 decisioni esposte direttamente alla policy, il regret medio cross-fitted e'
`+0,195` e la mediana `0`, ma nessun intervallo al 99% ha limite inferiore positivo. I
casi con medie apparenti piu' alte hanno intervalli larghi: non formano una classe
ripetibile e il report non costruisce alcun cluster.

Le 14 decisioni affidabili fuori dal gruppo policy-only confermano invece che i layer
runtime hanno un compito reale:

- 9 sono nella finestra in cui il prodotto consulta PIMC belief 16x8;
- 5 sono nel finale in cui il prodotto ignora la policy e usa il minimax esatto.

Le etichette piu' frequenti sono rinunciare alla presa corrente (7 casi) e passare da
briscola a non-briscola (6), ma si sovrappongono e appartengono tutte ai due layer
runtime. Non riaprono la dose PIMC: il precedente confronto 16x8/32x8 ha gia' misurato
forza e costo del runtime reale.

Il secondo run completo ha prodotto un JSON byte-identico al primo, SHA-256
`d14bfb9112c65a2352e63f696ec00b14803daf4d11e505f90acce145e431fa6f`.

## Decisione

Il routing preregistrato restituisce `no_policy_error_signal`. Questo non significa che
v14 sia perfetta: significa che questa suite non individua un difetto early/mid abbastanza
chiaro e ripetibile da sapere cosa insegnarle.

**STOP** a un nuovo training, reward, ottimizzatore o architettura scelti per inerzia.
V14 con PIMC belief 16x8 e solver resta la baseline. Una pista modello si riapre soltanto
con nuova evidenza ripetibile, per esempio dal replay live o da un teacher realmente piu'
informativo. Nel frattempo il prossimo lavoro utile e' metodologico: eliminare lo split
casuale per record da BC/value prima di riutilizzare quelle pipeline.

## Esecuzione formale

```bash
uv run python scripts/probe_policy_regret.py \
  --model data/models/best_a2c_v14.npz \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --num-observations 192 \
  --max-games 400 \
  --determinizations 64 \
  --seed 20260720 \
  --min-regret-points 1.0 \
  --confidence-z 2.576 \
  --out benchmarks/experiments/policy_regret_v14_v0_20260714/policy_regret_v14_o192_d64_seed20260720.json
```

Il report prodotto contiene tutte le decisioni, valori per carta, aggregati per fase,
posizione, avversario e layer runtime, top case, hash degli asset e verdetto automatico.
