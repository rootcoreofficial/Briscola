# Distillazione del teacher simmetrico v0 (2026-07-11)

> Pipeline completata e promossa come `best_a2c_v14.npz` nella release `0.36.0`. Evidenza:
> [suit_distillation_v0.v1.json](../reports/evidence/suit_distillation_v0.v1.json).

## Domanda e verdetto

La media dei logits v13 sulle 24 rinomine gioca meglio della policy originale ma richiede
un batch 24x. Possiamo trasferire quel comportamento in una sola MLP v4, mantenendo un
forward normale a inference?

**Verdetto screening: sì.** Il candidato distillato su 10.000 partite:

- porta l'accordo col teacher sull'holdout da `86,83%` a **`92,88%`**;
- riduce il flip dei semi da `18,19%` di v13 a **`10,23%`**, superando il gate `<12%`;
- batte direttamente v13 di **`+0,51` punti/partita**, CI95 `+0,11..+0,92`;
- resta neutro contro il teacher 24x: `-0,23`, CI95 `-0,59..+0,13`;
- conserva il miglioramento comportamentale: overkill su piatto povero `5,5%`, fra v13
  (`8,0%`) e teacher (`3,9%`).

**Il corpus indipendente da 50.000 partite e il successivo PIMC 16x8 medium hanno superato
tutti i gate: GO tecnico alla promozione come nuova policy ufficiale, subordinato al bump
di release e all'audit catalogo/report.**

## Pipeline implementata

La distillazione non riusa il JSONL BC: 380.000 osservazioni v4 complete sarebbero molto
più grandi e lente da parsare. Il nuovo formato numerico compresso contiene:

- feature `(N, 369)` float32 e action mask `(N, 40)`;
- probabilità soft del teacher e relativo argmax;
- `game_id` e split per ogni esempio;
- metadati essenziali su modello, hash, seed, roster e temperatura.

Ogni partita produce 38 esempi: le due mosse finali con una sola carta vengono escluse
perché la softmax ha gradiente nullo. Il teacher riceve soltanto `PlayerObservation`.

### Split senza leakage

Le 10.000 partite sono assegnate deterministicamente prima della raccolta:

| split | partite | esempi |
|---|---:|---:|
| train | 8.000 | 304.000 |
| validation | 1.000 | 38.000 |
| test | 1.000 | 38.000 |

Tutte le decisioni di una partita restano nello stesso split. Il test non partecipa alla
selezione del checkpoint: viene letto sul modello iniziale e poi sul migliore per KL di
validation. Questo chiude il rischio di leakage per la nuova pipeline; `train_bc.py` e il
value trainer generico conservano il debito storico dello split per record.

### Distribuzione delle traiettorie

V13 occupa un posto alternato; l'altro agente viene campionato dal roster preregistrato:

| avversario | peso | partite osservate |
|---|---:|---:|
| mirror v13 | 50% | 4.905 |
| heuristic_trump_saver | 20% | 2.040 |
| heuristic_v1 | 15% | 1.572 |
| heuristic_v2 | 10% | 998 |
| random | 5% | 485 |

Ogni stato visitato viene etichettato dal teacher simmetrico, anche quando la traiettoria
è stata prodotta dall'euristica. Il corpus copre quindi stili differenti senza introdurre
informazione nascosta.

## Training

Il candidato parte esattamente da v13 (`369 -> 256 -> 40`) e usa Adam, 5 epoche, batch
1024, learning rate `2e-4`, weight decay `1e-6`. Per ogni minibatch viene affiancata una
copia supervisionata con una delle 23 rinomine non identità: feature, mask, probabilità e
argmax vengono trasformati insieme.

Questa copia è valida dove la paired augmentation RL non lo era: il target soft deriva
direttamente dal teacher sulla posizione, non da un'azione campionata da un'altra policy.

| checkpoint | val KL | val agreement |
|---:|---:|---:|
| v13 iniziale | 0,62658 | 86,69% |
| epoca 1 | 0,34103 | 90,01% |
| epoca 3 | 0,21874 | 91,63% |
| **epoca 5** | **0,17078** | **92,50%** |

Sul test indipendente: KL `0,16861`, agreement `92,88%`. Il controllo senza augmentation
si ferma a KL `0,19635` e agreement `92,29%`. Nel direct match paired-vs-noaug la forza è
neutra (`+0,08`, CI `-0,11..+0,28`): la copia migliora imitazione e simmetria, ma questo
screening non attribuisce ad essa un vantaggio di gioco separato.

## Simmetria

Sonda canonica su 4.096 osservazioni e 94.208 rinomine non identità:

| modello | flip | JS media | stati con almeno un flip |
|---|---:|---:|---:|
| v13 | 18,19% | 0,14124 bit | 51,17% |
| distillato senza augmentation | 10,95% | 0,06773 bit | n/d |
| **distillato paired** | **10,23%** | **0,06143 bit** | **31,37%** |
| teacher 24x | 0% | 0 bit | 0% |

La MLP ordinaria non eredita la garanzia matematica del teacher, ma supera il gate `<12%`
senza comprimere il margine: gap top-2 medio `0,909`, vicino a v13 (`~0,93`).

## Forza policy-only

Tutte le righe usano la seed suite medium, 10.000 partite seat-fair e CI sulle coppie.

| agente A | agente B | punti A-B | CI95 |
|---|---|---:|---:|
| **distillato paired** | v13 | **+0,51** | **+0,11..+0,92** |
| distillato paired | teacher 24x | -0,23 | -0,59..+0,13 |
| distillato paired | no augmentation | +0,08 | -0,11..+0,28 |
| distillato paired | heuristic_v1 | +22,01 | +21,47..+22,55 |
| distillato paired | heuristic_trump_saver | +15,57 | +15,05..+16,09 |

Il candidato recupera circa il 57% del vantaggio diretto teacher-vs-v13 (`0,51/0,90`),
ma la stima è solo descrittiva: i due direct match condividono la suite ma non costituiscono
una singola differenza statistica paired a tre agenti.

## Qualità decisionale

Gate domain da 10.000 partite contro `heuristic_v1`:

| metrica | v13 | distillato | teacher 24x |
|---|---:|---:|---:|
| overkill complessivo | 22,05% | **21,86%** | 22,05% |
| overkill su piatto povero | 8,01% | **5,51%** | 3,91% |
| trump waste | 0,120% | **0,080%** | 0,050% |

Il candidato resta intermedio fra allievo iniziale e teacher anche nei comportamenti, senza
riaprire il difetto corretto da v13.

## Costi e artefatti locali

- Generazione 10k: `70,54 s` + `5,69 s` di compressione, 142 partite/s.
- Dataset compresso: `32.547.075` byte; SHA-256
  `cdcfb37ec20b32deb76fa785e118c1083a60d6e7cc8dba474787eec6ec443275`.
- Training paired: `6,01 s`.
- Modello: `424.740` byte; SHA-256
  `a837b3b9994a307bf58403797d4c72c7f038678c47c8ec2eef2fc8cb12c4cce5`.

Dataset, modelli candidati e report grezzi restano gitignored. L'evidenza sintetica
versionata contiene i numeri necessari a riprodurre la decisione.

## Estensione indipendente a 50.000 partite

Il corpus con seed `20260712`, stesso roster e split 80/10/10, contiene 50.000 partite e
1,9 milioni di esempi. La raccolta ha richiesto `341,75 s`, più `27,57 s` di compressione;
il file misura `162.757.578` byte. SHA-256:
`87673a305615d2ced4cb016f7fd26fc44e061803d030101857e20e09e4b8ec30`.

Il formato monolitico richiede circa 3,2 GB di RAM. Per iterazioni successive sarà
preferibile introdurre shard streaming; il gate singolo è terminato senza errori.

Stessa ricetta preregistrata del 10k, senza accordare epoche o learning rate sul risultato:

| modello | agreement test | KL test | flip semi | JS media |
|---|---:|---:|---:|---:|
| v13 iniziale sul corpus 50k | 86,51% | 0,62683 | 18,19% | 0,14124 |
| distillato 10k | 92,88% | 0,16861 | 10,23% | 0,06143 |
| **distillato 50k** | **95,39%** | **0,06633** | **6,04%** | **0,02867** |

Il gap top-2 del 50k resta `0,905`: la simmetria non deriva da una policy resa indecisa.
Soltanto un confronto su 94.208 mostra near-tie al threshold `1e-4`.

### Forza e comportamento 50k

| agente A | agente B | punti A-B | CI95 |
|---|---|---:|---:|
| **distillato 50k** | v13 | **+0,66** | **+0,24..+1,09** |
| distillato 50k | teacher 24x | -0,22 | -0,53..+0,09 |
| distillato 50k | distillato 10k | +0,16 | -0,16..+0,48 |
| distillato 50k | heuristic_v1 | +22,10 | +21,56..+22,64 |
| distillato 50k | heuristic_trump_saver | +15,78 | +15,26..+16,30 |

Più dati migliorano nettamente imitazione e simmetria; non dimostrano invece forza
aggiuntiva sul 10k, perché la CI dello scontro diretto include zero. Il confronto causale
con v13 resta positivo e quello col teacher neutro.

Nel gate qualità contro `heuristic_v1`, overkill complessivo `21,77%`, overkill su piatto
povero **`4,17%`** e trump waste `0,074%`: il candidato si avvicina ancora al teacher e
resta migliore di v13 sui comportamenti preregistrati.

Modello locale: `data/models/suit_distilled_v0_50k_seed20260712.npz`, 424.692 byte,
SHA-256 `c413a704fff42838714baff791f706d6fe4f008e77ea86c750b0c2770d445cec`.

## Gate PIMC

Il probe da 200 partite era inconcludente (`-0,44`, CI `-3,05..+2,17`). Lo screening small
da 2.000 partite è neutro ma senza regressione evidente: `+0,35`, CI `-0,53..+1,22`.
Una partita richiede circa `0,076 s` quando entrambi i lati usano PIMC belief 16x8; il
medium da 10.000 dura quindi circa 12-13 minuti e va lanciato dal maintainer.

Il confronto medium usa la stessa belief v0, `uniform_mix=0.10`, finestra 8, 16 determinizzazioni
e solver su entrambi i lati tramite il registry `bc_model_pimc_belief_16x8`; cambia soltanto
il file policy. È questo il gate che decide se la forza policy-only arriva al default reale.

| configurazione A | configurazione B | partite | punti A-B | CI95 |
|---|---|---:|---:|---:|
| distillato 50k PIMC belief 16x8 | v13 PIMC belief 16x8 | 2.000 | +0,35 | -0,53..+1,22 |
| **distillato 50k PIMC belief 16x8** | **v13 PIMC belief 16x8** | **10.000** | **+0,43** | **+0,03..+0,84** |

Nel medium il candidato vince 4.936 partite, ne perde 4.758 e pareggia 306; score rate
`50,89%`, CI `50,07..51,71%`. Il margine è piccolo ma positivo secondo il criterio
preregistrato: la CI sulle 5.000 coppie seat-fair esclude appena zero.

Il beneficio della policy simmetrica sopravvive quindi alla ricerca del default reale.
Non va sommato aritmeticamente al `+0,66` policy-only: sono due configurazioni e due match
diversi. Entrambi rispondono però nella stessa direzione, senza regressioni sulle baseline
o sui comportamenti di conservazione delle briscole.

**Decisione eseguita: promozione a `best_a2c_v14.npz` nella release `0.36.0`.** L'asset
ufficiale conserva esattamente i quattro array del candidato e aggiunge metadati sintetici;
SHA-256 `a67ed1d7f01ba1019f157134ade23fa9f822e442b671c83684bd4500e97695a8`.

Il controllo aggiuntivo col protocollo storico omogeneo big 100k, seat-fair, Numba e seed
`0..49.999` ottiene **`+21,75808` punti/partita** contro `heuristic_v1`, CI95
`+21,59..+21,93`. Questo risultato non sostituisce il confronto causale con v13: serve a
inserire v14 nel grafico di progressione senza mescolare il gate medium con i big precedenti.

## Criteri registrati

L'intera distillazione v0 è passata perché:

1. agreement test e KL migliorano rispetto al 10k;
2. flip resta sotto 10,23% o almeno non supera il gate 12%;
3. forza diretta contro v13 resta positiva e contro teacher non peggiora;
4. overkill povero non supera v13;
5. il PIMC medium dimostra un piccolo vantaggio su v13 (`+0,43`, CI `+0,03..+0,84`).

Artefatto grezzo finale:
`benchmarks/experiments/suit_distillation_v0_50k_seed20260712/pimc16x8_vs_v13_medium.json`.
