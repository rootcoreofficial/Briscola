# Distillazione sharded del teacher 20M (250k)

Data: 2026-07-17
Stato: **esperimento concluso; STOP promozione per neutralita' nel runtime PIMC 16x8**
Modello live invariato: `best_a2c_v14.npz`

## Perche' questo esperimento

Il checkpoint A2C 20M grezzo e' pari a v14 sui 100.000 game indipendenti. La sua media
esatta sulle 24 rinomine dei semi mostra invece un segnale nuovo e replicabile:

| confronto | delta teacher | CI95 a coppie |
|---|---:|---:|
| teacher 20M vs v14 | `+0,3293` | `+0,2295..+0,4291` |
| teacher 20M vs raw 20M | `+0,2606` | `+0,1532..+0,3680` |

Il gate autorizza a verificare se una singola MLP puo' assorbire questo vantaggio. Non
riapre lo scaling A2C, non promuove il 20M e non crea automaticamente v15.

## Configurazione congelata del corpus

| voce | valore |
|---|---|
| teacher base | `a2c_v14_scale50m_seed20260723_at20m.npz` |
| SHA-256 teacher | `b414b66dd5ab4c18cb0897e664f15ff56c9c1a5949dc8401a415f7e1d6342fe5` |
| target | media dei logits riallineati sulle 24 rinomine |
| partite / esempi | `250.000` / `9.500.000` |
| shard | `10 x 25.000` partite, `950.000` esempi ciascuno |
| seed globale | `20260724` |
| split globale | 80/10/10 per `game_id`: `200.000 / 25.000 / 25.000` partite |
| temperatura | `1,0` |
| posto policy base | alternato |
| mosse forzate | escluse, 2 per partita |

Roster delle traiettorie, uguale alla distillazione v14 per non introdurre una seconda
variabile:

| avversario | peso |
|---|---:|
| mirror raw 20M | 50% |
| `heuristic_trump_saver` | 20% |
| `heuristic_v1` | 15% |
| `heuristic_v2` | 10% |
| `random` | 5% |

Ogni decisione non forzata viene etichettata dal teacher anche quando lo stato e' stato
raggiunto dall'avversario. Teacher e policy ricevono soltanto `PlayerObservation`.

## Formato e resume

`generate_suit_distillation_shards.py` assegna prima lo split a tutte le 250.000
partite, poi genera ogni shard con un seed uint32 derivato da SHA-256 di seed globale e
indice. Ne segue che un resume non deve rigenerare o saltare RNG gia' consumato.

Il manifest rigoroso conserva:

- configurazione e fingerprint dei sorgenti che cambiano la raccolta;
- commit di partenza e ricevuta del teacher;
- intervallo contiguo di `game_id`, seed, split e roster per ogni shard;
- dimensione e SHA-256 di ogni NPZ;
- stato `in_progress` o `complete` e contatori aggregati.

Ogni shard viene compresso in un temporaneo e pubblicato con replace atomico. Il resume
verifica tutte le ricevute gia' registrate; recupera anche uno shard completo rimasto
senza riga nel manifest dopo un'interruzione fra i due replace. Il loader rifiuta path
fuori dalla directory del manifest, range sovrapposti, game id mancanti, split incoerenti
e partite con un numero diverso da 38 esempi.

Il formato atteso occupa circa 814 MB compressi in totale. Un singolo shard richiede
circa 1,6 GB quando decompresso, invece dei circa 16 GB del corpus monolitico.

## Gate del corpus

Prima del training devono essere veri tutti i punti:

1. manifest `complete`, esattamente 10 shard e fingerprint invariato;
2. 250.000 game, 9.500.000 esempi e split `200k/25k/25k`;
3. `game_id` contigui `0..249999`, uno split per partita e 38 esempi per game;
4. hash, dimensione e contenuto numerico validi per tutti gli shard;
5. nessun valore non finito, massa su azioni illegali o target diverso dal proprio argmax.

La raccolta e il training non sono concatenati. Il launcher si ferma dopo il corpus per
permettere questo controllo.

### Esito del corpus reale (2026-07-17)

La raccolta ha completato i dieci shard in `1.925,88 s` (circa 32 minuti), senza resume
necessario. Il manifest finale riporta:

| controllo | risultato |
|---|---:|
| shard | `10/10` |
| partite | `250.000` |
| esempi | `9.500.000` |
| split train / validation / test | `200.000 / 25.000 / 25.000` game |
| dimensione NPZ totale | `816.188.051` byte |
| fingerprint configurazione | `b626dd0761d1a06e5c67e6d4a1d1d05d326210f2f97f9e6d41cd0681e9f9f510` |
| SHA-256 manifest | `d4bc41a58b75c8405357f85c905b33cb0eab790d76339b7372cabdcf76e045d9` |

Il roster osservato (`124.979` mirror, `50.054` trump saver, `37.506` heuristic v1,
`25.077` heuristic v2, `12.384` random) e' coerente col campionamento preregistrato.

`verify-data` ha ricalcolato dimensione e SHA-256 dei dieci file e li ha decompressi uno
alla volta, ripetendo tutte le validazioni di feature, probabilita', action mask,
`game_id`, split e 38 esempi per partita. La ricevuta
`dataset_verified.sha256` coincide col digest del manifest. **Gate corpus: PASS;** il
training streaming congelato e' autorizzato.

## Training congelato

Solo dopo il gate del corpus:

| voce | valore |
|---|---|
| warm start | raw 20M, stesso SHA-256 del teacher base |
| epoche | 5 |
| batch | 1.024 esempi originali |
| Adam learning rate | `2e-4` |
| weight decay | `1e-6` |
| seed | `20260724` |
| augmentation | una copia paired con rinomina non identita' per minibatch |
| ordine | shard rimescolati, poi righe train rimescolate dentro lo shard |
| selezione | KL minima sulla validation globale |

Il trainer apre un solo shard alla volta. Validation e test sono aggregati su tutti gli
shard in ordine canonico; il test viene letto sul warm start e poi soltanto sul migliore
checkpoint scelto dalla validation. Il primo gate richiede KL validation/test migliori
del warm start, agreement test crescente e assenza di valori non finiti.

### Esito del training reale (2026-07-17)

Le cinque epoche sono terminate in `240,81 s`; ogni epoca ha migliorato la validation e
la quinta e' quindi il checkpoint selezionato. Il confronto sul test separato, mai usato
per scegliere o aggiornare i pesi, e':

| metrica test | warm start raw 20M | student epoca 5 | delta |
|---|---:|---:|---:|
| KL dal teacher | `0,090962` | `0,012784` | `-85,9%` |
| cross-entropy | `0,202143` | `0,123965` | `-38,7%` |
| accordo argmax | `94,350%` | `97,860%` | `+3,510 pp` |

Il modello misura 428.044 byte, e' caricabile come `MLPBCModel` 369x256x40, non contiene
valori non finiti e ha SHA-256
`8a0a03946c9413ed7e6c18059a6aa03f63a9476e0b603ad977f4955cb444199d`. Artefatti:

```text
benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/
  training.log
  training_report.json
  models/suit_distilled_20m_teacher24_250k_seed20260724.npz
```

**Gate imitation: PASS.** Questo dimostra che la singola MLP ha assorbito gran parte del
comportamento del teacher sulle osservazioni tenute fuori; non dimostra ancora che vinca
piu' partite di v14.

## Gate dello student

Il file prodotto resta un candidato senza nome di versione. I test usano il dominio
canonico finche' la divergenza Numba/domain non e' spiegata.

1. Sonda canonica da 4.096 osservazioni: identity esatta e flip non superiore al `6,04%`
   di v14.
2. Screen policy-only da 20.000 partite seat-fair contro v14, raw 20M e teacher 20M,
   con suite nuova a partire da `10.000.000`. Per proseguire, la stima contro v14 deve
   essere positiva e nessun confronto deve mostrare una sconfitta certa.
3. Conferma indipendente student-v14 da 100.000 partite, seed da `11.000.000`: il limite
   inferiore CI95 del delta deve essere sopra zero.
4. Decision quality sulla suite medium: overkill povero non superiore al `4,17%` di v14
   e nessuna regressione materiale su trump waste.
5. Solo allora PIMC belief 16x8 da 10.000 partite a configurazione identica. Una
   promozione richiede anche qui limite inferiore CI95 sopra zero contro v14.

La neutralita' o una regressione nello screen chiudono la promozione, ma restano
informative sulla quota di vantaggio del teacher comprimibile in una MLP.

### Esito dei gate pre-PIMC (2026-07-17)

La sonda canonica usa le stesse 4.096 osservazioni bilanciate di v14. Lo student cambia
argmax nel `2,9456%` delle 94.208 comparazioni identity-vs-rinomina, con CI95 bootstrap
`2,5709%..3,3150%`: meno della meta' del riferimento v14 (`6,04%`). Il controllo identity
e' esatto. **Gate simmetria: PASS.**

Lo screen policy-only domain usa 20.000 partite seat-fair e gli stessi 10.000 seed di
coppia consecutivi da `10.000.000`:

| avversario | delta punti student | CI95 a coppie | esito |
|---|---:|---:|---|
| v14 | `+0,3778` | `+0,16..+0,59` | positivo |
| raw 20M | `+0,3051` | `+0,07..+0,54` | positivo |
| teacher 20M 24x | `+0,0038` | `-0,16..+0,17` | pari |

La conferma indipendente contro v14 usa 100.000 partite e 50.000 nuovi seed da
`11.000.000`: `+0,18046` punti, CI95 `+0,08..+0,28`, 48.798 vittorie, 48.317 sconfitte e
2.885 pareggi. **Gate forza policy-only: PASS.** Il segnale dello screen si riduce ma
resta positivo con il limite inferiore sopra zero.

La decision quality medium contro `heuristic_v1`, ripetuta anche su v14 con seed `0`,
non mostra regressioni:

| metrica | student | v14 |
|---|---:|---:|
| overkill su piatto povero | `1,0637%` (78/7.333) | `4,1724%` (302/7.238) |
| trump waste | `0,0409%` (34/83.077) | `0,0739%` (61/82.545) |
| overkill complessivo | `21,0359%` | `21,7737%` |

**Gate decision quality: PASS.** Resta il solo confronto PIMC belief 16x8, cioe' il
runtime live effettivo. Usa 10.000 partite seat-fair, seed di coppia nuovi da
`12.000.000`, belief v0 e solver identici sui due lati. La promozione richiede ancora il
limite inferiore CI95 del delta sopra zero; la parita' non basta.

### Esito del gate PIMC finale (2026-07-17)

Il confronto domain e' terminato in 12 minuti e 35 secondi sugli esatti 5.000 seed di
coppia consecutivi da `12.000.000`:

| metrica | student PIMC 16x8 | v14 PIMC 16x8 |
|---|---:|---:|
| vittorie | `4.827` | `4.870` |
| pareggi | `303` | `303` |
| punti medi | `59,9908` | `60,0092` |
| delta punti student | `-0,0184` | CI95 `-0,33..+0,30` |
| score rate student | `49,785%` | CI95 `49,10%..50,47%` |

Belief v0, 16 determinizzazioni, finestra 8, solver, motore e seed sono identici sui due
lati; cambia soltanto la policy MLP. Il report JSON ha SHA-256
`e242480adb4568681462895c530ab72f395ecc0a9de0db89d0f215da7da776cf`.

**Gate PIMC 16x8: FAIL per neutralita'.** Il vantaggio policy-only dello student viene
assorbito dalla search live. Il protocollo ha vietato di aumentare il campione o ripetere
la suite per sostenere una promozione di forza: nessuna replica e v14 è rimasta ufficiale
in quella fase.
Lo student rimane un artefatto sperimentale utile: dimostra che il teacher 24x e'
comprimibile, migliora simmetria e stile e aggiunge un piccolo vantaggio senza search,
ma non migliora il prodotto effettivamente pubblicato.

Il follow-up separato di efficienza 8x8 dimezza la latenza search ma perde `-0,742` punti
contro v14 16x8 con CI95 interamente negativa. Anche questa alternativa si ferma allo
screen; dettagli in `suit-student-8x8-efficiency-2026-07-17.md`.

Il successivo punto intermedio 12x8 supera invece lo screen da 4.000 partite: delta
`-0,0495` (CI95 `-0,5666..+0,4676`) e latenza media/p95 circa `0,75x`, senza errori.
Anche la conferma indipendente da 20.000 partite passa: `+0,1052` punti (CI95
`-0,1126..+0,3230`), latenza media/p95 circa `0,75x` e integrita' perfetta. E' un PASS
di non inferiorita' ed efficienza, non una riapertura della promozione di forza. Dopo un
audit browser separato, questo diverso criterio ha portato alla promozione v15 in 0.38.0;
dettagli e gate in `suit-student-12x8-efficiency-2026-07-17.md`.

## Comando operativo

Il launcher congela tutti i parametri e mantiene PID/log. Le fasi gia' concluse erano:

```bash
scripts/run_suit_distillation_20m_250k.sh start-data
scripts/run_suit_distillation_20m_250k.sh verify-data
scripts/run_suit_distillation_20m_250k.sh start-train
```

Controllo senza seguire il log:

```bash
scripts/run_suit_distillation_20m_250k.sh status
```

Artefatti attesi:

```text
benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/
  data_generation.log
  dataset/manifest.json
  dataset_verified.sha256
  dataset/shards/shard-00000-of-00010.npz
  ...
  dataset/shards/shard-00009-of-00010.npz
```

Corpus, training e tutti i gate sono conclusi e non vanno ripetuti. Il comando storico
del gate finale era:

```bash
scripts/run_suit_distillation_20m_250k.sh start-pimc
```

Il report `pimc16x8_student_vs_v14_10k.json` e il log omonimo sono presenti nella
directory del run; il launcher rifiuta di sovrascriverli.

## Ricevuta di implementazione

Lo smoke 9 game / 3 shard ha verificato:

- stop dopo il primo shard e resume senza modificarne SHA-256;
- copertura globale e split disgiunti;
- verifica rigorosa di manifest, hash e contenuto;
- training streaming e caricamento del modello prodotto nel runtime.

Le regressioni vivono in `tests/test_suit_distillation_shards.py`; il percorso monolitico
v14 resta supportato e separato.
