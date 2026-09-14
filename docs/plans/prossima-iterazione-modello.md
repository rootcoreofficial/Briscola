# Piano di ricerca: prossima iterazione del modello

> Stato finale: esiti aggiornati al 2026-07-17; le sette piste e lo scaling 50M sono chiusi.
> Il ramo teacher 20M e' proseguito in una distillazione 250k e nel profilo 12x8,
> promosso come v15 nella release 0.38.0 per efficienza/non inferiorita'.
> Questo documento approfondisce le sette piste sintetizzate in `PLAN.md`; non autorizza
> automaticamente training lunghi né promozioni. `PLAN.md` resta la fonte di verità su
> quale fase eseguire adesso.

## 1. Obiettivo e baseline

La baseline da preservare durante gli esperimenti descritti in questo documento era:

- policy `best_a2c_v14.npz`, encoder v4, `369 -> 256 -> 40`, senza guard runtime;
- default prodotto `bc_model_pimc_belief_16x8` con belief v0 e solver finale;
- valutazioni seat-fair paired, stessi mazzi e posti scambiati;
- anti-cheat invariato: policy, belief, reward e search ricevono solo informazione lecita.

Il plateau v11-v13 non prova da solo che la rete sia troppo piccola. Può dipendere da
capacità, rumore del trainer, simmetrie non apprese, qualità della belief o dal fatto che
parte del vantaggio della search non sia comprimibile in una policy reattiva. L'ordine
del piano serve a distinguere queste cause prima di pagare un'altra run lunga.

## 2. Regole comuni degli esperimenti

Ogni ablation deve cambiare **una sola variabile** e registrare:

- commit, hash SHA-256 degli asset, encoder e metadati completi;
- seed di training, seed suite di evaluation, roster e configurazione dei wrapper;
- giochi, tempo totale, CPU media e p95 quando cambia il runtime;
- gate policy-only, default PIMC 16x8 e decision quality;
- CI paired e risultati di almeno tre training seed per una promozione del trainer.

Una CI di evaluation stretta misura l'incertezza delle partite di **quel checkpoint**,
non la variabilità del training. Per questo un singolo candidato fortunato non basta.

### Criterio di promozione

Una variante può essere promossa solo se:

1. non regredisce materialmente nei gate policy-only e PIMC 16x8;
2. non peggiora spreco/overkill o altri comportamenti già corretti;
3. il vantaggio si ripete su più seed di training, oppure l'intervento è una modifica
   runtime deterministica verificata sugli stessi campioni;
4. costo e latenza sono compatibili con il default web;
5. nessun test o nuova API introduce accesso a mani o mazzo nascosti.

## 3. Pista 1: salute e capacità della MLP

### Domanda

La hidden layer da 256 neuroni usa davvero la propria capacità, oppure contiene neuroni
quasi inattivi o contributi duplicati che indicano un problema di ottimizzazione?

### Stato del codice

Sono già disponibili i pesi `w1/b1/w2/b2` in
`src/briscola_ai/ai/models/bc_model.py`, l'encoder v4 e raccolte di osservazioni lecite
simili a quelle di `scripts/style_feature_probe.py`. `scripts/widen_mlp_net2net.py`
implementa invece Net2Net per copia: è una tecnica diversa dalla nuova ablation
zero-output e preserva esattamente la funzione solo con rumore zero.

### Da implementare

Una sonda riproducibile su suite fissa che, per fase della partita, misuri:

- frequenza con cui ogni pre-attivazione è positiva;
- distribuzione e magnitudine delle attivazioni;
- contributo del neurone alle sole azioni legali, rimuovendo la componente costante che
  non può cambiare la scelta;
- cambi dell'argmax quando il neurone viene ablatto;
- collegamento ai gradienti actor, critic e trunk durante piccoli smoke training.

Un neurone mai attivo nella suite va chiamato **inattivo sulla suite**, non morto in
assoluto. Anche 256 neuroni attivi non dimostrano che serva più capacità: la prova causale
resta un training controllato.

### Ablation di capacità

Se la diagnosi la giustifica, confrontare v13 con estensioni `256 -> 320` e `256 -> 384`.
I nuovi ingressi devono avere inizializzazione random/He, mentre le nuove righe verso
actor e critic devono partire da zero. Così la funzione iniziale è identica, ma i nuovi
neuroni possono attivarsi e ricevere gradiente. Azzerare sia ingressi sia uscite creerebbe
ReLU permanentemente inattive.

Test richiesti:

- logits e valore iniziali identici a v13 su una suite di osservazioni;
- nuovi neuroni effettivamente attivi;
- pesi uscenti nuovi aggiornati dopo almeno un optimizer step;
- save/load e metadati coerenti;
- confronto distinto con Net2Net `noise=0`, senza mescolare le due tecniche.

### Decisione

**GO** solo se widening zero-output migliora in modo ripetibile i gate su tre seed.
**STOP** se attivazioni e ablation non mostrano un collo di bottiglia oppure il widening
aumenta varianza senza vantaggio ripetibile.

### Esito 2026-07-12

La pista è chiusa con **STOP** prima del widening. V14 ha 123/256 unità quasi inattive;
l'ablation congiunta è neutra su 10.000 partite. Un controllo successivo ha
reinizializzato gruppi annidati di 8 e 16 unità: tutte diventano attive e apprendono, ma
reset 16 riduce la KL validation solo dello `0,328%` rispetto alla stessa continuazione
senza reset, sotto il gate preregistrato dell'`1%`. Report:
`hidden-unit-diagnostic-v0-2026-07-12.md` e
`dormant-reinitialization-screen-v0-2026-07-12.md`.

## 4. Pista 2: simmetria alla rinomina dei semi

### Domanda

Il modello cambia decisione quando denari, coppe, spade e bastoni vengono soltanto
rinominati, rinominando coerentemente anche briscola, tavolo e storia?

### Implementato il 2026-07-11

`ai/evaluation/suit_symmetry.py` permuta semanticamente un'intera `PlayerObservation`:

- mano, briscola, tavolo e `trick_history`;
- `seen_cards_onehot` e `out_of_play_cards_onehot`;
- action mask e action id, usando la convenzione `suit * 10 + rank`;
- output del modello, rimappato poi nei semi originali prima del confronto.

`scripts/probe_suit_symmetry.py` prova tutte le 24 permutazioni su una suite seat-fair
bilanciata per avversario e fase. L'identità è un controllo; agreement, flip dell'argmax
e divergenza Jensen-Shannon sono calcolati sulle altre 23 e sulle 276 coppie dell'orbita.

I test coprono:

- round trip permutazione/inversa senza perdita;
- carte legali e feature coerenti dopo il remapping;
- agente sintetico equivariant con agreement 100% e divergenza zero;
- near-tie controllato anche nella distribuzione rinominata e smoke CLI byte-riproducibile.

Il confine anti-cheat è strutturale: l'helper accetta soltanto `PlayerObservation`; il
`GameState` resta confinato al loop del motore e non viene conservato o serializzato.
Seed e posto compaiono nel report solo come metadati offline di riproduzione, mai come
input del modello.

Esito su v13: 4.096 osservazioni non forzate, 94.208 confronti non identità,
**18,19% di flip** (CI bootstrap per osservazione `17,38..18,93%`) e 51,17% degli
stati con almeno un cambio. Nessuna delle 98.304 distribuzioni sulle 24 rinomine è un
quasi-pareggio al threshold `1e-4`.
Il controllo identità è esatto e il secondo run produce un JSON byte-identico. Numeri,
metodo e limiti sono in `sonda-simmetria-semi-2026-07-11.md`; evidenza canonica in
`../reports/evidence/suit_symmetry_v13.v1.json`.

### Possibili interventi

In ordine di costo:

1. data augmentation paired: originale e copia rinominata nello stesso update; una sola
   sostituzione casuale non cambia l'obiettivo medio di mazzi già simmetrici;
2. loss di consistenza tra osservazione originale e permutata;
3. media delle 24 predizioni a inference, solo come upper bound perché costosa;
4. architettura esplicitamente equivariant, valutata soltanto nella pista 7.

Embedding assoluti separati per i quattro semi **non** garantiscono la simmetria. Una
futura architettura deve usare ruoli relativi alla briscola/lead oppure condivisione di
pesi equivariant.

Per l'augmentation paired, una singola permutazione deve trasformare **l'intera
traiettoria**: osservazioni, mask e action id; reward, return e advantage restano invariati.
Il riferimento domain deve passare da `permute_player_observation` e dall'encoder canonico.
Il collector fast/Numba non deve introdurre scambi ad hoc delle 369 feature: o conserva
campi strutturati sufficienti, oppure usa una trasformazione v4 esplicita verificata su
osservazioni casuali contro il riferimento semantico. Il flag disattivato deve essere
bit-identico al trainer attuale. La loss paired va mediata sui `2N` sample, non sommata,
per non raddoppiare implicitamente il learning rate; numero di traiettorie ambiente e
optimizer update restano uguali, mentre il costo extra di forward/backward va registrato.

### Decisione

**GO diagnostico** verso la prima ablation di augmentation paired dei semi: l'asimmetria
è grande, attraversa fasi e avversari e non dipende dai pareggi. Augmentation e consistency
vanno testate separatamente. Non è ancora un GO alla promozione: l'intervento deve ridurre
i flip senza regressione nei gate di forza e stile. La media delle 24 predizioni resta solo
un upper bound costoso, non una proposta runtime.

### Esito del test causale (2026-07-11)

Paired, forward-KL e hinge sono stati chiusi dopo gli screening documentati. La media
esatta sulle 24 rinomine ha invece flip zero, costa `1,45x` grazie a un solo batch e batte
v13 di `+0,90` punti/partita (CI `+0,47..+1,33`). La simmetria è quindi una leva di forza.
Il prossimo passo è distillare questo teacher in una singola MLP; dettagli e limiti in
`suit-symmetrized-v13-2026-07-11.md`.

La prima distillazione su 10.000 partite supera lo screening: flip `10,23%`, `+0,51`
punti/partita contro v13 e neutralità col teacher 24x. Il gate successivo è un corpus
indipendente da 50.000 partite; protocollo in `suit-distillation-v0-2026-07-11.md`.

Il 50k porta il flip a `6,04%` e resta positivo contro v13 (`+0,66`, CI
`+0,24..+1,09`). Il PIMC belief 16x8 small è neutro (`+0,35`, CI `-0,53..+1,22`):
prossima e unica decisione del ramo è il confronto PIMC medium a pari configurazione.

Il PIMC medium passa con `+0,43` punti/partita (CI `+0,03..+0,84`): il vantaggio arriva
anche al default reale. Il ramo è chiuso con GO tecnico alla promozione v14; restano
soltanto audit di release, catalogo e report.

## 5. Pista 3: dose e budget adattivo PIMC

### Domanda

Quanta forza aggiungono davvero 32 o 64 determinizzazioni nella finestra 8, e possiamo
spendere il budget alto soltanto sulle decisioni incerte?

### Stato del codice

`PIMCAgent`, `PIMCSearchDiagnostics` e `scripts/evaluate_pimc.py` forniscono la base.
Oggi manca la latenza per singola decisione, quindi non abbiamo p50/p95, e l'harness non
confronta ancora in modo simmetrico due configurazioni belief differenti: il belief
passato dalla CLI viene applicato soltanto al lato A.

### Fase fissa 16/32/64

Prima estendere l'harness affinché i due lati condividano:

- policy v13, belief, `uniform_mix`, finestra 8 e solver;
- stessi seed e stesso prefisso di campioni per le prime 16/32 determinizzazioni;
- diagnostica per decisione e latenza p50/p95.

Il confronto principale è forza a pari costo CPU, non semplice agreement con 64.

### Budget adattivo

Solo se 32/64 mostrano headroom, introdurre campionamento a blocchi con minimo e massimo.
Lo stop deve considerare stabilità del primo e secondo candidato, non soltanto il margine
medio dell'ultimo blocco. Controllare più volte una CI nominale al 95% introduce bias da
optional stopping: una prima versione va dichiarata euristica e validata direttamente
contro fixed-64; una regola sequenziale formale è un esperimento successivo.

Test richiesti:

- budget sempre nei limiti e modalità disattivata identica al fixed budget;
- nessuno stop su tie o sequenze sintetiche ambigue;
- stop anticipato su sequenze chiaramente separate;
- determinizzazioni effettive e latenza sempre registrate;
- anti-cheat invariato: tutti i campioni partono dalla sola osservazione.

### Decisione

**GO** al budget adattivo se esiste vantaggio 32/64 e la variante mantiene la forza con
CPU media/p95 inferiori. **STOP** se la curva 16/32/64 è piatta; non riaprire la finestra
10, già chiusa negativa.

### Esito 2026-07-12

Lo storico rendeva superfluo un nuovo sweep completo; è stato eseguito soltanto un probe
simmetrico v14 da 2.000 partite. PIMC belief 32×8 contro 16×8 fa `+0,298` punti, ma la CI95
paired `-0,025..+0,621` include zero, mentre la latenza media search raddoppia
(`28,93` vs `14,50 ms`). **STOP** a conferma 10k, 64×8 e budget adattivo; 16×8 resta il
default. Dettagli in `pimc-dose-v14-probe-2026-07-12.md`.

## 6. Pista 4: rendere A2C meno rumoroso

### Stato reale del trainer

`scripts/train_a2c.py` usa un actor-critic con return Monte Carlo completi e trunk
condiviso. Nei warm-start il critic `wv/bv` viene oggi sempre reinizializzato a zero.
Non esistono ancora normalizzazione degli advantage né gradient clipping globale.

### Prima fase: osservabilità

La diagnostica passiva è ora implementata e registra:

- media/std degli advantage per optimizer update;
- explained variance e loss del critic;
- norme dei gradienti separate per actor, critic e trunk;
- activation rate della hidden layer;
- rapporto tra aggiornamento e norma dei parametri.

Il probe preregistrato su tre seed × 2.000 partite supera tutti i gate. Nella metà finale
il critic raggiunge explained variance mediana `0,127..0,133`, gli advantage hanno bias
relativo mediano `0,147..0,174`, i gradienti restano a `1,92..2,00×` tra p95 e mediana e
i passi relativi p95 restano sotto `0,019%`. Circa il 48% delle unità non si attiva in
almeno un batch, coerentemente con la pista capacità già chiusa, ma il tasso medio di
attivazione resta stabile vicino all'`8,7%`. Report completo:
`a2c-health-diagnostic-v0-2026-07-14.md`.

### Ablation numeriche sospese

1. critic `reset` attuale contro `reuse` dal checkpoint;
2. normalizzazione advantage sull'intero optimizer update, non per singola partita;
3. global gradient clipping immediatamente prima di Adam;
4. learning-rate decay solo dopo aver isolato i tre punti precedenti;
5. stop-gradient o trunk separato actor/critic soltanto se la diagnostica mostra
   interferenza, perché cambia più profondamente capacità e costo.

I path per-step e Numba batch applicano la stessa osservazione passiva; test sintetici e
un controllo end-to-end garantiscono che attivare il report non cambi i pesi. Le ablation
restano disponibili, ma nessuna ha priorità finché una diagnostica futura non fallisce il
relativo gate.

### Decisione

**STOP**, per ora, a reuse, normalizzazione e clipping: il probe non mostra il difetto che
dovrebbero correggere. Se una misura futura riapre una variante, passerà prima uno
screening breve su tre seed. **GO** a una run lunga solo per un intervento con varianza
ridotta e mediana non regressiva; **STOP** se il beneficio appare in un solo seed o
richiede combinare più modifiche non interpretabili.

PPO non è il passo successivo: il controllo corrente non individua un collo di bottiglia
numerico nel trainer.

## 7. Pista 5: training davvero paired

### Problema

Il flag attuale `train_a2c.py --seat-fair` alterna il posto della policy, ma genera un
mazzo diverso per ogni partita. Non è quindi paired come l'evaluation.

### Implementazione completata

Una schedule pura e riproducibile `(game_seed, policy_seat, opponent)` in cui ogni coppia:

- usa lo stesso seed e lo stesso opponent campionato dal mix;
- gioca una volta per seat;
- viene consumata interamente prima dello stesso optimizer update.

`num_games` deve essere pari. Anche `update_every` deve essere pari oppure il buffer deve
garantire che una coppia non attraversi due versioni diverse della policy. I collector
Numba accettano già array `game_seeds` e `policy_seats`; la schedule va resa identica nei
path domain, fast Python e Numba.

### Disegno sperimentale

Il pairing dimezza i mazzi distinti a parità di partite. Confrontare quindi:

1. stesso numero totale di partite/optimizer update;
2. stesso numero di mazzi distinti, accettando il doppio delle partite paired;
3. almeno tre seed indipendenti di training per ciascun regime.

Metriche: varianza tra run, velocità di apprendimento, gate finali e costo. Il pairing può
ridurre bias e rumore, ma non è garantito che migliori il modello.

Test richiesti: seed/opponent uguali e seat `{0,1}` in ogni coppia, schedule deterministica,
nessuna coppia spezzata da un update e rifiuto esplicito di configurazioni dispari.

Questi vincoli sono implementati da `--training-schedule paired` e coperti nei path
dominio, fast Python e Numba. Flag omesso e `serial` esplicito producono pesi bit-per-bit
uguali. Il runner riprendibile confronta tre seed a pari partite e a pari mazzi; soglie,
comando, risultato ed evidenza sono in `a2c-paired-schedule-v0-2026-07-14.md`.

### Esito 2026-07-14

A pari 20.000 partite, il paired ottiene differenze dirette rispetto al seriale di
`-0,151`, `-0,462` e `+0,242` punti: mediana `-0,151` e un solo seed non negativo. La
deviazione standard tra seed della forza contro v14 cresce da `0,177` a `0,367`
(`2,08x`), mentre la variabilita' dei gradienti cresce leggermente (`1,038x`).

Il controllo paired a pari 20.000 mazzi usa 40.000 partite ed e' molto stabile tra seed,
ma resta neutro nel direct match con il seriale 20k (mediana `-0,148`) a circa il doppio
del costo. Verdetto preregistrato: **inconcludente, mantenere seriale**. Non e'
giustificato un altro run paired piu' lungo; nessun modello temporaneo e' candidabile.

## 8. Pista 6: belief v1 multi-stile

### Problema

La belief ufficiale `belief_v0_h128_50k_seed20260702.npz` è stata addestrata su 50k
partite mirror di v7 (350k record). Il vecchio dataset usava la stessa policy su entrambi
i lati e non misurava generalizzazione a stili diversi.

### Implementazione congelata (2026-07-14)

- roster esplicito: v14 dominante, v13, anchor nominata v11 ed euristiche con stili differenti;
- `opponent_id` per record come solo metadato di split, mai input della rete;
- split per partita e fold leave-one-opponent-out;
- metriche BCE/top-k già esistenti più Brier score e calibrazione/ECE;
- harness A/B simmetrico belief v0-v1 nello stesso PIMC 16x8.

`generate_belief_dataset.py`, `train_belief.py`, `summarize_belief_folds.py` e
`run_belief_v1_gate.py` implementano il protocollo. Il roster versionato assegna pesi
`4:2:1:1:1:1:1` a v14, v13, v11, heuristic_v1, heuristic_v2, trump_saver e random.
Il runner usa 66.000 partite, esegue i sette fold in sequenza e allena l'all-styles solo
dopo un GO offline. Metodi, soglie, comando e limiti sono in
`docs/plans/belief-v1-multistile-2026-07-14.md`.

Il fold va assegnato in base allo stile della mano avversaria che la belief cerca di
inferire, non al generico nome del matchup. Il full-state è lecito solo per costruire la
label `y`; input `x` e inferenza restano osservazioni parziali.

Il pilot da 770 partite/5.390 record ha validato l'intera pipeline e lo stop automatico.
Come previsto non è competitivo con v0 addestrata su 50.000 partite (BCE macro `0,6103`
contro `0,5514`): il verdetto è soltanto `pilot_pipeline_validated`, non un test dell'idea.

La sigmoid per-carta non impone l'esatta cardinalità della mano, quindi una metrica
offline migliore non basta. Il gate decisivo mantiene identici policy v14, D=16,
finestra 8, solver e uniform mix, cambiando soltanto belief v0/v1.

### Decisione

**GO** solo se v1 migliora calibrazione sui fold esclusi e poi batte/non regredisce v0
nel PIMC paired. **STOP** se migliora BCE/top-k ma non le decisioni della search.

### Esito 2026-07-14

Il gate offline completo usa 66.000 partite/462.000 record e passa tutti i controlli:
BCE macro `-7,63%` relativo, top-k `+3,29` punti, Brier `-0,0130` ed ECE `-0,0259`
rispetto a belief v0. Il fold v14 e' l'unico con BCE peggiore (`+1,45%`), entro il tetto
preregistrato del 2%, quindi autorizza lo screen runtime ma non una promozione.

Nel PIMC 16x8 seat-fair da 2.000 partite il candidato fa `-0,224` punti/partita contro
v0, CI95 `-0,572..+0,124`, con 950 vittorie, 971 sconfitte e 79 pareggi. Integrita' e
costo sono perfetti, ma falliscono punto stimato e limite basso richiesti dallo screen.
**STOP** alla conferma 10k e alla pista belief v1; belief v0 resta ufficiale. Report ed
evidenze: `belief-v1-multistile-2026-07-14.md`.

## 9. Pista 7: nuova architettura o Q Monte Carlo

Questa fase parte solo se le piste precedenti dimostrano un limite di rappresentazione.
Non va combinata con modifiche al trainer o alla belief nello stesso esperimento.

### Opzione A: ramo residuale zero-output

È l'estensione minima della MLP e riusa la logica della pista 1. Deve partire con funzione
identica a v13, avere formato `.npz` esplicito e restare compatibile con inference NumPy.

### Opzione B: carte e storia con pesi condivisi

Usare uno scorer condiviso per carta e ruoli relativi a briscola/lead, più un piccolo
encoder sequenziale della sola `trick_history` pubblica. Embedding assoluti dei quattro
semi non bastano a garantire equivarianza. La storia non deve contenere ordine del mazzo
né mani nascoste.

### Opzione C: `Q(observation, card)` / Deep Monte Carlo

Una vera Q condivisa assegna un valore con lo stesso scorer a ogni carta legale. Può
chiamarsi Deep Monte Carlo solo se i target sono ritorni Monte Carlo stato-azione con
esplorazione sufficiente. Allenarla soltanto sulle azioni scelte crea selection bias;
usare action-values PIMC è invece distillazione del teacher, un esperimento diverso.

### Costo architetturale

Il repository non usa PyTorch/JAX e i kernel Numba assumono una MLP flat a una hidden
layer. Prima di implementare va quindi deciso:

- trainer manuale NumPy oppure dipendenza solo-training;
- formato export NumPy stabile per il runtime web;
- loader/catalogo, inferenza batch/singola e benchmark CPU;
- strategia fast/Numba senza portare framework pesanti nel processo web.

Test minimi: mask delle azioni illegali, save/load identico, equivarianza dei semi,
parità della storia pubblica, assenza di full-state e gate separati policy/PIMC/qualità.

### Decisione

Iniziare dal ramo residuale, che ha il minor numero di variabili. Embedding/history e Q
Monte Carlo sono due programmi distinti. **STOP** se il costo di tooling domina il
segnale oppure se il widening controllato non mostra alcun limite di capacità.

## 10. Sequenza operativa e criteri di stop

| Ordine | Fase | Costo iniziale | Output richiesto | Stop immediato |
|---:|---|---|---|---|
| fatto: GO v14 | Sonda simmetria semi | basso | report su 4.096 osservazioni | ramo chiuso con distillazione |
| fatto: STOP | Augmentation paired | medio | smoke, sonda e gate | forza/simmetria non congiunte |
| fatto: STOP | Sonda salute MLP | basso | JSON per fase/neurone + ablation | nessun collo di capacità |
| fatto: STOP | Dose PIMC 16/32/64 | medio | forza + CPU media | curva piatta rispetto al costo |
| fatto: sano | Strumentazione/A2C | medio | gradienti, critic, tre ablation separate | nessun difetto numerico |
| fatto: inconcludente | Training paired | medio | schedule + confronto multi-seed | varianza/forza non migliori |
| fatto: STOP | Belief v1 | medio-alto | fold multi-stile + A/B PIMC | solo metriche offline migliori |
| fatto: STOP | Scaling A2C 50M | alto | 5 checkpoint + audit domain 400k | nessun vantaggio replicato su v14 |
| fatto: GO corpus | Teacher 20M 24x | medio | equivarianza + 200k domain | `+0,3293` vs v14 e `+0,2606` vs raw |
| fatto: corpus PASS | Distillazione teacher 20M | alto | 10 shard da 25k, 9,5M esempi | manifest e contenuti verificati |
| fatto: imitation PASS | Student 20M 250k | medio | streaming 5 epoche + test separato | KL `-85,9%`, agreement `+3,51 pp` |
| fatto: pre-PIMC PASS | Student vs v14 | medio | simmetria + 20k/100k domain + stile | `+0,1805`, CI95 `+0,08..+0,28` |
| fatto: STOP | Student live 16x8 | alto | 10k domain, belief/solver identici | `-0,0184`, CI95 `-0,33..+0,30` |
| fatto: STOP | Student PIMC 8x8 | basso | screen efficienza 2k vs v14 16x8 | costo `0,51x`, forza `-0,742` |
| fatto: efficienza PASS | Student PIMC 12x8 | basso | screen 4k + conferma 20k vs v14 16x8 | costo `0,75x`, conferma `+0,1052` |
| sospeso | Nuova architettura | alto | prototipo esportabile e gate completi | nessun limite di rappresentazione isolato |

Non avviare la fase successiva per inerzia. Ogni fase deve produrre un artefatto piccolo
e versionabile (JSON di sonda, manifest o nota tecnica) che consenta a una sessione futura
di ricostruire la decisione senza dipendere dalla memoria della conversazione.

## 11. Comandi già validi e lavoro mancante

Baseline riproducibili già disponibili:

```bash
uv run python scripts/behavior_profile.py \
  --model data/models/best_a2c_v14.npz \
  --opponents heuristic_trump_saver,mirror,heuristic_v1 \
  --num-games 2000

uv run python scripts/evaluate_pimc.py \
  --model data/models/best_a2c_v14.npz \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --determinizations 16 \
  --max-unknown-cards 8 \
  --num-games 2000

uv run python scripts/probe_suit_symmetry.py \
  --model data/models/best_a2c_v14.npz \
  --out-json data/suit_symmetry_v14.json
```

Sonda MLP, simmetria, dose, roster belief, diagnostica A2C e schedule paired hanno script
ed evidenze stabili. Critic reuse, normalizzazione, clipping e una nuova architettura non
sono lavoro mancante: sono ipotesi sospese perche' i test non hanno mostrato il difetto
che dovrebbero correggere. Anche l'audit 192x64 degli errori residui e lo scaling A2C da
50M sono chiusi senza autorizzare altro reinforcement learning.

L'unico ramo attivo nasce dal gate separato del teacher 20M a 24 viste: su 200.000 nuove
partite domain supera v14 di `+0,3293` punti e il 20M grezzo di `+0,2606`, con entrambi i
limiti CI95 positivi. Il corpus reale da 250.000 partite e 9,5 milioni di esempi e'
completo: hash e contenuti dei dieci shard sono verificati, con split globale
`200k/25k/25k`. Lo student ha completato le cinque epoche congelate: sul test separato la
KL scende dell'`85,9%` e l'accordo sale di `3,51` punti percentuali rispetto al raw 20M.
Anche i gate successivi sono completati: flip `2,9456%`, conferma policy-only
`+0,18046` punti contro v14 (CI95 `+0,08..+0,28`) e decision quality migliore. Resta
pero' neutro il confronto live PIMC belief 16x8 da 10.000 partite: `-0,0184` punti,
CI95 `-0,33..+0,30`. Come preregistrato, la parita' chiude la promozione **di forza** e
vieta repliche della stessa prova. Protocollo e ricevute sono in
`suit-distillation-20m-250k-2026-07-17.md`.

Il follow-up di efficienza student 8x8 contro v14 16x8 dimezza davvero media e p95 della
search, ma perde `-0,742` punti con CI95 `-1,457..-0,027`. Lo screen fallisce prima della
conferma 20k: quel profilo non e' stato promosso. Protocollo in
`suit-student-8x8-efficiency-2026-07-17.md`.

Il punto intermedio student 12x8 passa sia lo screen da 4.000 sia la conferma indipendente
da 20.000 partite. La conferma misura `+0,1052` punti (CI95 `-0,1126..+0,3230`) con media
e p95 di latenza a circa `0,75x`, zero fallimenti e zero mosse corrette. E' un PASS di
non inferiorita' ed efficienza, non evidenza di forza superiore. Anche l'audit di
integrazione con asset reali passa:
catalogo, API, WebSocket, browser desktop/mobile e una partita completa esercitano
fallback, search e solver senza errori. Ricevute in
`suit-student-12x8-efficiency-2026-07-17.md` e
`suit-student-12x8-release-audit-2026-07-17.md`.

La decisione esplicita successiva ha confezionato lo student come `best_a2c_v15.npz` e
promosso `bc_model_pimc_belief_12x8` come default della release 0.38.0. V14 16x8 resta
selezionabile: la conclusione e' un miglior compromesso costo/forza, non una vittoria
statisticamente certa del nuovo runtime.

## 12. Prerequisito dati BC/value (completato 2026-07-14)

Lo split casuale per record e' stato sostituito da uno split deterministico per partita
in BC, value scalare e value pairwise. I trainer usano per default 80/10/10, valutano il
test solo dopo la scelta del modello e salvano conteggi + digest dell'assegnazione. I
dataset NPZ value e leaf PIMC hanno formato v2 con `game_ids`; quelli storici privi del
confine fra partite vengono rifiutati e vanno rigenerati.

Questo chiude un rischio di validation ottimistica, ma non costituisce un segnale per
riaprire il training o usare automaticamente i dati live. Dettagli e
migrazione: `dataset-split-per-partita-2026-07-14.md`.
