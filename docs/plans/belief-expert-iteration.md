# Piano storico completato: Belief → Determinizzazioni Pesate → Expert Iteration

> **Stato retrospettivo (2026-07-11): programma di ricerca concluso.** Questo documento
> conserva ipotesi, gate e risultati nel loro ordine storico; non è una roadmap corrente.
> Per le prossime azioni fa fede esclusivamente `PLAN.md`.
>
> Piano di ricerca dettagliato per la progressione del modello dopo `best_a2c_v7`.
> Fonte: indagine approfondita del 2026-07-02 (analisi encoder/training/esperimenti falliti + letteratura).
> `PLAN.md` resta la fonte di verità operativa: questo documento è l'archivio dell'ipotesi.

## Matrice conclusiva

| Fase | Stato | Decisione consolidata |
|---|---|---|
| 0 — sonde | Completata | Warm-start confermato; classe v3 quasi satura; headroom PIMC di circa +3.8. |
| 1 — encoder v4 | Completata | Storia pubblica delle prese aggiunta; 369 feature e parità dominio/fast/Numba; valore netto misurato +0.27. |
| 2 — belief | Completata | Belief valida offline; input diretto alla policy chiuso negativo; sampling PIMC pesato validato e rilasciato nel ruolo corretto. |
| 3 — Expert Iteration | Conclusa | Distillazione BC chiusa; sparring A2C a rendimento decrescente; expert full-game depth-1 chiuso; scelto il miglioramento del runtime PIMC. |
| 4 — nuova architettura | Chiusa senza esecuzione | Era una proposta condizionata al successo del loop ExIt; non è più il passo successivo di questo piano e va rivalutata dal piano corrente. |

L'esito di prodotto del programma fu `bc_model_pimc_belief_64x10` nella v0.23.0. Le
configurazioni di produzione sono cambiate in seguito: i riferimenti a quel candidato e
alla sequenza delle fasi, nel corpo sotto, descrivono il momento dell'esperimento.

## 1. Perché siamo a un punto morto (diagnosi)

Il ristagno post-v7 è **sovradeterminato**: tre soffitti indipendenti si rinforzano a vicenda. Ogni
esperimento recente (distillazione PIMC, value retrain v1 e leaf-pairwise, v7 stesso) ha sbattuto
contro almeno uno di essi.

### 1.1 Soffitto informativo (il più grave)

L'osservazione è **aggregata-markoviana**: `seen_cards_onehot` / `out_of_play_cards_onehot` sono
bitmask globali sulle 40 carte, senza ordine né attribuzione. `make_player_observation`
(`domain/observation.py`) comprime `p.captured_cards` — che nel dominio È già segmentato per
giocatore — in due bitmask, cancellando:

- la storia ordinata delle prese (chi ha giocato cosa, in che ordine);
- chi ha vinto ogni presa e quanti punti valeva;
- la sequenza comportamentale dell'avversario (es. "non ha mai tagliato a coppe"),
  che è l'informazione centrale per inferirne la mano.

Conseguenza: **l'opponent modeling comportamentale è irrappresentabile con questi input, per
qualsiasi rete di qualsiasi capacità**. Non è un vincolo anti-cheat (tutto ciò che manca è stato
pubblicamente osservato): è una perdita introdotta dall'encoding.

### 1.2 Soffitto di rappresentazione

La policy è una MLP `310 → 128 → 40` (~45k parametri, 1 hidden layer, ReLU), **identica per 5
generazioni** (v3..v7). Input flat one-hot: nessun embedding di carta, nessuna condivisione tra
semi, nessun meccanismo relazionale. Il blocco v3 dell'encoder (22 feature hand-crafted) esiste
proprio per iniettare a mano le relazioni che l'architettura non può derivare. Il value model
(`310 → 128 → 1`) condivide encoder e architettura.

### 1.3 Soffitto del segnale di apprendimento

- **Catena warm-start senza controllo**: `bc_v3 → v3 → v4 → v5 → v6 → v7`, ogni generazione
  inizializzata dalla precedente, iperparametri congelati (lr 3e-4 costante, entropy 5e-4,
  γ=1.0, update_every 20, hidden 128). Anchor CE verso `bc_v3` (β=0.01) per v3–v6: la ricerca è
  confinata nel bacino del teacher BC originale. **Non esiste alcun run from-scratch di
  confronto** a scala reale.
- **A2C Monte-Carlo puro**: advantage = return non scontato − V(s), nessun GAE/n-step, nessuna
  normalizzazione dell'advantage, nessun gradient clipping, value head lineare su trunk condiviso.
  Alta varianza: gran parte delle 5M partite/generazione paga la varianza, non l'apprendimento.
- **Avversari congelati della stessa famiglia**: v7 è allenato contro un unico opponent statico
  (`bc_model_value_lookahead_8x8` su base v6). Risultato documentato in PLAN.md: v7 batte il
  target statico ma NON supera la search (`v7+solver` vs `v6+solver` = −0.64). La ricetta ha
  saturato ciò che può estrarre **senza search nel loop di training**.
- **Nessun gating intra-run**: si promuove l'ultimo checkpoint del run, non il migliore.

### 1.4 La rilettura del fallimento della distillazione (il dato che inchioda)

Il "val acc 57%" della distillazione PIMC è misurato **sul subset di disaccordo** (PIMC ≠ base),
dove la chance è ≈50% (restano ~2 alternative plausibili): **è un coin-flip**. Evidenza raccolta:

- **capacità esclusa**: hidden=512 memorizza il train senza migliorare la val;
- **nitidezza label esclusa**: i soft-label peggiorano scaldando (T=2→56.9%, T=5→55.6%, T=10→52.8%);
- **causa primaria — non-identificabilità**: la correzione PIMC è un integrale sulle mani
  avversarie nascoste; non è una funzione dell'osservazione v3;
- **cause aggravanti**: covariate shift BC classico (dataset on-policy del base, mai del
  candidato: no DAgger) e teacher disallineato (probe teacher-v6 su base-v7 perde);
- **rumore reale ma secondario**: solo 2935/18184 record search sono strong/reliable
  (`margin≥2 ∧ ci_low≥0`); l'argmax su 16–64 determinizzazioni è instabile proprio dove serve.

Stesso schema per i value retrain (v1, leaf-pairwise): metriche offline su (medie dominate dai
root facili), A/B runtime nullo/negativo (deciso dalle poche decisioni pivotali a basso margine).

### 1.5 Il margine esiste

`PIMC(v6,16×8)+solver` batte `v6+solver` di **+3.59** (CI95 +2.48..+4.70). La search sa giocare
meglio della policy reattiva di ~3-4 punti/partita: il problema non è il tetto del gioco, è che
nessuna delle strade tentate riesce a travasare quel vantaggio nella policy.

### 1.6 Un dettaglio tecnico che sblocca la Fase 3

`pimc.py` calcola la matrice `per_determinization_scores[det][azione]` (righe ~449, ~497) ma
**non la persiste**: la diagnostica salva solo media per azione + margine best-vs-second. Per
target soft/distribuzionali affidabili serve persisterla.

## 2. Obiettivo e definizione di "progresso deciso"

- **Obiettivo primario**: una policy reattiva `.npz` (o un agente runtime a pari CPU) che superi
  `bc_model_value_lookahead_8x8` (l'attuale campione runtime) nel confronto seat-fair con CI
  su coppie — cioè assorbire nella policy il vantaggio oggi esclusivo della search.
- **Obiettivo secondario**: alzare il tetto dell'agente search stesso (belief-weighted PIMC).
- **Scala attesa**: la sonda dell'oracolo (Fase 0) fissa il target numerico. Ordine di grandezza
  ragionevole: +3..+6 punti/partita vs v7, contro i +2.3 dell'ultimo giro incrementale.
- Ogni fase ha criteri di kill espliciti: se una fase fallisce il suo gate, si ferma quel ramo,
  non si "insiste con più dati simili" (lezione dei value retrain).

## 3. Fase 0 — Sonde diagnostiche (1-2 giorni, quasi solo script esistenti)

Prima di investire, tre misure che cambiano le priorità:

### 0.a Sonda di exploitability
Allena un best-response A2C (ricetta identica, 1-2M partite) **contro v7 congelato**
(`train_a2c.py --opponent bc_model --opponent-model best_a2c_v7.npz`).
- Lettura: se `BR vs v7 >> v7 vs v6` (es. BR vince di +6 o più), il self-play converge a
  strategie sfruttabili → la diversità degli avversari (league/population o ExIt) sale di priorità.
- Costo: una notte di CPU. Nessun codice nuovo.

### 0.b Sonda dell'oracolo (tetto realistico)
`PIMC(v7, 64×10)+solver` vs `v7+solver`, 2-4k partite seat-fair (CI su coppie).
- Lettura: il margine misura quanto vale, al massimo, "portare la search nella policy" con
  l'attuale qualità di determinizzazione. Se il margine con budget 4x resta ~+3-4, quello è il
  target della Fase 3; se cresce molto, il soffitto è più alto del previsto.
- Costo: qualche ora di CPU. Nessun codice nuovo.

### 0.c Controllo from-scratch
`train_a2c.py` con ricetta v7 ma **senza `--init`** (e senza anchor), 5M partite.
- Lettura: se from-scratch ≈ v7, la catena warm-start non è il problema (e si smette di
  sospettarlo); se from-scratch < v7 di molto, il warm-start è un asset da conservare; se
  from-scratch > v7, la catena era un ottimo locale (improbabile ma va escluso).
- Costo: una notte di CPU. Nessun codice nuovo.

Gate di fase: nessuno blocca le fasi successive; i risultati **orientano** (0.a → quanta
diversità di avversari serve; 0.b → target numerico; 0.c → init della Fase 4).

### Risultati Fase 0 (eseguita 2026-07-02)

Nota di costo: grazie al throughput Numba (~4.150 partite/s per il training, 0,28 s/partita per
PIMC 64×10) l'intera fase è costata **~45 minuti di CPU**, non "notti". Le stime di effort delle
fasi successive vanno lette di conseguenza: il collo di bottiglia di ExIt sarà la search del
teacher (~0,08 s/decisione search), non il training.

**0.a Exploitability — v7 è quasi inespugnabile nella sua classe.**
Best-response A2C (2M partite, init=v7, opponent=v7 congelato, seed 20260702) vs v7,
10k seat-fair: **+0.70** (CI95 coppie +0.22..+1.19), score rate 0.511.
Un avversario allenato SPECIFICAMENTE contro v7 lo batte di meno di un terzo del margine
v7-su-v6 (+2.46). Lettura: il self-play non sta convergendo a strategie sfruttabili;
league/population play NON è la leva; la classe "MLP 128×1 su encoder v3" è satura attorno a v7.

**0.b Tetto oracolo — la search vale ~+3.8, e satura con il budget.**
`PIMC(v7, 64×10)+solver` vs `v7+solver`, 4000 partite seat-fair: **+3.76**
(CI95 coppie +3.40..+4.12), score rate 0.564. Con budget 4× rispetto al runtime (16×8) il
margine cresce solo marginalmente rispetto al +3.59 storico → la search a determinizzazioni
UNIFORMI satura intorno a +3.8: per alzare il tetto serve migliorare la qualità delle
determinizzazioni (belief, Fase 2), non il loro numero. Dettaglio: le mosse search sono il
17,5% delle decisioni (14k/80k) — tutto il vantaggio vive nella finestra di fine partita.
Artefatto: `benchmarks/experiments/fase0/oracle_pimc_v7_64x10_vs_control_4k.json` (locale).

**0.c From-scratch — la catena warm-start è un asset, non una trappola.**
Ricetta v7 identica (opponent value-lookahead, 5M partite, seed 20260702) ma init casuale:
vs v7 = **−5.42** (CI95 coppie −5.91..−4.92); vs `heuristic_v1` = +12.58 (v7 fa +18.73).
A parità di budget il from-scratch arriva a livello ~v2/v3. Lettura: il valore accumulato nella
catena è reale; non c'è evidenza di ottimo locale indotto dal warm-start. Implicazioni:
ExIt (Fase 3) parte da `policy_0 = v7`; l'eventuale from-scratch profondo della Fase 4 richiederà
più budget e/o curriculum, non è gratis.

**Sintesi Fase 0**: le tre sonde convergono sulla stessa conclusione del piano — l'unico
margine dimostrato (+3.8) vive nella search ed è irraggiungibile dall'interno della classe
reattiva attuale (0.a); il budget di search non lo alza (0.b); e non c'è scorciatoia
"ripartire da zero" (0.c). Restano esattamente le leve delle Fasi 1-3: informazione (encoder
v4), qualità delle determinizzazioni (belief) e trasferimento iterato (ExIt).

## 4. Fase 1 — Encoder v4: restituire la storia (prerequisito di tutto)

### 4.1 Modifiche a `PlayerObservation` (dominio)

Aggiungere campi **pubblicamente osservabili** (anti-cheat per costruzione):

- `trick_history`: tupla ordinata di prese completate, ognuna
  `(cards_in_order: tuple[(card_id, player_index)], winner_index, points)`;
- eventualmente `draw_order_known`: nulla di nuovo da esporre (l'ordine di pesca non è pubblico),
  esplicitamente NON incluso.

Nota di fattibilità: `GameState.players[i].captured_cards` conserva già le carte per giocatore;
serve arricchire `engine.step` perché registri la presa completa nell'ordine di gioco (oggi
l'ordine intra-presa è ricostruibile solo dal tavolo corrente). Aggiornare serializzazione
(bump `SERIALIZATION_SCHEMA`), fast path e test di parità.

### 4.2 Feature v4 (`observation_encoder.py`)

Blocco aggiuntivo rispetto a v3 (dimensioni indicative):

- **Conteggi comportamentali per-avversario** (il cuore): per ogni seme, quante carte l'avversario
  ha giocato di quel seme; quante volte ha tagliato di briscola; quante volte ha risposto al seme
  di uscita vs scartato; punti che ha ceduto su lead bassi (~16-24 feature);
- **Ultime K prese** (K=3-5): per presa, one-hot compatti di (carta lead, carta risposta, chi ha
  vinto, punti/11) (~K×(8-12) feature);
- **Traiettoria punti**: punti per presa recenti invece del solo cumulativo (~K feature);
- fix igiene v1: normalizzare i blocchi `hand_points`/`hand_strength` (oggi grezzi 0-11 contro
  scalari /120 — asimmetria di scala documentata).

Vincoli: parità dict/oggetto (come v2/v3), traduzione nel fast path + kernel Numba
(`numba/observation.py`), test di parità nuovi. Il contratto `feature_dim_for_encoder_version`
propaga v4 automaticamente a policy e value model.

### 4.3 Gate di fase

- Retrain BC/A2C **a parità di tutto il resto** (128×1, ricetta v7) su encoder v4:
  atteso un guadagno anche piccolo ma positivo vs v7 (holdout + CI su coppie).
- Kill: se v4 a parità di ricetta è ≤ v3 E la belief (Fase 2) su v4 non supera nettamente la
  belief su v3, il design delle feature va rivisto prima di procedere.
- Stima effort: 3-5 sessioni di lavoro (dominio+engine, encoder, fast/numba, test parità).

### 4.4 Stato Fase 1 (2026-07-02): stadio dominio+encoder COMPLETATO

Implementato e testato (431 test verdi):

- **Dominio**: `TrickRecord` (carte in ordine di gioco + vincitore + punti), `GameState.trick_history`
  popolato da `engine.step` a ogni presa completata; serializzazione **schema 2** con migrazione
  tollerante dei dump legacy schema 1 (storia vuota, stato di gioco corretto);
- **Osservazione**: `PlayerObservation.trick_history` (pubblica per costruzione, default vuoto
  per compatibilità) + `TrickRecordDTO` in `ObservationDTO` (additivo, client vecchi lo ignorano);
- **Encoder v4** (`FEATURE_DIM_2P_V4 = 369 = v3 + 59`): blocco A comportamento avversario
  (per-seme, lead, tagli, risposte a seme, scarti, lead briscola/carichi, prese completate)
  + blocco B ultime 4 prese (12 feature l'una); path dict e oggetto con parità verificata su
  partite reali; degradazione a zero su dataset pre-v4 (come v2/v3);
- riconoscimento v4 in `bc_model` e `value_model` (feature_dim 369): **gli agenti runtime
  (BCModelAgent, PIMC, value-lookahead, backend) possono già usare modelli v4** — girano sul
  path dominio, che è completo.

**Scelta di sequenza (deliberata)**: il path Numba v4 (encoder arrays + tracking incrementale
della storia nei kernel di rollout) è rinviato a quando la Fase 3 richiederà il training A2C
su v4. La Fase 2 (belief + determinizzazioni pesate) gira interamente sul path dominio, quindi
NON è bloccata; il gate 4.3 "A2C a parità su v4" si sposta di conseguenza dopo il kernel Numba
(oppure si valuta prima via BC su dataset v4, che non richiede Numba).

Igiene rinviata con consapevolezza: la normalizzazione dei blocchi grezzi v1
(`hand_points`/`hand_strength` non scalati) non è stata toccata — gli encoder sono cumulativi
e cambiare v1 dentro v4 romperebbe la proprietà di prefisso; se ne riparla con un eventuale
encoder v5 non-cumulativo (es. per l'architettura della Fase 4).

## 5. Fase 2 — Belief network e determinizzazioni pesate

### 5.1 La rete belief

`B(observation_v4) → P(carta c in mano avversaria)` per le 40 carte (sigmoid multi-label,
mascherata sulle carte non-ignote: per le carte in `seen`/mano propria la label è nota).

- **Dataset**: self-play numerico (riuso di `generate_value_dataset_numba`-style): input =
  osservazione, label = mano avversaria vera dal full-state. Lecito: il full-state è usato SOLO
  per le label a training; a inference la rete vede solo l'osservazione.
- **Architettura iniziale**: stessa MLP (v4_dim → 128/256 → 40) per coerenza con l'infrastruttura.
- **Gate offline**: log-loss e top-k recall vs baseline uniforme-sulle-ignote. Se la belief non
  batte nettamente l'uniforme, l'encoder v4 non porta segnale comportamentale → tornare a 4.3.

### 5.2 Uso 1 — Determinizzazioni pesate (attacca subito il campione runtime)

Oggi `determinize_observation` campiona le mani avversarie **uniformemente** tra le carte ignote.
Sostituire con campionamento ∝ belief (senza rimpiazzo, rinormalizzando), in PIMC e
value-lookahead. Precedente diretto in letteratura: *Policy Based Inference in Trick-Taking Card
Games* (Rebstock, Solinas, Buro, Sturtevant 2019, arXiv:1905.10911) migliora Kermit (SOTA Skat)
esattamente con questo meccanismo.

- **Gate runtime**: `PIMC-belief(16×8)+solver` vs `PIMC-uniforme(16×8)+solver`, 4k+ seat-fair,
  CI su coppie. Idem per value-lookahead.
- Kill: se il pesatura non migliora (CI compatibile con zero) con belief offline sana, indagare
  la calibrazione della belief prima di buttare l'idea (il campionamento è sensibile alle code).
- Stima effort: 2-3 sessioni (dataset+training belief, sampling pesato, eval).

### Risultati 5.1-5.2 (eseguiti 2026-07-02)

**Gate offline: SUPERATO nettamente.** `belief_v0` (h128, dataset 50k partite self-play v7
ε=0.05, 350k record in finestra u≤10, encoder v4): `val_bce 0.492` vs `0.612` uniforme;
`top-k recall 0.593` vs `0.399` uniforme. La rete legge davvero il comportamento avversario
dalle feature v4. Latenza a runtime: invariata (0.0136 vs 0.0135 s/mossa search).

**Gate runtime (uso 1, determinizzazioni pesate): NEGATIVO per il deploy, con spiegazione.**
Head-to-head 4000 partite seat-fair (CI su coppie), base v7:

- belief 16×8 vs uniforme 16×8 (config campione): `+0.15` (CI `−0.08..+0.39`) — nullo;
- belief 16×12 vs uniforme 16×12: `+0.56` (CI `+0.15..+0.97`) — POSITIVO: la belief migliora
  davvero il campionamento dove il pool di carte ignote è grande (non è un problema di
  calibrazione);
- belief 16×12 vs uniforme 16×8: `−0.23` (CI `−0.62..+0.17`) — allargare la finestra resta
  un netto svantaggio (rumore dei rollout più lunghi) che la belief compensa solo in parte.

Diagnosi strutturale: nella finestra di deploy (u≤8) la mano avversaria sono 3 carte da un
pool ≤8 (≤56 combinazioni): 16 determinizzazioni uniformi coprono già lo spazio, e il valore
marginale dell'inferenza è piccolo. L'inferenza varrebbe di più a inizio partita (pool 15-25),
ma lì la search non opera affatto (fallback alla policy): è il limite architetturale già noto
("il vantaggio della search vive tutto nel finale").

**Decisione**:
- NON promuovere un agente `pimc_belief`: nessun guadagno nella config di deploy;
- TENERE `belief_v0` e l'infrastruttura (determinizzazioni pesate restano disponibili via
  `card_weights`): il gate offline è superato con margine e l'effetto a u=12 dimostra che
  l'inferenza è reale;
- il valore della belief si sposta sull'**uso 2** (input/auxiliary per la policy, §5.3) e
  sulla **Fase 3**: è lì che l'inferenza può agire su TUTTA la partita, non solo nella
  finestra di search. Questo è coerente con la diagnosi originale: il soffitto della policy
  è informativo, e la belief è il condensato dell'informazione mancante.
- Artefatti: `benchmarks/experiments/fase2/*.json` (locali), `data/models/belief_v0_h128_50k_seed20260702.npz`.

### 5.3 Uso 2 — Input/auxiliary per la policy

- Concatenare l'output della belief (40 valori) alle feature della policy, oppure
- testa ausiliaria belief nel training A2C (loss ausiliaria che plasma il trunk condiviso).
- Gate: A2C a parità di ricetta con/senza belief input, holdout + CI coppie.

## 6. Fase 3 — Expert Iteration (il volano)

Sostituire la distillazione one-shot con il loop iterato (Anthony/Tian/Barber, NIPS 2017 — il
principio di AlphaZero):

```
policy_0 = v7 (o from-scratch v4, decisione informata dalla Fase 0.c)
repeat k:
  expert_k   = PIMC(policy_k, belief_k) + solver         # più forte della policy per costruzione
  dataset_k  = partite AVANZATE DA expert_k (o dal candidato: DAgger)   # niente covariate shift
               target = distribuzione soft dagli action_values per determinizzazione
  policy_k+1 = train su dataset_k (BC/A2C misto) partendo da policy_k
  belief_k+1 = retrain belief su self-play di policy_k+1
  gate_k     = policy_k+1 vs policy_k E expert_k+1 vs expert_k (seat-fair, CI coppie)
```

Correzioni specifiche ai fallimenti documentati:

1. **Covariate shift** → gli stati vengono dalle partite dell'expert/candidato, non del base
   (`advance_with_teacher=True`, già supportato da `generate_pimc_teacher_dataset.py`).
2. **Teacher disallineato** → il teacher è SEMPRE la policy corrente (mai v6 su base v7).
3. **Rumore label** → persistere `per_determinization_scores` in `PIMCSearchDiagnostics`
   (modifica a `pimc.py`); budget adattivo: più determinizzazioni solo dove il margine è
   incerto; target soft dalla distribuzione, pesati per affidabilità.
4. **Non-identificabilità** → mitigata da encoder v4 + belief input (Fasi 1-2): il target
   diventa molto più funzione dell'input osservabile.

**Design dell'expert — estendere il ragionamento a TUTTA la partita (nota 2026-07-02).**
L'expert PIMC attuale giudica ogni ipotesi giocando la partita fino in fondo ("rollout"):
affidabile solo nel finale (simulazioni corte, poco rumore), inutilizzabile a inizio partita
(35 mosse e 17 pescate di rumore sommergono il segnale — confermato: allargare la finestra
peggiora, −0.23 anche con belief). La via per un expert a tutta partita è sostituire il
rollout con **lookahead corta + value network di foglia** (lo schema motori-di-scacchi /
AlphaZero, e la generalizzazione del value-lookahead esistente):
- la belief (Fase 2) fornisce determinizzazioni verosimili proprio dove le ignote sono tante;
- serve però un **value model allenato su tutte le fasi di partita** su encoder v4(+belief):
  `value_v0` è tarato sulla finestra finale. È il primo upgrade da fare dentro il loop ExIt
  (il value si riallena comunque a ogni iterazione).

- **Gate per iterazione**: entrambe le curve (policy e expert) devono salire; 2 iterazioni
  consecutive piatte = stop e analisi.

### Risultati iterazione-0 (2026-07-02): BC NON è l'operatore di improvement giusto

Setup: dataset expert 200k record (PIMC(v7, d=64, u=8)+solver, partite AVANZATE dal teacher,
25k decisioni search di cui 11.276 correzioni a v7 e 4.411 strong/reliable; 145k coverage v7;
30k solver), osservazioni v4 (trick_history nel DTO). `train_bc` esteso con encoder v4 e
upgrade warm-start v3→v4 (padding a zero: init ≡ v7). Gate: 10k seat-fair vs v7 (CI coppie).

| Variante | vs v7 |
|---|---|
| hard, warm-start v7, lr 3e-4 | **−2.33** (CI −2.73..−1.94) |
| soft T=2, warm-start v7 | **−9.19** (i target sfumati distruggono la policy) |
| hard + anchor ai logit v7 (β=0.1, lr 1e-4) | **−1.52** (meglio, ma perde) |
| hard, hidden=512 from scratch | **−3.71** (train 94% / val 81%: memorizza) |

Diagnosi (coerente con §1.4 e col fallimento storico su v3, che ora si spiega meglio):
la cross-entropy su label argmax è un operatore LOSSY su una policy raffinata da RL.
La forza di v7 non è nel suo argmax ma nella struttura fine dei logit tarata da 5M partite
a reward; ri-fittarla da 200k campioni argmax perde più fedeltà sulla copertura (val acc
satura a ~85% col warm-start: underfitting alla capacità 45k; memorizzazione a 512) di
quanto le ~11k correzioni aggiungano. L'anchor ai logit mitiga ma non inverte il segno.

Decisione:
- **kill del ramo "improvement via BC"** (in tutte le forme provate: hard/soft/anchored/capacità);
- l'operatore di improvement che HA già funzionato in questo progetto è **A2C con l'expert
  come avversario** (è esattamente come è nato v7: +2.3 su v6 allenando contro
  value-lookahead(v6)). L'iterazione-1 naturale è quindi: A2C(policy v4+belief) contro
  expert(v7) congelato — che richiede il **kernel Numba v4** (encoder + tracking storia nei
  rollout JIT). Il pezzo di infrastruttura rinviato in Fase 1 è ora il percorso critico;
- il dataset expert e il padding v3→v4 restano asset riusabili (l'expert-as-opponent li
  riutilizzerà per eval e per eventuali auxiliary loss a peso basso).
- Artefatti: `data/exit/iter0_models/*` (locali). Il dataset `iter0_teacher_v7_d64u8_200k.jsonl` (798MB) è stato rimosso nella pulizia del 2026-07-03: rigenerabile con `generate_pimc_teacher_dataset.py` (ricetta nei commit).

### Sequenza delle iterazioni (archivio della decisione del 2026-07-02)

**Kernel Numba v4: COMPLETATO (2026-07-02).** Storia delle prese mantenuta in tutti i loop
JIT (play/quality/collector/value-dataset; simulazioni determinizzate con storia dummy:
i modelli interni restano ≤ v3), `_v4_block_numba` replica esatta dell'helper di dominio,
parità a tre motori (dominio↔fast↔numba) protetta da test su partite specchiate a ogni
profondità. La policy in training può essere v4; eval numba accetta modelli v4 (gate veloci).
Lezione operativa: la cache numba locale può mascherare firme rotte — la CI (compilazione
fredda) resta il gate di verità per i kernel.

**Iterazione 1 — occhi nuovi a taglia FISSA.** A2C contro expert(v7) congelato
(value-lookahead/PIMC su base v7), policy warm-start da `best_a2c_v7_padded_v4.npz`
(hidden 128 invariato; input 369 v4, opzionalmente +40 belief). La taglia resta quella di v7
PER SCELTA SPERIMENTALE: la domanda dell'iterazione è "quanto vale la nuova informazione a
parità di tutto il resto?" — cambiare informazione e capacità insieme renderebbe il risultato
non interpretabile (una variabile alla volta). Prerequisito: kernel Numba v4
(encoder + tracking incrementale storia nei rollout; belief forward opzionale).

**Risultati iterazione 1 (2026-07-02).** Run: A2C 5M partite, seed 20260713, opponent
`value_lookahead(v7)` congelato; braccio v4 (init v7-padded) e braccio di controllo v3
(init v7), identici in tutto tranne l'encoder. Gate big holdout 100k, CI su coppie:

| Confronto | Esito |
|---|---|
| iter1-v4 vs v7 | **+0.72** (CI +0.56..+0.87) |
| controllo-v3 vs v7 | **+0.44** (CI +0.28..+0.59) |
| iter1-v4 vs controllo-v3 (testa a testa) | **+0.27** (CI +0.12..+0.42) |

Letture:
1. **Le feature v4 hanno valore misurabile nella policy: +0.27 netto a parità di tutto.**
   È la prima evidenza runtime positiva dell'intero programma encoder (la belief nel PIMC
   era neutra nel deploy; qui la storia delle prese paga direttamente). L'investimento
   Fase 1 è validato.
2. **L'operatore sparring mostra rendimenti fortemente decrescenti**: la stessa ricetta
   che diede +2.46 (v7 vs expert-su-v6) ora rende +0.44 (vs expert-su-v7). Atteso: più la
   policy si avvicina al livello dell'expert (~+2-3), più il segnale di sparring si
   assottiglia. L'expert deve migliorare perché il loop continui a rendere (→ expert a
   tutta partita, §6 nota; e/o belief-weighted dove serve).
3. iter1 (+0.72 su v7) NON supera il campione runtime `value_lookahead(v7)`: obiettivo
   primario non raggiunto in una iterazione, come previsto dal design iterativo.

Le leve allora ordinate per rapporto valore/costo erano:
- **iterazione 1b — belief come input della policy** (40 feature in più): richiede il
  forward della belief nei kernel di rollout; è l'informazione mancante più promettente
  (l'inferenza vale nelle prime 15 prese, dove la policy gioca da sola);
- **iterazione 2 — net2net widening** (128→256/512): banale nei kernel (hidden è un
  parametro), serve solo lo script di widening; testa se il +0.27 è strozzato dalla capacità;
- **expert migliore**: value model a tutta partita su v4 per alzare il teacher (§6 nota).

Artefatti: `data/models/exit_iter1_*`, `benchmarks/experiments/fase3/*.json` (locali).

**Risultati iterazione 2 / braccio capacità (2026-07-02).** Net2Net widening 128→256
(`scripts/widen_mlp_net2net.py`, funzione preservata esattamente: 200/200 scelte invariate
sul modello reale) + 5M partite stessa ricetta. Gate big 100k: iter2-h256 vs iter1 **+0.18**
(CI +0.03..+0.33); vs v7 **+0.89** (CI +0.74..+1.05). Il +0.18 confonde capacità e un
ulteriore giro di operatore: il contributo marginale della capacità è ~0. **La capacità
non è il collo di bottiglia.**

**Risultati iterazione 1b / belief come input della policy (2026-07-02).** Input 409 =
v4 + 40 probabilità belief congelata (kernel `_policy_input_v4_belief_numba`, artefatto
self-contained), init iter1 con pad a zero (start identico), 5M stessa ricetta. Gate
medium 10k (engine domain): iter1b vs iter1 **−0.56** (CI −1.05..−0.06) — NEGATIVO;
vs v7 +0.82 (residuo dell'istinto di iter1). **Kill della belief-as-policy-input**, con
diagnosi: la belief è una funzione deterministica delle stesse feature v4 che la policy
già riceve — come input NON aggiunge informazione, solo una base ridondante, e il
gradiente che la insegue erode l'istinto accumulato. Dove la belief aggiunge valore vero
è dentro l'EXPERT (pesare le determinizzazioni: un'operazione che la policy non può
replicare strutturalmente, §5).

**Bilancio leve lato allievo (tutte misurate, 2026-07-02):** feature v4 **+0.27 netto**
(reale, validata); capacità 256 **~0 marginale**; belief input **−0.56** (dannosa).
Il vincolo attivo è il MAESTRO: la prossima leva è l'expert a tutta partita
(value model v4 + lookahead dalla prima carta, nota §6), eventualmente con belief-weighted
determinizations, per ridare pendenza al segnale di sparring.

**Risultati capitolo "expert a tutta partita" (2026-07-03) — NEGATIVO, con curva dose-risposta.**
Costruito e misurato: value model v4 full-game (dataset 50k partite × tutte le fasi, storia
reale anche nelle etichette; MAE 14.1 vs baseline 16.2, sign 0.743), storia reale nelle
simulazioni determinizzate del V-lookahead (kernel + runtime + determinize), finestra
estendibile a tutta partita. Gate vs controllo v8+solver (400 partite, det 16):

| Finestra (unknown) | Esito |
|---|---|
| 8 | **+1.80** (CI +0.86..+2.75) |
| 12 | +0.57 n.s. |
| 16 | −0.88 n.s. |
| 22 | −2.56 |
| 34 (tutta partita) | **−5.16** (CI −7.80..−2.51) |
| 8 con value_v0 (v3) | +1.78 — identico a v1 |

Diagnosi: a inizio/metà partita il valore di una posizione è dominato dal caso del mazzo
(MAE ~14 punti è irriducibile lì); il depth-1 argmax su V rumorosa amplifica il rumore e
perde dall'istinto della policy. Il V-lookahead funziona SOLO dove V è affidabile (finestra
endgame), e lì le feature v4 non aggiungono nulla (la storia conta poco quando le carte
viste dicono già quasi tutto). **Kill dell'expert full-game via value depth-1.**

Conseguenza strategica: l'edge del maestro VL sul suo allievo si restringe per costruzione
(+2.27 su v6 → +1.80 su v8): lo sparring contro VL(v8) comprerebbe ~+0.2-0.4. L'headroom
DIMOSTRATO residuo è l'oracle PIMC 64×10 (+3.76, Fase 0.b): search a ROLLOUT (media sul
rumore invece di fidarsi di una stima V), costosa. Strade per un eventuale prossimo ciclo:
(a) PIMC-as-teacher nei kernel numba (rollout veloci esistono, serve l'orchestrazione);
(b) investire sulla search RUNTIME del prodotto (l'avversario avanzato che i giocatori
affrontano) invece che sul training della policy. Nota positiva collaterale: la storia
reale nelle simulazioni VL rimuove la nota "base smemorata" della promozione v8.

**Sonde search-a-rollout su base v8 (2026-07-03) — la strada (b) è validata.**
PIMC vs controllo v8+solver (400 partite, seed 42, ~73 ms/mossa pensata):

| Config | Esito |
|---|---|
| PIMC 64×10 uniforme | +3.03 (CI +1.98..+4.08) |
| **PIMC 64×10 belief-weighted** | **+3.83 (CI +2.74..+4.92)** |
| PIMC 64×14 belief | +3.33 (finestra più larga non paga) |
| (riferimento: VL 16×8, avversario avanzato attuale) | +1.80 |

La belief vale ~+0.8 nel suo posto naturale (pesare le determinizzazioni), e il rollout
degrada con grazia dove la V collassava. Candidato prodotto: **avversario avanzato
PIMC 64×10 belief su base v8**.

**CONFERMATO e RILASCIATO (v0.23.0, 2026-07-03).** Conferme su 4.000 partite seat-fair,
stessi semi per entrambi i candidati vs controllo v8+solver:
- `bc_model_pimc_belief_64x10`: **+3.66** (CI +3.32..+4.00), 0.267 s/partita;
- VL 16×8 (avanzato precedente): +2.12 (CI +1.83..+2.41).
CI disgiunte: il nuovo avanzato è più forte di ~+1.5 sull'attuale. In produzione come
opzione selezionabile (`bc_model_pimc_belief_64x10`), belief provisionata via
`BRISCOLA_BELIEF_MODEL_URL/SHA256`. La belief network è così arrivata in produzione
nel suo ruolo giusto: pesare i mondi simulati, non imboccare la policy.

**Iterazione 2 — capacità, con warm start preservato.** Se l'iterazione 1 è positiva ma
modesta, si allarga l'hidden (256/512) SENZA ripartire da zero (la Fase 0.c mostra che
from-scratch costa −5.4): widening **net2net** (clonazione dei neuroni esistenti + rumore
simmetrico), che riproduce esattamente la funzione del modello di partenza. Allargare un
1-hidden-layer è banale nei kernel (è un parametro); approfondire (2-3 layer, embedding)
resta Fase 4.

**Nuance importante (da non perdere):** "hidden=512 memorizza" è dimostrato SOLO per la
copia supervisionata (BC/distillazione: iterazione-0 e probe storico). La capacità extra
con apprendimento dal GIOCO (A2C, milioni di partite a reward) non è mai stata testata in
questo progetto — ed è il regime in cui la letteratura (es. DouZero, ~50x i nostri parametri)
mostra che paga. Non usare il risultato BC come argomento contro la capacità in RL.
- **Criterio di successo del piano**: `policy_K` (reattiva) ≥ `bc_model_value_lookahead_8x8`
  attuale, oppure `expert_K` > campione attuale di un margine ≥ sonda oracolo/2.
- Stima effort: 3-4 sessioni per l'harness del loop (molti pezzi esistono già:
  teacher dataset, soft-label in `train_bc`, warm-start) + run multi-notte.

## 7. Fase 4 — Architettura (proposta non avviata)

Questa fase non fu avviata dentro il programma: il loop ExIt si fermò sui gate precedenti.
Il disegno conservato per archivio era:

- MLP più profonda (2-3 hidden, 256-512) con **embedding di carta** condivisi (40×d) e
  aggregazione delle feature per-carta, opzionalmente encoder della storia (stile DouZero:
  LSTM/attenzione sulle giocate — DouZero, ICML 2021, arXiv:2106.06135, ha dominato DouDizhu
  con Deep MC + storia encodata, from scratch).
- **Costo reale da pianificare**: i kernel Numba (`numba/observation.py`, `numba/mlp.py`)
  hardcodano il forward a 1 hidden layer; servono kernel generalizzati (o un forward a profondità
  fissa 2-3). Da fare una volta, riusato ovunque.
- Gate: from-scratch v4-deep dentro ExIt vs v4-128×1 dentro ExIt, a parità di budget partite.

## 8. Cosa NON fare (kill già decisi)

- **Niente altri fine-tune del value model** su dataset simili: tre fallimenti (v1,
  leaf-pairwise v6 e v7) con causa strutturale ora identificata (gate offline dominato dai casi
  facili; runtime deciso dalle code).
- **Niente v8 warm-start contro avversario congelato**: v7 dimostra che si impara a battere il
  target statico senza superare la search.
- **Niente aumento di capacità sulle feature v3**: escluso sperimentalmente (hidden=512).
- **Niente lavoro sul 4-player o su PPO/GAE** finché questo asse non è concluso (PPO resta
  l'ottimizzazione della Fase 3/4 SE la varianza si rivela il collo di bottiglia, non prima).

## 9. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Encoder v4 rompe la parità fast/numba | test di parità nuovi PRIMA del training (pattern test-àncora già in uso) |
| Belief mal calibrata peggiora il sampling | gate offline con calibrazione (reliability diagram), fallback misto uniforme+belief (λ) |
| ExIt amplifica bias della search (strategy fusion/non-locality di PIMC — Long et al., AAAI 2010) | la belief riduce l'errore di determinizzazione; monitorare con decision-quality; valutare idee EPIMC (arXiv:2408.02380) solo se emerge il problema |
| Costo CPU del loop | budget per iterazione fisso (es. 200-500k stati teacher); il collo è la search, non il training |
| Regressione stile di gioco (overkill ecc.) | gate decision-quality vs `heuristic_v1` già standard per ogni promozione |

## 10. Riferimenti

- R. Long, N. Sturtevant, M. Buro, M. Bowling — *Understanding the Success of Perfect Information
  Monte Carlo Sampling in Game Tree Search* (AAAI 2010) — limiti PIMC: strategy fusion, non-locality.
- D. Rebstock, C. Solinas, M. Buro, N. Sturtevant — *Policy Based Inference in Trick-Taking Card
  Games* (CoG 2019, arXiv:1905.10911) — determinizzazioni pesate da un modello dell'avversario
  migliorano il SOTA Skat.
- D. Rebstock, C. Solinas, M. Buro — *Learning Policies from Human Data for Skat*
  (arXiv:1905.10907).
- T. Anthony, Z. Tian, D. Barber — *Thinking Fast and Slow with Deep Learning and Tree Search*
  (Expert Iteration, NIPS 2017) — il loop policy↔search iterato vs distillazione one-shot.
- D. Zha et al. — *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*
  (ICML 2021, arXiv:2106.06135) — Deep MC + storia encodata, from scratch, gioco di carte.
- J. Cotarelo et al. — *Perfect Information Monte Carlo with Postponing Reasoning*
  (EPIMC, arXiv:2408.02380) — mitigazione strategy fusion.

## 11. Sequenza operativa originale (archivio)

La lista seguente era il piano ex ante. Gli esiti nella matrice conclusiva la sostituiscono;
non va usata come coda di lavoro corrente.

1. Fase 0 (a+b+c in parallelo, notti CPU) → fissa target e priorità.
2. Fase 1 encoder v4 → gate 4.3.
3. Fase 2 belief + determinizzazioni pesate → primo artefatto promuovibile
   (`bc_model_pimc_belief_*` come nuovo campione runtime).
4. Fase 3 ExIt → l'obiettivo primario.
5. Fase 4 architettura dentro ExIt, solo se/quando il loop satura per capacità.

Le promozioni del programma hanno seguito i criteri standard (seat-fair con CI su coppie,
holdout non peggiore, decision-quality, report modelli). Per i criteri e le azioni correnti
fa fede `PLAN.md`.
