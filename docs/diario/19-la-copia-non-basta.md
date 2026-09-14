# Approfondimento - La copia non basta

**Approfondimento del diario:** [Capitolo 18](https://ai.briscola.dev/diario) - **Periodo:** 11 luglio 2026

## Dalla sonda all'ablation

La sonda sui semi aveva trovato una proprietà misurabile: v13 cambia argmax nel 18,19%
delle rinomine semanticamente equivalenti. La prima ipotesi causale era una data
augmentation paired: nello stesso update A2C, affiancare alla traiettoria originale una
copia trasformata con una delle 23 permutazioni non identità.

L'implementazione trasforma feature v1-v4, action mask e action id con una sola
permutazione per traiettoria. Reward e return restano invariati. La media del loss usa gli
step originali e quelli copiati, quindi la scala non raddoppia. Un RNG separato evita di
cambiare mazzi, azioni e avversari del ramo originale.

## Controlli prima del training

I test coprono quattro invarianti:

1. per tutte le 24 permutazioni e gli encoder v1-v4, il vettore trasformato coincide con
   `PlayerObservation -> permute -> encode`;
2. trasformazione e inversa ricostruiscono esattamente batch, mask e action id;
3. originale+copia identità, mediati su `2N`, producono lo stesso gradiente
   dell'originale mediato su `N`;
4. default storico e `--suit-augmentation off` producono array e metadata identici.

Il paired è disponibile solo per input v1-v4. Una policy `v4+belief` da 409 feature viene
rifiutata: le 40 probabilità belief embedded richiedono un contratto di permutazione
separato e verificato.

## Esperimento controllato

Sono stati allenati tre controlli e tre trattamenti per 10.000 partite ciascuno, con seed
`20260711..20260713`. Tutti partono da v13 e conservano ricetta, BC anchor v11, reward
shaping, seat alternata e mix di avversari. Cambia soltanto
`--suit-augmentation paired`.

| ramo medio su 3 seed | flip | JS media | stati con flip |
|---|---:|---:|---:|
| controllo | 18,32% | 0,14124 bit | 51,20% |
| paired | 18,84% | 0,14575 bit | 52,28% |
| paired - controllo | **+0,53 pp** | **+0,00451 bit** | **+1,07 pp** |

Ogni modello è poi stato valutato su 10.000 partite seat-fair della suite medium. Il
paired perde in media `-0,15` punti/partita contro il controllo dello stesso seed. I
controlli restano sostanzialmente pari a v13 (`-0,01`), mentre i paired fanno `-0,33`
contro v13. Le differenze sono piccole, ma non esiste un segnale favorevole sulla metrica
primaria che giustifichi gate più costosi.

## Interpretazione

Nel supervised learning si può duplicare liberamente un esempio se la trasformazione
preserva l'etichetta. Il policy gradient ha un vincolo ulteriore: l'azione dovrebbe essere
campionata dalla stessa distribuzione di policy di cui si sta stimando il gradiente. Qui
l'azione copiata proviene dalla policy sull'osservazione originale, non da quella sulla
versione rinominata. Proprio perché v13 è asimmetrica, le due distribuzioni differiscono.

Questa osservazione non dimostra da sola la causa del risultato, ma rende la formulazione
paired v0 un'approssimazione off-policy. Un run più lungo potrebbe teoricamente cambiare
segno; i tre seed brevi dicono però che non merita il costo senza una formulazione più
diretta.

## Decisione

**STOP** al paired A2C v0: nessuna promozione e nessun training lungo. Il flag resta
spento per default per riproducibilità sperimentale.

La prossima ablation separerà i due obiettivi. Il loss A2C resterà on-policy sull'esperienza
originale. Un termine di consistency confronterà invece la distribuzione originale,
fermata come target, con quella della copia dopo il riallineamento delle 40 azioni. Prima
di allenare, un test su batch congelato dovrà mostrare che un update riduce davvero la
divergenza.

Protocollo, comandi, CI per seed e hash degli artefatti:
[`docs/plans/suit-augmentation-paired-v0-2026-07-11.md`](../plans/suit-augmentation-paired-v0-2026-07-11.md).
Evidenza sintetica:
[`suit_augmentation_paired_v0.v1.json`](../reports/evidence/suit_augmentation_paired_v0.v1.json).

## Esito del tentativo successivo

La consistency loss separata è stata implementata e provata con beta `0.001`, `0.01` e
`0.1`, sempre su tre seed da 10.000 partite. Il controllo `beta=0` produce un artefatto
identico al trainer precedente. La risposta è dose-dipendente: il flip medio scende a
`18,00%`, `17,42%` e infine **`15,64%`**; con beta `0.1` la JS media scende da `0,14124`
a `0,10402` bit.

Sul gioco non emerge ancora un aumento dimostrato. Beta `0.1` fa `+0,27` punti/partita
contro i controlli brevi dello stesso seed, ma nel confronto diretto contro v13 fa in
media `-0,08`; tutte le CI dei tre seed includono zero. Il risultato corretto è quindi:
**coerenza migliorata, forza conservata, promozione non autorizzata**.

Il prossimo gate è un run intermedio da 50.000 partite con checkpoint. Report completo:
[`docs/plans/suit-consistency-v0-2026-07-11.md`](../plans/suit-consistency-v0-2026-07-11.md).

## Esito del run intermedio

Il risultato a 50.000 partite impedisce la promozione. A 30k il flip medio resta quasi
fermo a `15,46%` e la forza è ancora pari a v13. A 50k la JS scende ulteriormente, ma il
flip risale a `16,47%` e tutti e tre i seed perdono forza: `-0,77` punti/partita medi.

La rete sta rendendo le probabilità più vicine soprattutto diventando meno decisa. Il
margine medio fra prima e seconda carta scende da circa `0,93` in v13 a `0,80`; così una
piccola differenza residua può cambiare più facilmente l'argmax. La prossima loss dovrà
proteggere direttamente la carta scelta e il suo margine, non soltanto ridurre una distanza
fra distribuzioni.

## Esito della loss sul margine

La hinge successiva fa meglio sul difetto specifico e non rende la rete indecisa. Con beta
`0.3` il flip medio scende a `14,42%`, mentre il gap top-2 resta `0,915` contro `0,929` dei
controlli. La forza contro v13 è neutra (`-0,14` punti/partita medi). Aumentare beta a `1.0`
non porta altro beneficio: il flip resta `14,49%`.

Anche questo ramo si ferma, perché manca il gate `<12%` e la curva ha raggiunto un plateau.
Il prossimo esperimento non allenerà una nuova policy: costruirà una vista di v13 che media
le 24 rinomine e garantisce simmetria esatta. Questo separerà finalmente la domanda “possiamo
eliminare il difetto?” dalla domanda più importante “eliminarlo migliora il gioco?”.
