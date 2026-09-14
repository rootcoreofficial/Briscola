# Diagnostica delle unità nascoste v0 (2026-07-12)

> Verdetto: **non allargare la rete**. Evidenza completa:
> [hidden_units_v14.v1.json](../reports/evidence/hidden_units_v14.v1.json).

## Domanda

V14 usa davvero tutte le 256 unità ReLU del suo livello nascosto, oppure una parte della
capacità è già disponibile ma non viene sfruttata dal training? E il residuo `6,04%` di
dipendenza dai nomi dei semi è concentrato in pochi neuroni rimovibili?

La risposta serve a decidere se abbia senso provare una rete larga 320/384. Una rete più
grande aumenta costo e spazio delle ipotesi; senza un collo di bottiglia misurato sarebbe
soltanto un'altra variabile.

## Protocollo

La raccolta riusa le quote validate dalla sonda di simmetria:

- 64 seed della suite `small`, due seat e quattro avversari (`mirror`, conservatore di
  briscole, `heuristic_v1`, `random`);
- 256 osservazioni non forzate per ciascuna delle 16 celle avversario/fase: **4.096 stati**;
- traiettorie generate da v14; v14 e v13 valutate sulle stesse identiche osservazioni;
- sottoquota deterministica di 16 stati per cella, 256 totali, valutata sulle 24 rinomine
  dei semi per l'ablation di simmetria.

Il modello riceve sempre e soltanto `PlayerObservation`. Per ogni stato ricostruiamo il
forward esatto `relu(xW1+b1)W2+b2`. Spegnere l'unità `j` significa sottrarre
`hidden[j] * W2[j]` dai logits in memoria: i file `.npz` non vengono modificati.

Soglie preregistrate:

- unità *morta*: attiva in non più dello `0,1%` degli stati non forzati;
- unità sempre attiva: almeno `99,9%`;
- coppia ridondante: correlazione assoluta delle attivazioni almeno `0,995`;
- dipendenza fragile da una unità: la sua rimozione cambia almeno il `5%` delle scelte.

Queste etichette sono descrittive. Un'unità morta sul campione non è dimostrata morta su
ogni stato matematicamente possibile; per questo il report non propone potatura o modifica
dei pesi live.

## Risultati

| metrica | v13 | v14 |
|---|---:|---:|
| unità morte su 256 | 139 (54,30%) | **123 (48,05%)** |
| unità con attivazione variabile | 136 | **163** |
| unità sempre attive | 0 | 0 |
| coppie con correlazione ≥0,995 | 0 | 0 |
| rango effettivo delle attivazioni | 39,09 | 35,60 |
| componenti che spiegano il 95% della varianza | 60 | 57 |
| massimo flip spegnendo una unità | 4,08% | 4,03% |
| flip dei semi sulla sottoquota | 19,43% | **6,06%** |

Le liste delle unità morte si sovrappongono molto: **109 unità** sono quasi inattive sia
in v13 sia in v14. La distillazione risveglia 30 unità prima morte e ne rende quasi
inattive 14, per un recupero netto di 16. Ha quindi redistribuito parte del calcolo, ma
non ha esaurito la larghezza disponibile.

Il rango effettivo non equivale al numero di “concetti” appresi: dipende anche dalla
struttura naturale degli stati di Briscola. Conferma però che l'attività osservata occupa
una porzione molto più piccola delle 256 direzioni disponibili. L'assenza di coppie quasi
identiche dice che le unità attive non sono semplici copie una dell'altra.

## Ablation causale

Nessuna unità supera la soglia di fragilità del 5%. La più influente di v14 è la `170`:
spegnendola cambia il `4,03%` delle decisioni. Questo esclude l'immagine di una rete retta
da un singolo interruttore, ma non rende le unità interpretabili come regole autonome.

Anche il residuo di asimmetria è distribuito:

- baseline v14 sulla sottoquota: `6,063%` di flip;
- migliore rimozione singola, unità `34`: `5,571%`, appena `-0,493` punti percentuali;
- peggiore rimozione singola, unità `52`: `9,851%`, `+3,787` punti percentuali.

Quindi eliminare il neurone apparentemente più “asimmetrico” non risolve il problema e
rischia di danneggiare altre decisioni. L'unità 52, per esempio, è importante anche fuori
dalla sonda: la sua rimozione cambia il `3,39%` delle scelte normali.

## Decisione

**STOP al widening 256→320/384.** Quasi metà delle unità è già disponibile sul campione;
non c'è evidenza che la larghezza impedisca a v14 di migliorare. Non bisogna neppure
potare subito: gli stati rari e fuori distribuzione non sono coperti da una singola suite.

Il controllo causale successivo è stato completato su seed indipendenti: l'ablation
congiunta concorda con v14 nel `99,9512%` delle 4.096 decisioni e il direct match da 10.000
partite è neutro (`+0,031`, CI95 `-0,018..+0,080`). Tutti i gate passano, ma 86 stati
attivano almeno una delle unità selezionate: non è una prova di potabilità universale.

Anche la prova autorizzata di riattivazione è conclusa. Le unità reinizializzate imparano,
ma reset 16 migliora la KL validation solo dello `0,328%` rispetto al controllo continuato,
sotto il gate dell'`1%`. La capacità dormiente non è quindi una leva pratica misurabile con
questo protocollo. Catena completa in `docs/plans/dormant-unit-ablation-v0-2026-07-12.md`
e `docs/plans/dormant-reinitialization-screen-v0-2026-07-12.md`.

## Riproduzione

```bash
uv run python scripts/diagnose_hidden_units.py
```

Il JSON conserva hash di modelli, seed suite e manifest di selezione, soglie, metriche per
tutte le 256 unità e confronto v14-v13. Non contiene mani avversarie, mazzo o osservazioni
serializzate.
