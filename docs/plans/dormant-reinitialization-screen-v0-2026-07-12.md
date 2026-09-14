# Screening di riattivazione delle unità dormienti v0 (2026-07-12)

> Protocollo preregistrato prima del training reale. Esito: **STOP alla
> reinizializzazione delle unità dormienti**. Evidenza:
> [dormant_reinitialization_screen_v14.v1.json](../reports/evidence/dormant_reinitialization_screen_v14.v1.json).

## Domanda

La distillazione v14 lascia capacità inutilizzata perché il trainer converge nello stesso
bacino, oppure alcune unità oggi dormienti possono imparare il residuo del teacher
simmetrico se ricevono una nuova inizializzazione?

Questo è un controllo di **ottimizzazione**, non ancora un tentativo di insegnare una
strategia nuova. Riutilizziamo il teacher v13 mediato sulle 24 rinomine perché fornisce un
target apprendibile e già validato. Non usiamo etichette PIMC: i precedenti esperimenti
mostrano che parte delle loro correzioni dipende dalle determinizzazioni di carte nascoste
e non è identificabile dalla sola osservazione lecita della student.

## Selezione congelata

La sorgente è `best_a2c_v14.npz`, SHA-256
`a67ed1d7f01ba1019f157134ade23fa9f822e442b671c83684bd4500e97695a8`.
Intersechiamo senza nuove soglie:

- le 93 unità con frequenza di attivazione esattamente zero nella diagnostica originale;
- le 90 unità mai attive nel holdout indipendente;
- risultato: **79 unità zero in entrambi i campioni**.

L'ordine non usa forza, KL o peso uscente. È il ranking crescente dello SHA-256 della
stringa `briscola.dormant_selection.v1:20260712:<unità>`, così gruppi e nesting sono
riproducibili anche senza NumPy:

- reset 8: `[160, 142, 72, 69, 104, 157, 237, 127]`;
- reset 16: le 8 precedenti più `[7, 113, 64, 137, 86, 136, 135, 219]`.

Per ogni unità selezionata, `w1` riceve un'inizializzazione He deterministica derivata da
seed e indice, `b1=0` e la riga `w2=0`. La stessa unità parte con gli stessi pesi nel reset
8 e nel reset 16. I logits iniziali coincidono con l'ablation del sottoinsieme, già
ritenuta neutra dal controllo causale; l'agreement iniziale viene comunque misurato su
validation e test e ha un gate proprio.

## Esperimento controllato

Tre varianti partono da v14:

1. `control`: nessuna reinizializzazione, ma stessa continuazione di training;
2. `reset_8`: riattiva le prime 8 unità congelate;
3. `reset_16`: riattiva le prime 16 unità congelate.

Tutte usano `suit_teacher_v13_10k_seed20260711.npz`, SHA-256
`cdcfb37ec20b32deb76fa785e118c1083a60d6e7cc8dba474787eec6ec443275`. Il corpus da
10.000 partite non è quello da 50.000 usato per addestrare v14. Contiene 380.000 decisioni
e split per partita: 8.000 train, 1.000 validation, 1.000 test. Il test non sceglie variante
o epoca.

Configurazione identica: 5 epoche massime, batch 1.024, Adam `lr=2e-4`, weight decay
`1e-6`, seed training `20260712` e augmentation paired dei semi. Per ogni variante la
validation sceglie l'epoca con KL minima. Fra reset 8 e 16 viene scelto soltanto quello con
KL validation minore; il test è letto dopo questa scelta come conferma.

## Gate

Il candidato scelto passa soltanto se tutti i controlli sono veri:

1. agreement iniziale con v14 almeno `99,9%` sia su validation sia su test;
2. KL validation almeno `1%` più bassa del controllo continuato;
3. agreement validation non peggiore di oltre `0,05` punti percentuali;
4. sul test, KL non peggiore del controllo e agreement non peggiore di oltre `0,05` punti;
5. almeno il `75%` delle unità reinizializzate è attivo in almeno l'`1%` della validation;
6. almeno il `75%` ha peso uscente non costante con norma centrata almeno `0,001`.

Un **GO** non prova maggiore forza: autorizza la sonda completa di simmetria e un direct
match contro controllo e v14. Uno **STOP** chiude il riuso delle unità dormienti: il passo
successivo torna al confronto runtime PIMC 16/32/64, senza widening né ulteriori reset.

## Risultati

Tutte le varianti raggiungono il checkpoint migliore alla quinta epoca:

| variante | KL validation | agreement validation | KL test | agreement test |
|---|---:|---:|---:|---:|
| controllo continuato | `0,059976` | **`95,6158%`** | `0,062310` | `95,5605%` |
| reset 8 | `0,059881` | `95,6000%` | `0,062196` | **`95,5737%`** |
| **reset 16** | **`0,059780`** | `95,6132%` | **`0,062063`** | `95,5632%` |

La validation seleziona correttamente reset 16. Rispetto al controllo, riduce la KL di
`0,000197`, cioè soltanto **`0,328%`**: meno di un terzo del gate preregistrato dell'`1%`.
Il test conferma un vantaggio altrettanto piccolo (`-0,000247`, circa `0,40%`) e un
agreement praticamente identico (`+0,0026` punti percentuali).

La parte meccanica funziona. Il reset 16 differisce inizialmente da v14 in appena 2 delle
38.000 decisioni test (`99,9947%` di agreement); dopo il training tutte le 16 unità sono
attive in oltre l'`1%` degli stati e tutte hanno appreso un peso uscente non costante. Non
sono quindi rimaste morte: semplicemente aggiungono pochissimo a ciò che il resto della
rete apprende già.

Il confronto col controllo chiarisce la causa. Senza alcun reset, la sola continuazione
riduce sul test la KL iniziale di v14 da `0,068745` a `0,062310` e porta l'agreement da
`95,4368%` a `95,5605%`. Quasi tutto il progresso viene dunque dalle altre cinque epoche,
non dal riuso delle unità.

## Decisione

Passano sei gate su sette; fallisce quello centrale sul miglioramento KL della validation.
Il verdetto automatico è **`stop_dormant_reinitialization`**. Non eseguiamo sonda di
simmetria o direct match: sarebbero valutazioni costose di un candidato che non ha superato
lo screening previsto.

Anche se il divario cresce gradualmente nelle cinque epoche, prolungare il run dopo aver
visto il risultato cambierebbe il protocollo per inseguire un effetto sotto soglia. La
pista capacità è quindi chiusa: niente widening, potatura o altri reset. Il prossimo
esperimento operativo è isolare la dose PIMC 16/32/64 a finestra 8 e pari condizioni.

## Riproduzione

```bash
uv run python scripts/run_dormant_reinitialization_screen.py
```

I tre `.npz` restano locali sotto
`benchmarks/experiments/dormant_reinitialization_v14_v0/models/`. L'evidenza sintetica è
`docs/reports/evidence/dormant_reinitialization_screen_v14.v1.json` e registra hash,
provenienza, lista completa, metriche per epoca e verdetto automatico.
