# Approfondimento - Il limite della strada che conosciamo

**Approfondimento del diario:** [Capitolo 20](https://ai.briscola.dev/diario) - **Periodo:** 12-14 luglio 2026

## La domanda corretta dopo v14

La promozione di v14 non dimostra che il giocatore sia ottimo. Dimostra che una
correzione precisa, la simmetria dei semi, migliora sia la policy sia il giocatore
completo con PIMC e solver. Dopo la promozione la domanda è diventata diversa: esiste un
altro difetto misurabile abbastanza bene da progettare v15?

Sette controlli hanno esplorato i candidati più vicini:

1. raddoppiare le determinizzazioni PIMC da 16 a 32 costa quasi il doppio e porta solo
   `+0,298` punti, con intervallo compatibile con la parità;
2. 123 unità ReLU quasi inattive possono essere rimosse insieme senza una perdita
   misurabile, quindi non sono un collo di bottiglia evidente;
3. riattivare 8 o 16 di quelle unità funziona meccanicamente, ma migliora la loss meno
   dell'1% richiesto per proseguire;
4. belief v1 riconosce meglio le carte nascoste sui sette holdout, ma dentro PIMC perde
   `0,224` punti a partita contro belief v0;
5. critic, gradienti e passi di A2C risultano sani su tre seed: non emerge un guasto
   numerico da correggere;
6. la schedule paired non riduce la variabilità del training e non supera il seriale;
7. l'audit automatico del regret non trova errori affidabili nelle 96 decisioni
   early/mid esposte direttamente alla policy.

Il replay sulle osservazioni live concorda con questo quadro: non localizza una
regressione di v14 e vede meno sprechi, ma il volume di campo è ancora troppo piccolo per
fare affermazioni sulla forza contro le persone.

## Cosa non abbiamo dimostrato

Questi risultati non sono una soluzione matematica della Briscola. Il progetto non sa
calcolare quanto v14 sia distante dalla migliore strategia possibile e non dispone di una
misura di exploitability dell'intera partita. Il PIMC resta un'approssimazione: immagina
molti mondi compatibili con le carte nascoste e ne media gli esiti, ma non costruisce una
strategia unica che tenga conto di come l'informazione cambia durante tutte le
continuazioni.

Anche il residuo di simmetria di v14, circa il 6% di cambi dell'argmax sotto le 24
rinomine, ricorda che il modello non è perfetto. Sappiamo però che ridurre quel numero non
basta da solo: serve dimostrare un vantaggio sul tavolo.

Il verdetto corretto è quindi più limitato: **si è appiattita la famiglia di interventi
che parte dall'attuale policy e le cambia una cosa vicina alla volta**. Più partite, più
neuroni, più mondi PIMC o un diverso ordine dei campioni non hanno oggi un segnale che ne
giustifichi il costo.

## Come si potrebbe riaprire la ricerca

Una nuova iterazione sulla forza dovrebbe partire da un maestro più informativo, non da
un training più lungo. Tre segnali sarebbero sufficienti per riaprire il lavoro:

- un cluster ripetibile di errori policy-only, trovato su nuove osservazioni o con una
  stima controfattuale più precisa;
- un piccolo benchmark a informazione nascosta con soluzione o limite superiore noto,
  sul quale misurare direttamente il margine lasciato da PIMC;
- un metodo che ragioni sull'intero insieme delle informazioni, per esempio una forma
  di ricerca o apprendimento per regret, prima validato su stati ridotti e solo dopo sul
  gioco completo.

Questa sarebbe una nuova linea di ricerca, con costi e strumenti diversi. Non è
autorizzata dai risultati correnti, ma mostra perché un plateau dell'attuale ricetta non
equivale al limite strategico del gioco.

## Misurare prima di ripartire

Gli ultimi due interventi non hanno modificato v14. Lo split dei dataset supervisionati
è stato portato dal singolo record alla partita intera, impedendo che mosse correlate
finiscano sia nell'allenamento sia nell'esame. Inoltre decision quality usa ora gli stessi
flussi casuali con uno o più worker: il parallelismo cambia il tempo, non il risultato.

Sono lavori meno visibili di un nuovo modello, ma rendono credibile il prossimo eventuale
confronto. I protocolli completi sono:

- [`policy-regret-audit-v14-2026-07-14.md`](../plans/policy-regret-audit-v14-2026-07-14.md);
- [`dataset-split-per-partita-2026-07-14.md`](../plans/dataset-split-per-partita-2026-07-14.md);
- [`decision-quality-rng-2026-07-14.md`](../plans/decision-quality-rng-2026-07-14.md).

La release `0.37.0` conserva v14 come campione. Non chiude il progetto: chiude la pretesa
che basti continuare a girare le stesse manopole per ottenere automaticamente v15.

## Addendum: un ultimo test della scala

Dopo la release, la disponibilita' di calcolo a costo monetario nullo ha cambiato il
rapporto costo/informazione di un esperimento, non l'evidenza strategica. E' stato quindi
pianificato uno scouting A2C seriale da 50 milioni di partite con checkpoint e gate
preregistrati. Un singolo seed non potra' essere promosso e la distillazione restera'
condizionata a un vantaggio prima misurato.

Il trainer e' stato quindi reso adatto a un job di circa due giorni: schedule e metriche
a memoria costante, telemetria campionata, checkpoint atomico e resume bit-identico. Il
primo blocco da 10 milioni resta una prova controllata, non una promozione anticipata.
Protocollo: [`a2c-super-training-50m-2026-07-14.md`](../plans/a2c-super-training-50m-2026-07-14.md).
