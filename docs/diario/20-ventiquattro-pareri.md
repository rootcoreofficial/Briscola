# Approfondimento - Ventiquattro pareri, una sola mossa

**Approfondimento del diario:** [Capitolo 19](https://ai.briscola.dev/diario) - **Periodo:** 11 luglio 2026

## Il test che separa causa e sintomo

Tre modi di insegnare la coerenza avevano ridotto il problema senza risolverlo. Restava
possibile che il 18% di cambi di carta fosse soltanto una stranezza interna: due mosse
diverse ma equivalenti nella partita reale.

Il nuovo test non allena nulla. La v13 guarda tutte le 24 ristampe della stessa posizione,
le risposte vengono riportate ai nomi originali e mediate. Per costruzione, rinominare i
semi non può più cambiare la decisione. Questo permette di confrontare direttamente la
stessa v13 con e senza il difetto.

## Ventiquattro non significa ventiquattro volte

Le 24 posizioni entrano nella rete come un unico batch. L'encoder gira una volta e NumPy
esegue insieme le moltiplicazioni. Su 10.000 decisioni la latenza media passa da 0,051 a
0,074 millisecondi: circa 1,45 volte, non 24. È un costo piccolo per una sonda offline,
anche se andrà rimisurato dentro il PIMC prima di pensare al sito.

I test confrontano il batch veloce con 24 trasformazioni complete dell'osservazione. Su
160 posizioni reali e 3.680 rinomine non banali, il cambio di carta è zero e anche il
residuo numerico massimo dei logits riallineati è zero nell'ambiente di misura.

## Il risultato sul gioco

Su 10.000 partite seat-fair, con lo stesso mazzo giocato da entrambi i posti, la versione
simmetrica batte la v13 originale di `+0,90` punti a partita. L'intervallo di confidenza
va da `+0,47` a `+1,33`: il risultato non è compatibile con una semplice parità.

Anche i controlli esterni vanno nella stessa direzione. Il vantaggio sulla vecchia
`heuristic_v1` cresce da `+21,59` a `+22,29`; contro il conservatore di briscole cresce da
`+15,20` a `+15,88`. Sono conferme secondarie, perché il confronto diretto resta quello
statisticamente più pulito.

La qualità delle mosse non paga il guadagno. Contro `heuristic_v1`, l'overkill di briscola
su piatti poveri scende dall'8,0% al 3,9%, mentre lo spreco di una briscola quando bastava
una carta normale scende da 99 a 41 casi. L'overkill complessivo resta identico al 22,0%.

## Cosa cambia nel piano

Ora sappiamo che i nomi accidentali dei semi costavano davvero forza. Non promuoviamo però
il wrapper 24x: il giocatore predefinito usa PIMC e richiama la policy molte volte dentro
le simulazioni, un contesto non ancora misurato.

La strada successiva è usare i 24 pareri come maestro. Una singola rete riceverà come
target la risposta media e proverà a riprodurla con un solo forward; questa tecnica si
chiama *distillazione*. Se conserva simmetria, forza e qualità, avremo comprato il beneficio
senza portare le 24 copie nel runtime. Se la rete ordinaria non ci riesce, servirà una
struttura che condivida esplicitamente i pesi fra i quattro semi.

Protocollo ed evidenza completa:
[`docs/plans/suit-symmetrized-v13-2026-07-11.md`](../plans/suit-symmetrized-v13-2026-07-11.md).
