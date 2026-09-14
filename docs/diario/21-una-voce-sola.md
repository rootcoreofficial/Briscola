# Approfondimento - Una voce sola

**Approfondimento del diario:** [Capitolo 19](https://ai.briscola.dev/diario) - **Periodo:** 11–12 luglio 2026

## Dal consiglio alla singola rete

Il capitolo precedente aveva trovato un giocatore migliore chiedendo ventiquattro pareri
alla stessa v13. Il nuovo esperimento prova a farne a meno: raccoglie la risposta media del
consiglio e la usa come maestro per una sola rete. È distillazione supervisionata.

Il corpus contiene 10.000 partite e 380.000 decisioni utili. Metà delle traiettorie sono
v13 contro se stessa; il resto mescola un conservatore di briscole, due euristiche e il
giocatore casuale. Le partite, non le singole mosse, vengono separate in training,
validazione e test. Così una presa vista durante l'allenamento non può riapparire poco dopo
nel compito in classe sotto forma di un'altra decisione della stessa partita.

## Cosa impara

Prima del fine-tuning v13 sceglie la stessa carta del maestro nel 86,83% delle decisioni
del test. Dopo cinque passaggi sul corpus arriva al 92,88%. La distanza fra le due
distribuzioni scende di quasi quattro volte.

Durante il training ogni gruppo di esempi riceve anche una ristampa coi semi rinominati.
Questa volta la copia è corretta: non stiamo fingendo che una mossa campionata altrove sia
stata giocata dalla policy, ma trasformiamo insieme posizione e risposta esplicita del
maestro. Il controllo senza copie imita peggio e resta un po' più dipendente dai nomi dei
semi.

La singola rete non diventa matematicamente perfetta come il consiglio a ventiquattro.
Riduce però il cambio di carta dal 18,19% della v13 al 10,23%, superando per la prima volta
la soglia del 12% che le loss precedenti non avevano raggiunto. E non lo fa diventando
indecisa: il vantaggio medio della prima carta sulla seconda resta quasi uguale a v13.

## La prova sul tavolo

Su 10.000 partite con gli stessi mazzi e i posti scambiati, il modello distillato batte
v13 di 0,51 punti a partita. L'intervallo va da +0,11 a +0,92, quindi la parità resta appena
fuori. Contro il maestro a ventiquattro pareri perde 0,23 punti, ma l'intervallo da -0,59 a
+0,13 comprende la parità: con questa misura non possiamo distinguerli.

Il comportamento resta pulito. L'overkill di briscola sui piatti poveri passa dall'8,0%
di v13 al 5,5%; il maestro era al 3,9%. Anche qui l'allievo finisce fra il punto di partenza
e il consiglio che sta imitando.

Il modello non viene ancora pubblicato. Diecimila partite sono uno screening riuscito, non
la fine del confronto. Il passo successivo userà 50.000 partite indipendenti. Se accordo,
simmetria, forza e qualità reggono, allora potremo provarlo dentro il PIMC del sito. Fino a
quel momento resta un candidato locale, non una nuova generazione ufficiale.

Protocollo completo:
[`docs/plans/suit-distillation-v0-2026-07-11.md`](../plans/suit-distillation-v0-2026-07-11.md).

## Esito delle cinquantamila partite

Il corpus indipendente più grande rafforza il risultato. L'accordo con il maestro sale al
95,39% e il cambio di carta dovuto ai nomi dei semi scende ancora, dal 10,23% al 6,04%.
La distanza fra le distribuzioni si riduce di oltre metà rispetto al modello da diecimila
partite, senza rendere la rete meno decisa.

Sul tavolo batte v13 di 0,66 punti a partita, con intervallo da +0,24 a +1,09. Contro il
maestro a ventiquattro resta indistinguibile: -0,22, con intervallo da -0,53 a +0,09.
L'overkill sui piatti poveri scende al 4,17%, ormai vicino al 3,9% del maestro.

Il primo controllo dentro il PIMC, su duemila partite, dà +0,35 ma con un margine largo
che comprende la parità. Non è una bocciatura e non è una promozione. Serve l'ultimo test
da diecimila partite nella configurazione esatta del sito.

## L'ultimo test

Nelle diecimila partite finali entrambi i giocatori usano la stessa ricerca PIMC belief
16x8, la stessa stima delle carte nascoste, lo stesso numero di mondi simulati e lo stesso
solver. Cambia soltanto la policy: distillato 50k contro v13.

Il candidato guadagna 0,43 punti a partita. L'intervallo va da +0,03 a +0,84: per poco,
ma la parità resta fuori anche nel giocatore completo del sito. Vince 4.936 partite, ne
perde 4.758 e ne pareggia 306.

Questo chiude la catena iniziata dai nomi dei semi: difetto misurato, causa verificata,
maestro simmetrico, distillazione compatta e infine vantaggio conservato dentro la search.
Il modello ha superato i criteri tecnici ed è stato promosso come `best_a2c_v14.npz` nella
release `0.36.0`. Il catalogo lo riconosce come policy compatibile e continua a nascondere
gli asset interni value e belief.
