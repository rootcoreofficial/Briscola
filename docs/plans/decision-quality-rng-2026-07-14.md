# Riproducibilità di decision quality tra worker

Data: 2026-07-14  
Stato: corretto e verificato

## Problema

`evaluate_decision_quality.py` separa il seme del mazzo dal generatore casuale usato
dagli agenti. Il percorso seriale manteneva un solo flusso casuale per tutta la suite;
quello parallelo assegnava invece un flusso indipendente a ogni coppia seat-fair.

Con agenti deterministici la differenza restava invisibile. Con un agente stocastico,
come `random`, cambiare `--workers` cambiava le carte scelte e quindi anche il risultato.
Il parallelismo non introduceva un bias noto, ma impediva di ripetere lo stesso
esperimento a velocità diverse.

## Riproduzione prima della correzione

Entrambi i comandi usavano 2.000 partite, `seed=123`, agenti `random` contro `random` e
la stessa generazione dei mazzi. L'unica differenza era il numero di processi.

| Metrica | 1 worker | 3 worker |
|---|---:|---:|
| vittorie A / B / pareggi | 975 / 991 / 34 | 1.014 / 950 / 36 |
| differenza punti media A-B | +0,09 | +0,82 |
| decisioni da secondo | 19.930 | 19.848 |
| sprechi di briscola | 801 | 740 |
| overkill di briscola | 866 | 844 |

La differenza non descriveva una variazione del gioco o del modello: i due percorsi
stavano semplicemente fornendo agli agenti sequenze casuali diverse.

## Correzione

La coppia seat-fair è ora anche l'unità del caso:

1. ogni coppia riceve un generatore delle azioni derivato dal seed principale e dal suo
   indice globale;
2. le due partite a posti scambiati consumano quello stesso flusso in sequenza;
3. seriale e parallelo attraversano lo stesso core;
4. il processo che esegue la coppia non entra mai nel calcolo del seed.

Il numero di worker modifica quindi soltanto la distribuzione del lavoro. Estendere una
suite conserva inoltre gli stream delle coppie già presenti, perché il loro indice non
cambia.

## Verifica

Dopo la correzione, gli stessi due comandi producono entrambi:

- vittorie A / B / pareggi: `1.014 / 950 / 36`;
- differenza punti media: `+0,82`;
- decisioni da secondo: `19.848`;
- risposte vincenti disponibili: `12.524`;
- sprechi di briscola: `740`;
- vittorie con briscola: `4.555`;
- overkill: `844`, di cui `514` su piatti poveri.

Il test automatico usa 15 coppie, agenti casuali e tre worker. La divisione in chunk
obbliga il parallelo a ricostruire correttamente l'indice globale; match e metriche devono
essere identici campo per campo al percorso con un worker.

## Compatibilità

Le valutazioni con policy argmax ed euristiche deterministiche non cambiano. Per agenti
stocastici, i vecchi risultati seriali non sono byte-comparabili con il nuovo schema: la
suite di mazzi resta la stessa, ma cambia una volta la sequenza delle azioni. Da questa
release il contratto stabile è più forte: stesso seed e stessi input producono gli stessi
aggregati con qualunque valore positivo di `--workers`.
