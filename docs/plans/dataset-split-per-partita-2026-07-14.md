# Split dei dataset per partita

Data: 2026-07-14
Stato: implementato e verificato

## Problema

Una partita di Briscola produce molte righe correlate: la mano cambia di una carta, ma
storia, punteggio e carte viste restano in larga parte gli stessi. I trainer BC e value
mescolavano le singole righe e ne riservavano una frazione alla validation. In questo modo
alcune mosse di una partita potevano essere usate per allenare il modello e altre per
valutarlo.

Non era un accesso a carte nascoste e non rende invalido il runtime di v14. Era pero' una
forma di *data leakage*: validation e test potevano risultare piu' facili del previsto,
quindi non erano una base abbastanza rigorosa per scegliere un futuro modello.

## Decisione

`ai/training/dataset_split.py` assegna ora ogni `game_id` interamente a uno dei tre split:

- train: 80% delle partite per default;
- validation: 10%, usata durante il training o per scegliere il checkpoint;
- test: 10%, letto soltanto dopo che il modello finale e' stato scelto.

Le percentuali si applicano alle **partite**, non alle righe. L'algoritmo ordina gli ID
unici, li mescola con il seed dichiarato e garantisce almeno una partita al train e a ogni
holdout richiesto. Riordinare le righe del dataset non cambia l'assegnazione.

## Pipeline coperte

| Pipeline | Provenance richiesta | Comportamento nuovo |
|---|---|---|
| `train_bc.py` | `game_id` in ogni record JSONL allenabile | split train/validation/test per partita |
| `train_value.py` JSONL | `game_id` per record | stesso split per partita |
| value Numba | `game_ids` + `game_seeds` nel formato `value_dataset_npz_v2` | gli NPZ v1 senza ID vengono rifiutati |
| value PIMC pairwise | `game_ids` nel formato `pimc_leaf_value_dataset_v2` | tutte le root della stessa partita restano insieme |

Il pairwise teneva gia' insieme le carte candidate della stessa root. Ora raggruppa anche
root diverse provenienti dalla stessa partita, chiudendo la contaminazione residua.

## Provenance salvata

Ogni modello BC/value nuovo include `dataset_split` nel `metadata_json`:

- algoritmo e seed;
- frazioni richieste;
- numero di partite e record per split;
- chiave di raggruppamento;
- SHA-256 dell'assegnazione completa.

Il digest permette di confrontare due run senza inserire tutti i `game_id` nel modello.
Gli identificativi non vengono usati come feature. Il test finale e' salvato separatamente
(`test_metrics` o `test_eval`) e non partecipa alla scelta del checkpoint.

## Compatibilita' e migrazione

I JSONL prodotti dagli exporter e dal generatore value canonico contenevano gia'
`game_id`, quindi non richiedono conversioni. I vecchi NPZ value e leaf PIMC non possono
invece essere separati in modo affidabile: il confine fra partite e' stato perso. Il trainer
fallisce con un messaggio esplicito e richiede di rigenerarli, invece di usare in silenzio
lo split storico per record.

Questa modifica non riaddestra e non cambia v14. Serve a rendere onesto il prossimo
esperimento supervisionato; nessun nuovo training e' autorizzato dal solo fatto che la
pipeline sia ora corretta.

## Verifiche

I test coprono:

- assenza di `game_id` condivisi fra train, validation e test;
- determinismo indipendente dall'ordine delle righe;
- dataset troppo piccoli e ID mancanti;
- round trip BC, value JSONL, value Numba e value pairwise;
- rifiuto esplicito degli NPZ storici senza provenance;
- conteggi, digest e metriche test nei metadati dei modelli.

Gate completo del repository: ruff format/check, mypy su 77 file sorgente,
`make docs-check` e 659 test superati. Resta un solo warning di deprecazione esterno
Starlette/httpx, gia' presente e non collegato a questa modifica.
