# Probe dose PIMC 32×8 su v14 (2026-07-12)

> Protocollo preregistrato prima del run. Esito: **STOP, confermato il default 16×8**.
> Evidenza: [pimc_dose_v14_probe.v1.json](../reports/evidence/pimc_dose_v14_probe.v1.json).

## Perché non ripetere lo storico

Il progetto ha già mostrato il Pareto generale: su v10, PIMC belief 16×8 conservava
l'87% dell'edge del 64×10 a circa un sesto del costo. Ha anche escluso l'estensione della
finestra: 16×10 perde `0,25` punti contro 16×8 con CI95 interamente sotto zero.

Manca soltanto il punto intermedio isolato: **32 determinizzazioni con la stessa finestra
8**, sulla v14 e con belief identica su entrambi i lati. Questo probe serve a verificare
che il cambio di policy non abbia creato un guadagno sorprendentemente grande; non cerca
di misurare effetti piccoli.

## Protocollo

- agente A: v14, belief v0, PIMC `32×8`, solver finale;
- agente B: stessa v14, stessa belief v0 e mix uniforme `0,10`, PIMC `16×8`, stesso solver;
- search Python su entrambi i lati, come nel runtime web; nessun kernel Numba;
- 2.000 partite seat-fair, cioè 1.000 coppie con stesso mazzo e posti scambiati;
- seed evaluation `20260712`;
- CI95 calcolata sulle coppie, non sulle singole partite correlate.

L'unica variabile intenzionale è il numero di determinizzazioni. Il modello v14 ha
SHA-256 `a67ed1d7f01ba1019f157134ade23fa9f822e442b671c83684bd4500e97695a8`; la belief
`belief_v0_h128_50k_seed20260702.npz` ha SHA-256
`4100b23b65a5566e047230ced665b91eef1942ea31e4a4cbe201b64545e7d035`.

L'evaluator usa un unico flusso RNG per le azioni delle partite: il seat swap rimuove il
bias di posizione, ma le prime 16 determinizzazioni dei due agenti non sono garantite
identiche. Il probe è quindi un confronto prodotto diretto, non una sonda di agreement a
common-random-numbers.

## Gate

Si passa a una conferma da 10.000 partite soltanto se valgono insieme:

1. margine medio 32×8 meno 16×8 almeno `+0,30` punti/partita;
2. limite inferiore della CI95 del margine strettamente maggiore di zero;
3. rapporto fra secondi medi per decisione search 32/16 non oltre `2,5`;
4. nessuna determinizzazione fallita e nessuna mossa corretta difensivamente.

Qualsiasi altro esito è **STOP**: resta il default 16×8 e la pista dose viene considerata
coperta dallo storico. Il probe registra soltanto la latenza media; p50/p95 diventano
necessari esclusivamente se il risultato supera questo gate e giustifica una conferma.

## Risultati

Su 2.000 partite, 32×8 ottiene 986 vittorie, 16×8 ne ottiene 954 e ci sono 60 pareggi.
Il segnale è piccolo e non conclusivo:

| metrica | risultato |
|---|---:|
| margine medio 32×8 - 16×8 | `+0,298` punti/partita |
| CI95 paired del margine | `-0,025..+0,621` |
| score rate 32×8 | `50,80%` |
| CI95 paired dello score rate | `49,93..51,67%` |
| latenza media search 32×8 | `28,93 ms` |
| latenza media search 16×8 | `14,50 ms` |
| rapporto di latenza | `1,995×` |
| tempo totale | `222,01 s` (`111,0 ms/partita`) |

Entrambi i lati eseguono 5.000 decisioni search. Il 32×8 completa 160.000
determinizzazioni e il 16×8 ne completa 80.000; non ci sono fallimenti, rollout falliti o
mosse corrette difensivamente. Hash e nomi degli agenti nel JSON confermano che policy,
belief e mix uniforme sono identici.

## Decisione

Passano il gate di costo (`1,995× <= 2,5×`) e quello di integrità. Falliscono entrambi i
gate di forza: `+0,298` è appena sotto `+0,30` e, soprattutto, il limite inferiore della
CI95 è negativo. Il vantaggio osservato può quindi essere rumore e non giustifica una
conferma lunga né il raddoppio del costo live.

Il verdetto è **STOP**: niente sweep 64×8, budget adattivo o cambio del default. Il 16×8
resta il punto Pareto, coerentemente con i risultati storici 16×8/64×10. La pista dose
PIMC è chiusa.

## Comando

```bash
uv run python scripts/evaluate_pimc.py \
  --model data/models/best_a2c_v14.npz \
  --num-games 2000 --seed 20260712 \
  --determinizations 32 --max-unknown-cards 8 \
  --opponent pimc --opponent-determinizations 16 --opponent-max-unknown-cards 8 \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --opponent-belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --belief-uniform-mix 0.10 --opponent-belief-uniform-mix 0.10 \
  --out-json docs/reports/evidence/pimc_dose_v14_probe.v1.json
```
