# Ablation congiunta delle unità dormienti v0 (2026-07-12)

> Protocollo preregistrato prima del run holdout. Esito: **GO circoscritto al riuso
> sperimentale di una piccola parte della capacità dormiente**. Evidenza:
> [dormant_unit_ablation_v14.v1.json](../reports/evidence/dormant_unit_ablation_v14.v1.json).

## Domanda

La diagnostica in-sample classifica 123 unità ReLU di v14 come quasi inattive su 4.096
decisioni non forzate. Sono davvero capacità disponibile, oppure alcune servono su stati
rari non coperti dal campione?

Non modifichiamo v14 live e non promuoviamo un modello potato. Creiamo una copia locale
in cui le 123 righe corrispondenti di `w2` sono azzerate insieme: questo equivale a
spegnere le unità lasciando intatti tutti gli altri pesi.

## Indipendenza

L'elenco delle unità è letto senza modifiche da
`docs/reports/evidence/hidden_units_v14.v1.json`. Il holdout usa:

- seed sequenziali `1.000.000..1.000.063`, mai usati per selezionare le unità;
- due seat e lo stesso roster preregistrato (`mirror`, conservatore, `heuristic_v1`,
  `random`);
- 256 osservazioni per ciascuna delle 16 celle avversario/fase: 4.096 stati;
- tutte le 24 rinomine dei semi sugli stessi 4.096 stati: 94.208 confronti non identità;
- direct match Numba seat-fair da 10.000 partite sui seed `1.000.000..1.004.999`.

Policy e raccolta ricevono soltanto `PlayerObservation`. Hash di modello, evidenza,
candidato e manifest dei seed vengono salvati nel JSON.

## Gate

L'ablation passa soltanto se soddisfa contemporaneamente:

1. agreement delle carte sul holdout almeno `99,9%`;
2. variazione assoluta del flip dei semi non oltre `0,5` punti percentuali;
3. margine medio del candidato contro v14 entro `±0,20` punti/partita;
4. limite inferiore della CI95 del margine almeno `-0,30`.

Un **GO** autorizza soltanto un piccolo screening successivo: reinizializzare una parte
delle unità dormienti e ripetere la distillazione. Non autorizza potatura, promozione o
modifica del default. Un solo gate fallito chiude il riuso causale della capacità dormiente
e riporta il piano alla dose PIMC 16/32/64.

## Risultati

Il controllo holdout conferma che il gruppo selezionato contribuisce molto raramente, ma
non è matematicamente spento in ogni stato:

| controllo | risultato | gate | esito |
|---|---:|---:|:---:|
| agreement delle azioni | **99,9512%** (2 differenze su 4.096) | >=99,9% | PASS |
| unità mai attive sul holdout | 90/123 | descrittivo | - |
| unità ancora sotto lo 0,1% | 115/123 | descrittivo | - |
| variazione flip dei semi | **-0,0382 punti percentuali** | <=0,5 in valore assoluto | PASS |
| margine medio vs v14 | **+0,031 punti/partita** | <=0,20 in valore assoluto | PASS |
| limite inferiore CI95 del margine | **-0,018** | >=-0,30 | PASS |

Sulle 98.304 decisioni dell'orbita completa delle rinomine, le due policy concordano nel
`99,9502%` dei casi (49 differenze). Il flip passa da `5,9199%` a `5,8817%`: una variazione
troppo piccola per attribuire alle unità dormienti il residuo di asimmetria.

Nel direct match Numba seat-fair da 10.000 partite, il candidato ablated ottiene 4.844
vittorie, v14 ne ottiene 4.839 e ci sono 317 pareggi. Il margine `+0,031` ha CI95
`-0,018..+0,080`: l'intervallo contiene lo zero, quindi il test non mostra una differenza
reale di forza nelle condizioni misurate.

Un dettaglio impedisce di interpretare il risultato come via libera alla potatura: in 86
stati almeno una delle unità selezionate si attiva e nel caso più sensibile la probabilità
di una carta cambia di `0,873`. L'effetto globale è neutro, ma esistono stati rari in cui
il gruppo conta. Il candidato resta quindi locale, diagnostico e non selezionabile dalla UI.

## Decisione

Tutti e quattro i gate preregistrati passano e hanno autorizzato un piccolo screening con
8 e 16 unità. Lo screening è ora concluso: le unità reinizializzate diventano attive e
imparano pesi in uscita, ma la migliore variante riduce la KL validation soltanto dello
`0,328%` rispetto allo stesso training senza reset, sotto il gate dell'`1%`.

La pista è quindi chiusa senza potatura, widening o modifica di v14 live. Protocollo e
risultati completi sono in
`docs/plans/dormant-reinitialization-screen-v0-2026-07-12.md`; il piano passa alla dose
PIMC 16/32/64.

## Comandi

Prima fase, crea candidato ed evidenza holdout:

```bash
uv run python scripts/evaluate_dormant_unit_ablation.py
```

Direct match standard:

```bash
uv run python scripts/evaluate_agents.py \
  --engine numba --num-games 10000 --seat-fair --seed-suite-range-start 1000000 \
  --agent0 bc_model --agent0-model data/models/v14_dormant123_ablated_holdout_v0.npz \
  --agent1 bc_model --agent1-model data/models/best_a2c_v14.npz \
  --out-json benchmarks/experiments/dormant_unit_ablation_v14_v0/match_vs_v14_holdout10k.json
```

Seconda fase, incorpora il match e applica i gate:

```bash
uv run python scripts/evaluate_dormant_unit_ablation.py \
  --match-json benchmarks/experiments/dormant_unit_ablation_v14_v0/match_vs_v14_holdout10k.json
```
