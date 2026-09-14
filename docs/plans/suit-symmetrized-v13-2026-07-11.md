# Policy v13 simmetrizzata sui semi (2026-07-11)

> Test causale policy-only, nessuna promozione runtime. Evidenza:
> [suit_symmetrized_v13.v1.json](../reports/evidence/suit_symmetrized_v13.v1.json).

## Domanda e verdetto

Le loss di consistency riducevano l'asimmetria ma non la eliminavano, lasciando aperta la
domanda principale: una policy davvero indipendente dai nomi dei semi gioca meglio oppure
la metrica sta misurando un difetto innocuo?

La v13 è stata valutata sulle 24 rinomine dei semi. I 40 logits di ogni copia sono stati
riallineati agli action id originali e mediati prima dell'argmax.

**Verdetto: il difetto non è innocuo.** La policy simmetrizzata batte direttamente v13 di
`+0,90` punti/partita, CI95 paired `+0,47..+1,33`, su 10.000 partite seat-fair. Il flip è
zero per costruzione e il controllo numerico trova zero flip su 3.680 rinomine non banali.
**GO alla distillazione del teacher; non ancora GO alla promozione nel default PIMC.**

## Implementazione e correttezza

`SuitSymmetrizedBCModelAgent` resta fuori dal catalogo UI ed è disponibile soltanto negli
strumenti offline. L'implementazione:

1. encoda una sola volta la `PlayerObservation` lecita;
2. costruisce le 24 copie numeriche con gather precalcolate e verificate contro il percorso
   semantico `permute -> encode`;
3. esegue un solo forward MLP sul batch `(24, 369)`;
4. riallinea i logits con una seconda gather e calcola la media float64;
5. applica la action mask originale e sceglie l'argmax.

I test confrontano ogni riga del batch col forward semantico sulle 24 rinomine, verificano
l'identità bit per bit e l'equivarianza dell'intera media. Modelli con belief embedded e
guard runtime vengono rifiutati: questa ablation deve isolare la sola policy v13.

Su 160 osservazioni reali, pari a 3.680 confronti non identità, il controllo esteso misura:

- flip dell'azione: `0/3.680`;
- massimo delta assoluto dei logits riallineati: `0,0` nell'ambiente di misura;
- nessun accesso a stato completo, mazzo o mano avversaria.

## Costo reale

Benchmark su Apple Silicon, 160 osservazioni reali e 10.000 decisioni cronometrate per
agente, dopo warm-up:

| policy | media | p95 | decisioni/s |
|---|---:|---:|---:|
| v13 | 0,0513 ms | 0,0568 ms | 19.505 |
| v13 simmetrizzata 24x batch | 0,0744 ms | 0,0820 ms | 13.444 |
| rapporto | **1,45x** | **1,44x** | 0,69x |

Non costa 24 volte: le moltiplicazioni della rete sono molto più efficienti in batch e
l'encoder resta singolo. La misura riguarda la policy NumPy isolata; il costo nel PIMC va
misurato separatamente, perché la policy viene richiamata dentro molti rollout.

## Forza policy-only

Ogni riga usa la seed suite medium versionata: 5.000 mazzi, due posti per mazzo, CI sulle
coppie. Il confronto diretto è la prova causale; le due baseline sono controlli secondari.

| agente A | agente B | punti A-B | CI95 | score rate A |
|---|---|---:|---:|---:|
| **v13 sym 24x** | v13 | **+0,90** | **+0,47..+1,33** | 51,45% |
| v13 sym 24x | heuristic_v1 | +22,29 | +21,75..+22,84 | 78,84% |
| v13 | heuristic_v1 | +21,59 | +21,04..+22,14 | 77,85% |
| v13 sym 24x | heuristic_trump_saver | +15,88 | +15,36..+16,40 | 72,77% |
| v13 | heuristic_trump_saver | +15,20 | +14,68..+15,73 | 71,52% |

I delta di circa `+0,70` sulle baseline sono coerenti col direct match ma non vanno trattati
come una seconda CI del vantaggio: sono match separati, non una differenza paired fra le
due righe.

## Qualità decisionale

Un secondo gate domain da 10.000 partite contro `heuristic_v1`, stessi seed e un worker,
controlla che la media non riapra lo spreco corretto da v13.

| metrica su A | v13 | v13 sym 24x |
|---|---:|---:|
| punti A-B | +21,19 | **+22,72** |
| trump waste | 99 / 82.298 (0,120%) | **41 / 82.705 (0,050%)** |
| overkill complessivo | 5.564 / 25.235 (22,05%) | 5.575 / 25.287 (22,05%) |
| overkill su piatto povero | 600 / 7.488 (**8,01%**) | 283 / 7.229 (**3,91%**) |

L'overkill complessivo è invariato, mentre il caso più chiaramente inutile si dimezza. Il
vantaggio di forza non è quindi ottenuto tornando al comportamento che v13 aveva corretto.

## Decisione e prossimo passo

La simmetria dei semi è una leva reale di forza, non soltanto eleganza rappresentazionale.
Il wrapper 24x resta però un teacher diagnostico: introdurlo direttamente nel default senza
un gate PIMC mescolerebbe il beneficio della policy con un nuovo costo dentro la ricerca.

Il prossimo esperimento è distillare i logits medi del teacher in una singola MLP v4:

- corpus di osservazioni separato per partita, per evitare leakage fra stati correlati;
- target soft sui 40 logits/probabilità legali, non soltanto l'argmax;
- controllo iniziale che il modello distillato riproduca il teacher su holdout;
- gate su flip, direct match contro v13 e teacher, baseline e qualità decisionale;
- nessuna promozione se il vantaggio sparisce o l'asimmetria torna alta.

Se una MLP ordinaria non riesce a imitare il teacher senza perdere il beneficio, la pista
successiva è un'architettura con pesi condivisi fra semi. Non serve un altro tentativo di
consistency loss sulla stessa architettura prima di questo confronto.

## Comandi

```bash
uv run python scripts/benchmark_suit_symmetrized.py \
  --model data/models/best_a2c_v13.npz --observations 160 --decisions 10000

uv run python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model_suit_symmetrized --agent0-model data/models/best_a2c_v13.npz \
  --agent1 bc_model --agent1-model data/models/best_a2c_v13.npz

uv run python scripts/evaluate_decision_quality.py --benchmark medium --engine domain \
  --agent-a bc_model_suit_symmetrized --agent-a-model data/models/best_a2c_v13.npz \
  --agent-b heuristic_v1 --workers 1
```
