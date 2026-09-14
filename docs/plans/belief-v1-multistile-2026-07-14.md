# Protocollo belief v1 multi-stile

**Data:** 2026-07-14  
**Stato:** gate completo chiuso con **STOP runtime**; belief v0 resta ufficiale.  
**Roster congelato:**
[`docs/plans/belief-v1-roster-2026-07-14.json`](belief-v1-roster-2026-07-14.json)  
**Evidenza pilot:**
[`docs/reports/evidence/belief_v1_pilot.v1.json`](../reports/evidence/belief_v1_pilot.v1.json)  
**Gate offline:**
[`docs/reports/evidence/belief_v1_offline_gate.v1.json`](../reports/evidence/belief_v1_offline_gate.v1.json)  
**Screen runtime:**
[`docs/reports/evidence/belief_v1_runtime_screen.v1.json`](../reports/evidence/belief_v1_runtime_screen.v1.json)

## Obiettivo in parole semplici

La belief network prova a indovinare quali carte possiede l'avversario usando soltanto
cio' che un giocatore puo' davvero vedere. Quella attuale, belief v0, ha studiato 50.000
partite giocate da copie dello stesso vecchio modello v7. Potrebbe quindi riconoscere bene
quel modo di giocare e generalizzare meno bene a persone o strategie diverse.

Belief v1 studia una popolazione piu' varia. Prima di usarla nel gioco, sosteniamo sette
esami: a turno nascondiamo completamente uno stile durante il training e misuriamo come
si comporta proprio su quello stile mai visto.

## Roster e volume congelati

Il gate completo usa 66.000 partite, multiplo esatto del peso totale 11:

| Stile mirror | Peso | Partite |
|---|---:|---:|
| v14, comportamento prodotto corrente | 4 | 24.000 |
| v13, predecessore diretto | 2 | 12.000 |
| v11, anchor ML piu' distante | 1 | 6.000 |
| heuristic_v1 | 1 | 6.000 |
| heuristic_v2 | 1 | 6.000 |
| heuristic_trump_saver | 1 | 6.000 |
| random | 1 | 6.000 |

Ogni partita e' `mirror`: lo stesso stile occupa entrambi i posti. Questo garantisce che
tutti gli stati di una partita abbiano un solo `opponent_id`, quindi una partita non puo'
finire in parte nel training e in parte nell'esame. E' un test conservativo di un intero
stile mai visto; non isola invece l'effetto dello stile avversario da quello del giocatore
che osserva. Se questa pista proseguira', una matrice di matchup con observer fissato sara'
un'eventuale ablation successiva, non una modifica da mescolare a questo gate.

Per ogni partita registriamo circa sette osservazioni nella finestra `max_unknown=10`, per
un totale atteso di circa 462.000 esempi. `opponent_id` serve solo a separare i dati: non
entra mai nelle 369 feature della rete. La mano vera dell'avversario viene usata solo come
risposta corretta offline; a runtime restano disponibili esclusivamente informazioni lecite.

## Esami e soglie offline

Ogni fold allena una MLP 369 -> 128 -> 40 per 30 epoche, con lo stesso Adam, learning
rate `1e-3`, batch 512 e seed. Belief v0 e belief v1 vengono misurate sullo stesso identico
holdout. Il riepilogo usa una macro-media: ogni stile conta uno, indipendentemente dal peso.

Il gate e' **GO** soltanto se tutte le condizioni sono vere:

- BCE macro migliore di almeno 1% relativo rispetto a belief v0;
- top-k recall macro non peggiore;
- Brier score macro non peggiore;
- errore di calibrazione ECE macro non peggiore;
- nessun singolo stile peggiora la BCE di oltre 2% relativo.

Un fallimento produce `stop_offline_gate` e il runner non allena il candidato finale. Un
GO autorizza soltanto il modello all-styles e il successivo confronto PIMC; non autorizza
promozione, deploy o sostituzione dell'asset live.

## Pilot completato

Il pilot ha usato 770 partite, 5.390 esempi e 10 epoche. Ha verificato con successo:

- scheduling pesato e deterministico;
- separazione completa per partita e stile;
- sette fold leave-one-out;
- confronto v0-v1 sullo stesso holdout;
- BCE, top-k, Brier, ECE e aggregazione con JSON rigoroso;
- stop automatico prima del candidato.

Il candidato pilot e' peggiore di v0: BCE macro `0,6103` contro `0,5514`, top-k `-0,0393`
e Brier `+0,0252`; soltanto ECE migliora di `0,0098`. Non e' un risultato sulla qualita'
dell'idea: v0 ha visto 50.000 partite, il pilot appena 770. Il verdetto salvato e'
`pilot_pipeline_validated`, che per contratto non puo' diventare GO.

## Job completo

Il runner esegue i fold in sequenza per non moltiplicare memoria e contesa CPU. La run
canonica e' terminata in modo integro; il comando resta la ricetta di riproduzione:

```bash
nohup caffeinate -i uv run python scripts/run_belief_v1_gate.py \
  --resume \
  > data/belief/belief_v1_gate_20260714.log 2>&1 &
```

Controllo del progresso:

```bash
tail -f data/belief/belief_v1_gate_20260714.log
```

`--resume` riusa solo dataset e modelli i cui hash, split e iperparametri coincidono col
protocollo; un file incompatibile causa un errore esplicito. Gli artefatti locali vivono in
`data/belief/belief_v1_gate_20260714/`:

- `belief_v1_multistyle_g66000_seed20260714.npz`;
- sette modelli sotto `folds/`;
- `belief_v1_leave_one_out_g66000_seed20260714.json` con il verdetto;
- `belief_v1_all_styles_h128_g66000_seed20260714.npz`, creato dopo il GO offline.

## Risultato offline completo

Il dataset contiene 66.000 partite e 462.000 esempi. Tutti i cinque gate preregistrati
passano:

| Macro-media sui sette stili esclusi | belief v0 | belief v1 | Differenza v1-v0 |
|---|---:|---:|---:|
| BCE, minore e' meglio | 0,5457 | 0,5041 | -7,63% relativo |
| Top-k recall | 54,87% | 58,16% | +3,29 punti |
| Brier, minore e' meglio | 0,1846 | 0,1715 | -0,0130 |
| ECE, minore e' meglio | 0,0463 | 0,0204 | -0,0259 |

Belief v1 migliora nettamente sugli stili euristici e random. Sul fold v14 e' invece
peggiore dell'`1,45%` relativo in BCE e di `1,10` punti in top-k; la regressione resta
entro il limite preregistrato del 2%, quindi il verdetto offline corretto e' GO al solo
screen PIMC. Il candidato all-styles migliora anche sul validation mix interno (BCE
`0,4995` contro `0,5351`, top-k `58,20%` contro `55,47%`).

## Gate runtime preregistrato

Se il gate offline e' GO, il runner stampa senza avviarlo un confronto seat-fair tra lo
stesso v14 PIMC 16x8 con belief v1 e belief v0. Lo screening usa 2.000 partite e passa alla
conferma da 10.000 soltanto se:

- differenza media almeno `+0,10` punti/partita;
- limite basso CI95 non inferiore a `-0,50`;
- zero determinizzazioni/rollout falliti e zero mosse corrette forzatamente;
- tempo per decisione search di v1 non oltre `1,10x` v0.

La conferma da 10.000 richiede differenza media almeno `+0,20`, limite basso CI95 maggiore
di zero, integrita' completa e lo stesso tetto di costo. Se il test non li soddisfa, belief
v0 resta l'asset ufficiale anche se le metriche offline di v1 sono migliori.

## Risultato runtime e decisione

Lo screen da 2.000 partite e' terminato in 155,4 secondi:

- belief v1: 950 vittorie; belief v0: 971; pareggi: 79;
- differenza media v1-v0: `-0,224` punti/partita, CI95 `-0,572..+0,124`;
- score rate v1 `49,475%`, CI95 `48,595..50,355%`;
- zero determinizzazioni o rollout falliti e zero mosse corrette forzatamente;
- latenza search `15,034 ms` v1 contro `15,062 ms` v0, rapporto `0,998x`.

Integrita' e costo passano, ma falliscono entrambi i gate di forza dello screen: il punto
stimato non raggiunge `+0,10` e il limite basso e' sotto `-0,50`. La CI include zero,
quindi non dimostriamo che v1 sia certamente peggiore; dimostriamo pero' che non ha
fornito l'headroom richiesto per continuare. Coerentemente con il protocollo:

- **STOP** alla conferma da 10.000 partite;
- nessuna promozione o copia del candidato in `data/models/`;
- belief v0 resta l'asset prodotto;
- niente reweighting post-hoc del roster per salvare lo stesso esperimento.

La conclusione didattica e' che una stima delle carte migliore in media non garantisce
scelte PIMC migliori. Il fold v14, unico leggermente negativo offline, anticipava inoltre
la direzione dello screen self-play con policy v14.
