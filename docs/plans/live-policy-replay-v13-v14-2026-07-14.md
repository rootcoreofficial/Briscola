# Replay appaiato v13-v14 sulle partite live

**Data:** 2026-07-14  
**Verdetto:** nessun nuovo difetto specifico di v14; audit utile ma non causale e non bloccante.  
**Evidenza canonica:**
[`docs/reports/evidence/live_policy_replay_v13_v14.v1.json`](../reports/evidence/live_policy_replay_v13_v14.v1.json)

## Domanda

Le vittorie e le sconfitte raccolte in produzione non sono un confronto corretto tra v13
e v14: cambiano giocatore umano, carte e momento della raccolta. Il replay risponde a una
domanda piu' stretta e controllabile:

> Se mostriamo a v13 e v14 esattamente le stesse osservazioni live lecite, quanto spesso
> scelgono carte diverse e in quale parte del runtime succede?

Non ricostruiamo partite alternative dopo una mossa diversa. Il test confronta la singola
decisione nello stesso punto osservato e non afferma quale scelta avrebbe vinto la partita.

## Dati e integrita'

- 70 partite complete: 59 giocate contro v13 e 11 contro v14;
- 2.800 azioni totali, sempre 40 per partita e 20 per lato;
- 2.660 decisioni non forzate confrontate; 140 mani da una carta escluse;
- 52 vittorie IA, 17 vittorie umane e un pareggio;
- observation presente in ogni record, consenso presente e nessuna mossa IA corretta a
  posteriori dal backend.

Lo script ricostruisce `PlayerObservation` solo dal DTO pubblico. Mano avversaria e ordine
del mazzo non vengono letti. Il report conserva soltanto aggregati e hash: nessun game id,
nome, client id o observation.

## Metodo

Ogni osservazione viene valutata quattro volte:

1. policy v13 senza search;
2. policy v14 senza search;
3. stack prodotto v13 con belief v0, PIMC 16x8 e solver;
4. stesso stack con la sola policy sostituita da v14.

Nella finestra PIMC usiamo quattro replay con seed comuni. Una differenza e' detta
`stable` solo se resta tale in tutti i replay; questo separa meglio il cambio di policy
dalla normale casualita' delle determinizzazioni.

La fedelta' al log e' completa sulle parti deterministiche: ramo runtime 100%, fallback
100% e solver 100%. La search riproduce la carta live nell'81,3% dei casi v13 e
nell'89,3% dei casi v14, risultato atteso per una scelta campionaria senza il seed
originale del processo web.

## Risultati

| Confronto v13-v14 | Accordo | Differenze | Differenze stabili |
|---|---:|---:|---:|
| Policy pura | 87,78% | 325 | 325 |
| Runtime completo | 89,40% | 282 | 250 |
| Runtime, solo fallback | 87,98% | 244 | 244 |
| Runtime, solo search | 89,14% | 38 | 6 |
| Runtime, solo solver | 100,00% | 0 | 0 |

La CI95 bootstrap per partita colloca il tasso di differenza runtime tra 9,40% e 11,80%.
Le differenze sono quindi reali, ma il solver rende identici i due modelli nel finale e
gran parte delle differenze instabili della search viene dal campionamento.

Non emerge una concentrazione specifica nelle sconfitte umane: la differenza runtime e'
circa 10,99% negli stati di partite vinte dall'umano e 10,27% in quelli di partite vinte
dall'IA. Il pareggio e' uno solo e non va interpretato.

## Stile di gioco

Sulle stesse 2.660 decisioni v14 usa meno spesso una briscola piu' costosa del necessario:

| Metrica aggregata | v13 | v14 |
|---|---:|---:|
| Overkill, policy pura | 20,85% | 18,16% |
| Overkill su piatto povero, policy pura | 13,43% | 8,82% |
| Sprechi di briscola, policy pura | 3 | 1 |
| Overkill, runtime completo | 22,91% | 20,54% |
| Sprechi di briscola, runtime completo | 4 | 1 |
| Carichi in apertura, runtime completo | 14,96% | 14,89% |

Il miglioramento di overkill compare sia nel fallback (`19,67% -> 17,08%`) sia nella
search (`18,31% -> 14,49%`). Nel solver i due runtime scelgono la stessa carta: il suo
`40,98%` descrive mosse equivalenti scelte dal solver e non un difetto nuovo della policy.

## Decisione

- Non riaprire training anti-overkill, guard runtime o dose PIMC.
- Non usare il rapporto vittorie live 59/11 per stimare la forza relativa.
- Considerare confermata, su questo piccolo campione, la direzione comportamentale attesa
  di v14: meno briscole sovradimensionate e nessuna regressione localizzata evidente.
- Aggiornare l'audit quando arriveranno nuove partite, senza bloccare la ricerca: il volume
  live e' diagnostico e non diventera' consistente nel breve.

## Riproduzione

```bash
uv run python scripts/audit_live_policy_replay.py \
  --input data/prod_live_actions_v13.jsonl \
  --input data/prod_live_actions_v14.jsonl \
  --model-a data/models/best_a2c_v13.npz \
  --model-b data/models/best_a2c_v14.npz \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --runtime-repeats 4 \
  --seed 20260714 \
  --bootstrap-samples 5000 \
  --out-json data/live_policy_replay_v13_v14.json
```

