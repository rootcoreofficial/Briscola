# Piano operativo — Briscola AI

> Questo file è volutamente breve: fotografa lo stato reale e le prossime decisioni.
> Cronologia e numeri completi vivono in `docs/plans/`, nel
> [diario di bordo](https://ai.briscola.dev/diario) e in `docs/diario/`.

## Stato Corrente

- Release corrente del repository: `0.39.0`. Il push di `master` attiva automaticamente il deploy su
  <https://ai.briscola.dev>; include la UI replay mobile e `POST /api/replay/advice`. Dopo ogni release verificare
  `/version`, l'endpoint replay, asset ausiliari ed event log Postgres.
- Il default effettivo della UI è `bc_model_pimc_belief_12x8` su `best_a2c_v15.npz`, senza guard runtime
  anti-overkill. V14 con 16×8 resta selezionabile, insieme a v13/v11/v10; 64×10 resta la variante massima.
- `best_a2c_v15.npz` usa encoder v4 (`feature_dim=369`, hidden 256) e un solo forward normale. È distillata dalla
  media esatta dei logits del checkpoint 20M sulle 24 rinomine dei semi, su 250.000 partite e 9,5 milioni di
  decisioni. Il flip dell'argmax scende al **`2,9456%`** e l'overkill sui piatti poveri al **`1,0637%`**.
- I gate v15 separano forza ed efficienza: policy-only vs v14 **`+0,18046`** (CI95 `+0,08..+0,28`) su 100.000
  partite; PIMC belief 12×8 vs v14 16×8 **`+0,1052`** (CI95 `-0,1126..+0,3230`) su 20.000, con latenza media e p95
  circa `0,75x`. È una promozione di non inferiorità/costo, non una prova di forza runtime superiore. Il controllo
  omogeneo big Numba contro `heuristic_v1` fa `+21,86688` ed è l'ultima riga del grafico del report.
- I tentativi precedenti paired-RL, forward-KL e margin hinge restano chiusi: non hanno raggiunto insieme il gate di
  simmetria e quello di forza. Cronologia, criteri e artefatti sono in `docs/plans/suit-*.md`; il resoconto completo
  della policy promossa è `docs/plans/suit-distillation-v0-2026-07-11.md`.
- La pista capacità ReLU è chiusa. V14 ha 123/256 unità quasi inattive; l'ablation congiunta è neutra su 10.000
  partite. Reset 8/16 le riattiva davvero, ma reset 16 migliora la KL validation solo dello **`0,328%`** rispetto
  allo stesso training senza reset, sotto il gate dell'`1%`. **STOP widening, potatura e altri reset.** Report:
  `docs/plans/hidden-unit-diagnostic-v0-2026-07-12.md` e `docs/plans/dormant-reinitialization-screen-v0-2026-07-12.md`.
- Runtime web zero-Numba: dominio, search PIMC e solver usano Python nel processo web; Numba resta per training,
  valutazioni e benchmark. In produzione: FastAPI Cloud, Redis per stato/pub-sub, Postgres in modalità `dataset`.
- Il catalogo modelli espone v15 come policy compatibile e non espone value/belief (`value_mlp_v1`/`belief_mlp_v1`),
  che sono asset interni. Policy v10/v11/v13/v14/v15, value e belief necessari al runtime sono tracciati in Git; gli altri
  `.npz`, dataset e benchmark restano locali e gitignored.
- L'event log live contiene `human_action`, `ai_action`, `game_finished`, consenso e metadati modello.
  `export_live_actions.py` produce la sequenza unica umano+IA e può filtrare versione, agente, modello e bot.
- Il primo replay appaiato di campo usa 70 partite complete (59 v13, 11 v14) e 2.660 decisioni non forzate. Sulle
  stesse osservazioni i runtime concordano nell'`89,40%` dei casi; non emerge un difetto v14 localizzato e v14 riduce
  ancora overkill (`22,91% -> 20,54%`) e sprechi (`4 -> 1`). Il dato live resta diagnostico e non bloccante:
  `docs/plans/live-policy-replay-v13-v14-2026-07-14.md`.
- La pista belief v1 multi-stile è chiusa con **STOP runtime**. Sui sette holdout migliora BCE macro del `7,63%`
  relativo e top-k di `+3,29` punti, ma nel PIMC 16×8 perde `-0,224` punti/partita contro v0 (CI95
  `-0,572..+0,124`) e fallisce lo screen preregistrato. Nessuna conferma 10k: belief v0 resta ufficiale. Report:
  `docs/plans/belief-v1-multistile-2026-07-14.md`.
- La diagnostica passiva A2C su tre seed × 2.000 partite supera tutti i gate. Il critic, pur ripartendo da zero,
  raggiunge explained variance mediana finale `0,127..0,133`; bias advantage `0,147..0,174`, picco gradiente
  `p95/mediana 1,92..2,00` e passi relativi sotto `0,019%` non indicano instabilità. **STOP**, per ora, a critic
  reuse, normalizzazione e clipping. Report:
  `docs/plans/a2c-health-diagnostic-v0-2026-07-14.md`.
- La schedule A2C realmente paired e' implementata e validata, ma il confronto 3 seed e' **inconcludente**. A pari
  20k partite perde direttamente in due seed su tre (mediana `-0,151`) e la dispersione della forza e' `2,08x`
  quella seriale; a pari 20k mazzi richiede 40k partite e resta direttamente neutra (`-0,148` mediano). Nessun run
  piu' lungo: `serial` resta il default. Report: `docs/plans/a2c-paired-schedule-v0-2026-07-14.md`.
- L'audit policy regret 192x64 e' chiuso con `no_policy_error_signal`. Su 96 decisioni early/mid trova 36 alternative
  candidate ma **zero errori affidabili al 99%**; i 14 errori confermati sono tutti gia' nelle finestre PIMC (9) o
  solver (5). Nessun cluster autorizza training o architetture nuove. Report:
  `docs/plans/policy-regret-audit-v14-2026-07-14.md`.
- Lo split supervisionato BC/value e' ora per partita, non per record: default 80/10/10, test finale separato,
  provenance con conteggi/digest e rifiuto degli NPZ storici senza `game_ids`. Anche il value pairwise raggruppa
  tutte le root della stessa partita. Protocollo: `docs/plans/dataset-split-per-partita-2026-07-14.md`.
- `decision_quality` assegna ora lo stesso stream RNG a ogni coppia seat-fair nel percorso seriale e parallelo:
  `--workers` cambia solo la velocita', anche con agenti stocastici. Riproduzione e compatibilita':
  `docs/plans/decision-quality-rng-2026-07-14.md`.
- Lo scouting A2C seriale da 50M e' concluso con **STOP scaling**. Nessuno dei checkpoint 10/20/30/40/50M supera
  insieme forza e simmetria sulla suite preregistrata; il gate finale resta sigillato. Un audit esplorativo separato
  da 400k partite seleziona 20M, ma la conferma indipendente lo trova pari a v14 (`+0,0286`, CI95
  `-0,0741..+0,1313`) e nettamente sopra v13 (`+0,8918`, CI95 `+0,7568..+1,0268`). Quindi la scala conserva v14
  ma non aggiunge forza misurabile; niente repliche o promozione del checkpoint grezzo. Protocollo e ricevute:
  `docs/plans/a2c-super-training-50m-2026-07-14.md`.
- Il percorso teacher 20M -> corpus sharded 250k -> student è concluso e promosso. Il corpus contiene 9.500.000
  esempi; sul test separato la KL cala dell'`85,9%`, l'accordo argmax arriva al `97,86%` e l'asset ufficiale compatto
  `best_a2c_v15.npz` ha SHA-256 `2f2dca3d4e77a363783124feeb30f482a85a740077222936b025b37b865f2eb6`.
  Protocollo: `docs/plans/suit-distillation-20m-250k-2026-07-17.md`.
- Il gate PIMC 16×8 dello student era neutro e ha chiuso la pretesa di maggiore forza. Il follow-up separato 8×8 è
  fallito; 12×8 ha invece superato screen, conferma 20k e audit browser. La decisione finale è quindi v15 12×8 come
  miglior compromesso costo/forza, con v14 16×8 conservato. Ricevute:
  `docs/plans/suit-student-12x8-efficiency-2026-07-17.md` e
  `docs/plans/suit-student-12x8-release-audit-2026-07-17.md`.
- Il diario pubblico arriva al capitolo 21: dopo il plateau della forza racconta perche' v15 e' una promozione di
  efficienza, non la prova di un giocatore piu' forte. Approfondimento: `docs/diario/23-dodici-mondi-bastano.md`.
- La produzione ha attualmente l'override `BRISCOLA_DEBUG_STATE_ENDPOINT=unsafe-full-state`: il tasto `S` funziona,
  ma chi conosce un game id può leggere mani e prossima carta. Il codice resta chiuso per default; rimuovere l'override
  quando il debug pubblico non serve più.

## Prossima Decisione

Le sette piste, l'audit degli errori residui e l'eccezione 50M hanno un esito chiuso. La semplice continuazione A2C
non e' una leva di forza oltre v14; v15 migliora il compromesso di esecuzione tramite distillazione e search ridotta.
Rami precedenti e motivazioni complete in
`docs/plans/prossima-iterazione-modello.md`.

1. **Verificare la release 0.39.0 dopo il push.** `/version` deve riportare v15 presente e
   `POST /api/replay/advice` deve restituire un suggerimento; il catalogo deve mostrare v15 compatibile, nascondere
   value/belief e mantenere selezionabili v14/v13/v11/v10. Una partita reale deve usare
   `bc_model_pimc_belief_12x8` senza errori client o server.
2. **Monitorare v15 senza riaprire il gate.** Raccogliere latenza, errori e replay live; non interpretare il piccolo
   `+0,1052` come prova di forza superiore e non ripetere la stessa suite finché non emerge un segnale indipendente.
3. **Non ridurre PIMC a 8x8.** Lo screen student 8x8 contro v14 16x8 dimezza latenza media e p95 (`~0,51x`) senza
   errori, ma perde `-0,742` punti/partita con CI95 interamente negativa (`-1,457..-0,027`). **STOP** prima della
   conferma 20k: il risparmio non mantiene la forza quasi equivalente. Protocollo:
   `docs/plans/suit-student-8x8-efficiency-2026-07-17.md`.
4. **Chiarire la divergenza Numba/domain.** Sullo stesso screen 50M da 4k, Numba non riproduce gli aggregati domain.
   Fino a una parita' policy end-to-end, i confronti di forza sotto il punto usano il dominio canonico.
5. **Aggiornare periodicamente il replay live.** Le nuove partite v15 aumentano la confidenza comportamentale, ma il
   basso volume previsto non blocca la diagnostica e non diventa training senza una nuova decisione su privacy e qualita'.
6. **Per altre linee di forza, misurare prima il soffitto.** Serve un cluster di errori ripetibile oppure un benchmark
   ridotto a informazione nascosta con riferimento piu' forte di PIMC. Solo un gap dimostrato autorizza ricerca su
   information set/regret; niente altro training massivo della ricetta corrente.

Critic reuse, normalizzazione, clipping, nuove architetture e Q Monte Carlo restano sospesi finché una misura non
mostra il problema specifico che dovrebbero risolvere.

## Audit Di Campo

- Le piste carichi guidati, timing dell'asso di briscola e cavata con mano lunga sono chiuse dalle ablation
  controfattuali: non riaprirle sulla base di singoli aneddoti. Riferimenti:
  `docs/plans/audit-campo-2026-07-07.md` e capitolo 17.
- Il replay appaiato corrente non confronta i rapporti vittorie grezzi: mostra v13 e v14 sulle stesse osservazioni e
  separa fallback, search e solver. Aggiornarlo quando il volume v14 cresce, separando versione e bot/load test.
- I dati umani restano diagnostici: niente training prima di rivalutare volume, consenso, qualità e privacy.

## Vincoli Operativi

- Anti-cheat: agenti, reward e modelli ricevono solo `PlayerObservation`; la vista full-state è debug opt-in e deve
  restare `403` di default.
- Se cambia una regola o l'osservazione, mantenere allineati dominio canonico, fast path e Numba con test di parità.
- Valutazioni serie: seat-fair paired, stesso mazzo e posti scambiati, CI sulle coppie. Miglioramenti sotto il punto
  richiedono anche più seed di training: la CI di evaluation non misura la varianza del training.
- Confrontare varianti search a pari CPU media e p95, non soltanto a pari numero di determinizzazioni.
- I job oltre ~5 minuti vanno preparati per il maintainer con `nohup`, log e path artefatti; non vanno avviati
  dall'agente restando in attesa.
- Gli artefatti e le ricevute del run 50M restano immutabili. Per confronti policy sotto il punto usare `domain` finche'
  la divergenza osservata con il percorso Numba non e' spiegata e coperta da un test end-to-end su modelli reali.
- Ogni bump richiede `pyproject.toml` + `uv.lock`, tag annotato e rigenerazione/verifica di
  `docs/reports/model_progress.xlsx`; il catalogo deve mostrare la policy compatibile e nascondere value/belief.

## Debito Aperto

- Cold start residuo (~13.7s) è il pavimento della piattaforma; leve reali: keep-alive sotto l'idle timeout o piano
  diverso. Il runtime applicativo non ha più compilazione JIT.

## Comandi Utili

Quality gate:

```bash
uv run ruff format src tests scripts
uv run ruff check --fix src tests scripts
uv run mypy src
make docs-check
uv run pytest
```

Verifica deploy e catalogo:

```bash
curl -sS https://ai.briscola.dev/version
curl -sS https://ai.briscola.dev/api/ai/models
```

Export live v15 per il prossimo audit:

```bash
DATABASE_URL=... uv run python scripts/export_live_actions.py \
  --code-version 0.38.0 \
  --ai-agent bc_model_pimc_belief_12x8 \
  --ai-model-id best_a2c_v15.npz \
  --exclude-client-id loadtest-bot \
  --out data/prod_live_actions_v15.jsonl
```

Gate locale e report:

```bash
uv run python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model --agent0-model data/models/best_a2c_v15.npz \
  --agent1 bc_model --agent1-model data/models/best_a2c_v14.npz
uv run python scripts/build_model_report.py
```

Sonda riproducibile di simmetria dei semi:

```bash
uv run python scripts/probe_suit_symmetry.py \
  --model data/models/best_a2c_v15.npz \
  --out-json data/suit_symmetry_v15.json
```

Avvio locale:

```bash
briscola-server --reload
```
