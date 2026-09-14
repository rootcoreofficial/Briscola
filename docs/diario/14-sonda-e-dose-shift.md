# Approfondimento — Dal test sullo stile umano alla v11: più maestro, meno volume

**Capitolo del diario:** [Capitolo 14](https://ai.briscola.dev/diario) · **Periodo:** 7–8 luglio 2026

## Le quattro regole del conservatore di briscole (`75da196`)

Lo stile dei 7 vincitori umani, codificato in `heuristic_trump_saver` (nel registry, non
esposta in UI — non è un avversario offerto ai giocatori, è uno strumento di misura):

1. aprire **liscio** (carte a 0 punti, mai carichi) e sondare senza regalare;
2. tenere assi e tre e incassarli **da secondi**: in 2-player chi risponde non può essere
   tagliato, quindi il carico giocato su presa vinta è punti in cassaforte;
3. **conservare le briscoline** per tagliare i carichi che l'avversario prima o poi guida;
4. non sprecare mai briscole su piatti poveri (≤2 punti) durante le pescate.

Come tutti gli agenti vede solo `PlayerObservation` (il card counting usa
`seen_cards_onehot`, informazione pubblica).

## Il presunto exploit umano non regge il test completo

Risultati (medium 10k seat-fair, `benchmarks/experiments/trump_saver/`):

| Confronto | Delta punti |
|---|---|
| trump_saver vs heuristic_v1 | **+10.47** (CI +9.88..+11.06) |
| trump_saver vs heuristic_v2 | +7.94 |
| trump_saver vs v10 | **−13.79** (vince ~30% delle partite) |
| heuristic_v1 vs v10 (stessa suite) | −20.87 |
| heuristic_v2 vs v10 (stessa suite) | −18.26 |
| trump_saver vs PIMC belief 64×10 su v10 (2k) | −15.90 |

La sonda è la rule-based più forte mai scritta nel repo. Ma il test per cui era nata dà
esito **negativo**: se v10 fosse davvero vulnerabile allo stile, contro la sonda avrebbe
dovuto rendere meno di quanto prevede la transitività (−20.87 + 10.47 ≈ **−10.4**);
misurato **−13.79**, cioè v10 fa MEGLIO della predizione, non peggio. Nessun exploit
differenziale a livello di policy. Diagnosi: **bias di selezione** — avevamo analizzato
solo le 7 vittorie, non le 29 sconfitte di giocatori dallo stile simile contro lo stesso
avversario. La perdita sui carichi guidati è reale, ma v10 compensa altrove. La sonda
resta in squadra con un mestiere nuovo: **−13.79 è la baseline anti-regressione**, il
numero da battere *in differenziale* per dire che una generazione futura ha chiuso il buco.

## Il maestro con search ha ancora qualcosa da insegnare

Seconda misura della giornata: PIMC belief su BASE v10 contro v10+solver, 4.000 seat-fair
(ricetta identica alla conferma v0.23.0):

| Config | Edge | CI 95% | Costo |
|---|---|---|---|
| 16×8 | **+3.37** | +3.06..+3.69 | 6.2 ms/mossa pensata |
| 64×10 | **+3.87** | +3.51..+4.23 | 36.7 ms/mossa pensata |

Edge **invariato** rispetto a base v8 (+3.66): 30M partite di sparring non l'hanno
assorbito, perché il vantaggio è strutturale (la search media sul rumore del mazzo, cosa
che una rete reattiva non può replicare). E la config agile 16×8 tiene l'**87%
dell'edge a 1/6 del costo**. Il ramo PIMC-as-teacher esce dal frigo.

## Il training v11: più maestro, non più partite

Ipotesi dose-shift: spostare dose dal maestro consumato (value-lookahead: +1.80 su v8 e
in calo) a quello intatto. A2C **5M partite** (6× meno di v10), base e maestri su v10,
seed 20260707, iperparametri v10 (lr 3e-4, entropy 5e-4, gamma 1.0), mix:
`bc_model:0.15, bc_model_pimc_belief 16×8:0.40, VL_8x8:0.20, heuristic_v1:0.08,
heuristic_v2:0.12, random:0.05`. Throughput ~26k partite/min, run completato in 3h33m.
Gate dichiarato prima del lancio: big seat-fair vs v10, successo = +0.3..+0.5, sotto
+0.2 il ramo sparring si chiude.

| Esame (big 100k seat-fair, `fase3/v11_vs_*.json`) | Esito |
|---|---|
| vs v10 | **+0.85** (CI coppie +0.71..+0.99) — sopra la banda di successo |
| vs heuristic_v1 | **+20.80** (CI +20.62..+20.97) — record assoluto (era +20.52) |
| vs trump_saver | +14.34 (v10: 13.79, misurato però su medium 10k domain) |

La curva dei rendimenti generazionali (+2.46 → +0.97 → +0.66) si è **rialzata** a +0.85:
conta la dose e la qualità del maestro, non il volume. Contro la sonda il guadagno è
proporzionale alla forza generale — nessun progresso differenziale sul fianco "umano",
atteso: la sonda non era nel mix di training. Promosso `best_a2c_v11` (release v0.31.0).

## La vecchia protezione anti-spreco ora fa perdere punti

Il gate era girato con v11 senza `overkill_guard` e v10 col guard: prima di promuovere,
la variabile andava isolata. Rifatto il gate con v11+guard: **+0.32** contro il +0.85
senza — il guard oggi COSTA ~0.5 punti a partita. La protesi anti-spreco del capitolo 3,
utile alla v6, è diventata una stampella su una gamba guarita: gli "overkill" di v11 sono
scelte deliberate. Promosso senza guard; ogni aiuto scritto a mano va rimisurato a ogni
generazione.

## v12: aggiungere il conservatore al training non cambia lo stile

Prerequisito fatto in giornata (`d2888be`): la sonda tradotta nel fast path e nei kernel
Numba con parità ESATTA a tre motori (il +10.47 su `heuristic_v1` riprodotto identico
campo-a-campo con `--engine fast`), quindi usabile in `--opponent-mix`. Bonus della
traduzione: trovato e corretto un range check hardcoded nel collector A2C numba
(`codes <= 3`) che avrebbe rifiutato in silenzio qualsiasi nuovo agente nel mix.

Run v12 lanciato nella notte: **10M partite**, seed 20260708, base e maestri su v11, mix
`bc_model:0.15, pimc_belief 16×8:0.40, VL_8x8:0.20, trump_saver:0.12, heuristic_v1:0.04,
heuristic_v2:0.06, random:0.03` — il castigatore al 12%, dose presa dalla quota bar.
Esito (2026-07-08, `fase3/v12_vs_*.json`): **NEGATIVO, niente promozione**.

| Termometro | v12 | v11 | Lettura |
|---|---|---|---|
| big vs v11 | **+0.11** (CI −0.03..+0.25) | — | non significativo |
| big vs trump_saver | +14.80 | +14.34 | +0.46, in parte familiarità col partner deterministico |
| big vs heuristic_v1 | +20.74 | +20.80 | record invariato |
| carichi guidati (2k strumentate vs saver) | 11.1% | 10.4% | identico |
| carichi guidati persi | 71.3% | 71.4% | identico |
| briscole su piatti ≤2 punti | 6.7% | 6.8% | identico |

Il contatore comportamentale è la parte più eloquente: dopo 10 milioni di partite con il
castigatore nella rosa, la v12 guida i carichi **esattamente come la v11** e li perde
nella stessa proporzione. Due diagnosi da tenere:

1. **il maestro non si consuma, ma l'allievo si satura**: da +0.85 a +0.11 con edge
   maestro identico (+3.36 confermato su base v11);
2. **la punizione diluita non sposta un comportamento che paga in media**: guidare un
   carico costa ~−11 punti attesi contro il saver (12% del mix di training) ma paga contro il
   restante 88% — una policy *incondizionale* fa la media e resta ferma. Il vizio n.1
   (carichi contro i conservatori di briscole) è intrinsecamente **condizionale**:
   richiede il riconoscimento dello stile avversario dalla storia della partita (Fase 4),
   oppure resta coperto dalla search a runtime.

v12 resta come artefatto locale non promosso. La prossima ipotesi in coda (v13, scelta
dal maintainer) attacca l'altro vizio misurato sul campo — l'economia di briscola — con
un reward shaping potential-based; razionale in `PLAN.md` e in
`docs/plans/audit-campo-2026-07-07.md` §7.
