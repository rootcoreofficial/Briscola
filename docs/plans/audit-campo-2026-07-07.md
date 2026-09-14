# Audit dei dati di campo e sonda trump-saver (2026-07-07)

> Documento di dettaglio della sessione 2026-07-07: quantificazione della fortuna,
> primo audit delle partite umane di produzione, analisi delle 7 vittorie contro v10,
> sonda di exploitability `heuristic_trump_saver`, sonda edge-maestro su v10 e lancio
> del run v11. Serve come base per l'analisi dei risultati (gate v11/v12) e per il
> diario di bordo. Il riassunto operativo vive in `PLAN.md`; qui ci sono numeri,
> metodo e ragionamenti completi.

## 1. Quanto pesa la fortuna in una partita 1v1 (quantificazione)

Domanda di partenza: "v10 è più forte ma non imbattibile: è normale?" Sì, e i gate
da 100k partite permettono di quantificarlo:

- **Deviazione standard della differenza punti per partita: ~28 punti** (misurata sia
  su v10 vs v9 sia su v10 vs heuristic_v1, 100k partite ciascuno; 120 punti in palio,
  vittoria a 61). Il rumore di una singola partita è più grande di quasi ogni divario
  di abilità: la skill non decide la partita, inclina la moneta.
- Traduzione in win rate (100k partite, `benchmarks/experiments/fase3/`):
  - v10 vs v9: +0.66 punti medi → 49.8% vs 47.3% (quasi coin flip);
  - PIMC belief 64×10 vs v10: +4.2 → vince solo il 56.5%;
  - v10 vs heuristic_v1: +20.5 → vince il 75.9% (perde comunque 1 su 4-5).
- Il value model full-game ha **MAE ~14 punti irriducibile a inizio partita**: prima
  di giocare, ~±14 punti del risultato sono già scritti nell'ordine del mazzo. Un
  giocatore imbattibile a briscola non può esistere.
- Corollario: per distinguere statisticamente due giocatori servono centinaia/migliaia
  di partite (da qui i gate seat-fair a coppie).

## 2. Volume dei dati di campo: soglie d'uso

- **~500–1.000 partite pulite**: audit e decision quality (l'uso di questo documento).
- **~4.000–10.000**: stime aggregate con CI decenti (winrate vs umani, segmentazioni).
- **≥100.000**: soglia minima per usi da training (BC su mosse umane, opponent
  modeling) — anni di traffico al ritmo attuale, e il BC da imitazione è già stato
  chiuso negativo due volte anche con teacher forti. Il valore dei dati di campo non è
  alimentare il training ma **orientarlo** (errore ricorrente → ipotesi misurabile).

## 3. Primo audit di produzione (export 2026-07-07)

Congruenza export ↔ API verificata: 427 partite in `/version` = 427 nell'export;
326 finite in entrambi; `code_version` più recente nel log = 0.29.2 deployata;
consenso 395/424. Filtro bot: `client_id != 'loadtest-bot'` (53 partite complete bot).

Partite umane complete: **123** (25 client distinti). Vs v10: **40 partite, 7 vittorie
umane (17.5%)** — quasi tutte contro `bc_model_pimc_belief_64x10` (7/36 = 19%), in
linea con l'atteso ~20% per giocatori di livello ~heuristic. Nessuna vittoria umana
su `bc_model` liscio (0/4, campione piccolo).

Artefatti (locali, gitignored): `data/prod_*_20260707.*` (export grezzi),
`data/field_audit_20260707/` (script e risultati dell'analisi: riepilogo partite,
fortuna, giudizi mossa-per-mossa, pattern umani).

## 4. Analisi delle 7 vittorie umane (metodo e verdetti)

**Metodo.** Fortuna del mazzo ricostruita esattamente (unione delle `my_hand` di
entrambi i lati = 40/40 carte); indicatore: punti+briscole ricevuti. Giudice: stack di
produzione potenziato — PIMC belief su base v10, **128 determinizzazioni × 5 seed**,
finestra estesa a tutta partita; a mazzo vuoto verdetto esatto col solver.
"Disaccordo forte" = giudice unanime 5/5 contro la mossa giocata. Scoperta tecnica da
non perdere: il DTO azzera `trump_card` a deck=0 → per giudicare l'endgame va
ricostruita la briscola da `seen − out_of_play` (fix negli script di analisi).

**Verdetti per partita** (dettaglio completo in `data/field_audit_20260707/`):

| Partita | Esito | Verdetto |
|---|---|---|
| 44895007 | 78-42 | Fortuna del mazzo dominante (91/29 punti ricevuti) |
| 27664781 | 68-52 | Fortuna (87/33) + taglio del carico guidato dall'IA |
| 3a053071 | 62-58 | Fortuna moderata + asso di briscola tenuto fino a t20 |
| 240a5bd0 | 68-52 | Gioco umano efficace, fortuna ~pari |
| 7e64d700 | 86-34 | Fortuna lieve + endgame umano dominante |
| 895ceaa3 | 70-50 | Gioco umano forte, fortuna pari |
| **7678099e** | 61-59 | **Pistola fumante: vinta CONTRO la fortuna (42/78, asso+3 di briscola all'IA), exploit puro** |

**Risultato centrale.** Zero errori interni dell'IA (140 mosse: nessun disaccordo
forte; 21 mosse endgame tutte esattamente ottime al solver). MA il giudice è la stessa
famiglia della policy (stessi rollout, stessa belief): è cieco ai bias condivisi.
L'evidenza comportamentale li mostra:

- l'IA ha **aperto 9 carichi perdendone 8 (~111 punti)**, quasi sempre tagliati da una
  briscolina conservata; 7/8 nella finestra fallback (deck 22→8), dove gioca la policy
  nuda. Il giudice APPROVA quelle mosse: i rollout self-play assumono un avversario
  che non conserva briscoline per tagliare;
- specchio: 22/35 dei "disaccordi forti" sulle mosse umane sono il giudice che chiede
  di giocare subito il carico che l'umano tiene — e gli umani hanno vinto lo stesso;
- economia briscole: umani 11 tagli → 119 punti; IA 20 tagli di cui **8 su piatti da
  ≤2 punti** (lo spreco è il vizio residuo misurabile);
- pattern dei vincitori: aprire liscio, incassare i carichi da secondi, conservare
  asso di briscola e briscoline per tagliare.

## 5. Sonda di exploitability `heuristic_trump_saver` (commit 75da196)

Lo stile dei vincitori codificato in un'euristica di dominio (registry, non in UI).
Risultati (medium 10k seat-fair, `benchmarks/experiments/trump_saver/`):

| Confronto | Esito |
|---|---|
| trump_saver vs heuristic_v1 | **+10.47** (CI +9.88..+11.06) |
| trump_saver vs heuristic_v2 | +7.94 |
| trump_saver vs v10 | **−13.79** (vince il 30%) |
| heuristic_v1 vs v10 (stessa suite) | −20.87 |
| heuristic_v2 vs v10 (stessa suite) | −18.26 |
| trump_saver vs PIMC-belief 64×10 su v10 (2k) | −15.90 |

**Lettura (negativa per l'ipotesi exploit).** La sonda è la rule-based più forte del
repo, ma contro v10 rende −13.79, PEGGIO della predizione per transitività (~−10.4):
nessun exploit differenziale a livello di policy. Diagnosi: **bias di selezione** —
avevamo analizzato solo le 7 vittorie, non le 29 sconfitte di giocatori dallo stile
simile. La perdita sui carichi guidati è reale ma v10 compensa altrove. La sonda resta
preziosa come baseline anti-regressione: **−13.79 è il numero da battere in
differenziale** per dire che un modello futuro ha chiuso il buco.

## 6. Sonda edge-maestro su v10 e run v11 (in corso)

PIMC belief su BASE v10 vs v10+solver, 4.000 seat-fair (ricetta identica alla conferma
v0.23.0): **16×8 +3.37** (CI +3.06..+3.69, 6.2 ms/mossa), **64×10 +3.87**
(CI +3.51..+4.23, 36.7 ms). Edge INVARIATO da base v8 (+3.66): lo sparring non assorbe
il vantaggio strutturale della search (media sul rumore del mazzo) — il maestro non si
consuma, e a 16×8 costa 1/6 del 64×10 tenendo l'87% dell'edge.

**Run v11 lanciato 2026-07-07 ~18:02** — ipotesi: spostare dose dal maestro consumato
(VL: +1.80 su v8 e in calo) a quello intatto (PIMC). A2C 5M, base e maestri su v10,
mix `bc_model:0.15, bc_model_pimc_belief:0.40 (16 det, finestra 8), VL_8x8:0.20,
heuristic_v1:0.08, heuristic_v2:0.12, random:0.05`, iperparametri v10 (lr 3e-4,
entropy 5e-4, gamma 1.0), seed 20260707, `--metrics-mode summary`, checkpoint ogni 1M.
Throughput osservato ~26k partite/min (1° checkpoint a +38'): fine attesa ~21:15.
Out: `data/models/a2c_v11_pimc16x8_dose40_5M_seed20260707.npz`,
log `data/train_v11_pimc16x8_dose40_5M_seed20260707.log`.
**Gate: big seat-fair vs v10; successo = +0.3..+0.5; sotto +0.2 il ramo sparring si
chiude.** Scelte scartate con evidenza: 16×10 (Pareto: +0.25 n.s. vs 16×8), mix di più
config PIMC nello stesso run (non supportato dal trainer: una sola
`--opponent-pimc-determinizations`; beneficio atteso minimo, 64×10 costa 6×).

## 7. Ipotesi v12: insegnare a non farsi fregare (e la differenza con "saper fregare")

Il bias di famiglia (carichi guidati "sicuri" perché il curriculum self-play non
contiene avversari che conservano briscoline) si cura con **diversità nel cartellone**:
trump_saver come avversario di training. Punti fermi del ragionamento:

- **Lo sparring insegna a battere, non a imitare** (lezione già dimostrata dal
  programma distillazione): contro trump_saver la policy impara la DIFESA (non offrire
  carichi tagliabili) e le contromisure, non lo stile in sé.
- L'ATTACCO (tagliare carichi guidati) v10 già lo possiede (reward immediato nel
  self-play). Il pezzo fine è l'**economia** (non sprecare briscole su piatti poveri):
  credit assignment lungo, il vizio residuo osservato sul campo.
- "Applicarla quando serve" è un **dosaggio condizionale**, non uno stile fisso: ha
  valore solo contro chi offre prede, e regredisce quando la popolazione impara la
  difesa (equilibrio). L'encoder v4 (storia delle prese) dà alla policy la materia
  prima per condizionarsi allo stile dell'avversario, ma solo un roster con ENTRAMBI
  gli stili (chi guida carichi e chi no) crea il segnale per impararlo.
- Canali alternativi valutati e messi in coda: reward shaping "non guidare carichi con
  briscole ignote" (rapido ma rischia di proibire guide giuste — seconda leva);
  rollout diversi nella search (non copre la finestra fallback 22→8, dove serve);
  belief fine-tune su mosse umane (dataset ancora troppo piccolo, ~6.8k azioni).

**Gate del ciclo v12 (tre termometri):**
1. differenziale vs trump_saver (baseline: v10 = −13.79 per la sonda → il nuovo
   modello deve guadagnare contro la sonda PIÙ che in generale);
2. big seat-fair vs v10/v11 (forza generale, no regressioni);
3. contatore comportamentale: briscole spese su piatti ≤2 punti (oggi 8/20 sul campo).

Prerequisito ingegneristico: traduzione fast/numba di trump_saver con parità a tre
motori — **FATTO** (2026-07-07, commit d2888be): parità esatta (eval fast 10k identica
al dominio campo-a-campo), agganciata a `--engine fast` e a `--opponent-mix` di
train_a2c (smoke train ok), 530 test verdi con compilazione numba fredda. Bonus: trovato
e corretto un range check hardcoded nel collector A2C numba (`codes <= 3`) che avrebbe
rifiutato qualsiasi nuovo codice agente nel mix.

## 8. Note operative

- Bump proposto: 0.30.0 (nuovo agente + `evaluate_agents` esteso a tutti gli agenti
  `.npz` del catalogo); decidere se accorpare alla promozione v11.
- Materiale diario: §1 (fortuna), §4 (le 7 partite, la pistola fumante 7678099e),
  §5 (l'exploit che non c'era: onestà del metodo), §7 (difesa ≠ imitazione).
