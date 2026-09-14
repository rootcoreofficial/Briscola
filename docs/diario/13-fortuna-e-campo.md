# Approfondimento — Quanto pesa il mazzo e cosa mostrano le prime vittorie umane

**Capitolo del diario:** [Capitolo 13](https://ai.briscola.dev/diario) · **Periodo:** 7 luglio 2026

## Quanto pesa la fortuna in una partita 1v1

I gate da 100k partite permettono di rispondere con precisione alla domanda del giocatore
("è più forte, ma non imbattibile: è normale?"). **La deviazione standard della differenza
punti per partita è ~28 punti** — misurata sia su v10 vs v9 sia su v10 vs heuristic_v1,
100k partite ciascuno, su 120 punti in palio e vittoria a 61. Il rumore di una singola
partita è più grande di quasi ogni divario di abilità reale. Traduzione in win rate
(artefatti in `benchmarks/experiments/fase3/`):

| Confronto (big 100k seat-fair) | Delta punti | Win rate |
|---|---|---|
| v10 vs v9 | +0.66 | 49.8% vs 47.3% (quasi coin flip) |
| PIMC belief 64×10 vs v10 | +4.2 | 56.5% |
| v10 vs heuristic_v1 | +20.5 | 75.9% (perde comunque 1 su 4–5) |

C'è anche un limite teorico già misurato altrove nel progetto: il value model full-game ha
**MAE ~14 punti irriducibile a inizio partita**. Prima ancora di giocare la prima carta,
circa ±14 punti del risultato sono scritti nell'ordine del mazzo. Un giocatore imbattibile
a briscola non può esistere; la skill non decide la partita, inclina la moneta. Corollario
operativo: per distinguere due giocatori servono centinaia o migliaia di partite — è il
motivo dei gate seat-fair a coppie.

## Il primo audit dell'event log di produzione

Per la prima volta il registro di produzione (event log Postgres, modalità `dataset`) è
stato usato come fonte di analisi. Prima di leggerlo, la congruenza export ↔ API: 427
partite riportate da `/version` = 427 nell'export; 326 finite in entrambi; `code_version`
più recente nel log = 0.29.2, la versione deployata; consenso 395/424. Le partite dei bot
del load test si escludono con `client_id != 'loadtest-bot'` (53 partite complete di bot).

Il raccolto umano: **123 partite complete** da 25 client distinti. Contro v10: **40
partite, 7 vittorie umane (17.5%)** — quasi tutte contro `bc_model_pimc_belief_64x10`
(7/36 = 19%), in linea con l'atteso ~20% per giocatori di livello ~heuristic. Nessuna
vittoria umana su `bc_model` liscio (0/4, campione minuscolo).

Nota sulle soglie d'uso di questi volumi: ~500–1.000 partite pulite bastano per audit e
decision quality (l'uso di questa sessione); ~4.000–10.000 per stime aggregate con CI
decenti; **≥100.000** è la soglia minima per usi da training — anni di traffico al ritmo
attuale, e il BC da imitazione è già stato chiuso negativo due volte anche con teacher
forti. Il valore dei dati di campo non è alimentare il training ma **orientarlo**: un
errore ricorrente diventa un'ipotesi misurabile.

## Il metodo dell'analisi delle 7 vittorie

- **Fortuna ricostruita esattamente**: in una partita completa l'unione delle `my_hand`
  dei due lati copre 40/40 carte, quindi il mazzo si ricostruisce senza stime.
  Indicatore: punti e briscole ricevuti da ciascun lato.
- **Giudice**: lo stack di produzione potenziato — PIMC belief su base v10 con
  **128 determinizzazioni × 5 seed** e finestra estesa a tutta partita; a mazzo vuoto
  verdetto **esatto** col solver. "Disaccordo forte" = giudice unanime 5/5 contro la
  mossa giocata.
- **Scoperta tecnica da non perdere**: il DTO azzera `trump_card` a `deck=0` (al client
  non serve più), quindi per rigiudicare l'endgame dagli eventi la briscola va
  ricostruita da `seen − out_of_play`. Fix negli script di analisi
  (`data/field_audit_20260707/`, locali).

## I verdetti, partita per partita

| Partita | Esito | Verdetto |
|---|---|---|
| 44895007 | 78–42 | Fortuna del mazzo dominante (91/29 punti ricevuti) |
| 27664781 | 68–52 | Fortuna (87/33) + taglio del carico guidato dall'IA |
| 3a053071 | 62–58 | Fortuna moderata + asso di briscola tenuto fino alla presa 20 |
| 240a5bd0 | 68–52 | Gioco umano efficace, fortuna ~pari |
| 7e64d700 | 86–34 | Fortuna lieve + endgame umano dominante |
| 895ceaa3 | 70–50 | Gioco umano forte, fortuna pari |
| **7678099e** | 61–59 | **Pistola fumante: vinta CONTRO la fortuna (42/78 punti ricevuti, asso e tre di briscola all'IA). Exploit puro.** |

## Il risultato centrale: zero errori, e un bias che il giudice non può vedere

Sulle 140 mosse dell'IA nelle 7 partite: **nessun disaccordo forte**; le 21 mosse di
endgame sono tutte **esattamente ottime** al solver. Ma il giudice appartiene alla stessa
famiglia della policy — stessi rollout self-play, stessa belief — ed è quindi **cieco ai
bias condivisi**. L'evidenza comportamentale li mostra:

- l'IA ha **aperto 9 carichi perdendone 8 (~111 punti)**, quasi sempre tagliati da una
  briscolina conservata; 7/8 nella finestra fallback (deck 22→8), dove gioca la policy
  nuda senza search. Il giudice APPROVA quelle aperture: nei suoi rollout l'avversario
  gioca come la famiglia self-play, e nessuno in famiglia conserva briscoline per tagliare;
- lo specchio: 22 dei 35 "disaccordi forti" sulle mosse umane sono il giudice che chiede
  di giocare subito il carico che l'umano tiene in mano — e gli umani hanno vinto lo stesso;
- economia delle briscole: gli umani hanno tagliato 11 volte incassando 119 punti; l'IA
  ha tagliato 20 volte, di cui **8 su piatti da ≤2 punti** — lo spreco è il vizio residuo
  misurabile;
- il pattern dei vincitori, in quattro gesti: aprire liscio, incassare i carichi da
  secondi (chi risponde, in due, non può essere tagliato), conservare l'asso di briscola
  e le briscoline per tagliare.

Questo pattern, codificato in un'euristica, è la sonda del capitolo successivo. Artefatti
locali (gitignored): `data/prod_*_20260707.*`, `data/field_audit_20260707/`; metodo e
numeri completi in `docs/plans/audit-campo-2026-07-07.md`.
