# Approfondimento — Perché v12 non ha cambiato stile e cosa misurare adesso

**Capitolo del diario:** [Capitolo 16](https://ai.briscola.dev/diario) · **Periodo:** 8 luglio 2026

## La domanda dopo v11

v11 ha migliorato la forza generale, ma l'audit delle partite umane aveva lasciato aperto
un punto preciso: contro giocatori che conservano le briscoline, l'IA tende ancora a
guidare carichi e a farseli tagliare. La sonda `heuristic_trump_saver` non aveva confermato
un exploit generale, però il comportamento restava misurabile. La domanda naturale era:
se mettiamo quel conservatore nel training, l'allievo impara a riconoscerlo?

Il run v12 ha testato proprio questo. Ricetta: 10M partite, seed 20260708, base e maestri
su v11, con `heuristic_trump_saver` nel 12% degli incontri di training:

```text
bc_model:0.15
pimc_belief 16x8:0.40
value_lookahead_8x8:0.20
trump_saver:0.12
heuristic_v1:0.04
heuristic_v2:0.06
random:0.03
```

## Risultato: forza quasi uguale, comportamento uguale

I gate non giustificano una promozione:

| Termometro | v12 | Lettura |
|---|---|---|
| vs v11 | **+0.11** (CI −0.03..+0.25) | compatibile con zero |
| vs trump_saver | +14.80 (v11: +14.34) | piccolo guadagno, anche da familiarità |
| vs heuristic_v1 | +20.74 (v11: +20.80) | record invariato |

La parte più importante non è il +0.11. È il profilo delle mosse, misurato su 2k partite
strumentate contro il saver:

| Contatore | v12 | v11 |
|---|---|---|
| carichi guidati | 11.1% | 10.4% |
| carichi guidati persi | 71.3% | 71.4% |
| briscole su piatti ≤2 punti | 6.7% | 6.8% |

In pratica v12 non ha cambiato abitudine. Ha visto il conservatore in allenamento, ma lo ha
visto solo come una quota del mondo. Contro di lui guidare un carico costa; contro molti
altri avversari può ancora pagare. Una policy che non sa distinguere gli stili fa la media
del gruppo di avversari, quindi resta quasi dov'era.

## La pagella della nonna: non forza, ma abitudini

Per capire cosa guardare dopo v12 è nato `scripts/behavior_profile.py`. Il nome è leggero,
ma il ruolo è serio: non valuta se il modello vince, valuta *come* gioca. Le metriche sono
prese dai consigli tradizionali e dall'audit di campo:

- apre liscio, cioè con carte a zero punti non di briscola;
- guida carichi e misura quanti vengono persi;
- taglia piatti poveri con briscola;
- regala punti quando perde una presa da risponditore;
- scarta dal seme corto, cioè prova a "sbiancarsi";
- tiene l'asso di briscola per il finale;
- "cava le briscole", cioè esce a briscola bassa quando ne ha tante.

Profilo v11 su 2k partite per avversario (`data/behavior_profile_v11_20260708.json`):

| Metrica | Saver | Specchio | heuristic_v1 |
|---|---:|---:|---:|
| apertura liscia | 39.51% | 39.08% | 39.18% |
| carichi guidati | 10.42% | 13.27% | 11.44% |
| carichi guidati persi | 71.38% | 72.20% | 50.08% |
| briscola su piatto povero | 6.75% | 5.86% | 6.29% |
| punti regalati per presa persa | 0.97 | 0.95 | 0.86 |
| cavate con mano lunga di briscole | 0.00% | 0.00% | 0.00% |
| cavate con mano corta di briscole | 18.58% | 16.19% | 18.75% |
| scarto dal seme corto | 76.90% | 71.19% | 76.78% |
| presa media dell'asso di briscola | 16.58 | 14.82 | 16.65 |

La lettura è mista. v11 ha imparato da sola alcune abitudini buone: regala pochissimo nelle
prese perse, si sbianca spesso dal seme corto e tende a tenere l'asso di briscola. Restano
due punti deboli: i carichi contro i conservatori e qualche spreco di briscole su piatti
poveri.

La sorpresa vera è la cavata. Con mano lunga di briscole (≥4), la regola tradizionale dice
di uscire ogni tanto a briscola bassa per far consumare quelle dell'altro. v11 fa l'opposto:
0.0% su tre avversari. Con mano corta invece esce a briscola circa il 18% delle volte. Non
va corretto a mano senza prova: potrebbe essere un vizio, ma potrebbe anche essere una
buona specializzazione del 1v1. Serve una sonda col giudice PIMC.

## Perché la prossima ipotesi è l'encoder v5

Il punto comune dei risultati è che il profilo cambia poco con l'avversario. Questa è la
diagnosi più utile: la policy non ha abbastanza materia prima per riconoscere lo stile che
ha davanti. Sa lo stato del gioco, ma non riceve contatori espliciti su *come* l'altro ha
giocato finora.

L'ipotesi v5 è aggiungere pochi segnali pubblici all'osservazione:

- quante volte l'avversario taglia carichi guidati;
- percentuale di aperture lisce;
- briscole spese su piatti poveri;
- punti rifiutati o lasciati passare;
- eventuale tendenza a cavare briscole o a conservarle.

Sono informazioni anti-cheat: derivano solo dalla storia visibile della partita. Il gate
deve essere specifico. Non basta che il modello guidi meno carichi in assoluto; quello
sarebbe solo prudenza media. Deve guidarne meno **contro il saver** e non necessariamente
contro lo specchio. Solo così si può dire che ha imparato adattamento, non una nuova regola
rigida.

## Cosa aspettare prima del prossimo run

La decisione corrente è non lanciare subito un altro training. Prima servono altre partite
umane complete contro v11 in produzione, idealmente ~50-100, poi lo stesso audit del
2026-07-07:

- quante aperture di carico fa l'IA;
- quante vengono tagliate;
- in quale fase della partita succede;
- se gli umani vincenti stanno davvero conservando briscoline;
- se la cavata delle briscole è un errore dell'IA o una differenza legittima tra teoria
  tradizionale e 1v1.

Solo dopo quel controllo ha senso scegliere tra encoder v5, potential shaping anti-spreco,
o una sonda mirata sulla cavata delle briscole.

## Nota retrospettiva: l'encoder v5 è stato chiuso

Questa era la decisione aperta l'8 luglio. La sonda riproducibile del giorno successivo ha
mostrato che l'ipotesi di partenza era troppo forte: l'encoder v4 (`feature_dim=369`)
contiene già contatori pubblici di stile e la policy li usa nel verso corretto, riducendo
la massa sui carichi contro il conservatore. L'effetto è però piccolo (circa −0.8 punti
percentuali e ~1% di flip dell'argmax) perché tagli e aperture di carico sono eventi rari,
non perché manchino le feature.

Decisione conclusiva: **nessun encoder v5 dedicato allo stile**. La pista dei carichi è
stata chiusa come causa generale; la correzione promossa è stata invece lo shaping
anti-overkill di v13, che agisce su un comportamento più frequente e misurabile senza
reintrodurre un guard runtime. Metodo e caveat sono nella
[nota tecnica sulla sensibilità allo stile](../plans/sonda-stile-finestra-2026-07-08.md);
l'esito v13 è in [Stessa forza, comportamento migliore](17-stessa-forza-comportamento-migliore.md).
