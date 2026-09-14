# Approfondimento - I semi non hanno nome

**Capitolo del diario:** [Capitolo 18](https://ai.briscola.dev/diario) - **Periodo:** 11 luglio 2026

## La domanda

In Briscola i nomi `clubs`, `cups`, `coins` e `swords` non hanno un valore strategico
assoluto. Se rinominiamo coerentemente ogni carta, compresa la briscola, il tavolo e la
storia pubblica, otteniamo la stessa posizione con etichette diverse. Una policy
equivariant dovrebbe quindi rinominare anche la propria scelta, senza cambiare piano.

Questa proprietà non era garantita dall'architettura v13: encoder e spazio azioni hanno
blocchi distinti per i quattro semi e la MLP può imparare associazioni accidentali con
la loro posizione numerica.

## La sonda

La diagnostica ha tre livelli di difesa:

1. trasforma semanticamente una `PlayerObservation`, non il `GameState` completo e non un
   vettore di feature manipolato a mano;
2. prova tutte le 24 biiezioni dei semi e riporta le 40 probabilità nelle coordinate
   originali prima del confronto;
3. esclude le mosse obbligate e bilancia 256 osservazioni per ciascuna combinazione di
   quattro avversari e quattro fasi della partita.

La suite comprende 64 mazzi, entrambi i posti e quattro avversari (`mirror`,
`heuristic_trump_saver`, `heuristic_v1`, `random`): 512 partite, 4.096 osservazioni e
94.208 confronti tra stato originale e rinomine non banali. Il controllo identità è
esatto e due esecuzioni complete producono lo stesso JSON byte per byte.

## Risultati v13

| Metrica | Risultato |
|---|---:|
| Flip dell'argmax, identità vs altre 23 | **18,19%** |
| CI95 bootstrap per osservazione | 17,38..18,93% |
| Stati con almeno un flip | **51,17%** |
| Divergenza JS media, softmax T=1 | 0,142 bit |
| Flip sulle 276 coppie dell'orbita | 18,31% |
| Quasi-pareggi, gap top-2 <= 1e-4 | **0 / 98.304 distribuzioni** |

Il segnale non appartiene a un solo momento o avversario: il flip resta tra 17,69% e
18,70% nelle quattro fasi e tra 16,13% e 20,92% nei quattro matchup. È più forte quando
v13 apre la presa (22,66%) che quando risponde (13,45%), e con tre carte legali (19,19%)
rispetto a due (11,18%).

La distribuzione è molto discontinua: la JS mediana è circa 0,00009 bit, ma il p95 è
0,948 bit. In molti confronti cambia pochissimo; in una minoranza la policy passa quasi
completamente da una carta a un'altra. La JS usa il softmax dei logits grezzi a
temperatura 1: descrive la policy, non è una probabilità calibrata di vittoria. Il
segnale principale resta il cambio dell'argmax.

## Cosa dimostra e cosa no

La sonda dimostra che v13 non tratta i nomi dei semi come etichette intercambiabili. Non
dimostra quale delle due mosse sia migliore, né che una policy più simmetrica vincerà
automaticamente di più: due mosse diverse possono anche avere valore di gioco simile.
Inoltre le CI ricampionano osservazioni, che nella stessa partita sono correlate; servono
come misura descrittiva della stabilità, non come gate di forza.

Il verdetto è quindi **GO diagnostico** alla prima ablation, non alla promozione di un
modello. Il prossimo esperimento deve affiancare, nello stesso update, una copia rinominata
alla traiettoria originale, senza costo a inference. Sostituire ogni esempio con una sola
rinomina casuale non basta, perché i mazzi casuali sono già simmetrici in distribuzione.
L'ablation passa soltanto se riduce nettamente i flip e non regredisce nei gate policy-only,
PIMC 16x8 e decision quality.

**Aggiornamento del seguito:** l'ablation paired è stata eseguita su tre seed da 10.000
partite e non ha superato il primo gate: flip medio `18,32% -> 18,84%` e `-0,15`
punti/partita contro i controlli corrispondenti. Il risultato e la nuova direzione sono nel
[secondo approfondimento del capitolo 18](19-la-copia-non-basta.md) e nella
[nota di ablation](../plans/suit-augmentation-paired-v0-2026-07-11.md).

Metodo completo: [`docs/plans/sonda-simmetria-semi-2026-07-11.md`](../plans/sonda-simmetria-semi-2026-07-11.md).
Evidenza canonica:
[`suit_symmetry_v13.v1.json`](../reports/evidence/suit_symmetry_v13.v1.json),
SHA-256 `72ed32b863262bee3602663713ecd123ae8462e16323186d943e451bbb475a05`.
