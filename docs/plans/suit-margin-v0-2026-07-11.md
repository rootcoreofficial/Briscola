# Ablation suit margin consistency v0 (2026-07-11)

> Screening causale, nessuna promozione. Evidenza:
> [suit_margin_v0.v1.json](../reports/evidence/suit_margin_v0.v1.json).

## Domanda e verdetto

La forward-KL avvicinava le distribuzioni comprimendo il margine top-2 e a 50k perdeva
forza. La margin consistency sostituisce quel termine con una hinge: nella copia rinominata
la carta argmax originale deve battere la migliore alternativa con il margine teacher,
limitato a `2.0` logit.

**Verdetto: STOP, nessun run lungo.** Beta `0.3` riduce il flip medio `18,32% -> 14,42%`,
conserva il gap (`0,915` contro `0,929`) ed è neutro contro v13 (`-0,14` punti/partita).
Non supera però il gate `<12%`; beta `1.0` resta a `14,49%`, quindi la curva è satura.

## Contratto e controlli

- Il teacher stop-gradient fornisce carta argmax e differenza di log-probabilità dalla
  migliore alternativa; per softmax questa differenza coincide con il margine dei logits.
- Il target è `min(margine_teacher, 2.0)`, per non imporre confidenze estreme.
- Le mosse con una sola carta legale sono escluse.
- Se il margine student è sufficiente, il gradiente ausiliario è zero; altrimenti la hinge
  alza il logit della carta teacher e abbassa quello della migliore alternativa.
- Actor e trunk ricevono il gradiente; critic, return e advantage restano invariati.
- Un test congelato verifica riduzione della hinge e aumento del margine student; beta zero
  produce un `.npz` identico al trainer precedente.

## Screening

Cinque beta (`0.001`, `0.01`, `0.1`, `0.3`, `1.0`), tre seed da 10.000 partite, stesso
warm-start v13 e stessa ricetta dei controlli precedenti. Ogni modello è stato valutato su
4.096 osservazioni x 24 rinomine. I candidati rilevanti sono passati anche in direct match
seat-fair da 10.000 partite.

| beta | flip medio | JS media (bit) | gap top-2 medio |
|---:|---:|---:|---:|
| controllo | 18,32% | 0,14124 | 0,929 |
| 0,001 | 18,28% | 0,13998 | 0,928 |
| 0,01 | 17,32% | 0,12815 | 0,921 |
| 0,1 | 14,92% | 0,10128 | 0,913 |
| **0,3** | **14,42%** | 0,09756 | **0,915** |
| 1,0 | 14,49% | **0,09747** | 0,915 |

La dose-risposta si arresta tra `0.1` e `0.3`; moltiplicare ancora il gradiente non riduce
il flip. I tre seed del migliore misurano `14,60%`, `13,80%`, `14,86%`: nessuno raggiunge
il gate `<12%`.

## Forza

Beta `0.3` fa `+0,35`, `-0,12`, `+0,45` contro i controlli dello stesso seed, media
`+0,23`. Contro v13 fa `-0,09`, `-0,37`, `+0,05`, media `-0,14`; tutte le CI95 includono
zero. La formulazione evita la regressione netta della KL a 50k, ma non dimostra maggiore
forza e non risolve abbastanza il difetto da giustificare un run intermedio.

## Prossimo test

Prima di un'altra modifica al training, costruire una policy symmetrized che valuti v13 su
tutte le 24 rinomine, riallinei i 40 output e ne faccia la media. La media sul gruppo
garantisce simmetria esatta per costruzione. Un confronto policy-only seat-fair contro v13
risponderà direttamente se rimuovere l'asimmetria aumenta, conserva o riduce la forza. Se
il risultato è favorevole si potrà distillare quel teacher o progettare pesi condivisi;
altrimenti la simmetria resterà un difetto rappresentazionale senza valore come leva di
forza.
