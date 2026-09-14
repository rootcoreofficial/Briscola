# Nota tecnica - Sonda di simmetria dei semi su v13 (2026-07-11)

> Nota interna di decisione. La misura autorizza una piccola ablation di training, non una
> promozione del modello. Evidenza completa: [suit_symmetry_v13.v1.json](../reports/evidence/suit_symmetry_v13.v1.json).

## Domanda

Le regole della Briscola non attribuiscono un valore speciale ai nomi *bastoni*, *coppe*,
*denari* e *spade*: se li rinominiamo ovunque in modo coerente, la situazione strategica
resta identica. Una policy perfettamente simmetrica dovrebbe quindi assegnare le stesse
probabilità alle stesse azioni, una volta riportate le carte ai semi originali.

La sonda chiede se la policy ufficiale `best_a2c_v13.npz` rispetta questa proprietà oppure
ha imparato preferenze accidentali per gli identificatori assoluti dei semi. La domanda è
diagnostica: un cambio di decisione prova sensibilità alla rappresentazione, non prova da
solo che quella decisione faccia perdere punti.

## Metodo e anti-cheat

- Sono state giocate **512 partite**: 64 seed, entrambi i posti e quattro avversari
  (`mirror`, `heuristic_trump_saver`, `heuristic_v1`, `random`). La policy è stata osservata
  in **10.240 decisioni**; 9.728 avevano almeno due carte legali e 512 decisioni forzate con
  una sola carta sono state escluse.
- Il campione finale contiene **4.096 `PlayerObservation`**, bilanciate in 16 celle
  avversario x fase: 256 osservazioni per cella. In ogni cella la selezione usa i 256 digest
  SHA-256 più bassi, quindi non dipende dall'ordine della raccolta.
- Per ogni osservazione sono state costruite tutte le **24 permutazioni** dei quattro semi.
  La trasformazione comprende mano, briscola, tavolo, storia delle prese, carte viste e
  carte fuori gioco. Anche action mask e output della rete sono rimappati; il confronto
  finale avviene sempre nel sistema di semi originale.
- La metrica primaria confronta l'osservazione originale con le altre **23** rinomine. Come
  controllo di robustezza sono confrontate anche tutte le **276 coppie** tra le 24 versioni.
  Il controllo negativo è la permutazione identità.
- Si misurano logits grezzi della policy, dopo la maschera delle azioni legali e prima di
  qualsiasi post-processing runtime. La scelta è l'argmax; la Jensen-Shannon (JS) usa la
  softmax a temperatura `T=1`.
- Un confronto è classificato come quasi-pareggio se il gap top-2 è sotto `1e-4` in almeno
  una delle due distribuzioni. Il filtro controlla quindi entrambi i lati, anche nelle 276
  coppie, e rimuove singoli confronti senza scartare gli altri dello stesso stato.
- L'anti-cheat resta intatto: il `GameState` completo serve soltanto al motore durante la
  partita e non viene conservato o serializzato. Policy e metriche ricevono soltanto la
  `PlayerObservation`. I casi estremi aggiungono `game_seed`, posto e ordinale come
  metadati offline di riproduzione, mai passati al modello; non serializzano mazzo o mano
  avversaria. Il seed permette comunque di ricostruire il mazzo e va trattato come dato di
  audit, non come input lecito di un agente.
- Gli intervalli al 95% ricampionano **osservazioni intere**, non le 23 o 276 comparazioni
  correlate della stessa osservazione. Sono state usate 2.000 repliche, seed `20260711`.

Comando esatto:

```bash
uv run python scripts/probe_suit_symmetry.py \
  --model data/models/best_a2c_v13.npz \
  --seed-suite seed_suites/small_1000.txt \
  --seed-count 64 \
  --samples-per-cell 256 \
  --opponents mirror heuristic_trump_saver heuristic_v1 random \
  --bootstrap-reps 2000 \
  --bootstrap-seed 20260711 \
  --near-tie-threshold 0.0001 \
  --worst-cases 20 \
  --out-json docs/reports/evidence/suit_symmetry_v13.v1.json
```

## Copertura e identità degli artefatti

| elemento | valore |
|---|---|
| codice / regole | `0.35.1` / `1` |
| commit di riferimento | `622ccd1e65979a08f850d5ffa3493a0dd5efd817` |
| worktree durante la sonda | dirty, perché sonda, modulo e test erano ancora da committare |
| modello | `data/models/best_a2c_v13.npz` |
| SHA-256 modello | `5b1c6ea0bca7fd2c868e01d4d583cbc5df7bbef2ab86bbb3ded4b18b14c9f1cf` |
| seed suite | `seed_suites/small_1000.txt`, primi 64 seed su 1.000 |
| SHA-256 seed suite | `6e2a49a424c2ff183df71cbf226130aaa1d9b85006be55f6654f5bdce8701cd1` |
| manifest dei 4.096 campioni | `d9c1dbeef8ccfbf4549f146797fd01790379e7af002984fa0b344bb972149c93` |
| SHA-256 script | `1634f4f8b0e711e2ed36341f0b139b68732f3df7e8a1a801b9d6bd7de0921ee9` |
| SHA-256 modulo di simmetria | `e3c40faac9a1cb2450167b17ad46d28997bc2eeb708ffad8b921cd23dbfc0554` |

Tutte le 16 celle raggiungono la quota prevista. Le fasi e gli avversari hanno quindi
1.024 osservazioni ciascuno; questa è una suite deliberatamente bilanciata, non una stima
della frequenza naturale degli stati nel traffico reale.

## Risultato complessivo

Gli intervalli della tabella sono bootstrap al 95% con un valore per osservazione.

| metrica | stima | CI 95% |
|---|---:|---:|
| flip dell'argmax, identità vs 23 | **18,19%** | **[17,38%; 18,93%]** |
| osservazioni con almeno un flip | **51,17%** | **[49,54%; 52,73%]** |
| JS media, identità vs 23 | **0,14175 bit** | **[0,13553; 0,14749]** |
| media della JS massima per osservazione | **0,43352 bit** | **[0,41976; 0,44639]** |
| massimo delta assoluto medio di probabilità | **0,17983** | **[0,17278; 0,18668]** |
| flip tra tutte le 276 coppie | **18,31%** | **[17,65%; 18,94%]** |
| JS media tra tutte le 276 coppie | **0,14085 bit** | **[0,13578; 0,14572]** |

Il risultato non è una piccola oscillazione confinata a pochi pareggi: circa una rinomina
su cinque cambia la carta scelta e oltre metà delle osservazioni ha almeno una rinomina
che cambia l'argmax. Il controllo sulle 276 coppie restituisce quasi lo stesso tasso di
flip, quindi il dato non dipende dal solo confronto con una particolare codifica originale.

La distribuzione JS è fortemente asimmetrica. Nel confronto identità vs 23 la **mediana è
0,000090 bit**, quasi zero, mentre il **p95 è 0,94823 bit**, vicino al massimo teorico di
1 bit. In parole semplici: molte rinomine cambiano pochissimo l'output, ma una coda non rara
sposta quasi tutta la massa tra azioni diverse. La media di 0,14175 bit riassume entrambe
le popolazioni e, da sola, nasconderebbe questa struttura.

Queste probabilità derivano da una softmax con `T=1` e **non sono calibrate**: `0,99` non
significa che una carta sia corretta nel 99% dei casi. JS misura quanto cambiano due output
numerici della stessa rete; non misura direttamente punti, vittorie o qualità strategica.

## Breakdown

Le colonne `flip` e CI si riferiscono sempre al confronto identità vs le altre 23
permutazioni. `Almeno uno` è la quota di osservazioni in cui almeno una rinomina cambia
la carta scelta.

| gruppo | osservazioni | flip | CI 95% | almeno uno | JS media |
|---|---:|---:|---:|---:|---:|
| fase: early | 1.024 | 17,69% | [16,16%; 19,26%] | 49,12% | 0,13870 |
| fase: mid | 1.024 | 18,70% | [17,19%; 20,25%] | 54,59% | 0,14981 |
| fase: pimc_window | 1.024 | 18,56% | [16,98%; 20,20%] | 51,56% | 0,14593 |
| fase: endgame | 1.024 | 17,78% | [16,27%; 19,36%] | 49,41% | 0,13254 |
| avversario: mirror | 1.024 | 18,04% | [16,55%; 19,57%] | 52,44% | 0,14250 |
| avversario: heuristic_trump_saver | 1.024 | 17,65% | [16,05%; 19,12%] | 49,80% | 0,13831 |
| avversario: heuristic_v1 | 1.024 | 16,13% | [14,62%; 17,53%] | 47,56% | 0,12380 |
| avversario: random | 1.024 | 20,92% | [19,30%; 22,66%] | 54,88% | 0,16237 |
| posizione: apertura | 2.104 | **22,66%** | [21,52%; 23,83%] | 61,98% | 0,17476 |
| posizione: risposta | 1.992 | **13,45%** | [12,48%; 14,50%] | 39,76% | 0,10688 |
| azioni legali: 2 | 512 | 11,18% | [9,31%; 13,04%] | 33,20% | 0,08306 |
| azioni legali: 3 | 3.584 | 19,19% | [18,33%; 20,00%] | 53,74% | 0,15013 |

L'asimmetria compare in tutte le fasi e contro tutto il roster: non è un singolo caso
limite. La differenza più netta è maggiore quando la policy apre la presa e quando deve
scegliere fra tre carte. Sono contesti con più libertà d'azione, quindi il dato è
compatibile con una preferenza appresa per codici di seme assoluti, ma non ne identifica da
solo la causa. Il picco contro `random` non va interpretato causalmente: gli avversari
generano distribuzioni di osservazioni diverse, anche con quote numeriche uguali.

## Pareggi numerici e controllo identità

La soglia di quasi-pareggio è `gap(top1, top2) <= 1e-4`. Il tasso osservato è **0%** sia
nelle 4.096 baseline sia nelle **98.304 distribuzioni** sulle 24 rinomine. Di conseguenza
tutti i 94.208 confronti identità-vs-23 e tutte le 1.130.496 coppie restano nel calcolo
filtrato, con metriche identiche. Il gap mediano sulle 24 rinomine è `0,9999905`: la
softmax è spesso molto satura. I flip non sono quindi spiegati dal criterio arbitrario
con cui `argmax` risolve probabilità quasi uguali.

La permutazione identità passa il controllo su tutte le **4.096** osservazioni:
agreement `100%`, JS massima `0` e delta massimo `0`. Questo esclude un errore numerico
di base nel percorso encode -> mask -> softmax -> remapping, pur non sostituendo i test
unitari sulle permutazioni non banali.

## Limiti

- La diagnostica dimostra asimmetria della policy v13, **non** dimostra che augmentation o
  consistency loss aumenteranno la forza di gioco. Rendere due output più simili può
  anche peggiorare l'ottimizzazione o interferire con altri obiettivi.
- Le osservazioni della stessa partita sono correlate. Il bootstrap usa l'osservazione,
  anziché le 23 comparazioni, come unità; non raggruppa però per partita
  o seed. Le CI possono quindi essere più strette di una block bootstrap per partita.
- Il campione è bilanciato per avversario e fase. L'overall descrive questa suite di
  stress test, non una miscela di utilizzo reale.
- La sonda riguarda i logits grezzi della policy MLP v13. Non misura il comportamento del
  default completo con belief, PIMC e solver, né gli effetti sulle traiettorie successive.
- La JS a `T=1` dipende dalla scala dei logits non calibrati. Argmax flip e JS sono segnali
  complementari, non sostituti di una evaluation seat-fair a punti.
- Il checkout era dirty perché lo strumento era appena stato implementato. Gli hash di
  script, modulo, modello, seed suite e manifest rendono comunque questa esecuzione
  identificabile; una ripetizione dopo il commit deve confrontare anche lo SHA del JSON.

## Verdetto

**GO** alla fase successiva: augmentation **paired** (originale e copia rinominata nello
stesso update) e consistency loss sono giustificate come **ablation controllate**. Una
semplice sostituzione con una sola permutazione casuale non è sufficiente: il mazzo casuale
espone già i semi in modo simmetrico e l'obiettivo atteso resterebbe lo stesso. Le due
ablation vanno provate separatamente contro lo stesso warm-start, budget, seed di training
e suite di evaluation; il primo gate è ridurre flip/JS, ma la decisione finale resta la
forza seat-fair policy-only e PIMC 16x8, insieme ai gate comportamentali già esistenti. Un
output più simmetrico non basta per la promozione.

La copia deve usare la stessa permutazione per tutta la traiettoria e trasformare anche
mask e action id; reward, return e advantage restano invariati. Nel path fast/Numba ogni
trasformazione delle 369 feature deve essere verificata contro il riferimento semantico
`PlayerObservation -> permute -> encoder v4`, non ricostruita per tentativi sugli indici.

**Nessun training è stato avviato da questa sonda.** Il risultato autorizza l'esperimento,
non sceglie ancora tra augmentation e consistency e non modifica il modello ufficiale.

**Esito successivo:** il paired v0 è stato poi provato su tre seed da 10.000 partite e ha
fallito il gate (flip medio `18,32% -> 18,84%`, direct match `-0,15` punti/partita). Non è
stato promosso né esteso. Report:
`docs/plans/suit-augmentation-paired-v0-2026-07-11.md`.

## Integrità dell'evidenza

- Artefatto: `docs/reports/evidence/suit_symmetry_v13.v1.json`
- SHA-256 finale: `72ed32b863262bee3602663713ecd123ae8462e16323186d943e451bbb475a05`
