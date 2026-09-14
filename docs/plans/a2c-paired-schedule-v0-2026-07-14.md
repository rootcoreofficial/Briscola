# Training A2C realmente paired v0 (2026-07-14)

> Verdetto: **inconcludente, mantenere la schedule seriale**. Il pairing a pari
> partite non riduce la variabilita' e non supera il controllo diretto. Evidenza
> completa:
> [a2c_paired_schedule_v0.v1.json](../reports/evidence/a2c_paired_schedule_v0.v1.json).

## Domanda

Il vecchio `--seat-fair` alterna il posto della policy, ma ogni partita riceve un mazzo
nuovo e ricampiona l'avversario. I due esempi consecutivi non controllano quindi la
stessa situazione da entrambi i lati.

Vogliamo capire se un training realmente appaiato riduca la dipendenza dalla fortuna del
mazzo e la variabilità tra run, senza indebolire il modello. Non assumiamo che sia meglio:
vedere due volte lo stesso ambiente riduce infatti la varietà dei mazzi a parità di
partite.

## Contratto implementato

`train_a2c.py --training-schedule paired` costruisce l'intera schedule prima del primo
update. Ogni coppia adiacente:

- usa lo stesso `game_seed`, quindi lo stesso mazzo e la stessa distribuzione iniziale;
- usa lo stesso tipo di avversario campionato dall'opponent mix;
- assegna la policy prima alla seat 0 e poi alla seat 1;
- resta interamente nello stesso optimizer update, quindi vede gli stessi pesi.

Le due partite non devono avere le stesse mosse: scambiando la seat cambiano mano,
osservazioni e sviluppo della partita. Il pairing controlla ambiente e avversario, non
forza traiettorie artificialmente uguali.

La modalità `serial` resta il default e conserva l'alternanza storica di `--seat-fair`.
La modalità paired rifiuta:

- `num_games` dispari;
- `update_every` dispari;
- un numero di partite non multiplo di `update_every`, perché il trainer storico non
  applica un update finale parziale.

Il modello conserva nei metadati modalità, numero di ambienti, ordine seat, seed degli
RNG e SHA-256 della schedule. Dominio, fast Python e batch Numba leggono lo stesso oggetto
schedule. I test coprono i tre path, la riproducibilità, le coppie non spezzate e
l'uguaglianza bit-per-bit tra default seriale e `--training-schedule serial` esplicito.

## Tre regimi

Per ognuno dei seed `20260717`, `20260718`, `20260719`:

| regime | partite | estrazioni ambiente (mazzo + opponent) | update |
|---|---:|---:|---:|
| `serial_same_games` | 20.000 | 20.000 | 1.000 |
| `paired_same_games` | 20.000 | 10.000 | 1.000 |
| `paired_same_decks` | 40.000 | 20.000 | 2.000 |

Il runner ricostruisce indipendentemente le schedule e rifiuta il job se:

- gli ambienti di `paired_same_games` non coincidono col prefisso dei primi 10.000
  ambienti seriali;
- i 20.000 ambienti di `paired_same_decks` non coincidono esattamente, nello stesso
  ordine, con quelli seriali.

La ricetta A2C resta quella del probe di salute: init e anchor v14, encoder v4, fast
Numba, PIMC belief 16×8 al 40%, modello puro 15%, value-lookahead 20%, conservatore 12%,
heuristic v1/v2 e random per il resto, beta anchor `0,01` e penalità overkill gap `0,3`.
Critic reset, advantage, clipping e learning rate non cambiano.

## Valutazione

Ogni modello temporaneo viene valutato su 4.000 partite seat-fair deterministiche,
tutte sulla suite `range(4_000_000, 4_002_000)`:

- contro v14, per misurare forza finale e dispersione tra i tre seed;
- direttamente contro il controllo seriale dello stesso seed.

Il report registra inoltre coefficiente di variazione della norma gradiente, dispersione
della media degli advantage, explained variance del critic, distanza dall'init e tempo.
Le CI95 delle valutazioni usano la coppia di partite come unità statistica.

## Gate preregistrato

Il confronto primario è `paired_same_games` contro `serial_same_games`. È **GO a un solo
screen paired più lungo** soltanto se valgono insieme:

1. mediana dei tre direct match paired-minus-serial `>= 0` punti;
2. almeno due seed su tre non negativi;
3. deviazione standard tra seed della forza contro v14 non superiore al seriale;
4. mediana del coefficiente di variazione dei gradienti non superiore al seriale.

È **STOP paired** se:

- la mediana diretta è `<= -0,25` punti e almeno due seed sono negativi; oppure
- sia la dispersione di forza sia quella dei gradienti crescono di almeno il `10%`.

Negli altri casi il verdetto è **inconcludente e serial resta default**. Il regime a pari
mazzi è solo di supporto: usa il doppio delle partite e non può salvare un fallimento del
confronto primario.

Nessuno dei nove modelli può essere promosso: il probe misura la schedule, non produce
v15.

## Risultati

Il job ha completato nove training, 240.000 partite complessive e quindici valutazioni
da 4.000 partite. Per tutti i seed il runner ha verificato che:

- `paired_same_games` usa esattamente il prefisso dei primi 10.000 ambienti seriali;
- `paired_same_decks` usa, nello stesso ordine, tutti i 20.000 ambienti seriali;
- modelli, diagnostiche e valutazioni corrispondono alle ricevute e agli hash attesi.

Il confronto primario paired-minus-seriale a pari numero di partite e' il seguente:

| seed | differenza punti | CI95 paired | esito rispetto a zero |
|---:|---:|---:|---|
| 20260717 | -0,151 | -0,615..+0,313 | inconcludente |
| 20260718 | -0,462 | -0,919..-0,004 | negativo |
| 20260719 | +0,242 | -0,200..+0,684 | inconcludente |

La media e' `-0,124`, la mediana `-0,151` e soltanto un seed su tre e' non
negativo. Le singole CI ricordano che due confronti sono compatibili con la parita', ma
il gate era intenzionalmente multi-seed: non possiamo scegliere soltanto il run
favorevole.

Contro la stessa v14 di partenza:

| regime | punti medi | mediana | deviazione standard tra seed |
|---|---:|---:|---:|
| seriale 20k | +0,160 | +0,211 | 0,177 |
| paired 20k | +0,136 | +0,334 | 0,367 |
| paired 40k | +0,164 | +0,155 | 0,035 |

Il paired 20k varia quindi circa `2,08x` piu' del seriale nella forza finale. Anche la
variabilita' relativa dei gradienti e' leggermente peggiore (`1,038x`), non migliore.
Il critic resta sano e simile nei tre regimi: non emerge un guasto numerico nascosto.

Il paired 40k e' molto coerente tra seed, ma usa il doppio delle partite e degli update.
Nel confronto diretto con il seriale 20k ottiene differenze `-0,238`, `+0,146` e
`-0,148`, con mediana `-0,148`: la maggiore stabilita' non produce maggiore forza. Il
tempo medio passa da `52,4 s` per il seriale 20k a `55,9 s` per il paired 20k e
`110,8 s` per il paired 40k.

## Decisione

Il **GO** fallisce tutti e quattro i requisiti: mediana diretta non negativa, almeno due
seed non negativi, minore dispersione della forza e minore variabilita' dei gradienti.
Non scatta pero' lo **STOP forte** preregistrato: la mediana non arriva a `-0,25` e il
rapporto dei gradienti non peggiora di almeno il 10%. Il verdetto formale e' quindi
`inconclusive_keep_serial`.

In pratica, il pairing e' implementato correttamente ma non ha mostrato il vantaggio per
cui era stato introdotto. `serial` resta il default; non e' giustificato un altro run
paired piu' lungo o un tuning post-hoc. La modalita' resta disponibile come strumento
sperimentale, non come ricetta consigliata e non come base di v15.

## Esecuzione

Il job previsto supera cinque minuti. Preparazione e avvio:

```bash
mkdir -p benchmarks/experiments/a2c_paired_schedule_v0_20260714
nohup uv run python scripts/run_a2c_paired_schedule_probe.py --resume \
  > benchmarks/experiments/a2c_paired_schedule_v0_20260714/run.log 2>&1 &
```

Controllo del log:

```bash
tail -f benchmarks/experiments/a2c_paired_schedule_v0_20260714/run.log
```

Artefatto locale prodotto:

`benchmarks/experiments/a2c_paired_schedule_v0_20260714/a2c_paired_schedule_g20000_3seeds.json`.

`--resume` verifica hash di modelli, diagnostiche e input delle valutazioni prima di
saltare un sottopasso già completato.
