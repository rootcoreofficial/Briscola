# Probe di efficienza student PIMC 8x8

Data: 2026-07-17
Stato: **STOP dopo lo screen 2k; latenza dimezzata ma regressione di forza certa**
Modello live invariato: `best_a2c_v14.npz` con PIMC belief 16x8

## Domanda

Lo student 250k e' migliore di v14 come policy pura ma pari nel confronto live quando
entrambi usano PIMC belief 16x8. Questo probe non riapre la promozione di forza: verifica
se la policy migliore permette di dimezzare le determinizzazioni mantenendo una forza
praticamente equivalente, con un risparmio misurabile di CPU e latenza.

## Configurazione congelata

| lato | policy | search | belief | solver |
|---|---|---|---|---|
| A | student teacher20M 250k | PIMC `8x8` Python | v0, mix uniforme `0,10` | attivo |
| B | v14 ufficiale | PIMC `16x8` Python | v0, mix uniforme `0,10` | attivo |

Asset:

- student SHA-256 `8a0a03946c9413ed7e6c18059a6aa03f63a9476e0b603ad977f4955cb444199d`;
- v14 SHA-256 `a67ed1d7f01ba1019f157134ade23fa9f822e442b671c83684bd4500e97695a8`;
- belief v0 SHA-256 `4100b23b65a5566e047230ced665b91eef1942ea31e4a4cbe201b64545e7d035`.

Il motore e' il dominio canonico. Il confronto e' seat-fair: ogni mazzo viene giocato
due volte scambiando i posti. I due agenti usano lo stesso asset belief e la stessa
finestra; cambiano intenzionalmente policy e numero di determinizzazioni, perche' la
domanda riguarda il prodotto completo meno costoso. Non e' quindi un'ablation causale di
una sola variabile ne' usa common random numbers fra le singole determinizzazioni.

## Strumentazione richiesta

`evaluate_pimc.py` accetta una policy distinta con `--opponent-model` e avvolge ciascun
PIMC soltanto durante l'evaluation per registrare la durata di ogni decisione search. Il
report conserva conteggio, media, p50, p95 e massimo senza aggiungere memoria crescente
al runtime web. Restano anche tempo totale, determinizzazioni, rollout, fallback, solver,
fallimenti e mosse corrette difensivamente.

## Screen 2k

- 2.000 partite / 1.000 coppie;
- seed radice `20260725`;
- CI95 calcolata sulle coppie seat-fair;
- output `efficiency_8x8/screen_student8_vs_v14_16_2k.json`.

Si autorizza la conferma soltanto se valgono insieme:

1. delta medio student 8x8 meno v14 16x8 almeno `-0,25` punti/partita;
2. limite superiore CI95 sopra zero, quindi lo screen non dimostra gia' una sconfitta;
3. rapporto student/v14 non oltre `0,60` sulla latenza media search e `0,70` sul p95;
4. zero determinizzazioni fallite, rollout falliti e mosse corrette difensivamente.

Qualsiasi altro esito chiude il probe. Lo screen non puo' autorizzare un cambio live.

### Esito dello screen

Lo screen e' terminato in `114,18 s` senza errori numerici o fallback difensivi:

| metrica | student 8x8 | v14 16x8 | rapporto / delta |
|---|---:|---:|---:|
| vittorie | `925` | `1.011` | 64 pareggi |
| delta punti | `-0,742` | - | CI95 `-1,457..-0,027` |
| score rate | `47,85%` | - | CI95 `46,27..49,43%` |
| latenza search media | `7,393 ms` | `14,553 ms` | `0,508x` |
| latenza search p50 | `7,338 ms` | `14,576 ms` | `0,503x` |
| latenza search p95 | `10,935 ms` | `21,665 ms` | `0,505x` |
| decisioni search | `5.032` | `4.968` | copertura equivalente |

Il gate di costo passa nettamente: media e p95 scendono di circa il `49%`. Anche il gate
di integrita' passa con zero determinizzazioni fallite, rollout falliti e mosse corrette.
Falliscono pero' entrambi i criteri di forza preregistrati: la stima e' molto sotto
`-0,25` e perfino il limite superiore CI95 resta sotto zero.

**Decisione: STOP.** La conferma 20k non viene aperta. Il miglioramento della policy non
compensa il dimezzamento dei mondi simulati; v14 PIMC 16x8 resta il punto operativo. Il
report ha SHA-256
`d43943a26475dd93f80f83bb614a1e8b55d77bd9dc340b42ba90517490fc06dc`.

## Conferma condizionata 20k (non aperta)

Solo dopo uno screen positivo:

- 20.000 partite / 10.000 coppie, seed nuovo `20260726`;
- stessa configurazione e stessi asset;
- output `efficiency_8x8/confirm_student8_vs_v14_16_20k.json`.

Il margine di non inferiorita' e' congelato a `-0,25` punti/partita: circa lo `0,21%`
dei 120 punti totali e inferiore al vantaggio live v14-v13 misurato in passato. Il nuovo
default e' tecnicamente eleggibile soltanto se:

1. stima del delta almeno `-0,10` e limite inferiore CI95 strettamente sopra `-0,25`;
2. rapporti di latenza media e p95 ancora non oltre `0,60` e `0,70`;
3. integrita' perfetta come nello screen;
4. nessun peggioramento dei vincoli gia' chiusi su asset, solver e anti-cheat.

Un eventuale GO indica un miglior punto costo/forza, non un modello piu' forte. Prima di
toccare il catalogo servono comunque audit di latenza web, capacita' concorrente e una
decisione esplicita del maintainer.

## Comando dello screen

```bash
uv run python scripts/evaluate_pimc.py \
  --model benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/models/suit_distilled_20m_teacher24_250k_seed20260724.npz \
  --num-games 2000 --seed 20260725 \
  --determinizations 8 --max-unknown-cards 8 \
  --opponent pimc \
  --opponent-model data/models/best_a2c_v14.npz \
  --opponent-determinizations 16 --opponent-max-unknown-cards 8 \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --opponent-belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --belief-uniform-mix 0.10 --opponent-belief-uniform-mix 0.10 \
  --out-json benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/efficiency_8x8/screen_student8_vs_v14_16_2k.json
```
