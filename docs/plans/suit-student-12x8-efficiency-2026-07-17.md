# Probe di efficienza student PIMC 12x8

Data: 2026-07-17
Stato: **conferma 20k e audit integrazione PASS; promosso come v15 in 0.38.0**
Default deciso: `best_a2c_v15.npz` con PIMC belief 12x8

## Perche' 12x8

Lo student 8x8 dimezza la latenza search ma perde `-0,742` punti/partita contro v14
16x8, con CI95 interamente negativa. Dodici determinizzazioni sono il solo punto
intermedio naturale rimasto: costo teorico `0,75x` e piu' campioni per stabilizzare la
decisione. Questo e' un nuovo probe di efficienza dichiarato dopo lo STOP 8x8; non cambia
retroattivamente quel risultato e non riapre la promozione di forza dello student.

## Configurazione congelata

| lato | policy | search | belief | solver |
|---|---|---|---|---|
| A | student teacher20M 250k | PIMC `12x8` Python | v0, mix uniforme `0,10` | attivo |
| B | v14 ufficiale | PIMC `16x8` Python | v0, mix uniforme `0,10` | attivo |

Motore domain, seat-fair, stessa belief, stessa finestra e stesso solver. Asset e hash
sono identici al probe 8x8 documentato in
`suit-student-8x8-efficiency-2026-07-17.md`; cambiano soltanto budget A e seed. Come negli
altri confronti PIMC, le singole determinizzazioni non usano common random numbers.

## Screen 4k

- 4.000 partite / 2.000 coppie, seed radice nuovo `20260727`;
- CI95 sulle coppie seat-fair;
- output `efficiency_12x8/screen_student12_vs_v14_16_4k.json`;
- latenza search: media, p50, p95 e massimo per entrambi i lati.

Si apre la conferma soltanto se valgono insieme:

1. delta medio student 12x8 meno v14 16x8 almeno `-0,25` punti/partita;
2. limite superiore CI95 sopra zero, quindi lo screen non dimostra gia' una sconfitta;
3. rapporto student/v14 non oltre `0,82` sulla latenza media e `0,85` sul p95;
4. zero determinizzazioni fallite, rollout falliti e mosse corrette difensivamente.

Il costo deve quindi scendere almeno del `18%` in media e del `15%` in coda. Qualsiasi
altro esito e' STOP; lo screen non autorizza modifiche live.

### Esito dello screen (2026-07-17)

Lo screen e' terminato in 4 minuti e 33 secondi e supera tutti i criteri preregistrati:

| metrica | student 12x8 | v14 16x8 | rapporto / intervallo |
|---|---:|---:|---:|
| vittorie | `1.913` | `1.964` | 123 pareggi |
| delta punti student | `-0,0495` | - | CI95 `-0,5666..+0,4676` |
| score rate student | `49,3625%` | - | CI95 `48,2694%..50,4556%` |
| latenza media search | `11,282 ms` | `15,074 ms` | `0,7485x` |
| latenza p50 | `11,206 ms` | `14,938 ms` | `0,7502x` |
| latenza p95 | `16,892 ms` | `22,439 ms` | `0,7528x` |

Le 280.024 determinizzazioni complessive e gli 840.072 rollout non registrano fallimenti
o mosse corrette difensivamente. Il delta supera `-0,25`, la CI include esiti positivi e
il risparmio misurato e' circa il `25%` sia in media sia al p95: **GO alla sola conferma
20k**. Il report ha SHA-256
`082150cf9feaab5b3fbc371b6a61a7d945f5e7dc0921dea8dbcac8583ccab54f`.

## Conferma condizionata 20k

Solo dopo un GO dello screen: 20.000 partite, seed nuovo `20260728`, stessi asset e
configurazione. Il margine di non inferiorita' resta `-0,25` punti/partita. Sono richiesti
insieme stima almeno `-0,10`, limite inferiore CI95 sopra `-0,25`, rapporti di latenza
entro `0,82/0,85` e integrita' perfetta.

### Esito della conferma (2026-07-17)

La conferma e' terminata in 22 minuti e 13 secondi e supera tutti i gate:

| metrica | student 12x8 | v14 16x8 | rapporto / intervallo |
|---|---:|---:|---:|
| vittorie | `9.689` | `9.621` | 690 pareggi |
| delta punti student | `+0,1052` | - | CI95 `-0,1126..+0,3230` |
| score rate student | `50,1700%` | - | CI95 `49,7018%..50,6382%` |
| latenza media search | `11,040 ms` | `14,666 ms` | `0,7528x` |
| latenza p50 | `10,952 ms` | `14,583 ms` | `0,7510x` |
| latenza p95 | `16,440 ms` | `21,826 ms` | `0,7532x` |

La stima supera `-0,10`, il limite inferiore CI95 (`-0,1126`) resta sopra il margine di
non inferiorita' `-0,25` e i due rapporti di latenza restano ampiamente sotto `0,82/0,85`.
Le 1.400.156 determinizzazioni e i 4.200.468 rollout complessivi hanno zero fallimenti e
zero mosse corrette difensivamente. Il report ha SHA-256
`018ed55c68763864d721dff55df73238143448be7ba49c8cdaacf5ba03da6d32`.

**Gate finale: PASS di non inferiorita' ed efficienza.** Il risultato identifica un
punto costo/forza migliore, non dimostra che lo student sia piu' forte: la CI del delta
include ancora zero. Catalogo e produzione sono rimasti invariati fino all'audit web
separato e alla successiva decisione esplicita del maintainer.

L'audit web separato e' ora concluso con PASS: factory, dipendenze, catalogo isolato,
API, WebSocket, layout desktop/mobile e una partita completa con decisioni
fallback/search/solver funzionano senza errori. La successiva release 0.38.0 ha promosso il profilo; ricevuta in
`suit-student-12x8-release-audit-2026-07-17.md`.

Avvio protetto della conferma:

```bash
scripts/run_suit_distillation_20m_250k.sh start-12x8-confirm
```

Il launcher ha usato `nohup` e `caffeinate`; log e report conclusivi sono rispettivamente in
`efficiency_12x8/confirm_student12_vs_v14_16_20k.log` e
`efficiency_12x8/confirm_student12_vs_v14_16_20k.json`. Un nuovo avvio viene rifiutato
per non sovrascrivere la ricevuta.

## Comando dello screen

```bash
uv run python scripts/evaluate_pimc.py \
  --model benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/models/suit_distilled_20m_teacher24_250k_seed20260724.npz \
  --num-games 4000 --seed 20260727 \
  --determinizations 12 --max-unknown-cards 8 \
  --opponent pimc \
  --opponent-model data/models/best_a2c_v14.npz \
  --opponent-determinizations 16 --opponent-max-unknown-cards 8 \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --opponent-belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --belief-uniform-mix 0.10 --opponent-belief-uniform-mix 0.10 \
  --out-json benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/efficiency_12x8/screen_student12_vs_v14_16_4k.json
```
