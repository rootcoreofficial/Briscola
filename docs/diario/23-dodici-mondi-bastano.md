# Approfondimento - Dodici mondi bastano

**Approfondimento del diario:** [Capitolo 21](https://ai.briscola.dev/diario) - **Periodo:** 14-17 luglio 2026

## Da 50 milioni di partite a un altro tipo di progresso

Lo scouting A2C seriale da 50 milioni di partite partiva da v14 e conservava la stessa
ricetta per cinque blocchi da 10 milioni. Nessuno dei checkpoint 10M, 20M, 30M, 40M e
50M ha superato insieme i gate preregistrati di forza e simmetria. Un ranking esplorativo
piu' ampio ha selezionato 20M, ma la conferma indipendente da 100.000 partite lo ha trovato
pari a v14: `+0,0286` punti per partita, CI95 `-0,0741..+0,1313`.

Quindi il run massivo non ha prodotto direttamente v15. Ha pero' lasciato un checkpoint
utile come teacher. Applicando al 20M la media esatta delle risposte sulle 24 rinomine dei
semi, il teacher ha battuto v14 di `+0,3293` punti su 200.000 nuove partite domain (CI95
`+0,2295..+0,4291`) e il 20M grezzo di `+0,2606` (CI95 `+0,1532..+0,3680`).

Il numero 24 descrive queste viste rinominate della policy. Non e' una configurazione
PIMC e non indica 24 mondi immaginati durante una partita live.

## La distillazione da 250.000 partite

Il corpus finale del teacher contiene 250.000 partite e 9.500.000 decisioni, divise per
partita in `200k/25k/25k`. Dieci shard compressi permettono di allenare lo student senza
caricare l'intero dataset in memoria. Dopo cinque epoche:

- la KL sul test separato scende da `0,090962` a `0,012784` (`-85,9%`);
- l'accordo argmax col teacher sale dal `94,35%` al `97,86%`;
- i flip sotto rinomina dei semi scendono al `2,9456%`, contro `6,04%` di v14;
- l'overkill sui piatti poveri scende al `1,0637%`, contro `4,1724%` di v14.

Come policy diretta lo student supera v14 di `+0,18046` punti su 100.000 partite
indipendenti, con CI95 `+0,08..+0,28`. Nel giocatore completo PIMC belief 16x8, invece,
la differenza su 10.000 partite e' `-0,0184`, con CI95 `-0,33..+0,30`: parita', non una
promozione di forza.

## Cercare efficienza senza cambiare il verdetto

Dopo la neutralita' 16x8 e' stato aperto un ramo separato. La domanda non era piu'
"lo student e' piu' forte?", gia' chiusa, ma "puo' conservare la stessa forza con una
search meno costosa?".

Lo screen 8x8 da 2.000 partite dimezza media e p95 della latenza (`circa 0,51x`), ma
perde `-0,742` punti con CI95 `-1,457..-0,027`. La perdita e' statisticamente visibile:
STOP prima della conferma lunga.

Il punto intermedio 12x8 e' stato preregistrato con soglie di non inferiorita' e costo.
Lo screen da 4.000 partite misura `-0,0495` punti (CI95 `-0,5666..+0,4676`) e latenza
media/p95 a circa `0,75x`, autorizzando la conferma indipendente.

## La conferma 12x8

Su 20.000 partite seat-fair con seed nuovi, policy student contro v14 e belief v0
identica sui due lati:

| metrica | student 12x8 | v14 16x8 |
|---|---:|---:|
| vittorie | `9.689` | `9.621` |
| pareggi | `690` | `690` |
| delta punti student | `+0,1052` | CI95 `-0,1126..+0,3230` |
| score rate student | `50,1700%` | CI95 `49,7018%..50,6382%` |
| latenza media search | `11,040 ms` | `14,666 ms` |
| latenza p95 | `16,440 ms` | `21,826 ms` |
| rapporto medio / p95 | `0,7528x` | `0,7532x` |

Il protocollo richiedeva stima almeno `-0,10`, limite inferiore CI95 sopra `-0,25`,
rapporti di latenza non oltre `0,82/0,85` e integrita' perfetta. Tutti i gate passano.
Le 1.400.156 determinizzazioni e i 4.200.468 rollout complessivi registrano zero
fallimenti e zero mosse corrette difensivamente.

L'intervallo del delta include zero: non possiamo affermare che 12x8 sia piu' forte.
Possiamo affermare, entro il margine dichiarato prima del test, che e' non inferiore e
che ogni decisione di search costa circa il 25% in meno.

## Dal benchmark al browser

L'audit di rilascio usa gli asset reali congelati e una directory modelli isolata. Ha
verificato catalogo, dipendenza belief, factory 12x8, risoluzione sicura del model id,
API, WebSocket e layout desktop/mobile. Una partita completa pilotata dal browser ha
prodotto 14 decisioni fallback, 3 search e 3 solver: tutte le tre fasi del giocatore sono
state attraversate senza errori console, pagina o rete.

Durante l'audit il candidato e' rimasto nascosto per default ed e' stato esposto soltanto
da un token esplicito. Solo dopo il PASS e' stato confezionato come
`best_a2c_v15.npz`, reso pubblico nel catalogo e scelto come default della release 0.38.0.

## Cosa significa v15

V15 identifica una nuova policy ufficiale e un nuovo profilo di esecuzione predefinito:
student distillato + PIMC belief 12x8. La promozione riguarda il compromesso costo/forza,
non una vittoria statisticamente certa su v14.

L'asset ufficiale pesa circa 0,43 MB, conserva soltanto i quattro tensori della MLP e
metadati pubblici riproducibili; non contiene dataset, storia completa del training o
percorsi del computer usato per l'esperimento.

La riproducibilita' resta parte del prodotto: v14 con PIMC 16x8 e le policy ufficiali
v13, v11 e v10 rimangono selezionabili; resta disponibile anche PIMC belief 64x10. Non
esisteva invece un PIMC 24x8: le 24 viste appartengono soltanto al teacher offline.

## Ricevute

- [`a2c-super-training-50m-2026-07-14.md`](../plans/a2c-super-training-50m-2026-07-14.md);
- [`suit-distillation-20m-250k-2026-07-17.md`](../plans/suit-distillation-20m-250k-2026-07-17.md);
- [`suit-student-8x8-efficiency-2026-07-17.md`](../plans/suit-student-8x8-efficiency-2026-07-17.md);
- [`suit-student-12x8-efficiency-2026-07-17.md`](../plans/suit-student-12x8-efficiency-2026-07-17.md);
- [`suit-student-12x8-release-audit-2026-07-17.md`](../plans/suit-student-12x8-release-audit-2026-07-17.md).
