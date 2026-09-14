# Approfondimento — Il fast path e Numba (14×)

**Capitolo del diario:** [Capitolo 5](https://ai.briscola.dev/diario) · **Periodo:** 3–9 giugno 2026

## L'architettura a tre livelli

1. **Dominio canonico**: puro, leggibile, fonte di verità.
2. **Fast path Python** (`ai/fast/`): stesse regole su interi/array, mutabile.
3. **Kernel Numba** (`ai/numba/`): loop di partita interamente JIT-compilati, `prange` per
   il parallelismo. Encoder, forward MLP, sampling e perfino il backprop A2C vettorizzato.

Ogni livello è ancorato al precedente da **test di parità su partite specchiate** (stesso
seme → stesso mazzo → stesse mosse → encoder e risultati identici).

## Come è protetta la parità

Il metodo dei test è "partite specchiate": stesso seme → stesso mazzo per costruzione
(entrambe le implementazioni mescolano con `random.Random(seed).shuffle`), stesse mosse
scelte da un RNG condiviso → a ogni profondità di partita gli encoder devono produrre
vettori identici e gli stati devono coincidere carta per carta, per entrambi i giocatori.
Ogni nuova versione dell'encoder (v1→v4) ha ripetuto questo rito su tre motori:
dominio ↔ fast Python ↔ kernel Numba.

## Dettagli del kernel

- `@njit(cache=True)` su tutte le funzioni calde; batch di partite in `prange`
  (parallelismo sui core, tipicamente 8-14 worker).
- Anche il **backprop A2C** è vettorizzato NumPy (nessun framework: i gradienti sono
  scritti a mano) e il collector di traiettorie è full-JIT.
- Lezione operativa: la cache Numba locale può mascherare firme rotte tra kernel — la CI
  (compilazione fredda) è il giudice di verità.

## I numeri

| Metrica | Prima | Dopo |
|---|---|---|
| Training A2C | ~419 partite/s | **~5.900 partite/s** (14×) |
| 5M partite di training | ore | **~930 secondi** (`fef976c`) |
| Evaluation 100k partite | — | 35–45 secondi |

## L'aneddoto dei 244 MB

Il modello 5M seed19 pesava 244 MB: `np.savez` aveva serializzato 250.000 record di metriche
come stringa JSON NumPy (~4 byte/carattere). La copia promossa: **138 KB**, identica a giocare
(`7516708`). La dimensione del file non dice nulla della forza del modello.
