# Struttura del package AI

Questo package contiene tutto cio' che riguarda gli agenti e la pipeline ML. E' diviso
per responsabilita', cosi' chi studia il progetto puo' partire dal livello giusto senza
dover leggere subito i path ottimizzati.

## Mappa

- `agents/`: agenti giocabili nel backend/UI. La facciata `__init__.py` esporta l'API pubblica;
  l'implementazione e' separata in `base.py`, `rule_based.py`, `hybrid_endgame.py`, `registry.py`.
- `models/`: caricamento modelli `.npz`, agente BC/A2C, catalogo server-side e provisioning.
- `endgame/`: solver esatto del finale 2-player a mazzo vuoto; `solver.py` e' l'oracolo didattico
  su dominio canonico, `fast_solver.py` e' il solver completo numerico/Python, `numba_solver.py` e'
  il choose-only JIT per training, valutazioni e benchmark offline.
- `encoding/`: spazio azioni e encoder observation -> feature/mask per i modelli.
- `training/`: componenti di training condivisi (curriculum, reward shaping, opponent mix, regolarizzazioni).
- `evaluation/`: valutazione offline, matrici benchmark e metriche di qualita' decisionale.
- `fast/`: motore 2-player mutabile in Python/NumPy per rollout veloci.
- `numba/`: path JIT ad alto throughput esclusivamente offline. `core.py` contiene regole/euristiche numeriche,
  `observation.py` encoder e kernel condivisi, `value_lookahead.py` il core depth-1 su stati
  numerici determinizzati e il collector A2C value-aware, `mlp.py` wrapper MLP/A2C, `types.py` DTO.

## Confine del runtime web

Il backend/UI di produzione e' deliberatamente **zero-Numba**: search PIMC e solver usano
i path Python (`agents/pimc.py` e `endgame/fast_solver.py`) e il processo web non importa
`numba`/`llvmlite`. Evitiamo cosi' compilazione JIT, memoria aggiuntiva e warm-up a ogni
cold start. I kernel in `ai/numba/` restano fondamentali per self-play, training,
evaluation e benchmark, dove il costo di compilazione viene ammortizzato su milioni di stati.

## Regola didattica

Il dominio canonico resta in `briscola_ai.domain`. Gli agenti ricevono sempre
`PlayerObservation`, mai `GameState` completo, salvo moduli-oracolo espliciti come
`endgame.solver`/`endgame.fast_solver` che sono usati solo dopo ricostruzione lecita dell'informazione.
Il kernel `numba.value_lookahead` e' destinato a training/evaluation su stati gia' determinizzati:
non campiona information set e non sostituisce l'agente runtime anti-cheat. Quando e' usato come
opponent in `train_a2c.py`, la policy candidata continua a ricevere solo feature da osservazione
lecita; e' l'avversario di training a usare la determinizzazione numerica gia' presente nel rollout.

## Import

Il nuovo codice deve usare i percorsi organizzati sopra. I vecchi moduli root storici
sono stati rimossi per evitare ambiguita' didattica: ogni import deve rendere chiara
la responsabilita' del modulo che sta usando.
