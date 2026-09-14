# Approfondimento — BC, RL e la guerra all'overkill

**Capitolo del diario:** [Capitolo 3](https://ai.briscola.dev/diario) · **Periodo:** gennaio–febbraio 2026

## Le due scuole

- **Behavioral Cloning** (`45cc792`, `aeeafc1`): encoder osservazione → MLP → azione tra 40
  id carta con action mask. Impara in fretta, copia anche i difetti del teacher.
- **Reinforcement Learning**: policy gradient (`cb5a0b1`) poi A2C con reward shaping
  (`4c12e37`) su delta punti per presa. Primo `best_a2c`: **+9.71** su heuristic_v1
  (big holdout), cresciuto a ~+12.69 con league training (avversario campione congelato
  nel mix per evitare il "chasing" di due policy che cambiano insieme, `3da6a17`).

## I meccanismi, in breve

- **Spazio azioni**: 40 classi (una per carta del mazzo) + *action mask* che abilita solo
  le carte in mano. Vale per BC e RL.
- **Encoder v1** (248 feature): mano, tavolo, briscola e scalari di stato (punti, mazzo,
  primo/secondo di mano) in one-hot con punti/forza per carta.
- **BC**: cross-entropy sulle mosse del teacher (JSONL da self-play), MLP a 1 hidden layer.
- **A2C**: actor (policy) + critic `V(s)`; advantage `A = G − V(s)` come baseline; reward
  denso = delta `punti_policy − punti_opp` per presa, normalizzato — solo informazione
  pubblica (anti-cheat anche nel reward). Il critic condivide l'hidden layer con l'actor.
- **League training**: il campione precedente CONGELATO nel mix di avversari (con quote di
  euristiche e random), per evitare il "chasing" di due policy che cambiano insieme.

## La guerra all'overkill (10 febbraio)

Overkill = giocare una briscola alta per vincere una presa povera. Il best dell'epoca lo
faceva nel **20.3%** delle occasioni. Tre tentativi:

| Rimedio | Commit | Esito |
|---|---|---|
| Penalità flat in training | `ae94aa9` | "NON riduce in modo affidabile" |
| Penalità proporzionale al gap | `92f4f5f` | "NON migliora (tende anzi a peggiorarla)" |
| **Guard in inference** (post-processing: scegli la briscola minima che vince comunque) | `6cd1d23`, A/B `bdefa6a` | **Adottato: costo in forza ≈ 0** |

Come funziona il guard: a decisione presa, se la carta scelta è una briscola che vince una
presa con ≤2 punti in palio ed esiste una briscola più bassa che vince comunque, si gioca
quella. È un post-processing puro: la policy non viene toccata, e il flag vive nei metadati
del modello (`inference_overkill_guard`).

Lezione: correggere un vizio nella funzione di reward può distorcere tutto il resto;
un vincolo esplicito al momento della decisione è chirurgico.
