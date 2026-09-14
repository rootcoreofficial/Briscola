# Approfondimento — L'audit dei precetti d'apertura

**Capitolo del diario:** [Capitolo 10](https://ai.briscola.dev/diario) · **Periodo:** 3 luglio 2026

## La domanda

I precetti classici del briscolista (non uscire coi carichi, non sprecare briscole, non
regalare figure) vanno insegnati all'IA? Prima di insegnare, misurare.

## L'audit (1.000 partite, prime 5 prese, vs heuristic_v2)

| Precetto | v8 | heuristic_v2 (scritta CON i precetti) |
|---|---|---|
| Uscite col carico | 3.7% | 2.0% |
| Uscite di briscola | 19.8% | 19.2% |
| Briscola su presa povera (≤2 punti) | **7.0%** | 10.1% |
| Scarta figura senza vincere | **9.3%** | 10.6% |
| Regala carico senza vincere | 0.4% | 0.4% |

## Verdetto

v8 rispetta tutti i precetti, e su due è PIÙ ortodossa dell'euristica programmata a mano —
senza che nessuno glieli abbia mai scritti. Il 3.7% di uscite col carico non è
(probabilmente) indisciplina: è il carico giocato coperto, con controllo briscole — la
sfumatura che il precetto non sa esprimere. Nessuna regola umana da iniettare; l'audit resta
come cruscotto di stile per le generazioni future.
