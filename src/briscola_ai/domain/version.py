"""
Versione delle regole di gioco (dominio).

Perché esiste questo file?
--------------------------
Quando iniziamo a raccogliere dataset e addestrare modelli, è essenziale sapere
con quale versione delle regole è stata giocata una partita.

Esempi di cambi che richiedono bump di `RULES_VERSION`:
- cambia l'ordine di forza delle carte;
- cambia il calcolo punti;
- cambia la gestione della briscola (es. carta sotto al mazzo, pescata finale);
- cambia il comportamento di `step()` (semantica, non refactor interno).

Regola pratica:
- aumentiamo `RULES_VERSION` solo per cambi “semantici” del dominio.
- refactor, test, doc, ottimizzazioni: NON richiedono bump.
"""

# Versione delle regole. Il tipo è STRINGA, fissato per contratto: finisce nell'event log
# e nei metadati dataset, dove un confronto accidentale `"1" != 1` sarebbe un bug subdolo.
# Partiamo da "1": significa “regole attuali baseline”.
RULES_VERSION: str = "1"
