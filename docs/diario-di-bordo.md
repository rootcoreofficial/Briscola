# Diario di bordo

Il diario di bordo — la storia del progetto (scelte, errori, svolte) raccontata in tono
divulgativo — è una **pagina del sito**: [ai.briscola.dev/diario](https://ai.briscola.dev/diario).

La **fonte unica è il markdown** `src/briscola_ai/frontend/static/diario.md`: per aggiornare
il diario si modifica solo quel file (il server lo renderizza a richiesta dentro
`diario_template.html`). Da aggiornare ai grandi traguardi, in stile non tecnico; la
cronologia tecnica completa vive nei messaggi di commit e in `docs/plans/`.

## Convenzioni di stile (dalla revisione del maintainer, luglio 2026)

- Tono discorsivo e naturale: niente morale a effetto a fine capitolo, pochi trattini
  lunghi, contrasti solo dove reggono.
- Zero gergo non spiegato; i **termini tecnici corretti si citano tra parentesi accanto
  alla descrizione discorsiva** (es. "è quello che chiamiamo sparring (il nome tecnico è
  *apprendimento per rinforzo*)"), così il lettore impara anche il vocabolario.
- Piccoli dettagli tecnici concreti nei punti dove sono la storia (Python/Numba, seme,
  sensori della rete), con link esterni per approfondire.
- I numeri dettagliati vivono negli approfondimenti `docs/diario/*.md`. La loro numerazione segue l'ordine di
  scrittura: dopo condensazioni editoriali più approfondimenti possono riferirsi allo stesso capitolo pubblico,
  quindi numero del file e numero del capitolo non devono necessariamente coincidere.
