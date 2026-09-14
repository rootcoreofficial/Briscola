# Audit di rilascio dello student PIMC 12x8

Data: 2026-07-17
Stato: **PASS integrazione; promosso come v15 nella release 0.38.0**
Default deciso: `best_a2c_v15.npz` con PIMC belief 12x8

## Scopo e confine

La conferma da 20.000 partite ha certificato lo student 12x8 come non inferiore a v14
16x8 con circa il 25% di costo search in meno. Questo audit verifica il passaggio dagli
artefatti offline al prodotto: factory dell'agente, dipendenze, catalogo, API, WebSocket e
browser. Al momento dell'esecuzione non assegnava ancora il nome v15 e non cambiava il
default: la promozione è stata eseguita soltanto dopo il PASS completo.

Durante l'audit il percorso era protetto dal token esatto
`BRISCOLA_RELEASE_CANDIDATE_12X8=student-12x8`. La release 0.38.0 ha poi rimosso il gate:
`bc_model_pimc_belief_12x8` è ora un agente pubblico e il nuovo default della UI.

## Asset congelati

| asset | SHA-256 |
|---|---|
| student teacher20M 250k | `8a0a03946c9413ed7e6c18059a6aa03f63a9476e0b603ad977f4955cb444199d` |
| belief v0 | `4100b23b65a5566e047230ced665b91eef1942ea31e4a4cbe201b64545e7d035` |

Il server di audit usa una directory isolata con soltanto questi due file e indica lo
student come `BRISCOLA_DEFAULT_MODEL_ID`. Il catalogo restituisce una sola policy,
`is_compatible: true`, mentre la belief resta correttamente nascosta come asset interno.
`GET /version` conferma la presenza di policy e belief.

## Verifiche automatiche

I test bloccano i seguenti contratti:

1. la factory costruisce esattamente 12 determinizzazioni, finestra 8, mix uniforme
   `0,10`, belief v0, solver attivo e search Python;
2. durante l'audit il candidato era assente dalla lista pubblica ordinaria e compariva
   soltanto nella lista protetta;
3. senza belief l'opzione è non disponibile; con la belief diventa disponibile;
4. `POST /api/games` conserva nome agente e id della policy selezionata, passando dalla
   stessa risoluzione anti-path-traversal degli altri modelli.

Suite mirata: **55 test passati** in `tests/test_pimc_agent.py` e
`tests/test_api_integration.py`.

## Audit browser con asset reali

Playwright ha verificato il flusso su desktop `1440x1000` e mobile `390x844`:

- opzione agente presente e selezionabile soltanto nel server di audit;
- unico modello selezionabile e consigliato: lo student congelato;
- creazione partita HTTP 200 con coppia agente/modello esatta;
- prima azione HTTP 200, WebSocket connesso e layout senza sovrapposizioni;
- zero errori console, eccezioni pagina o richieste fallite.

Una seconda partita e' stata completata dal browser usando le API della stessa pagina:
20 azioni umane e 20 risposte IA. I frame `ai_card_reveal` contano 14 decisioni
`fallback`, 3 `search` e 3 `solver`; quindi il test attraversa davvero la finestra PIMC
12x8 e il finale esatto, non soltanto il caricamento della policy. Tutte le azioni hanno
risposto 200 e la partita e' terminata 44-76 senza errori client o rete.

## Verdetto e prossimo confine

**PASS integrazione e promozione eseguita.** Non sono emersi difetti che impedissero di
confezionare lo student 12x8 come variante ufficiale di efficienza. Il change atomico ha:

- assegnato i nomi `best_a2c_v15.npz` e release `0.38.0`;
- prodotto un NPZ compatto con metadati pubblici, senza path locali o diciture sperimentali;
- tracciato l'asset e aggiornato default agente/modello, provisioning e catalogo;
- rigenerato report Excel e diario prima del quality gate, tag e push.

Il PASS non cambia l'interpretazione statistica: lo student 12x8 e' un miglior compromesso
costo/forza, non un modello dimostrato piu' forte di v14 16x8.
