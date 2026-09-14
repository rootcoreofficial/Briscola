# Approfondimento — Chi decide quando l'IA "pensa"

**Capitolo del diario:** [Capitolo 1](https://ai.briscola.dev/diario) · **Periodo:** 9–13 gennaio 2026

## Il problema

L'IA decide la mossa in microsecondi; la pausa che l'utente vede è presentazione. La domanda
architetturale: il ritmo lo detta il client (browser) o il server?

## Le due architetture provate

| Approccio | Commit chiave | Esito |
|---|---|---|
| Client-driven: il browser chiama `/ai-turn` quando è "pronto" | `9fbee17` (10 gen) | Fragile: doppi trigger, race su riconnessione |
| Client-driven blindato: lock + `server_version` idempotente | `fee2b1e` (12 gen) | Funziona ma la complessità cresce a ogni edge case |
| **Server-driven: il backend avanza subito, coda eventi WS + hold nel frontend** | `ac1800d` (13 gen) | **Adottato, regge ancora oggi** |

## La lezione

Lo stato di gioco ha UNA fonte di verità (il server); la UX è un problema di *racconto*
degli eventi, non di controllo del flusso. Il backend non introduce delay di presentazione:
è una regola scritta in `AGENTS.md` (il documento di riferimento condiviso per gli agenti; `CLAUDE.md` ne è un alias) e mai più violata.
