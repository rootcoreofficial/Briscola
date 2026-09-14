"""
Solver esatto dell'endgame (minimax 2-player) — Fase 5G, step 1.

Idea
----
Quando il mazzo è finito (`deck_size == 0`) in una partita 2-player, restano al massimo
3 carte per mano (≤ 6 carte totali in gioco). L'albero di gioco è quindi minuscolo e possiamo
calcolare la mossa **ottima esatta** con un minimax completo, senza euristiche.

Cosa massimizza
---------------
Il solver ragiona sul **delta-punti finale dal punto di vista del player 0**:

    final_delta_p0_p1 = players[0].points - players[1].points   (a fine partita)

È una scelta deliberata (riferimento fisso) per evitare ambiguità di segno nella ricorsione:
- quando tocca al player 0 (`current_turn == 0`) la mossa ottima **massimizza** questo delta;
- quando tocca al player 1 (`current_turn == 1`) la mossa ottima **minimizza** questo delta.

Il gioco è a somma costante (totale punti del mazzo = 120), quindi minimizzare il delta p0-p1
equivale a massimizzare i punti del player 1: è un classico minimax zero-sum.

Coerenza col dominio
--------------------
Tutte le transizioni passano da `domain.engine.step`, che è la fonte canonica delle regole
(gestione tavolo parziale, vincitore della presa, ricalcolo punti, fine partita). Il solver non
duplica regole: esplora soltanto.

Nota su "anti-cheat"
--------------------
Questo modulo è un **oracolo di dominio**: opera su `GameState` completo, quindi vede entrambe
le mani. NON è ancora agent-safe. Diventa lecito solo nello step 2 (agente ibrido), quando lo
stato di endgame viene ricostruito a partire da `PlayerObservation` senza leggere informazione
nascosta (a mazzo vuoto la mano avversaria è comunque deducibile dall'informazione pubblica).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.engine import PlayCardAction, step
from ...domain.state import GameState

# Limite massimo di carte residue gestite dal solver (3 per mano in 2-player, più al più una
# carta già sul tavolo). Serve solo come guard contro usi sbagliati su stati artificiali.
_MAX_REMAINING_CARDS = 6


@dataclass(frozen=True, slots=True)
class EndgameSolution:
    """
    Risultato del solver per lo stato passato.

    Campi:
    - `best_card_index`: indice (nella mano di `state.current_turn`) della carta ottima da giocare.
    - `final_delta_p0_p1`: delta-punti finale `players[0].points - players[1].points` raggiungibile
      con gioco ottimo di **entrambi** i giocatori. È sempre dal punto di vista del player 0,
      indipendentemente da chi deve muovere.
    - `principal_variation`: sequenza ottima di mosse come coppie `(player_index, card_index)`.
      Usiamo `player_index` esplicito perché i `card_index` sono locali alla mano del momento e
      cambiano dopo ogni giocata: senza il player la sequenza sarebbe ambigua da rileggere.
    """

    best_card_index: int
    final_delta_p0_p1: int
    principal_variation: tuple[tuple[int, int], ...]


def _validate(state: GameState) -> None:
    """
    Verifica le precondizioni del solver e solleva `ValueError` con messaggio esplicito.

    Manteniamo il solver "strict": niente fallback silenziosi su stati fuori scope, così è
    impossibile usarlo per sbaglio (es. a mazzo non vuoto) ottenendo risultati senza senso.
    """
    if state.num_players != 2:
        raise ValueError(f"Il solver endgame supporta solo 2 giocatori (ricevuti {state.num_players})")
    # `num_players` e la tupla `players` potrebbero in teoria divergere su stati artificiali:
    # verifichiamo entrambi, perché più sotto indicizziamo direttamente `players[0]`/`players[1]`.
    if len(state.players) != 2:
        raise ValueError(f"Attesi 2 player, trovati {len(state.players)}")
    # Indici di player validi: senza questo controllo `current_turn=-1` verrebbe accettato e
    # l'indicizzazione negativa di Python produrrebbe risultati silenziosamente sbagliati.
    if state.current_turn not in (0, 1):
        raise ValueError(f"current_turn fuori range: {state.current_turn}")
    if state.game_over:
        raise ValueError("Partita già terminata: nessuna mossa da risolvere")
    if len(state.deck) != 0:
        raise ValueError(f"Il solver richiede il mazzo vuoto (deck_size={len(state.deck)})")
    if len(state.table_cards) not in (0, 1):
        raise ValueError(f"Stato tavolo non supportato: attese 0 o 1 carte, trovate {len(state.table_cards)}")
    for _card, player_idx in state.table_cards:
        if player_idx not in (0, 1):
            raise ValueError(f"Player id sul tavolo fuori range: {player_idx}")
    remaining = sum(len(p.hand) for p in state.players) + len(state.table_cards)
    if remaining == 0:
        raise ValueError("Nessuna carta residua: stato non terminale ma senza mosse possibili")
    if remaining > _MAX_REMAINING_CARDS:
        raise ValueError(f"Troppe carte residue per l'endgame esatto: {remaining} > {_MAX_REMAINING_CARDS}")

    # Coerenza delle mani: in un endgame 2-player ben formato le mani restano "bilanciate".
    # A tavolo vuoto i due giocatori hanno lo stesso numero di carte; con una carta sul tavolo
    # chi deve muovere (il non-apritore) ne ha esattamente una in più di chi ha aperto.
    # Senza questo controllo uno stato sbilanciato porterebbe la ricorsione a un mover con la
    # mano vuota e a un fallimento poco leggibile.
    h0, h1 = len(state.players[0].hand), len(state.players[1].hand)
    if len(state.table_cards) == 0:
        if h0 != h1:
            raise ValueError(f"Mani sbilanciate a tavolo vuoto: {h0} vs {h1}")
    else:
        leader = state.table_cards[0][1]
        if state.current_turn == leader:
            raise ValueError("Turno incoerente: chi ha aperto la mano non può rigiocare")
        if len(state.players[state.current_turn].hand) != len(state.players[leader].hand) + 1:
            raise ValueError("Mani sbilanciate rispetto alla carta sul tavolo")


def _minimax(state: GameState, memo: dict[GameState, tuple[int, int | None]]) -> tuple[int, int | None]:
    """
    Minimax esatto. Ritorna `(final_delta_p0_p1, best_card_index)` per lo stato dato.

    - `final_delta_p0_p1`: valore terminale ottimo dal punto di vista del player 0.
    - `best_card_index`: mossa ottima per `state.current_turn` (None nei nodi terminali).

    Memoization: usiamo direttamente `GameState` come chiave. È un dataclass `frozen` (quindi
    hashable) e cattura tutto ciò che serve (mani, tavolo+player, punti, turno, trump_card),
    evitando firme manuali incomplete. Con ≤ 6 carte la cache è più una rete di sicurezza che
    una necessità di performance.
    """
    if state.game_over:
        return state.players[0].points - state.players[1].points, None

    cached = memo.get(state)
    if cached is not None:
        return cached

    mover = state.current_turn
    maximize = mover == 0  # il player 0 massimizza il delta p0-p1, il player 1 lo minimizza

    best_value: int | None = None
    best_index: int | None = None

    # Le carte sono iterate in ordine crescente di indice: usando confronti stretti (>/<),
    # a parità di valore resta selezionato l'indice più basso => tie-break deterministico.
    for card_index in range(len(state.players[mover].hand)):
        next_state, _ = step(state, PlayCardAction(player_index=mover, card_index=card_index))
        child_value, _ = _minimax(next_state, memo)

        if best_value is None:
            best_value, best_index = child_value, card_index
        elif maximize:
            if child_value > best_value:
                best_value, best_index = child_value, card_index
        else:
            if child_value < best_value:
                best_value, best_index = child_value, card_index

    # `best_value` non può essere None: lo stato non è terminale, quindi la mano del mover ha
    # almeno una carta giocabile.
    assert best_value is not None and best_index is not None
    result = (best_value, best_index)
    memo[state] = result
    return result


def solve_endgame(state: GameState) -> EndgameSolution:
    """
    Risolve esattamente l'endgame a partire da `state` e ritorna la mossa ottima.

    Argomenti:
        state: stato di dominio 2-player con mazzo vuoto (vedi `_validate` per le precondizioni).

    Ritorna:
        Un `EndgameSolution` con la carta ottima per `state.current_turn`, il delta-punti finale
        ottimo (riferimento player 0) e la principal variation.
    """
    _validate(state)

    memo: dict[GameState, tuple[int, int | None]] = {}
    final_delta, best_index = _minimax(state, memo)
    assert best_index is not None  # garantito da `_validate` (stato non terminale)

    # Ricostruiamo la principal variation rigiocando le mosse ottime già memoizzate.
    principal_variation: list[tuple[int, int]] = []
    cursor = state
    while not cursor.game_over:
        _, move_index = _minimax(cursor, memo)
        if move_index is None:
            break
        mover = cursor.current_turn
        principal_variation.append((mover, move_index))
        cursor, _ = step(cursor, PlayCardAction(player_index=mover, card_index=move_index))

    return EndgameSolution(
        best_card_index=best_index,
        final_delta_p0_p1=final_delta,
        principal_variation=tuple(principal_variation),
    )
