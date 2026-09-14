"""
Test del solver esatto dell'endgame (Fase 5G, step 1).

Strategia di test
-----------------
- Stati costruiti a mano con poche carte e mazzo vuoto, tutti spiegati nei commenti.
- Invariante importante: `step` ricalcola i punti del vincitore da `captured_cards`, quindi
  ogni stato iniziale deve avere `points == somma punti delle captured_cards`. Negli scenari
  qui sotto le prese sono vuote (`points == 0`), così lo stato è banalmente coerente.
- La briscola (`trump_card`) serve solo a fissare il **seme** di briscola: a mazzo vuoto la sua
  identità non è più in gioco, quindi usiamo una carta del seme di briscola non presente in mano.
- Oltre agli scenari calcolati a mano, un test "integrazione" gioca una partita reale fino al
  mazzo vuoto e verifica la coerenza interna del solver (PV terminale, somma punti = 120).
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.rules import trick_points
from briscola_ai.domain.state import GameState, PlayerState, new_game_state

# Briscola = denari (COINS) in tutti gli scenari a mano. Usiamo il 7 di denari come `trump_card`:
# non compare in nessuna mano, quindi conta solo per definire il seme di briscola.
TRUMP_SUIT = Suit.COINS
TRUMP_CARD = Card(Suit.COINS, Rank.SEVEN)


def _endgame_state(
    hand0: tuple[Card, ...],
    hand1: tuple[Card, ...],
    *,
    current_turn: int,
    table_cards: tuple[tuple[Card, int], ...] = (),
    captured0: tuple[Card, ...] = (),
    captured1: tuple[Card, ...] = (),
) -> GameState:
    """Costruisce uno stato 2-player a mazzo vuoto, con punti coerenti con le prese."""
    players = (
        PlayerState(name="P0", hand=hand0, captured_cards=captured0, points=trick_points(captured0)),
        PlayerState(name="P1", hand=hand1, captured_cards=captured1, points=trick_points(captured1)),
    )
    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=players,
        deck=(),
        trump_card=TRUMP_CARD,
        table_cards=table_cards,
        current_turn=current_turn,
        first_player=current_turn,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def test_single_trick_value_is_exact() -> None:
    """Una sola presa, deterministica: l'Asso di coppe batte il Re di coppe e cattura 11+4."""
    state = _endgame_state(
        hand0=(Card(Suit.CUPS, Rank.ACE),),  # 11 punti
        hand1=(Card(Suit.CUPS, Rank.KING),),  # 4 punti
        current_turn=0,
    )
    solution = solve_endgame(state)
    assert solution.best_card_index == 0
    assert solution.final_delta_p0_p1 == 15  # P0 cattura entrambe le carte: +15
    assert solution.principal_variation == ((0, 0), (1, 0))


def test_tempo_matters_player0() -> None:
    """
    Il tempo conta: con due mosse a esito diverso, il solver sceglie la migliore.

    Briscola = denari. P0 (muove) ha Cavallo di denari (briscola, 3pt, ma più debole del Re)
    e Asso di coppe (11pt). P1 ha Re di denari (briscola, 4pt, più forte del Cavallo) e Due di
    bastoni (0pt). P0 è comunque in svantaggio: l'obiettivo è perdere il meno possibile.

    - Se P0 "incassa" subito l'Asso di coppe: P1 lo taglia col Re di briscola e cattura 11+4=15,
      poi conduce il Due di bastoni e P0 col Cavallo di briscola recupera 3. delta = 3 - 15 = -12.
    - Se P0 apre col Cavallo di briscola: P1 sovrataglia col Re (cattura 3+4=7), poi conduce il
      Due di bastoni e P0 (Asso di coppe, fuori seme) non può prendere: P1 cattura anche l'Asso.
      delta = 0 - 18 = -18.

    Quindi la mossa ottima è incassare l'Asso di coppe (indice 1), che limita il danno a -12.
    """
    state = _endgame_state(
        hand0=(Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)),
        hand1=(Card(Suit.COINS, Rank.KING), Card(Suit.CLUBS, Rank.TWO)),
        current_turn=0,
    )
    solution = solve_endgame(state)
    assert solution.best_card_index == 1  # Asso di coppe: limita il danno a -12
    assert solution.final_delta_p0_p1 == -12


def test_polarity_player1_minimizes() -> None:
    """
    Scenario speculare con `current_turn == 1`: blocca eventuali bug di polarità max/min.

    È lo scenario precedente con i ruoli scambiati: ora muove P1 (con Cavallo di briscola e Asso
    di coppe). La mossa ottima è la stessa (incassare l'Asso di coppe, indice 1) e, per simmetria,
    il delta dal punto di vista del player 0 è +12 (cioè -12 per il mover P1).

    Se il solver trattasse erroneamente `current_turn == 1` come massimizzatore di p0-p1,
    sceglierebbe la mossa sbagliata e otterrebbe un delta diverso: il test lo intercetta.
    """
    state = _endgame_state(
        hand0=(Card(Suit.COINS, Rank.KING), Card(Suit.CLUBS, Rank.TWO)),
        hand1=(Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)),
        current_turn=1,
    )
    solution = solve_endgame(state)
    assert solution.best_card_index == 1
    assert solution.final_delta_p0_p1 == 12


def test_tie_with_no_point_cards() -> None:
    """Solo carte da 0 punti: nessuno può segnare, il delta finale è 0."""
    state = _endgame_state(
        hand0=(Card(Suit.CLUBS, Rank.TWO),),
        hand1=(Card(Suit.SWORDS, Rank.TWO),),
        current_turn=0,
    )
    solution = solve_endgame(state)
    assert solution.final_delta_p0_p1 == 0


def test_second_to_play_with_card_on_table() -> None:
    """
    Il solver gestisce anche lo stato 'secondo di mano' (`len(table_cards) == 1`).

    Sul tavolo c'è il Due di bastoni giocato da P0; tocca a P1 (che ha una carta in più di P0,
    come in un endgame ben formato). P1 ha l'Asso di bastoni (stesso seme, più forte) e il Cinque
    di spade; P0 ha ancora il Quattro di spade.

    - Se P1 prende con l'Asso di bastoni: cattura 11 punti e poi vince anche la seconda presa
      (Cinque > Quattro di spade), 0 punti. delta = 0 - 11 = -11.
    - Se P1 scarta il Cinque di spade: il Due di bastoni di P0 vince la presa (0 punti), poi P0
      conduce il Quattro di spade e l'Asso di bastoni di P1 (fuori seme) non può prendere: P0
      cattura 11. delta = +11.

    La mossa ottima per P1 è quindi prendere con l'Asso di bastoni (indice 0).
    """
    state = _endgame_state(
        hand0=(Card(Suit.SWORDS, Rank.FOUR),),
        hand1=(Card(Suit.CLUBS, Rank.ACE), Card(Suit.SWORDS, Rank.FIVE)),
        current_turn=1,
        table_cards=((Card(Suit.CLUBS, Rank.TWO), 0),),
    )
    solution = solve_endgame(state)
    # P1 prende con l'Asso di bastoni (indice 0) e cattura i suoi 11 punti => delta p0-p1 = -11.
    assert solution.best_card_index == 0
    assert solution.final_delta_p0_p1 == -11


def test_principal_variation_is_consistent_and_terminal() -> None:
    """La principal variation, rigiocata, deve portare a fine partita e rispettare i turni."""
    state = _endgame_state(
        hand0=(Card(Suit.COINS, Rank.THREE), Card(Suit.CLUBS, Rank.TWO)),
        hand1=(Card(Suit.CUPS, Rank.ACE), Card(Suit.CLUBS, Rank.FOUR)),
        current_turn=0,
    )
    solution = solve_endgame(state)

    # La prima mossa della PV coincide con la mossa ottima dichiarata.
    assert solution.principal_variation[0] == (0, solution.best_card_index)

    cursor = state
    for player_index, card_index in solution.principal_variation:
        assert player_index == cursor.current_turn  # i turni della PV sono coerenti
        cursor, _ = step(cursor, PlayCardAction(player_index=player_index, card_index=card_index))
    assert cursor.game_over
    # Il delta finale raggiunto rigiocando la PV coincide con quello calcolato dal solver.
    assert cursor.players[0].points - cursor.players[1].points == solution.final_delta_p0_p1


def test_real_game_endgame_is_self_consistent() -> None:
    """
    Integrazione: gioca una partita reale fino al mazzo vuoto, poi risolve l'endgame.

    Verifica proprietà strutturali (senza calcolare l'ottimo a mano):
    - lo stato di endgame ha mazzo vuoto, tavolo pulito e 3 carte per mano;
    - la principal variation porta a `game_over` con somma punti == 120;
    - il delta finale rigiocando la PV coincide con `final_delta_p0_p1`.
    """
    state = new_game_state(2, seed=12345)
    # Politica fissa "gioca la prima carta" finché il mazzo non si svuota.
    while len(state.deck) > 0 and not state.game_over:
        state, _ = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))

    assert not state.game_over
    assert len(state.deck) == 0
    assert state.table_cards == ()
    assert tuple(len(p.hand) for p in state.players) == (3, 3)

    solution = solve_endgame(state)

    cursor = state
    for player_index, card_index in solution.principal_variation:
        assert player_index == cursor.current_turn
        cursor, _ = step(cursor, PlayCardAction(player_index=player_index, card_index=card_index))
    assert cursor.game_over
    assert cursor.players[0].points + cursor.players[1].points == 120
    assert cursor.players[0].points - cursor.players[1].points == solution.final_delta_p0_p1


@pytest.mark.parametrize(
    "make_state, message_fragment",
    [
        # Mazzo non vuoto.
        (
            lambda: GameState(
                num_players=2,
                is_team_game=False,
                teams=None,
                players=(
                    PlayerState("P0", (Card(Suit.CUPS, Rank.ACE),), (), 0),
                    PlayerState("P1", (Card(Suit.CUPS, Rank.KING),), (), 0),
                ),
                deck=(Card(Suit.SWORDS, Rank.TWO),),
                trump_card=TRUMP_CARD,
                table_cards=(),
                current_turn=0,
                first_player=0,
                game_over=False,
                winner_index=None,
                winning_team=None,
            ),
            "mazzo vuoto",
        ),
        # 4 giocatori non supportati.
        (
            lambda: GameState(
                num_players=4,
                is_team_game=True,
                teams=((0, 2), (1, 3)),
                players=tuple(PlayerState(f"P{i}", (Card(Suit.CUPS, Rank.ACE),), (), 0) for i in range(4)),
                deck=(),
                trump_card=TRUMP_CARD,
                table_cards=(),
                current_turn=0,
                first_player=0,
                game_over=False,
                winner_index=None,
                winning_team=None,
            ),
            "2 giocatori",
        ),
        # Partita già finita.
        (
            lambda: GameState(
                num_players=2,
                is_team_game=False,
                teams=None,
                players=(PlayerState("P0", (), (), 0), PlayerState("P1", (), (), 0)),
                deck=(),
                trump_card=TRUMP_CARD,
                table_cards=(),
                current_turn=0,
                first_player=0,
                game_over=True,
                winner_index=None,
                winning_team=None,
            ),
            "terminata",
        ),
    ],
)
def test_guards_reject_out_of_scope_states(make_state, message_fragment: str) -> None:
    """Il solver è strict: rifiuta stati fuori scope con messaggi espliciti."""
    with pytest.raises(ValueError, match=message_fragment):
        solve_endgame(make_state())


def test_guard_rejects_too_many_remaining_cards() -> None:
    """Più di 6 carte residue: stato artificiale fuori dall'endgame reale, rifiutato."""
    hand0 = tuple(Card(Suit.CLUBS, r) for r in (Rank.TWO, Rank.FOUR, Rank.FIVE, Rank.SIX))
    hand1 = tuple(Card(Suit.SWORDS, r) for r in (Rank.TWO, Rank.FOUR, Rank.FIVE, Rank.SIX))
    state = _endgame_state(hand0=hand0, hand1=hand1, current_turn=0)
    with pytest.raises(ValueError, match="Troppe carte"):
        solve_endgame(state)


def test_guard_rejects_current_turn_out_of_range() -> None:
    """`current_turn` negativo: senza guard l'indicizzazione negativa darebbe risultati errati."""
    state = _endgame_state(
        hand0=(Card(Suit.CUPS, Rank.ACE),),
        hand1=(Card(Suit.CUPS, Rank.KING),),
        current_turn=-1,
    )
    with pytest.raises(ValueError, match="current_turn fuori range"):
        solve_endgame(state)


def test_guard_rejects_table_player_out_of_range() -> None:
    """Player id incoerente sulla carta a tavolo: stato malformato, rifiutato."""
    state = _endgame_state(
        hand0=(Card(Suit.SWORDS, Rank.FOUR),),
        hand1=(Card(Suit.CLUBS, Rank.ACE), Card(Suit.SWORDS, Rank.FIVE)),
        current_turn=1,
        table_cards=((Card(Suit.CLUBS, Rank.TWO), 5),),
    )
    with pytest.raises(ValueError, match="Player id sul tavolo"):
        solve_endgame(state)


def test_guard_rejects_non_terminal_state_without_cards() -> None:
    """Mani vuote ma `game_over=False`: stato non terminale senza mosse, deve fallire pulito."""
    state = _endgame_state(hand0=(), hand1=(), current_turn=0)
    with pytest.raises(ValueError, match="Nessuna carta residua"):
        solve_endgame(state)


def test_guard_rejects_wrong_players_tuple_length() -> None:
    """`num_players == 2` ma tupla `players` incoerente: rifiutato prima di indicizzare."""
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(PlayerState("P0", (Card(Suit.CUPS, Rank.ACE),), (), 0),),
        deck=(),
        trump_card=TRUMP_CARD,
        table_cards=(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )
    with pytest.raises(ValueError, match="Attesi 2 player"):
        solve_endgame(state)
