#!/usr/bin/env python3
"""
Simulatore CLI: esegue N partite senza UI (self-play casuale).

Uso:
  python scripts/simulate_games.py --num-games 100 --seed 42
  python scripts/simulate_games.py --num-games 50 --seed 1 --num-players 4
"""

from __future__ import annotations

import argparse
import random

from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.state import GameState, new_game_state


def _build_result_payload(state: GameState) -> dict:
    """
    Converte lo stato finale in un dizionario "human-friendly" (CLI).

    Nota:
    - questo formato è pensato per stampa/diagnostica, non è un contratto API.
    - la UI/HTTP usano DTO specifici nel backend.
    """
    if not state.game_over:
        return {"game_in_progress": True}

    if state.is_team_game and state.teams is not None:
        team_0_points = sum(state.players[i].points for i in state.teams[0])
        team_1_points = sum(state.players[i].points for i in state.teams[1])

        if team_0_points > team_1_points:
            winning_team = 0
        elif team_1_points > team_0_points:
            winning_team = 1
        else:
            winning_team = None

        if winning_team is None:
            winner = "Pareggio"
        else:
            p0 = state.players[state.teams[winning_team][0]].name
            p1 = state.players[state.teams[winning_team][1]].name
            winner = f"Squadra {winning_team} ({p0} e {p1})"

        return {
            "game_over": True,
            "is_team_game": True,
            "winner": winner,
            "winning_team": winning_team,
            "team_points": {"Team 0": team_0_points, "Team 1": team_1_points},
            "individual_points": {p.name: p.points for p in state.players},
            "point_difference": abs(team_0_points - team_1_points),
        }

    p0 = state.players[0].points
    p1 = state.players[1].points
    if p0 > p1:
        winner_index = 0
    elif p1 > p0:
        winner_index = 1
    else:
        winner_index = None

    return {
        "game_over": True,
        "is_team_game": False,
        "winner": state.players[winner_index].name if winner_index is not None else "Pareggio",
        "winner_index": winner_index,
        "points": {p.name: p.points for p in state.players},
        "point_difference": abs(p0 - p1),
    }


def simulate_one_game(num_players: int, *, rng: random.Random) -> dict:
    """
    Simula una singola partita (self-play casuale).

    Argomenti:
        num_players: 2 o 4
        rng: RNG per riproducibilità (sceglie seed iniziale e azioni)

    Ritorna:
        Un dizionario con:
        - `status`: "ok" oppure "error"
        - `result`: (solo se ok) payload riassuntivo calcolato dal dominio
        - `error`: (solo se error) descrizione testuale del problema
    """
    seed = rng.randrange(0, 2**32)
    state = new_game_state(num_players=num_players, seed=seed)

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        hand_size = len(state.players[current].hand)
        if hand_size <= 0:
            return {
                "status": "error",
                "error": "Nessuna carta in mano per il turno corrente",
                "num_players": num_players,
            }
        card_index = rng.randrange(hand_size)
        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        if result.error is not None:
            return {"status": "error", "error": result.error, "num_players": num_players}

    if safety <= 0:
        return {"status": "error", "error": "Loop di sicurezza", "num_players": num_players}

    return {"status": "ok", "result": _build_result_payload(state), "num_players": num_players, "seed": seed}


def main() -> int:
    """Entry point CLI: simula N partite e stampa un riepilogo dei vincitori."""
    parser = argparse.ArgumentParser(description="Simula partite di Briscola senza UI (self-play casuale)")
    parser.add_argument("--num-games", type=int, default=10, help="Numero di partite da simulare")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (per riproducibilità)")
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 4], help="Numero di giocatori (2 o 4)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    ok = 0
    errors = 0
    winners = {}

    for _ in range(args.num_games):
        out = simulate_one_game(args.num_players, rng=rng)
        if out["status"] != "ok":
            errors += 1
            continue
        ok += 1
        winner = out["result"].get("winner", "Sconosciuto")
        winners[winner] = winners.get(winner, 0) + 1

    print(f"Simulazioni: {ok} OK, {errors} errori (seed={args.seed}, players={args.num_players})")
    for winner, count in sorted(winners.items(), key=lambda x: x[1], reverse=True):
        print(f"- {winner}: {count}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
