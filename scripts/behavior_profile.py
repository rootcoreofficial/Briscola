#!/usr/bin/env python3
"""
Pagella della nonna: profilo comportamentale di un modello su partite strumentate.

Misura, regola per regola, quanto un agente rispetta la saggezza briscolistica
tradizionale ("le regole della nonna"), giocando partite complete di dominio contro
un avversario a scelta e osservando OGNI decisione. Nato dall'audit di campo del
2026-07-07: i contatori formalizzano i vizi/virtù emersi confrontando l'IA coi
giocatori umani vincenti (docs/plans/audit-campo-2026-07-07.md).

Regole misurate (tutte in fase pescate, salvo dove indicato):
- apertura_liscia:      % aperture con carta a 0 punti non di briscola ("apri liscio");
- carichi_guidati:      % aperture con carico (asso/tre) non di briscola, e % persi;
- briscola_su_povero:   % risposte in cui taglia un piatto povero (lead <=2 punti);
- punti_regalati:       punti medi ceduti per risposta perdente, e % di risposte
                        perdenti in cui cede un carico avendo uno scarto povero;
- cavare_briscole:      con mano lunga di briscole (>=4) guida briscole basse piu'
                        spesso che con mano corta (<=2)? ("briscola chiama briscola");
- sbianchirsi:          negli scarti a 0 punti, % dal proprio seme non-briscola piu'
                        corto (chi cerca il vuoto per tagliare scarta dal seme corto);
- asso_di_briscola:     presa media in cui gioca l'asso di briscola e % giocato
                        durante le pescate ("l'asso di briscola si tiene per la fine").

Esempio:
  python scripts/behavior_profile.py --model data/models/best_a2c_v14.npz \\
      --opponents heuristic_trump_saver,mirror,heuristic_v1 --num-games 2000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state


def _is_trump(card: Card, trump_suit: Suit | None) -> bool:
    return trump_suit is not None and card.suit == trump_suit


def profile(model_agent: Agent, opponent: Agent, *, num_games: int, seed_base: int = 0) -> dict[str, float]:
    """Gioca `num_games` partite (seat alternato) e ritorna le metriche della pagella."""
    c: Counter[str] = Counter()
    ace_played_tricks: list[int] = []

    for game_seed in range(seed_base, seed_base + num_games):
        model_idx = game_seed % 2  # seat-fair grezzo: il modello alterna primo/secondo
        state = new_game_state(2, ["A", "B"], seed=game_seed)
        rngs = {
            model_idx: random.Random(10_000 + game_seed),
            1 - model_idx: random.Random(20_000 + game_seed),
        }
        trick_index = 0
        carico_pending = False

        while not state.game_over:
            turn = state.current_turn
            obs = make_player_observation(state, turn)
            agent = model_agent if turn == model_idx else opponent
            card_index = agent.choose_card_index(obs, rng=rngs[turn])
            hand = state.players[turn].hand
            played = hand[card_index]
            trump_suit = state.trump_card.suit if state.trump_card else None
            in_draw = len(state.deck) > 0
            is_lead = len(state.table_cards) == 0

            if turn == model_idx:
                trump_ace = trump_suit is not None and _is_trump(played, trump_suit) and played.rank.points == 11
                if trump_ace:
                    ace_played_tricks.append(trick_index + 1)
                    if in_draw:
                        c["ace_in_draw"] += 1

            if turn == model_idx and in_draw:
                if is_lead:
                    c["leads"] += 1
                    if played.rank.points == 0 and not _is_trump(played, trump_suit):
                        c["lead_liscio"] += 1
                    if played.rank.points >= 10 and not _is_trump(played, trump_suit):
                        c["carichi_led"] += 1
                        carico_pending = True
                    # "Cavare le briscole": guida di briscola bassa, condizionata alla
                    # lunghezza di briscola in mano (il consiglio vale con mano lunga).
                    my_trumps = sum(1 for card in hand if _is_trump(card, trump_suit))
                    low_trump_lead = _is_trump(played, trump_suit) and played.rank.points < 10
                    if my_trumps >= 4:
                        c["leads_trump_long"] += 1
                        if low_trump_lead:
                            c["cavate_long"] += 1
                    elif my_trumps <= 2:
                        c["leads_trump_short"] += 1
                        if low_trump_lead:
                            c["cavate_short"] += 1
                else:
                    c["responses"] += 1
                    lead_card, lead_player = state.table_cards[0]
                    if (
                        _is_trump(played, trump_suit)
                        and not _is_trump(lead_card, trump_suit)
                        and lead_card.rank.points <= 2
                    ):
                        c["trump_on_poor"] += 1
                    # Scarto a 0 punti non di briscola: da quale seme scarta?
                    if played.rank.points == 0 and not _is_trump(played, trump_suit):
                        non_trump_suits = {card.suit for card in hand if not _is_trump(card, trump_suit)}
                        if len(non_trump_suits) > 1:
                            lengths = {suit: sum(1 for card in hand if card.suit == suit) for suit in non_trump_suits}
                            shortest = min(lengths.values())
                            c["discards_with_choice"] += 1
                            if lengths[played.suit] == shortest:
                                c["discard_from_shortest"] += 1

            state, result = step(state, PlayCardAction(player_index=turn, card_index=card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio: {result.error}")

            if result.trick_completed:
                trick_index += 1
                if carico_pending:
                    if result.trick_winner != model_idx:
                        c["carichi_led_lost"] += 1
                    carico_pending = False
                # "Punti regalati": il modello ha appena chiuso la presa da risponditore
                # (turn e' chi ha giocato la seconda carta) e l'ha persa?
                # `trick_cards` e' in ordine di gioco: la carta del risponditore e' la seconda.
                if len(result.trick_cards) == 2 and result.trick_winner != model_idx and turn == model_idx:
                    model_card = next(card for card, player in result.trick_cards if player == model_idx)
                    c["losing_responses"] += 1
                    c["points_given"] += int(model_card.rank.points)

    def pct(num: str, den: str) -> float:
        return 100.0 * c[num] / c[den] if c[den] else 0.0

    return {
        "apertura_liscia_pct": round(pct("lead_liscio", "leads"), 2),
        "carichi_guidati_pct": round(pct("carichi_led", "leads"), 2),
        "carichi_guidati_persi_pct": round(pct("carichi_led_lost", "carichi_led"), 2),
        "briscola_su_povero_pct": round(pct("trump_on_poor", "responses"), 2),
        "punti_regalati_per_persa": round(c["points_given"] / c["losing_responses"], 2)
        if c["losing_responses"]
        else 0.0,
        "cavate_con_mano_lunga_pct": round(pct("cavate_long", "leads_trump_long"), 2),
        "cavate_con_mano_corta_pct": round(pct("cavate_short", "leads_trump_short"), 2),
        "scarto_dal_seme_corto_pct": round(pct("discard_from_shortest", "discards_with_choice"), 2),
        "asso_briscola_presa_media": round(sum(ace_played_tricks) / len(ace_played_tricks), 2)
        if ace_played_tricks
        else 0.0,
        "asso_briscola_in_pescate_pct": round(100.0 * c["ace_in_draw"] / len(ace_played_tricks), 2)
        if ace_played_tricks
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pagella della nonna: profilo comportamentale di un modello")
    parser.add_argument("--model", required=True, help="Path `.npz` del modello da profilare.")
    parser.add_argument(
        "--opponents",
        default="heuristic_trump_saver,mirror,heuristic_v1",
        help="Avversari separati da virgola; `mirror` = lo stesso modello.",
    )
    parser.add_argument("--num-games", type=int, default=2000, help="Partite per avversario.")
    parser.add_argument("--out-json", default="", help="Path JSON opzionale per salvare il profilo.")
    args = parser.parse_args()

    model_path = Path(args.model)
    model_agent = build_agent("bc_model", model_path=model_path)
    results: dict[str, dict[str, float]] = {}
    for name in [item.strip() for item in args.opponents.split(",") if item.strip()]:
        opponent = build_agent("bc_model", model_path=model_path) if name == "mirror" else build_agent(name)
        results[name] = profile(model_agent, opponent, num_games=int(args.num_games))
        print(f"=== vs {name} ({args.num_games} partite) ===")
        for metric, value in results[name].items():
            print(f"  {metric}: {value}")

    if args.out_json.strip():
        payload = {"model": str(model_path), "num_games": int(args.num_games), "profiles": results}
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON salvato in: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
