#!/usr/bin/env python3
"""
Fase 0 diagnostica per un eventuale guard sui carichi guidati.

Questo script NON implementa il guard. Misura se il fenomeno target esiste abbastanza
spesso da giustificare un esperimento eval-only:

- la policy e' leader;
- guida un carico non-briscola (Asso/Tre, oppure anche Re con `--load-points-min 4`);
- aveva una liscia non-briscola alternativa;
- il seme guidato e' "thin" (poche carte residue lecite di quel seme);
- la presa viene persa o tagliata.

La stima usa solo `PlayerObservation`: mano, seen/out-of-play e trick_history pubblica.
Serve come filtro economico prima di scrivere un lead-load guard vero.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.training.reward_shaping import card_conservation_cost
from briscola_ai.domain.card_id import card_to_id, id_to_card
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Suit
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.rules import who_wins_trick
from briscola_ai.domain.state import new_game_state


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def _is_trump(card: Card, trump_suit: Suit | None) -> bool:
    return trump_suit is not None and card.suit == trump_suit


def _is_load(card: Card, *, trump_suit: Suit, load_points_min: int) -> bool:
    return card.suit != trump_suit and int(card.rank.points) >= int(load_points_min)


def _smooth_non_trump_indices(hand: tuple[Card, ...], *, trump_suit: Suit) -> list[int]:
    return [i for i, card in enumerate(hand) if card.suit != trump_suit and int(card.rank.points) == 0]


def _replacement_index_base(hand: tuple[Card, ...], *, trump_suit: Suit) -> int | None:
    candidates = _smooth_non_trump_indices(hand, trump_suit=trump_suit)
    if not candidates:
        return None
    return min(candidates, key=lambda i: card_conservation_cost(hand[i]))


def _unknown_live_cards(observation: PlayerObservation) -> list[Card]:
    """Carte non viste pubblicamente e non in mano a noi: mano avversaria o mazzo, senza leak."""
    my_ids = {card_to_id(card) for card in observation.hand}
    return [
        id_to_card(card_id)
        for card_id, seen in enumerate(observation.seen_cards_onehot)
        if not seen and card_id not in my_ids
    ]


def _unknown_same_suit_count(observation: PlayerObservation, *, suit: Suit, trump_suit: Suit) -> int:
    """Conta le carte ignote vive del seme guidato, escludendo la briscola per definizione."""
    if suit == trump_suit:
        return 0
    return sum(1 for card in _unknown_live_cards(observation) if card.suit == suit)


def _opponent_has_skipped_suit(observation: PlayerObservation, *, suit: Suit, trump_suit: Suit) -> bool:
    """
    True se l'avversario ha gia' mostrato di non rispondere al seme richiesto.

    Nota di dominio: in Briscola NON c'è obbligo di rispondere al seme. Questo segnale
    quindi non prova un vuoto reale; è solo un indizio comportamentale pubblico. Per questo
    il guard eval-only usa `not_master` come trigger predefinito, non questo flag.
    """
    opponent = 1 - int(observation.player_index)
    for trick in observation.trick_history:
        if len(trick.cards) != 2:
            continue
        lead_card, lead_player = trick.cards[0]
        response_card, response_player = trick.cards[1]
        if lead_player == opponent:
            continue
        if response_player != opponent:
            continue
        if lead_card.suit != suit or lead_card.suit == trump_suit:
            continue
        if response_card.suit != suit:
            return True
    return False


def _is_master_against_unknown_live(observation: PlayerObservation, *, card: Card, trump_suit: Suit) -> bool:
    """
    True se `card`, guidata da leader, non può essere battuta da nessuna carta ignota viva.

    È un controllo pessimista ma anti-cheat: se una carta ignota potrebbe essere in mano
    all'avversario o nel mazzo e batterci, il carico non è considerato "sicuro".
    """
    opponent = 1 - int(observation.player_index)
    for other in _unknown_live_cards(observation):
        trick_cards = ((card, observation.player_index), (other, opponent))
        if who_wins_trick(trick_cards, trump_suit) != observation.player_index:
            return False
    return True


def _point_diff(observation: PlayerObservation) -> int:
    player = int(observation.player_index)
    opponent = 1 - player
    return int(observation.players_points[player]) - int(observation.players_points[opponent])


def _bucket(value: int, *, thresholds: tuple[int, int]) -> str:
    lo, hi = thresholds
    if value <= lo:
        return f"<= {lo}"
    if value <= hi:
        return f"{lo + 1}..{hi}"
    return f"> {hi}"


def _empty_counts() -> Counter[str]:
    return Counter(
        {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "points_for": 0,
            "points_against": 0,
            "point_diff": 0,
            "lead_decisions": 0,
            "lead_load": 0,
            "lead_load_with_smooth_alt": 0,
            "lead_load_thin": 0,
            "lead_load_thin_or_suit_skip": 0,
            "lead_load_master": 0,
            "lead_load_not_master": 0,
            "lead_load_lost": 0,
            "lead_load_cut": 0,
            "lead_load_thin_lost": 0,
            "lead_load_thin_cut": 0,
            "replacement_points_saved": 0,
        }
    )


def _pct(num: int, den: int) -> float:
    return round(100.0 * float(num) / float(den), 2) if den else 0.0


def _summarize(counter: Counter[str]) -> dict[str, float | int]:
    lead_load = int(counter["lead_load"])
    lead_load_alt = int(counter["lead_load_with_smooth_alt"])
    lead_load_thin = int(counter["lead_load_thin"])
    return {
        "lead_decisions": int(counter["lead_decisions"]),
        "lead_load": lead_load,
        "lead_load_pct": _pct(lead_load, int(counter["lead_decisions"])),
        "lead_load_with_smooth_alt": lead_load_alt,
        "lead_load_with_smooth_alt_pct_of_load": _pct(lead_load_alt, lead_load),
        "lead_load_thin": lead_load_thin,
        "lead_load_thin_pct_of_load": _pct(lead_load_thin, lead_load),
        "lead_load_thin_or_suit_skip": int(counter["lead_load_thin_or_suit_skip"]),
        "lead_load_master": int(counter["lead_load_master"]),
        "lead_load_master_pct_of_load": _pct(int(counter["lead_load_master"]), lead_load),
        "lead_load_not_master": int(counter["lead_load_not_master"]),
        "lead_load_not_master_pct_of_load": _pct(int(counter["lead_load_not_master"]), lead_load),
        "lead_load_lost_pct": _pct(int(counter["lead_load_lost"]), lead_load),
        "lead_load_cut_pct": _pct(int(counter["lead_load_cut"]), lead_load),
        "lead_load_thin_lost_pct": _pct(int(counter["lead_load_thin_lost"]), lead_load_thin),
        "lead_load_thin_cut_pct": _pct(int(counter["lead_load_thin_cut"]), lead_load_thin),
        "replacement_points_saved": int(counter["replacement_points_saved"]),
    }


def _summarize_match(counter: Counter[str]) -> dict[str, float | int]:
    games = int(counter["games"])
    return {
        "games": games,
        "wins": int(counter["wins"]),
        "losses": int(counter["losses"]),
        "draws": int(counter["draws"]),
        "avg_points_for": round(float(counter["points_for"]) / games, 3) if games else 0.0,
        "avg_points_against": round(float(counter["points_against"]) / games, 3) if games else 0.0,
        "avg_point_diff": round(float(counter["point_diff"]) / games, 3) if games else 0.0,
        "score_rate": round((float(counter["wins"]) + 0.5 * float(counter["draws"])) / games, 4) if games else 0.0,
    }


@dataclass(frozen=True, slots=True)
class LeadLoadGuardConfig:
    """
    Parametri del guard eval-only sui carichi guidati.

    Il default è volutamente conservativo:
    - si applica solo late/mid-late (`max_deck_size=8`);
    - non interviene se siamo sotto di più di 10 punti;
    - richiede una liscia non-briscola alternativa;
    - richiede che il carico non sia master secondo informazione pubblica.
    """

    load_points_min: int = 10
    thin_unknown_same_suit_max: int = 1
    max_deck_size: int | None = 8
    min_point_diff: int | None = -10
    trigger: str = "not_master"


def apply_lead_load_guard_index(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    config: LeadLoadGuardConfig,
) -> tuple[int, str]:
    """
    Post-processing eval-only per lead di carico non-briscola.

    Ritorna `(card_index, reason)`, dove `reason == "adjusted"` indica una sostituzione.
    La funzione usa solo `PlayerObservation`; non legge mai mano avversaria o ordine del mazzo.
    """
    if observation.num_players != 2:
        return chosen_card_index, "not_2p"
    if observation.game_over or observation.trump_card is None:
        return chosen_card_index, "inactive"
    if observation.current_turn != observation.player_index or observation.table_cards:
        return chosen_card_index, "not_lead"
    if chosen_card_index < 0 or chosen_card_index >= len(observation.hand):
        return chosen_card_index, "invalid_choice"

    trump_suit = observation.trump_card.suit
    chosen = observation.hand[chosen_card_index]
    if not _is_load(chosen, trump_suit=trump_suit, load_points_min=int(config.load_points_min)):
        return chosen_card_index, "not_load"

    if config.max_deck_size is not None and int(observation.deck_size) > int(config.max_deck_size):
        return chosen_card_index, "deck_too_large"
    if config.min_point_diff is not None and _point_diff(observation) < int(config.min_point_diff):
        return chosen_card_index, "too_far_behind"

    replacement_idx = _replacement_index_base(observation.hand, trump_suit=trump_suit)
    if replacement_idx is None:
        return chosen_card_index, "no_smooth_alt"

    unknown_same_suit = _unknown_same_suit_count(observation, suit=chosen.suit, trump_suit=trump_suit)
    thin = unknown_same_suit <= int(config.thin_unknown_same_suit_max)
    master = _is_master_against_unknown_live(observation, card=chosen, trump_suit=trump_suit)
    trigger = str(config.trigger)
    if trigger == "not_master":
        should_adjust = not master
    elif trigger == "thin":
        should_adjust = thin
    elif trigger == "thin_or_not_master":
        should_adjust = thin or not master
    elif trigger == "thin_and_not_master":
        should_adjust = thin and not master
    else:
        raise ValueError(f"trigger guard non supportato: {trigger!r}")

    if not should_adjust:
        return chosen_card_index, "safe_by_trigger"
    return int(replacement_idx), "adjusted"


class LeadLoadGuardAgent:
    """Wrapper eval-only: applica il lead-load guard a una policy già esistente."""

    def __init__(self, inner: Agent, *, config: LeadLoadGuardConfig) -> None:
        self.inner = inner
        self.config = config
        self.metrics: Counter[str] = Counter()

    @property
    def name(self) -> str:
        return f"{self.inner.name},lead_load_guard={self.config.trigger}"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        chosen = int(self.inner.choose_card_index(observation, rng=rng))
        guarded, reason = apply_lead_load_guard_index(observation, chosen_card_index=chosen, config=self.config)
        self.metrics[f"guard_reason_{reason}"] += 1
        if reason == "adjusted":
            self.metrics["guard_adjustments"] += 1
            self.metrics["guard_points_replaced"] += int(observation.hand[chosen].rank.points)
            self.metrics["guard_points_replacement"] += int(observation.hand[guarded].rank.points)
        return int(guarded)


def profile(
    model_agent: Agent,
    opponent: Agent,
    *,
    num_games: int,
    seed_base: int,
    load_points_min: int,
    thin_unknown_same_suit_max: int,
) -> dict[str, Any]:
    """Gioca partite seat-alternate e misura i lead di carico target per il guard."""
    c = _empty_counts()
    by_deck: dict[str, Counter[str]] = {}
    by_margin: dict[str, Counter[str]] = {}

    for game_seed in range(seed_base, seed_base + num_games):
        model_idx = game_seed % 2
        state = new_game_state(2, ["A", "B"], seed=game_seed)
        rngs = {
            model_idx: random.Random(10_000 + game_seed),
            1 - model_idx: random.Random(20_000 + game_seed),
        }
        pending: dict[str, Any] | None = None

        while not state.game_over:
            turn = state.current_turn
            obs = make_player_observation(state, turn)
            agent = model_agent if turn == model_idx else opponent
            card_index = agent.choose_card_index(obs, rng=rngs[turn])
            played = obs.hand[card_index]
            trump_suit = obs.trump_card.suit if obs.trump_card else None
            is_lead = len(obs.table_cards) == 0

            if turn == model_idx and is_lead and trump_suit is not None and obs.num_players == 2:
                c["lead_decisions"] += 1
                deck_bucket = _bucket(int(obs.deck_size), thresholds=(6, 16))
                margin_bucket = _bucket(_point_diff(obs), thresholds=(-10, 10))
                by_deck.setdefault(deck_bucket, _empty_counts())["lead_decisions"] += 1
                by_margin.setdefault(margin_bucket, _empty_counts())["lead_decisions"] += 1
                if _is_load(played, trump_suit=trump_suit, load_points_min=load_points_min):
                    c["lead_load"] += 1
                    replacement_idx = _replacement_index_base(obs.hand, trump_suit=trump_suit)
                    if replacement_idx is not None:
                        c["lead_load_with_smooth_alt"] += 1
                        replacement = obs.hand[replacement_idx]
                        c["replacement_points_saved"] += int(played.rank.points) - int(replacement.rank.points)

                    unknown_same_suit = _unknown_same_suit_count(obs, suit=played.suit, trump_suit=trump_suit)
                    suit_skip = _opponent_has_skipped_suit(obs, suit=played.suit, trump_suit=trump_suit)
                    thin = unknown_same_suit <= thin_unknown_same_suit_max
                    if thin:
                        c["lead_load_thin"] += 1
                    if thin or suit_skip:
                        c["lead_load_thin_or_suit_skip"] += 1
                    master = _is_master_against_unknown_live(obs, card=played, trump_suit=trump_suit)
                    if master:
                        c["lead_load_master"] += 1
                    else:
                        c["lead_load_not_master"] += 1

                    by_deck.setdefault(deck_bucket, _empty_counts())["lead_load"] += 1
                    by_margin.setdefault(margin_bucket, _empty_counts())["lead_load"] += 1
                    if replacement_idx is not None:
                        by_deck[deck_bucket]["lead_load_with_smooth_alt"] += 1
                        by_margin[margin_bucket]["lead_load_with_smooth_alt"] += 1
                        points_saved = int(played.rank.points) - int(obs.hand[replacement_idx].rank.points)
                        by_deck[deck_bucket]["replacement_points_saved"] += points_saved
                        by_margin[margin_bucket]["replacement_points_saved"] += points_saved
                    if thin:
                        by_deck[deck_bucket]["lead_load_thin"] += 1
                        by_margin[margin_bucket]["lead_load_thin"] += 1
                    if thin or suit_skip:
                        by_deck[deck_bucket]["lead_load_thin_or_suit_skip"] += 1
                        by_margin[margin_bucket]["lead_load_thin_or_suit_skip"] += 1
                    if master:
                        by_deck[deck_bucket]["lead_load_master"] += 1
                        by_margin[margin_bucket]["lead_load_master"] += 1
                    else:
                        by_deck[deck_bucket]["lead_load_not_master"] += 1
                        by_margin[margin_bucket]["lead_load_not_master"] += 1

                    pending = {
                        "thin": thin,
                        "played_suit": played.suit,
                        "deck_bucket": deck_bucket,
                        "margin_bucket": margin_bucket,
                    }

            state, result = step(state, PlayCardAction(player_index=turn, card_index=card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio: {result.error}")

            if result.trick_completed and pending is not None:
                lost = result.trick_winner != model_idx
                cut = False
                if lost and trump_suit is not None:
                    winner_card = next(card for card, player in result.trick_cards if player == result.trick_winner)
                    cut = winner_card.suit == trump_suit and pending["played_suit"] != trump_suit
                if lost:
                    c["lead_load_lost"] += 1
                if cut:
                    c["lead_load_cut"] += 1
                if pending["thin"]:
                    if lost:
                        c["lead_load_thin_lost"] += 1
                    if cut:
                        c["lead_load_thin_cut"] += 1
                deck_counter = by_deck[pending["deck_bucket"]]
                margin_counter = by_margin[pending["margin_bucket"]]
                if lost:
                    deck_counter["lead_load_lost"] += 1
                    margin_counter["lead_load_lost"] += 1
                if cut:
                    deck_counter["lead_load_cut"] += 1
                    margin_counter["lead_load_cut"] += 1
                if pending["thin"]:
                    if lost:
                        deck_counter["lead_load_thin_lost"] += 1
                        margin_counter["lead_load_thin_lost"] += 1
                    if cut:
                        deck_counter["lead_load_thin_cut"] += 1
                        margin_counter["lead_load_thin_cut"] += 1
                pending = None

        points_for = int(state.players[model_idx].points)
        points_against = int(state.players[1 - model_idx].points)
        c["games"] += 1
        c["points_for"] += points_for
        c["points_against"] += points_against
        c["point_diff"] += points_for - points_against
        if points_for > points_against:
            c["wins"] += 1
        elif points_for < points_against:
            c["losses"] += 1
        else:
            c["draws"] += 1

    return {
        "match": _summarize_match(c),
        "summary": _summarize(c),
        "by_deck_size": {key: _summarize(value) for key, value in sorted(by_deck.items())},
        "by_point_diff": {key: _summarize(value) for key, value in sorted(by_margin.items())},
    }


def _build_opponent(name: str, *, model_path: Path) -> Agent:
    return build_agent("bc_model", model_path=model_path) if name == "mirror" else build_agent(name)


def guard_ablation(
    *,
    model_path: Path,
    opponent_name: str,
    num_games: int,
    seed_base: int,
    load_points_min: int,
    thin_unknown_same_suit_max: int,
    guard_config: LeadLoadGuardConfig,
) -> dict[str, Any]:
    """Confronta baseline e lead-load guard sugli stessi seed iniziali."""
    baseline_agent = build_agent("bc_model", model_path=model_path)
    guarded_inner = build_agent("bc_model", model_path=model_path)
    guarded_agent = LeadLoadGuardAgent(guarded_inner, config=guard_config)

    baseline = profile(
        baseline_agent,
        _build_opponent(opponent_name, model_path=model_path),
        num_games=num_games,
        seed_base=seed_base,
        load_points_min=load_points_min,
        thin_unknown_same_suit_max=thin_unknown_same_suit_max,
    )
    guarded = profile(
        guarded_agent,
        _build_opponent(opponent_name, model_path=model_path),
        num_games=num_games,
        seed_base=seed_base,
        load_points_min=load_points_min,
        thin_unknown_same_suit_max=thin_unknown_same_suit_max,
    )
    baseline_match = baseline["match"]
    guarded_match = guarded["match"]
    return {
        "baseline": baseline,
        "guarded": guarded,
        "delta": {
            "avg_point_diff": round(
                float(guarded_match["avg_point_diff"]) - float(baseline_match["avg_point_diff"]), 3
            ),
            "score_rate": round(float(guarded_match["score_rate"]) - float(baseline_match["score_rate"]), 4),
            "lead_load_pct": round(
                float(guarded["summary"]["lead_load_pct"]) - float(baseline["summary"]["lead_load_pct"]), 3
            ),
            "lead_load_cut_pct": round(
                float(guarded["summary"]["lead_load_cut_pct"]) - float(baseline["summary"]["lead_load_cut_pct"]), 3
            ),
        },
        "guard_metrics": dict(sorted(guarded_agent.metrics.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fase 0 diagnostica per lead-load guard eval-only.")
    parser.add_argument(
        "--mode",
        choices=["phase0", "guard", "both"],
        default="phase0",
        help="`phase0` misura il fenomeno; `guard` confronta baseline vs wrapper eval-only; `both` fa entrambi.",
    )
    parser.add_argument("--model", default="data/models/best_a2c_v11.npz", help="Path `.npz` del modello.")
    parser.add_argument(
        "--opponents",
        default="heuristic_trump_saver,mirror,heuristic_v1",
        help="Avversari separati da virgola; `mirror` = lo stesso modello.",
    )
    parser.add_argument("--num-games", type=int, default=2000, help="Partite per avversario.")
    parser.add_argument("--seed", type=int, default=20260709, help="Seed base per le partite.")
    parser.add_argument(
        "--load-points-min",
        type=int,
        default=10,
        help="Soglia carico non-briscola: 10=Asso/Tre, 4=include Re.",
    )
    parser.add_argument(
        "--thin-unknown-same-suit-max",
        type=int,
        default=1,
        help="Seme thin se restano al massimo N carte ignote vive dello stesso seme.",
    )
    parser.add_argument(
        "--guard-trigger",
        choices=["not_master", "thin", "thin_or_not_master", "thin_and_not_master"],
        default="not_master",
        help="Trigger del guard eval-only. Default: carico non master secondo informazione pubblica.",
    )
    parser.add_argument(
        "--guard-max-deck-size",
        type=int,
        default=8,
        help="Applica il guard solo con deck_size <= N; usa -1 per disabilitare il filtro.",
    )
    parser.add_argument(
        "--guard-min-point-diff",
        type=int,
        default=-10,
        help="Applica il guard solo se point_diff >= N; usa -999 per disabilitare il filtro.",
    )
    parser.add_argument("--out-json", default="", help="Path JSON opzionale.")
    args = parser.parse_args()

    model_path = Path(args.model)
    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    guard_config = LeadLoadGuardConfig(
        load_points_min=int(args.load_points_min),
        thin_unknown_same_suit_max=int(args.thin_unknown_same_suit_max),
        max_deck_size=None if int(args.guard_max_deck_size) < 0 else int(args.guard_max_deck_size),
        min_point_diff=None if int(args.guard_min_point_diff) <= -999 else int(args.guard_min_point_diff),
        trigger=str(args.guard_trigger),
    )
    results: dict[str, Any] = {}
    for index, name in enumerate(opponents):
        seed_base = int(args.seed) + index * 100_000
        if args.mode == "phase0":
            model_agent = build_agent("bc_model", model_path=model_path)
            results[name] = profile(
                model_agent,
                _build_opponent(name, model_path=model_path),
                num_games=int(args.num_games),
                seed_base=seed_base,
                load_points_min=int(args.load_points_min),
                thin_unknown_same_suit_max=int(args.thin_unknown_same_suit_max),
            )
            print(f"=== phase0 vs {name} ({args.num_games} partite) ===")
            print(f"  match: {results[name]['match']}")
            for key, value in results[name]["summary"].items():
                print(f"  {key}: {value}")
        elif args.mode == "guard":
            results[name] = guard_ablation(
                model_path=model_path,
                opponent_name=name,
                num_games=int(args.num_games),
                seed_base=seed_base,
                load_points_min=int(args.load_points_min),
                thin_unknown_same_suit_max=int(args.thin_unknown_same_suit_max),
                guard_config=guard_config,
            )
            print(f"=== guard vs {name} ({args.num_games} partite) ===")
            print(f"  baseline_match: {results[name]['baseline']['match']}")
            print(f"  guarded_match: {results[name]['guarded']['match']}")
            print(f"  delta: {results[name]['delta']}")
            print(f"  guard_metrics: {results[name]['guard_metrics']}")
        else:
            model_agent = build_agent("bc_model", model_path=model_path)
            phase0 = profile(
                model_agent,
                _build_opponent(name, model_path=model_path),
                num_games=int(args.num_games),
                seed_base=seed_base,
                load_points_min=int(args.load_points_min),
                thin_unknown_same_suit_max=int(args.thin_unknown_same_suit_max),
            )
            ablation = guard_ablation(
                model_path=model_path,
                opponent_name=name,
                num_games=int(args.num_games),
                seed_base=seed_base,
                load_points_min=int(args.load_points_min),
                thin_unknown_same_suit_max=int(args.thin_unknown_same_suit_max),
                guard_config=guard_config,
            )
            results[name] = {"phase0": phase0, "guard": ablation}
            print(f"=== both vs {name} ({args.num_games} partite) ===")
            print(f"  phase0_match: {phase0['match']}")
            print(f"  guard_delta: {ablation['delta']}")
            print(f"  guard_metrics: {ablation['guard_metrics']}")

    payload = {
        "meta": {
            "git_commit": _git_commit(),
            "mode": str(args.mode),
            "model": str(model_path),
            "opponents": opponents,
            "num_games_per_opponent": int(args.num_games),
            "seed": int(args.seed),
            "load_points_min": int(args.load_points_min),
            "thin_unknown_same_suit_max": int(args.thin_unknown_same_suit_max),
            "guard": {
                "trigger": str(guard_config.trigger),
                "max_deck_size": guard_config.max_deck_size,
                "min_point_diff": guard_config.min_point_diff,
            },
        },
        "profiles": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out_json.strip():
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON salvato in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
