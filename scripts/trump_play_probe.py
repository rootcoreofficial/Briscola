#!/usr/bin/env python3
"""
Sonda diagnostica su due comportamenti di briscola del modello:

1. **Timing dell'asso di briscola**: quando il modello incassa l'asso di briscola e
   quanti punti rastrella la presa che lo contiene. L'asso è la carta più forte: vince
   *sempre* la presa in cui entra, quindi "timing" non è mai una mossa perdente, è solo
   *quando lo cavo e quanto vale la presa*. Un asso "sprecato" cattura ~solo i suoi 11
   punti (l'avversario scarta liscio); un asso "valorizzato" cattura anche il carico
   avversario.
2. **Cavata delle briscole con mano lunga**: quando il modello tiene molte briscole e
   guida, apre in briscola per prosciugare l'avversario? È una scelta buona o no?

Lo script segue lo stesso schema che ha chiuso la pista dei carichi guidati
(`lead_load_guard_probe.py`): la statistica descrittiva più l'ablation controfattuale
seat-fair sono lo strumento decisionale; il solver esatto è solo un cross-check *endgame*
(funziona solo a `deck_size == 0`), non il giudice della domanda mid-game.

Tre tier:

- **Tier A (descrittivo, tutte le fasi)**: distribuzioni e tassi, per fase di partita.
- **Tier B (ablation controfattuale seat-fair)**: agenti-trattamento eval-only che
  cambiano il comportamento (trattieni l'asso guidato presto; cava di più / di meno con
  mano lunga di briscola). Il trattamento **è** il guard candidato: se non migliora
  seat-fair, niente guard né shaping.
- **Tier C (cross-check endgame esatto)**: dove `deck_size == 0`, confronta la mossa del
  modello con l'ottimo del solver e misura la regret in punti. Valida solo l'esecuzione
  tardiva, non la decisione mid-game di *tenere* l'asso.

Tutte le stime usano solo `PlayerObservation` (mano, `seen`/`out-of-play`, `trick_history`
pubblica). Il Tier C usa lo `state` reale del simulatore: è una misura diagnostica, non un
agente, quindi non viola l'anti-cheat (nessun agente riceve lo stato completo).
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
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.ai.training.reward_shaping import card_conservation_cost
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import new_game_state

# Soglie fisse pre-registrate per la lettura per fase di partita (non derivate dai dati).
_DECK_THRESHOLDS = (6, 16)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def _is_trump(card: Card, trump_suit: Suit) -> bool:
    return card.suit == trump_suit


def _is_trump_ace(card: Card, trump_suit: Suit) -> bool:
    """L'asso di briscola: seme di briscola e rango asso. È la carta più forte del mazzo."""
    return card.suit == trump_suit and card.rank == Rank.ACE


def _trump_hand_size(hand: tuple[Card, ...], *, trump_suit: Suit) -> int:
    return sum(1 for card in hand if card.suit == trump_suit)


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


def _least_committal_non_trump_index(
    hand: tuple[Card, ...], *, trump_suit: Suit, exclude_index: int | None = None
) -> int | None:
    """
    Carta non-briscola meno utile da giocare: preferisce lisce da 0 punti, poi costo minimo.

    Serve sia come sostituzione per "trattieni l'asso" sia per "cava di meno". Usa
    `card_conservation_cost` (lo stesso ordinamento di `overkill_guard` e del reward shaping)
    per coerenza tra guard runtime, shaping e metriche.
    """
    indices = [i for i in range(len(hand)) if i != exclude_index and hand[i].suit != trump_suit]
    if not indices:
        return None
    smooth = [i for i in indices if int(hand[i].rank.points) == 0]
    pool = smooth if smooth else indices
    return min(pool, key=lambda i: card_conservation_cost(hand[i]))


def _cheapest_non_load_trump_index(hand: tuple[Card, ...], *, trump_suit: Suit, load_points_min: int) -> int | None:
    """
    Briscola più economica NON carico (points < load_points_min), per "cava di più".

    Escludiamo Asso/Tre di briscola: forzare un pull dumpando un carico di briscola sarebbe
    autolesionista. Se non c'è una briscolina liscia, il trattamento non interviene.
    """
    candidates = [
        i for i, card in enumerate(hand) if card.suit == trump_suit and int(card.rank.points) < int(load_points_min)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda i: card_conservation_cost(hand[i]))


# ---------------------------------------------------------------------------
# Tier B: configurazione e agente-trattamento eval-only
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrumpTreatmentConfig:
    """
    Parametri dei trattamenti controfattuali eval-only.

    - `ace_early_deck_min`: "asso guidato presto" = `is_lead` e `deck_size >= ace_early_deck_min`.
      Default 9 (cioè `deck_size > 8`), soglia pre-registrata prima di guardare i dati.
    - `long_trumps_min`: "mano lunga di briscola" = almeno N briscole in mano (default 2).
    - `load_points_min`: soglia carico (Asso/Tre = 10) usata per scegliere una briscolina non
      carico nel pull.
    """

    treatment: str = "ace_hold"
    ace_early_deck_min: int = 9
    long_trumps_min: int = 2
    load_points_min: int = 10


def apply_trump_treatment_index(
    observation: PlayerObservation,
    *,
    chosen_card_index: int,
    config: TrumpTreatmentConfig,
) -> tuple[int, str]:
    """
    Post-processing eval-only che modifica la mossa del modello secondo il trattamento scelto.

    Ritorna `(card_index, reason)`; `reason == "adjusted"` indica una sostituzione. Usa solo
    `PlayerObservation`: mai la mano avversaria o l'ordine del mazzo. Tutti i trattamenti
    agiscono **solo da primo di mano** (`is_lead`): da secondo, `overkill_guard` copre già
    l'asso-overkill e un asso che cattura un carico è quasi sempre una mossa buona.
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
    hand = observation.hand
    chosen = hand[chosen_card_index]
    treatment = str(config.treatment)

    if treatment == "ace_hold":
        # Trattieni l'asso di briscola guidato presto: gioca la carta meno impegnativa.
        if not _is_trump_ace(chosen, trump_suit):
            return chosen_card_index, "not_ace_lead"
        if int(observation.deck_size) < int(config.ace_early_deck_min):
            return chosen_card_index, "not_early"
        replacement = _least_committal_non_trump_index(hand, trump_suit=trump_suit, exclude_index=chosen_card_index)
        if replacement is None:
            # Solo briscole in mano oltre all'asso: nessuna alternativa sensata, non toccare.
            return chosen_card_index, "no_alt"
        return int(replacement), "adjusted"

    if treatment == "pull_more":
        # Con mano lunga di briscola, se NON stavo aprendo in briscola, forza una briscolina.
        if _trump_hand_size(hand, trump_suit=trump_suit) < int(config.long_trumps_min):
            return chosen_card_index, "hand_not_long"
        if _is_trump(chosen, trump_suit):
            return chosen_card_index, "already_pulling"
        replacement = _cheapest_non_load_trump_index(
            hand, trump_suit=trump_suit, load_points_min=int(config.load_points_min)
        )
        if replacement is None:
            return chosen_card_index, "no_cheap_trump"
        return int(replacement), "adjusted"

    if treatment == "pull_less":
        # Con mano lunga di briscola, se stavo aprendo in briscola, gioca invece una liscia.
        if _trump_hand_size(hand, trump_suit=trump_suit) < int(config.long_trumps_min):
            return chosen_card_index, "hand_not_long"
        if not _is_trump(chosen, trump_suit):
            return chosen_card_index, "not_pulling"
        replacement = _least_committal_non_trump_index(hand, trump_suit=trump_suit, exclude_index=chosen_card_index)
        if replacement is None:
            return chosen_card_index, "no_non_trump"
        return int(replacement), "adjusted"

    raise ValueError(f"trattamento non supportato: {treatment!r}")


class TrumpTreatmentAgent:
    """Wrapper eval-only: applica un trattamento controfattuale a una policy esistente."""

    def __init__(self, inner: Agent, *, config: TrumpTreatmentConfig) -> None:
        self.inner = inner
        self.config = config
        self.metrics: Counter[str] = Counter()

    @property
    def name(self) -> str:
        return f"{self.inner.name},trump_treatment={self.config.treatment}"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        chosen = int(self.inner.choose_card_index(observation, rng=rng))
        treated, reason = apply_trump_treatment_index(observation, chosen_card_index=chosen, config=self.config)
        self.metrics[f"reason_{reason}"] += 1
        if reason == "adjusted":
            self.metrics["adjustments"] += 1
        return int(treated)


# ---------------------------------------------------------------------------
# Contatori e sommari
# ---------------------------------------------------------------------------


def _empty_counts() -> Counter[str]:
    counter: Counter[str] = Counter(
        {
            # match
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "points_for": 0,
            "points_against": 0,
            "point_diff": 0,
            # asso di briscola (Tier A)
            "lead_decisions": 0,
            "ace_plays": 0,
            "ace_led": 0,
            "ace_led_early": 0,
            "ace_followed": 0,
            "ace_capture_points": 0,
            "ace_capture_le11": 0,
            "ace_capture_12_16": 0,
            "ace_capture_ge17": 0,
            # regret endgame (Tier C)
            "endgame_decisions": 0,
            "endgame_suboptimal": 0,
            "endgame_regret_points": 0,
            "endgame_ace_decisions": 0,
            "endgame_ace_suboptimal": 0,
            "endgame_ace_regret_points": 0,
            "endgame_trumplead_decisions": 0,
            "endgame_trumplead_suboptimal": 0,
            "endgame_trumplead_regret_points": 0,
            "endgame_unsolved": 0,
        }
    )
    # cavata con mano lunga (Tier A), per soglia di briscole in mano
    for t in (2, 3):
        counter[f"lead_dec_trump_ge{t}"] = 0
        counter[f"lead_trump_ge{t}"] = 0
        counter[f"lead_trump_ge{t}_won"] = 0
        counter[f"lead_trump_ge{t}_points"] = 0
    return counter


def _pct(num: int, den: int) -> float:
    return round(100.0 * float(num) / float(den), 2) if den else 0.0


def _avg(num: int, den: int) -> float:
    return round(float(num) / float(den), 3) if den else 0.0


def _summarize(counter: Counter[str]) -> dict[str, float | int]:
    ace_plays = int(counter["ace_plays"])
    endgame = int(counter["endgame_decisions"])
    summary: dict[str, float | int] = {
        "lead_decisions": int(counter["lead_decisions"]),
        # asso
        "ace_plays": ace_plays,
        "ace_led": int(counter["ace_led"]),
        "ace_led_early": int(counter["ace_led_early"]),
        "ace_followed": int(counter["ace_followed"]),
        "ace_led_pct_of_plays": _pct(int(counter["ace_led"]), ace_plays),
        "ace_avg_capture": _avg(int(counter["ace_capture_points"]), ace_plays),
        "ace_capture_le11_pct": _pct(int(counter["ace_capture_le11"]), ace_plays),
        "ace_capture_12_16_pct": _pct(int(counter["ace_capture_12_16"]), ace_plays),
        "ace_capture_ge17_pct": _pct(int(counter["ace_capture_ge17"]), ace_plays),
        # regret endgame
        "endgame_decisions": endgame,
        "endgame_suboptimal_pct": _pct(int(counter["endgame_suboptimal"]), endgame),
        "endgame_avg_regret": _avg(int(counter["endgame_regret_points"]), endgame),
        "endgame_ace_decisions": int(counter["endgame_ace_decisions"]),
        "endgame_ace_suboptimal_pct": _pct(
            int(counter["endgame_ace_suboptimal"]), int(counter["endgame_ace_decisions"])
        ),
        "endgame_trumplead_decisions": int(counter["endgame_trumplead_decisions"]),
        "endgame_trumplead_suboptimal_pct": _pct(
            int(counter["endgame_trumplead_suboptimal"]), int(counter["endgame_trumplead_decisions"])
        ),
        "endgame_unsolved": int(counter["endgame_unsolved"]),
    }
    # cavata con mano lunga
    for t in (2, 3):
        lead_dec = int(counter[f"lead_dec_trump_ge{t}"])
        lead_trump = int(counter[f"lead_trump_ge{t}"])
        summary[f"lead_dec_trump_ge{t}"] = lead_dec
        summary[f"lead_trump_ge{t}_pct"] = _pct(lead_trump, lead_dec)
        summary[f"lead_trump_ge{t}_won_pct"] = _pct(int(counter[f"lead_trump_ge{t}_won"]), lead_trump)
        summary[f"lead_trump_ge{t}_avg_points"] = _avg(int(counter[f"lead_trump_ge{t}_points"]), lead_trump)
    return summary


def _summarize_match(counter: Counter[str]) -> dict[str, float | int]:
    games = int(counter["games"])
    return {
        "games": games,
        "wins": int(counter["wins"]),
        "losses": int(counter["losses"]),
        "draws": int(counter["draws"]),
        "avg_points_for": _avg(int(counter["points_for"]), games),
        "avg_points_against": _avg(int(counter["points_against"]), games),
        "avg_point_diff": _avg(int(counter["point_diff"]), games),
        "score_rate": round((float(counter["wins"]) + 0.5 * float(counter["draws"])) / games, 4) if games else 0.0,
    }


# ---------------------------------------------------------------------------
# Tier C: regret esatto in endgame (deck_size == 0)
# ---------------------------------------------------------------------------


def _endgame_regret(state: Any, *, chosen_card_index: int) -> int | None:
    """
    Regret in punti della mossa scelta rispetto all'ottimo del solver, dal punto di vista del
    mover. Ritorna `None` se lo stato non è risolvibile (mazzo non vuoto o fuori scope).

    Convenzione solver: `final_delta_p0_p1` è sempre dal punto di vista del player 0. Il mover 0
    massimizza quel delta, il mover 1 lo minimizza; convertiamo entrambi in regret >= 0 dal punto
    di vista di chi muove.
    """
    if len(state.deck) != 0 or state.game_over or len(state.table_cards) not in (0, 1):
        return None
    mover = int(state.current_turn)
    try:
        best_value = solve_endgame(state).final_delta_p0_p1
        child, result = step(state, PlayCardAction(player_index=mover, card_index=chosen_card_index))
        if result.error:
            return None
        if child.game_over:
            chosen_value = int(child.players[0].points) - int(child.players[1].points)
        else:
            chosen_value = solve_endgame(child).final_delta_p0_p1
    except ValueError:
        return None
    regret = (best_value - chosen_value) if mover == 0 else (chosen_value - best_value)
    return max(0, int(regret))


# ---------------------------------------------------------------------------
# Simulazione + profilazione
# ---------------------------------------------------------------------------


def profile(
    model_agent: Agent,
    opponent: Agent,
    *,
    num_games: int,
    seed_base: int,
) -> dict[str, Any]:
    """Gioca partite seat-alternate e misura i comportamenti asso/cavata + regret endgame."""
    c = _empty_counts()
    by_deck: dict[str, Counter[str]] = {}
    pending: dict[str, Any] | None = None

    for game_seed in range(seed_base, seed_base + num_games):
        model_idx = game_seed % 2
        state = new_game_state(2, ["A", "B"], seed=game_seed)
        rngs = {
            model_idx: random.Random(10_000 + game_seed),
            1 - model_idx: random.Random(20_000 + game_seed),
        }
        pending = None

        while not state.game_over:
            turn = state.current_turn
            obs = make_player_observation(state, turn)
            agent = model_agent if turn == model_idx else opponent
            card_index = agent.choose_card_index(obs, rng=rngs[turn])
            played = obs.hand[card_index]
            trump_suit = obs.trump_card.suit if obs.trump_card else None
            is_lead = len(obs.table_cards) == 0

            if turn == model_idx and trump_suit is not None and obs.num_players == 2:
                deck_bucket = _bucket(int(obs.deck_size), thresholds=_DECK_THRESHOLDS)
                by_deck.setdefault(deck_bucket, _empty_counts())

                # Tier C: regret endgame sulla mossa effettiva (usa lo stato reale, deck==0).
                regret = _endgame_regret(state, chosen_card_index=card_index)
                if regret is None:
                    if len(state.deck) == 0 and not state.game_over:
                        c["endgame_unsolved"] += 1
                else:
                    suboptimal = 1 if regret > 0 else 0
                    c["endgame_decisions"] += 1
                    c["endgame_suboptimal"] += suboptimal
                    c["endgame_regret_points"] += regret
                    if _is_trump_ace(played, trump_suit):
                        c["endgame_ace_decisions"] += 1
                        c["endgame_ace_suboptimal"] += suboptimal
                        c["endgame_ace_regret_points"] += regret
                    if is_lead and _is_trump(played, trump_suit):
                        c["endgame_trumplead_decisions"] += 1
                        c["endgame_trumplead_suboptimal"] += suboptimal
                        c["endgame_trumplead_regret_points"] += regret

                # Tier A: asso di briscola.
                if _is_trump_ace(played, trump_suit):
                    c["ace_plays"] += 1
                    by_deck[deck_bucket]["ace_plays"] += 1
                    if is_lead:
                        c["ace_led"] += 1
                        by_deck[deck_bucket]["ace_led"] += 1
                        if int(obs.deck_size) >= 9:  # soglia descrittiva "presto"
                            c["ace_led_early"] += 1
                    else:
                        c["ace_followed"] += 1

                # Tier A: cavata con mano lunga (solo da primo di mano).
                if is_lead:
                    c["lead_decisions"] += 1
                    by_deck[deck_bucket]["lead_decisions"] += 1
                    trump_hand = _trump_hand_size(obs.hand, trump_suit=trump_suit)
                    led_trump = _is_trump(played, trump_suit)
                    for t in (2, 3):
                        if trump_hand >= t:
                            c[f"lead_dec_trump_ge{t}"] += 1
                            if t == 2:
                                by_deck[deck_bucket]["lead_dec_trump_ge2"] += 1
                            if led_trump:
                                c[f"lead_trump_ge{t}"] += 1
                                if t == 2:
                                    by_deck[deck_bucket]["lead_trump_ge2"] += 1

                # Pending per calcolare esito/punti della presa alla chiusura.
                pending = {
                    "is_ace": _is_trump_ace(played, trump_suit),
                    "is_trump_lead": is_lead and _is_trump(played, trump_suit),
                    "trump_hand": _trump_hand_size(obs.hand, trump_suit=trump_suit) if is_lead else 0,
                    "deck_bucket": deck_bucket,
                }

            state, result = step(state, PlayCardAction(player_index=turn, card_index=card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio: {result.error}")

            if result.trick_completed and pending is not None:
                won = result.trick_winner == model_idx
                trick_points = sum(int(card.rank.points) for card, _ in result.trick_cards)
                if pending["is_ace"]:
                    # L'asso di briscola vince sempre: `trick_points` è ciò che il modello rastrella.
                    c["ace_capture_points"] += trick_points
                    by_deck[pending["deck_bucket"]]["ace_capture_points"] += trick_points
                    if trick_points <= 11:
                        c["ace_capture_le11"] += 1
                    elif trick_points <= 16:
                        c["ace_capture_12_16"] += 1
                    else:
                        c["ace_capture_ge17"] += 1
                if pending["is_trump_lead"] and won:
                    for t in (2, 3):
                        if int(pending["trump_hand"]) >= t:
                            c[f"lead_trump_ge{t}_won"] += 1
                            c[f"lead_trump_ge{t}_points"] += trick_points
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
    }


def _build_opponent(name: str, *, model_path: Path) -> Agent:
    return build_agent("bc_model", model_path=model_path) if name == "mirror" else build_agent(name)


def treatment_ablation(
    *,
    model_path: Path,
    opponent_name: str,
    num_games: int,
    seed_base: int,
    config: TrumpTreatmentConfig,
) -> dict[str, Any]:
    """Confronta baseline e agente-trattamento sugli stessi seed iniziali."""
    baseline_agent = build_agent("bc_model", model_path=model_path)
    treated_inner = build_agent("bc_model", model_path=model_path)
    treated_agent = TrumpTreatmentAgent(treated_inner, config=config)

    baseline = profile(
        baseline_agent,
        _build_opponent(opponent_name, model_path=model_path),
        num_games=num_games,
        seed_base=seed_base,
    )
    treated = profile(
        treated_agent,
        _build_opponent(opponent_name, model_path=model_path),
        num_games=num_games,
        seed_base=seed_base,
    )
    b_match, t_match = baseline["match"], treated["match"]
    return {
        "baseline": baseline,
        "treated": treated,
        "delta": {
            "avg_point_diff": round(float(t_match["avg_point_diff"]) - float(b_match["avg_point_diff"]), 3),
            "score_rate": round(float(t_match["score_rate"]) - float(b_match["score_rate"]), 4),
        },
        "treatment_metrics": dict(sorted(treated_agent.metrics.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sonda diagnostica asso di briscola + cavata con mano lunga.")
    parser.add_argument(
        "--mode",
        choices=["phase0", "treat", "both"],
        default="phase0",
        help="`phase0` = descrittivo (Tier A/C); `treat` = ablation controfattuale (Tier B); `both`.",
    )
    parser.add_argument("--model", default="data/models/best_a2c_v11.npz", help="Path `.npz` del modello.")
    parser.add_argument(
        "--opponents",
        default="heuristic_trump_saver,mirror,heuristic_v1",
        help="Avversari separati da virgola; `mirror` = lo stesso modello.",
    )
    parser.add_argument("--num-games", type=int, default=1000, help="Partite per avversario.")
    parser.add_argument("--seed", type=int, default=20260709, help="Seed base per le partite.")
    parser.add_argument(
        "--treatment",
        default="all",
        help="Trattamento Tier B: `ace_hold`, `pull_more`, `pull_less` o `all` (tutti e tre).",
    )
    parser.add_argument("--long-trumps-min", type=int, default=2, help="Mano lunga di briscola: >= N briscole.")
    parser.add_argument(
        "--ace-early-deck-min",
        type=int,
        default=9,
        help="Asso guidato presto se deck_size >= N (default 9 = deck_size > 8).",
    )
    parser.add_argument("--load-points-min", type=int, default=10, help="Soglia carico (Asso/Tre=10).")
    parser.add_argument("--out-json", default="", help="Path JSON opzionale.")
    args = parser.parse_args()

    model_path = Path(args.model)
    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    if str(args.treatment) == "all":
        treatments = ["ace_hold", "pull_more", "pull_less"]
    else:
        treatments = [item.strip() for item in str(args.treatment).split(",") if item.strip()]

    results: dict[str, Any] = {}
    for index, name in enumerate(opponents):
        seed_base = int(args.seed) + index * 100_000
        entry: dict[str, Any] = {}

        if args.mode in ("phase0", "both"):
            model_agent = build_agent("bc_model", model_path=model_path)
            phase0 = profile(
                model_agent,
                _build_opponent(name, model_path=model_path),
                num_games=int(args.num_games),
                seed_base=seed_base,
            )
            entry["phase0"] = phase0
            print(f"=== phase0 vs {name} ({args.num_games} partite) ===")
            print(f"  match: {phase0['match']}")
            for key, value in phase0["summary"].items():
                print(f"  {key}: {value}")
            print("  by_deck_size:")
            for bucket, summ in phase0["by_deck_size"].items():
                print(
                    f"    {bucket}: ace_plays={summ['ace_plays']} ace_led={summ['ace_led']} "
                    f"ace_avg_capture={summ['ace_avg_capture']} "
                    f"lead_trump_ge2_pct={summ['lead_trump_ge2_pct']} "
                    f"(lead_dec_ge2={summ['lead_dec_trump_ge2']})"
                )

        if args.mode in ("treat", "both"):
            entry["treatments"] = {}
            for treatment in treatments:
                config = TrumpTreatmentConfig(
                    treatment=treatment,
                    ace_early_deck_min=int(args.ace_early_deck_min),
                    long_trumps_min=int(args.long_trumps_min),
                    load_points_min=int(args.load_points_min),
                )
                ablation = treatment_ablation(
                    model_path=model_path,
                    opponent_name=name,
                    num_games=int(args.num_games),
                    seed_base=seed_base,
                    config=config,
                )
                entry["treatments"][treatment] = ablation
                print(f"=== treat[{treatment}] vs {name} ({args.num_games} partite) ===")
                print(f"  baseline_match: {ablation['baseline']['match']}")
                print(f"  treated_match:  {ablation['treated']['match']}")
                print(f"  delta: {ablation['delta']}")
                print(f"  treatment_metrics: {ablation['treatment_metrics']}")

        results[name] = entry

    payload = {
        "meta": {
            "git_commit": _git_commit(),
            "mode": str(args.mode),
            "model": str(model_path),
            "opponents": opponents,
            "num_games_per_opponent": int(args.num_games),
            "seed": int(args.seed),
            "treatments": treatments,
            "long_trumps_min": int(args.long_trumps_min),
            "ace_early_deck_min": int(args.ace_early_deck_min),
            "load_points_min": int(args.load_points_min),
        },
        "profiles": results,
    }
    if args.out_json.strip():
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON salvato in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
