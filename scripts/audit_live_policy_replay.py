#!/usr/bin/env python3
"""
Confronta due policy sulle stesse osservazioni lecite esportate dal live.

Perche' esiste
--------------
Un confronto dei win-rate tra versioni giocate da persone diverse richiede molte
partite e resta confuso dalla diversa abilita' dei giocatori. Questo audit risponde a
una domanda piu' stretta ma gia' utile con pochi giochi: **v13 e v14 sceglierebbero la
stessa carta se vedessero esattamente la stessa situazione pubblica?**

Lo script valuta due livelli:

1. policy pura, deterministica;
2. runtime prodotto PIMC belief 16x8 con solver finale. Nella finestra stochasticamente
   campionata ripete gli stessi seed per entrambi i modelli e distingue i disaccordi
   stabili dalla normale variabilita' PIMC.

Anti-cheat e privacy
--------------------
L'input del modello e' sempre una ``PlayerObservation`` ricostruita dal DTO pubblico.
La carta di briscola, omessa dal DTO quando il mazzo e' vuoto, viene recuperata dal
primo snapshot della stessa partita: era gia' stata mostrata a entrambi i giocatori.
Il JSON finale contiene solo aggregati e hash; non salva game id, carte, nomi o
osservazioni individuali.

Esempio::

    uv run python scripts/audit_live_policy_replay.py \
      --input data/prod_live_actions_v13.jsonl \
      --input data/prod_live_actions_v14.jsonl \
      --model-a data/models/best_a2c_v13.npz \
      --model-b data/models/best_a2c_v14.npz \
      --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
      --runtime-repeats 4 --seed 20260714 \
      --out-json data/live_policy_replay_v13_v14.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents import PIMCAgent
from briscola_ai.ai.models.bc_model import BCModelAgent
from briscola_ai.ai.models.belief_model import load_belief_model_npz
from briscola_ai.ai.training.reward_shaping import card_conservation_cost
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation
from briscola_ai.domain.rules import who_wins_trick
from briscola_ai.domain.state import TrickRecord
from briscola_ai.versioning import get_code_version, get_rules_version


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Configurazione riproducibile del replay controfattuale."""

    input_paths: tuple[Path, ...]
    model_a_path: Path
    model_b_path: Path
    belief_model_path: Path
    out_json_path: Path
    seed: int = 20260714
    runtime_repeats: int = 4
    determinizations: int = 16
    max_unknown_cards: int = 8
    bootstrap_samples: int = 5000


@dataclass(frozen=True, slots=True)
class _LoadedRecord:
    """Record live con chiave partita locale, mai serializzata nel report."""

    game_key: str
    game_id: str
    event_id: int
    source_name: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RuntimeChoice:
    """Scelta modale del runtime e stabilita' rispetto ai replay PIMC."""

    card_id: int
    branch: str
    choices: tuple[int, ...]

    @property
    def stable(self) -> bool:
        """True se tutti i replay hanno scelto la stessa carta."""
        return len(set(self.choices)) == 1


@dataclass(frozen=True, slots=True)
class _DecisionComparison:
    """Risultato compatto per una singola osservazione non forzata."""

    game_key: str
    source_model: str
    source_version: str
    actor: str
    outcome: str
    branch: str
    policy_a: int
    policy_b: int
    runtime_a: int
    runtime_b: int
    runtime_a_stable: bool
    runtime_b_stable: bool


def _sha256(path: Path) -> str:
    """SHA-256 streaming di un artefatto locale."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    """Descrizione stabile di un artefatto senza includerne il contenuto."""
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _git_commit() -> str | None:
    """Commit corrente best-effort, utile a riprodurre l'audit."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError, subprocess.SubprocessError:
        return None


def _card_from_dto(raw: Any) -> Card:
    """Converte e valida una carta del contratto ``CardDTO``."""
    if not isinstance(raw, dict):
        raise ValueError(f"CardDTO non oggetto: {type(raw).__name__}")
    suit_raw = raw.get("suit")
    rank_raw = raw.get("rank")
    number_raw = raw.get("number")
    if not isinstance(suit_raw, str) or not isinstance(rank_raw, str):
        raise ValueError(f"CardDTO senza suit/rank validi: {raw!r}")
    try:
        card = Card(suit=Suit(suit_raw), rank=Rank[rank_raw])
    except (ValueError, KeyError) as exc:
        raise ValueError(f"CardDTO con suit/rank sconosciuti: {raw!r}") from exc
    if isinstance(number_raw, int) and int(card.rank.number) != number_raw:
        raise ValueError(f"CardDTO incoerente: rank={rank_raw} number={number_raw}")
    return card


def _trick_history_from_dto(raw: Any) -> tuple[TrickRecord, ...]:
    """Ricostruisce la storia pubblica completa usata dall'encoder v4."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("trick_history deve essere una lista")
    records: list[TrickRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("trick_history contiene un record non oggetto")
        cards_raw = item.get("cards")
        if not isinstance(cards_raw, list) or len(cards_raw) != 2:
            raise ValueError("Ogni presa 2-player deve contenere esattamente due carte")
        cards: list[tuple[Card, int]] = []
        for table_item in cards_raw:
            if not isinstance(table_item, dict):
                raise ValueError("Carta di trick_history non oggetto")
            cards.append((_card_from_dto(table_item.get("card")), int(table_item["player_index"])))
        records.append(
            TrickRecord(
                cards=tuple(cards),
                winner_index=int(item["winner_index"]),
                points=int(item["points"]),
            )
        )
    return tuple(records)


def observation_from_live_dto(raw: dict[str, Any], *, exposed_trump_card: Card) -> PlayerObservation:
    """
    Ricostruisce una ``PlayerObservation`` soltanto dall'informazione pubblica.

    ``ObservationDTO.trump_card`` diventa ``null`` a mazzo vuoto per la UI. Il seme
    resta pubblico, ma policy e solver ricevono nel dominio la carta originariamente
    esposta. Il chiamante la recupera da un record precedente della stessa partita.
    """
    if int(raw.get("num_players", 0)) != 2 or bool(raw.get("is_team_game", False)):
        raise ValueError("L'audit live supporta solo partite 2-player non a squadre")
    if not bool(raw.get("my_turn", False)):
        raise ValueError("Il record action deve contenere un decision-state con my_turn=true")

    player_index = int(raw["my_index"])
    players_raw = raw.get("players")
    if not isinstance(players_raw, list) or len(players_raw) != 2:
        raise ValueError("ObservationDTO senza due players")
    players = sorted(players_raw, key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in players] != [0, 1]:
        raise ValueError("Indici players ObservationDTO non canonici")

    dto_trump = raw.get("trump_card")
    if dto_trump is not None and _card_from_dto(dto_trump) != exposed_trump_card:
        raise ValueError("La carta di briscola cambia tra snapshot della stessa partita")
    trump_suit = raw.get("trump_suit")
    if trump_suit is not None and str(trump_suit) != exposed_trump_card.suit.value:
        raise ValueError("trump_suit incoerente con la carta di briscola esposta")

    table_cards: list[tuple[Card, int]] = []
    for item in raw.get("table_cards") or []:
        if not isinstance(item, dict):
            raise ValueError("table_cards contiene un elemento non oggetto")
        table_cards.append((_card_from_dto(item.get("card")), int(item["player_index"])))

    seen = tuple(int(value) for value in (raw.get("seen_cards_onehot") or []))
    out_of_play = tuple(int(value) for value in (raw.get("out_of_play_cards_onehot") or []))
    if len(seen) != 40 or len(out_of_play) != 40:
        raise ValueError("Le one-hot pubbliche devono avere lunghezza 40")

    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=player_index,
        player_name=str(players[player_index].get("name", f"player_{player_index}")),
        hand=tuple(_card_from_dto(card) for card in (raw.get("my_hand") or [])),
        trump_card=exposed_trump_card,
        deck_size=int(raw.get("cards_remaining_in_deck", 0)),
        table_cards=tuple(table_cards),
        current_turn=player_index,
        first_player=int(raw.get("first_player", table_cards[0][1] if table_cards else player_index)),
        game_over=bool(raw.get("game_over", False)),
        winner_index=None,
        winning_team=None,
        players_points=tuple(int(item.get("points", 0)) for item in players),
        players_hand_sizes=tuple(int(item.get("hand_size", 0)) for item in players),
        seen_cards_onehot=seen,
        out_of_play_cards_onehot=out_of_play,
        trick_history=_trick_history_from_dto(raw.get("trick_history")),
    )


def _load_records(paths: tuple[Path, ...]) -> tuple[list[_LoadedRecord], dict[str, Card]]:
    """Carica JSONL, rende uniche le chiavi partita e recupera la briscola pubblica."""
    records: list[_LoadedRecord] = []
    trump_by_game: dict[str, Card] = {}
    seen_events: set[tuple[str, int]] = set()
    for input_index, path in enumerate(paths):
        source_name = path.name
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_number}: record JSON non oggetto")
                game_id = str(raw.get("game_id", "")).strip()
                event_id = int(raw.get("event_id", -1))
                if not game_id or event_id < 0:
                    raise ValueError(f"{path}:{line_number}: game_id/event_id mancanti")
                game_key = f"{input_index}:{game_id}"
                event_key = (game_key, event_id)
                if event_key in seen_events:
                    raise ValueError(f"{path}:{line_number}: evento duplicato {event_id}")
                seen_events.add(event_key)
                observation = raw.get("observation")
                if not isinstance(observation, dict):
                    raise ValueError(f"{path}:{line_number}: observation mancante")
                dto_trump = observation.get("trump_card")
                if dto_trump is not None:
                    card = _card_from_dto(dto_trump)
                    previous = trump_by_game.setdefault(game_key, card)
                    if previous != card:
                        raise ValueError(f"{path}:{line_number}: briscola non stabile")
                records.append(
                    _LoadedRecord(
                        game_key=game_key,
                        game_id=game_id,
                        event_id=event_id,
                        source_name=source_name,
                        payload=raw,
                    )
                )

    missing_trump = sorted({record.game_key for record in records} - set(trump_by_game))
    if missing_trump:
        raise ValueError(f"{len(missing_trump)} partite non contengono alcuno snapshot della briscola")
    records.sort(key=lambda item: (item.game_key, item.event_id))
    return records, trump_by_game


def _stable_seed(base_seed: int, record: _LoadedRecord, repeat: int) -> int:
    """Seed cross-process stabile; non usa ``hash()``, randomizzato da Python."""
    raw = f"{base_seed}:{record.source_name}:{record.game_id}:{record.event_id}:{repeat}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _choose_card_id(agent: BCModelAgent, observation: PlayerObservation) -> int:
    """Scelta deterministica della policy convertita nell'id carta canonico."""
    card_index = int(agent.choose_card_index(observation, rng=random.Random(0)))
    if not 0 <= card_index < len(observation.hand):
        raise ValueError(f"Policy ha prodotto card_index={card_index} per mano {len(observation.hand)}")
    return card_to_id(observation.hand[card_index])


def _runtime_choice(
    agent: PIMCAgent,
    observation: PlayerObservation,
    *,
    record: _LoadedRecord,
    base_seed: int,
    repeats: int,
) -> _RuntimeChoice:
    """Esegue una volta fallback/solver e piu' volte la sola search PIMC."""
    choices: list[int] = []
    branch: str | None = None
    for repeat in range(repeats):
        before_search = agent.metrics.search_decisions
        before_solver = agent.metrics.endgame_solver_decisions
        before_fallback = agent.metrics.fallback_decisions
        card_index = int(
            agent.choose_card_index(
                observation,
                rng=random.Random(_stable_seed(base_seed, record, repeat)),
            )
        )
        if not 0 <= card_index < len(observation.hand):
            raise ValueError(f"Runtime ha prodotto card_index={card_index} per mano {len(observation.hand)}")
        choices.append(card_to_id(observation.hand[card_index]))

        if agent.metrics.search_decisions > before_search:
            current_branch = "search"
        elif agent.metrics.endgame_solver_decisions > before_solver:
            current_branch = "solver"
        elif agent.metrics.fallback_decisions > before_fallback:
            current_branch = "fallback"
        else:
            raise RuntimeError("Il runtime non ha incrementato alcun contatore di ramo")
        if branch is None:
            branch = current_branch
        elif branch != current_branch:
            raise RuntimeError(f"Ramo runtime instabile: {branch} vs {current_branch}")

        # Fallback e solver sono deterministici: le ripetizioni servono solo alla search.
        if current_branch != "search":
            break

    assert branch is not None
    counts = Counter(choices)
    modal = max(counts, key=lambda card_id: (counts[card_id], -card_id))
    return _RuntimeChoice(card_id=int(modal), branch=branch, choices=tuple(choices))


def _human_outcome(records: list[_LoadedRecord]) -> str:
    """Esito della partita dal punto di vista umano."""
    human_seats = {
        int(record.payload["player_index"])
        for record in records
        if record.payload.get("actor") == "human" and isinstance(record.payload.get("player_index"), int)
    }
    if len(human_seats) != 1:
        raise ValueError(f"Partita con posti umani inattesi: {sorted(human_seats)}")
    human_seat = next(iter(human_seats))
    metadata = records[0].payload.get("metadata") or {}
    points = metadata.get("final_points_by_player_index")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError("Partita senza punteggio finale 2-player")
    human_points = int(points[human_seat])
    ai_points = int(points[1 - human_seat])
    if human_points > ai_points:
        return "human_win"
    if human_points < ai_points:
        return "ai_win"
    return "draw"


def _choice_quality(observation: PlayerObservation, card_id: int) -> dict[str, int]:
    """Eventi di stile calcolabili dalla sola osservazione pubblica."""
    choice_index = next((idx for idx, card in enumerate(observation.hand) if card_to_id(card) == card_id), None)
    if choice_index is None:
        raise ValueError(f"Carta scelta {card_id} non presente nella mano")
    chosen = observation.hand[choice_index]
    trump_suit = observation.trump_card.suit if observation.trump_card is not None else None
    counters: Counter[str] = Counter()

    if not observation.table_cards:
        counters["lead_decisions"] += 1
        if trump_suit is not None and chosen.suit != trump_suit and int(chosen.rank.points) >= 10:
            counters["lead_load"] += 1
            if any(card.suit != trump_suit and int(card.rank.points) == 0 for card in observation.hand):
                counters["lead_load_with_smooth_alternative"] += 1
        return dict(counters)

    if len(observation.table_cards) != 1:
        return dict(counters)
    counters["second_hand_decisions"] += 1
    lead_card, lead_player = observation.table_cards[0]
    winning_indices = [
        idx
        for idx, card in enumerate(observation.hand)
        if who_wins_trick(
            ((lead_card, lead_player), (card, observation.player_index)),
            trump_suit,
        )
        == observation.player_index
    ]
    if winning_indices:
        counters["second_hand_with_winning_reply"] += 1
        winning_non_trump = any(
            trump_suit is None or observation.hand[idx].suit != trump_suit for idx in winning_indices
        )
        if trump_suit is not None and chosen.suit == trump_suit and winning_non_trump:
            counters["trump_waste"] += 1

    chosen_wins = (
        who_wins_trick(
            ((lead_card, lead_player), (chosen, observation.player_index)),
            trump_suit,
        )
        == observation.player_index
    )
    if trump_suit is None or chosen.suit != trump_suit or not chosen_wins:
        return dict(counters)

    counters["trump_wins"] += 1
    low_lead = int(lead_card.rank.points) <= 2
    if low_lead:
        counters["trump_wins_low_lead_points"] += 1
    winning_trumps = [observation.hand[idx] for idx in winning_indices if observation.hand[idx].suit == trump_suit]
    if winning_trumps and card_conservation_cost(chosen) > min(card_conservation_cost(card) for card in winning_trumps):
        counters["trump_overkill"] += 1
        if low_lead:
            counters["trump_overkill_low_lead_points"] += 1
    return dict(counters)


def _quality_summary(counter: Counter[str]) -> dict[str, int | float]:
    """Aggiunge tassi leggibili ai contatori di stile."""

    def rate(num: str, den: str) -> float:
        return float(counter[num]) / float(counter[den]) if counter[den] else 0.0

    return {
        **{key: int(value) for key, value in sorted(counter.items())},
        "lead_load_rate": rate("lead_load", "lead_decisions"),
        "trump_waste_rate": rate("trump_waste", "second_hand_with_winning_reply"),
        "trump_overkill_rate": rate("trump_overkill", "trump_wins"),
        "trump_overkill_low_lead_points_rate": rate("trump_overkill_low_lead_points", "trump_wins_low_lead_points"),
    }


def _choice_transition(observation: PlayerObservation, left_card_id: int, right_card_id: int) -> str:
    """Classifica il cambio A->B rispetto al costo di conservazione della carta."""
    if left_card_id == right_card_id:
        return "same"
    cards = {card_to_id(card): card for card in observation.hand}
    left = cards[left_card_id]
    right = cards[right_card_id]
    trump_suit = observation.trump_card.suit if observation.trump_card is not None else None

    def cost(card: Card) -> tuple[int, int, int]:
        return (int(trump_suit is not None and card.suit == trump_suit), *card_conservation_cost(card))

    left_cost = cost(left)
    right_cost = cost(right)
    if right_cost < left_cost:
        return "b_lower_conservation_cost"
    if right_cost > left_cost:
        return "b_higher_conservation_cost"
    return "different_equal_conservation_cost"


def _transition_summary(counter: Counter[str]) -> dict[str, int | float]:
    """Riepiloga la direzione dei soli disaccordi A->B."""
    disagreements = sum(value for key, value in counter.items() if key != "same")
    return {
        **{key: int(value) for key, value in sorted(counter.items())},
        "disagreements": int(disagreements),
        "b_lower_cost_share_of_disagreements": (
            float(counter["b_lower_conservation_cost"]) / float(disagreements) if disagreements else 0.0
        ),
        "b_higher_cost_share_of_disagreements": (
            float(counter["b_higher_conservation_cost"]) / float(disagreements) if disagreements else 0.0
        ),
    }


def _comparison_summary(items: list[_DecisionComparison], *, layer: str) -> dict[str, Any]:
    """Agreement aggregato per policy o runtime."""
    if layer not in {"policy", "runtime"}:
        raise ValueError(f"Layer non supportato: {layer}")
    if layer == "policy":
        values = [(item.policy_a, item.policy_b, True) for item in items]
    else:
        values = [(item.runtime_a, item.runtime_b, item.runtime_a_stable and item.runtime_b_stable) for item in items]
    agreements = sum(1 for left, right, _stable in values if left == right)
    stable_disagreements = sum(1 for left, right, stable in values if left != right and stable)
    total = len(values)
    return {
        "observations": total,
        "agreements": agreements,
        "disagreements": total - agreements,
        "agreement_rate": agreements / total if total else 0.0,
        "stable_disagreements": stable_disagreements,
    }


def _grouped_comparison(items: list[_DecisionComparison], *, layer: str, field: str) -> dict[str, Any]:
    """Applica lo stesso riepilogo a gruppi descrittivi non sovrapposti."""
    groups: dict[str, list[_DecisionComparison]] = defaultdict(list)
    for item in items:
        groups[str(getattr(item, field))].append(item)
    return {key: _comparison_summary(groups[key], layer=layer) for key in sorted(groups)}


def _bootstrap_disagreement_ci(
    items: list[_DecisionComparison], *, layer: str, seed: int, samples: int
) -> dict[str, float | int]:
    """CI bootstrap per partita: le decisioni della stessa partita restano unite."""
    counts: dict[str, tuple[int, int]] = {}
    grouped: dict[str, list[_DecisionComparison]] = defaultdict(list)
    for item in items:
        grouped[item.game_key].append(item)
    for game_key, game_items in grouped.items():
        if layer == "policy":
            disagreements = sum(item.policy_a != item.policy_b for item in game_items)
        else:
            disagreements = sum(item.runtime_a != item.runtime_b for item in game_items)
        counts[game_key] = (int(disagreements), len(game_items))

    game_keys = sorted(counts)
    if not game_keys or samples <= 0:
        return {"confidence": 0.95, "low": 0.0, "high": 0.0, "samples": int(samples)}
    rng = np.random.default_rng(seed)
    ratios = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        picked = rng.integers(0, len(game_keys), size=len(game_keys))
        numerator = denominator = 0
        for index in picked:
            disagreements, total = counts[game_keys[int(index)]]
            numerator += disagreements
            denominator += total
        ratios[sample_index] = numerator / denominator if denominator else 0.0
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {"confidence": 0.95, "low": float(low), "high": float(high), "samples": int(samples)}


def _build_runtime(policy: BCModelAgent, belief_model: Any, config: ReplayConfig, *, label: str) -> PIMCAgent:
    """Replica esplicitamente la configurazione prodotto, senza dipendere dall'env."""
    return PIMCAgent(
        rollout_agent=policy,
        fallback=policy,
        num_determinizations=int(config.determinizations),
        max_unknown_cards=int(config.max_unknown_cards),
        use_endgame_solver=True,
        belief_model=belief_model,
        belief_uniform_mix=0.10,
        use_numba_search=False,
        name=label,
    )


def run_audit(config: ReplayConfig) -> dict[str, Any]:
    """Esegue il replay e ritorna un report aggregato privo di dati individuali."""
    if config.runtime_repeats <= 0:
        raise ValueError("runtime_repeats deve essere > 0")
    if config.determinizations <= 0 or config.max_unknown_cards <= 0:
        raise ValueError("determinizations e max_unknown_cards devono essere > 0")

    records, trump_by_game = _load_records(config.input_paths)
    records_by_game: dict[str, list[_LoadedRecord]] = defaultdict(list)
    for record in records:
        records_by_game[record.game_key].append(record)
    outcomes = {game_key: _human_outcome(game_records) for game_key, game_records in records_by_game.items()}

    policy_a = BCModelAgent.from_npz(config.model_a_path)
    policy_b = BCModelAgent.from_npz(config.model_b_path)
    belief_model = load_belief_model_npz(config.belief_model_path)
    runtime_a = _build_runtime(policy_a, belief_model, config, label=f"live-replay:{config.model_a_path.name}")
    runtime_b = _build_runtime(policy_b, belief_model, config, label=f"live-replay:{config.model_b_path.name}")

    comparisons: list[_DecisionComparison] = []
    quality: dict[str, Counter[str]] = {
        "policy_a": Counter(),
        "policy_b": Counter(),
        "runtime_a": Counter(),
        "runtime_b": Counter(),
    }
    quality_by_branch: dict[str, dict[str, Counter[str]]] = {label: defaultdict(Counter) for label in quality}
    quality_by_outcome: dict[str, dict[str, Counter[str]]] = {label: defaultdict(Counter) for label in quality}
    transitions: dict[str, Counter[str]] = {"policy": Counter(), "runtime": Counter()}
    transitions_by_branch: dict[str, dict[str, Counter[str]]] = {
        "policy": defaultdict(Counter),
        "runtime": defaultdict(Counter),
    }
    transitions_by_outcome: dict[str, dict[str, Counter[str]]] = {
        "policy": defaultdict(Counter),
        "runtime": defaultdict(Counter),
    }
    counters: Counter[str] = Counter()
    native_fidelity: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.perf_counter()

    for record in records:
        counters["records_read"] += 1
        raw_observation = record.payload["observation"]
        observation = observation_from_live_dto(
            raw_observation,
            exposed_trump_card=trump_by_game[record.game_key],
        )
        if len(observation.hand) <= 1:
            counters["forced_decisions_skipped"] += 1
            continue
        counters["eligible_nonforced_decisions"] += 1

        policy_a_choice = _choose_card_id(policy_a, observation)
        policy_b_choice = _choose_card_id(policy_b, observation)
        runtime_a_choice = _runtime_choice(
            runtime_a,
            observation,
            record=record,
            base_seed=config.seed,
            repeats=config.runtime_repeats,
        )
        runtime_b_choice = _runtime_choice(
            runtime_b,
            observation,
            record=record,
            base_seed=config.seed,
            repeats=config.runtime_repeats,
        )
        if runtime_a_choice.branch != runtime_b_choice.branch:
            raise RuntimeError(
                f"Il ramo runtime dipende dal modello: {runtime_a_choice.branch} vs {runtime_b_choice.branch}"
            )

        metadata = record.payload.get("metadata") or {}
        source_model = str(metadata.get("ai_model_id") or "<unknown>")
        source_version = str(metadata.get("code_version") or "<unknown>")
        actor = str(record.payload.get("actor") or "<unknown>")
        item = _DecisionComparison(
            game_key=record.game_key,
            source_model=source_model,
            source_version=source_version,
            actor=actor,
            outcome=outcomes[record.game_key],
            branch=runtime_a_choice.branch,
            policy_a=policy_a_choice,
            policy_b=policy_b_choice,
            runtime_a=runtime_a_choice.card_id,
            runtime_b=runtime_b_choice.card_id,
            runtime_a_stable=runtime_a_choice.stable,
            runtime_b_stable=runtime_b_choice.stable,
        )
        comparisons.append(item)

        for label, choice in (
            ("policy_a", policy_a_choice),
            ("policy_b", policy_b_choice),
            ("runtime_a", runtime_a_choice.card_id),
            ("runtime_b", runtime_b_choice.card_id),
        ):
            choice_events = _choice_quality(observation, choice)
            quality[label].update(choice_events)
            quality_by_branch[label][runtime_a_choice.branch].update(choice_events)
            quality_by_outcome[label][outcomes[record.game_key]].update(choice_events)

        policy_transition = _choice_transition(observation, policy_a_choice, policy_b_choice)
        runtime_transition = _choice_transition(observation, runtime_a_choice.card_id, runtime_b_choice.card_id)
        transitions["policy"][policy_transition] += 1
        transitions["runtime"][runtime_transition] += 1
        transitions_by_branch["policy"][runtime_a_choice.branch][policy_transition] += 1
        transitions_by_branch["runtime"][runtime_a_choice.branch][runtime_transition] += 1
        transitions_by_outcome["policy"][outcomes[record.game_key]][policy_transition] += 1
        transitions_by_outcome["runtime"][outcomes[record.game_key]][runtime_transition] += 1

        if actor == "ai" and source_model in {config.model_a_path.name, config.model_b_path.name}:
            native = "a" if source_model == config.model_a_path.name else "b"
            selected_runtime = runtime_a_choice if native == "a" else runtime_b_choice
            selected_policy = policy_a_choice if native == "a" else policy_b_choice
            logged_branch = str((record.payload.get("ai") or {}).get("decision_type") or "<unknown>")
            actual_card = record.payload.get("action", {}).get("card")
            actual_card_id = card_to_id(_card_from_dto(actual_card))
            bucket = native_fidelity[f"model_{native}"]
            bucket["observations"] += 1
            bucket["branch_matches"] += int(logged_branch == selected_runtime.branch)
            bucket[f"branch_{selected_runtime.branch}"] += 1
            bucket[f"actual_runtime_agreement_{selected_runtime.branch}"] += int(
                actual_card_id == selected_runtime.card_id
            )
            if selected_runtime.branch == "fallback":
                bucket["actual_policy_agreement_on_fallback"] += int(actual_card_id == selected_policy)

    elapsed = time.perf_counter() - started
    games_summary: Counter[str] = Counter(outcomes.values())
    version_games: Counter[str] = Counter()
    model_games: Counter[str] = Counter()
    for game_records in records_by_game.values():
        metadata = game_records[0].payload.get("metadata") or {}
        version_games[str(metadata.get("code_version") or "<unknown>")] += 1
        model_games[str(metadata.get("ai_model_id") or "<unknown>")] += 1

    policy_summary = _comparison_summary(comparisons, layer="policy")
    runtime_summary = _comparison_summary(comparisons, layer="runtime")
    policy_summary["disagreement_rate_ci95_game_bootstrap"] = _bootstrap_disagreement_ci(
        comparisons,
        layer="policy",
        seed=config.seed ^ 0xA13,
        samples=config.bootstrap_samples,
    )
    runtime_summary["disagreement_rate_ci95_game_bootstrap"] = _bootstrap_disagreement_ci(
        comparisons,
        layer="runtime",
        seed=config.seed ^ 0xB14,
        samples=config.bootstrap_samples,
    )

    def fidelity_payload(counter: Counter[str]) -> dict[str, Any]:
        payload = {key: int(value) for key, value in sorted(counter.items())}
        observations = int(counter["observations"])
        payload["branch_match_rate"] = counter["branch_matches"] / observations if observations else 0.0
        for branch in ("fallback", "search", "solver"):
            denom = int(counter[f"branch_{branch}"])
            payload[f"actual_runtime_agreement_rate_{branch}"] = (
                counter[f"actual_runtime_agreement_{branch}"] / denom if denom else 0.0
            )
        fallback = int(counter["branch_fallback"])
        payload["actual_policy_agreement_rate_on_fallback"] = (
            counter["actual_policy_agreement_on_fallback"] / fallback if fallback else 0.0
        )
        return payload

    return {
        "schema": "briscola.live_policy_replay.v1",
        "scope": {
            "games": len(records_by_game),
            "records": len(records),
            "eligible_nonforced_decisions": int(counters["eligible_nonforced_decisions"]),
            "forced_decisions_skipped": int(counters["forced_decisions_skipped"]),
            "outcomes_human_perspective": dict(sorted(games_summary.items())),
            "games_by_code_version": dict(sorted(version_games.items())),
            "games_by_source_model": dict(sorted(model_games.items())),
        },
        "method": {
            "anti_cheat": (
                "input ricostruito esclusivamente da ObservationDTO pubblico; nessuna mano avversaria o deck order"
            ),
            "privacy": "solo aggregati; nessun game_id, nome, client_id o observation nel report",
            "forced_decisions_excluded": True,
            "paired_same_observations": True,
            "paired_same_runtime_seeds": True,
            "runtime": {
                "determinizations": int(config.determinizations),
                "max_unknown_cards": int(config.max_unknown_cards),
                "belief_uniform_mix": 0.10,
                "repeats_in_search_window": int(config.runtime_repeats),
                "endgame_solver": True,
                "numba_search": False,
            },
            "bootstrap": {"unit": "game", "samples": int(config.bootstrap_samples)},
        },
        "artifacts": {
            "inputs": [_artifact(path) for path in config.input_paths],
            "model_a": _artifact(config.model_a_path),
            "model_b": _artifact(config.model_b_path),
            "belief_model": _artifact(config.belief_model_path),
        },
        "comparison": {
            "labels": {"a": config.model_a_path.name, "b": config.model_b_path.name},
            "policy": {
                "overall": policy_summary,
                "by_runtime_branch": _grouped_comparison(comparisons, layer="policy", field="branch"),
                "by_source_model": _grouped_comparison(comparisons, layer="policy", field="source_model"),
                "by_actor": _grouped_comparison(comparisons, layer="policy", field="actor"),
                "by_human_outcome": _grouped_comparison(comparisons, layer="policy", field="outcome"),
            },
            "runtime": {
                "overall": runtime_summary,
                "by_branch": _grouped_comparison(comparisons, layer="runtime", field="branch"),
                "by_source_model": _grouped_comparison(comparisons, layer="runtime", field="source_model"),
                "by_actor": _grouped_comparison(comparisons, layer="runtime", field="actor"),
                "by_human_outcome": _grouped_comparison(comparisons, layer="runtime", field="outcome"),
            },
        },
        "choice_quality": {
            "overall": {label: _quality_summary(counter) for label, counter in sorted(quality.items())},
            "by_runtime_branch": {
                label: {
                    branch: _quality_summary(counter) for branch, counter in sorted(quality_by_branch[label].items())
                }
                for label in sorted(quality_by_branch)
            },
            "by_human_outcome": {
                label: {
                    outcome: _quality_summary(counter) for outcome, counter in sorted(quality_by_outcome[label].items())
                }
                for label in sorted(quality_by_outcome)
            },
        },
        "choice_transitions_a_to_b": {
            layer: {
                "overall": _transition_summary(transitions[layer]),
                "by_runtime_branch": {
                    branch: _transition_summary(counter)
                    for branch, counter in sorted(transitions_by_branch[layer].items())
                },
                "by_human_outcome": {
                    outcome: _transition_summary(counter)
                    for outcome, counter in sorted(transitions_by_outcome[layer].items())
                },
            }
            for layer in ("policy", "runtime")
        },
        "native_replay_fidelity": {
            label: fidelity_payload(counter) for label, counter in sorted(native_fidelity.items())
        },
        "runtime_metrics": {
            "a": {
                "total_decisions": runtime_a.metrics.total_decisions,
                "fallback_decisions": runtime_a.metrics.fallback_decisions,
                "search_decisions": runtime_a.metrics.search_decisions,
                "endgame_solver_decisions": runtime_a.metrics.endgame_solver_decisions,
                "failed_determinizations": runtime_a.metrics.failed_determinizations,
                "failed_rollouts": runtime_a.metrics.failed_rollouts,
                "coerced_moves": runtime_a.metrics.coerced_moves,
            },
            "b": {
                "total_decisions": runtime_b.metrics.total_decisions,
                "fallback_decisions": runtime_b.metrics.fallback_decisions,
                "search_decisions": runtime_b.metrics.search_decisions,
                "endgame_solver_decisions": runtime_b.metrics.endgame_solver_decisions,
                "failed_determinizations": runtime_b.metrics.failed_determinizations,
                "failed_rollouts": runtime_b.metrics.failed_rollouts,
                "coerced_moves": runtime_b.metrics.coerced_moves,
            },
        },
        "elapsed_seconds": elapsed,
        "seed": int(config.seed),
        "versions": {
            "code": get_code_version(),
            "rules": get_rules_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": _git_commit(),
        },
        "limitations": [
            (
                "Il replay misura differenze di scelta sulle stesse osservazioni, "
                "non quale scelta avrebbe vinto la partita."
            ),
            "Le partite provengono da giocatori umani diversi e non stimano causalmente la forza relativa v13-v14.",
            "La scelta PIMC resta campionaria; stable_disagreements richiede stabilita' in tutti i replay configurati.",
        ],
    }


def _print_summary(report: dict[str, Any]) -> None:
    """Stampa il sottoinsieme operativo del report senza dati sensibili."""
    scope = report["scope"]
    policy = report["comparison"]["policy"]["overall"]
    runtime = report["comparison"]["runtime"]["overall"]
    labels = report["comparison"]["labels"]
    print(f"=== Replay live {labels['a']} vs {labels['b']} ===")
    print(
        f"partite={scope['games']} record={scope['records']} "
        f"decisioni_non_forzate={scope['eligible_nonforced_decisions']}"
    )
    print(f"policy: agreement={policy['agreement_rate'] * 100.0:.2f}% disaccordi={policy['disagreements']}")
    print(
        f"runtime PIMC: agreement={runtime['agreement_rate'] * 100.0:.2f}% "
        f"disaccordi={runtime['disagreements']} stabili={runtime['stable_disagreements']}"
    )
    for branch, summary in report["comparison"]["runtime"]["by_branch"].items():
        print(
            f"  {branch}: n={summary['observations']} agreement={summary['agreement_rate'] * 100.0:.2f}% "
            f"disaccordi_stabili={summary['stable_disagreements']}"
        )
    print(f"tempo={report['elapsed_seconds']:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay v13-v14 sulle stesse osservazioni live")
    parser.add_argument("--input", action="append", required=True, help="Export live JSONL; opzione ripetibile.")
    parser.add_argument("--model-a", required=True, help="Prima policy .npz.")
    parser.add_argument("--model-b", required=True, help="Seconda policy .npz.")
    parser.add_argument("--belief-model", required=True, help="Belief network usata da entrambi i runtime.")
    parser.add_argument("--out-json", required=True, help="Report aggregato JSON.")
    parser.add_argument("--seed", type=int, default=20260714, help="Seed dei replay e bootstrap.")
    parser.add_argument(
        "--runtime-repeats",
        type=int,
        default=4,
        help="Replay per osservazione nella sola finestra PIMC (default 4).",
    )
    parser.add_argument("--determinizations", type=int, default=16, help="Dose PIMC comune (default 16).")
    parser.add_argument("--max-unknown-cards", type=int, default=8, help="Finestra PIMC comune (default 8).")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="Campioni bootstrap per partita.")
    args = parser.parse_args()

    config = ReplayConfig(
        input_paths=tuple(Path(path) for path in args.input),
        model_a_path=Path(args.model_a),
        model_b_path=Path(args.model_b),
        belief_model_path=Path(args.belief_model),
        out_json_path=Path(args.out_json),
        seed=int(args.seed),
        runtime_repeats=int(args.runtime_repeats),
        determinizations=int(args.determinizations),
        max_unknown_cards=int(args.max_unknown_cards),
        bootstrap_samples=int(args.bootstrap_samples),
    )
    report = run_audit(config)
    config.out_json_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(report)
    print(f"report={config.out_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
