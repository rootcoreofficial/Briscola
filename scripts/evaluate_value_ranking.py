#!/usr/bin/env python3
"""
Valuta un value model contro i valori per-carta prodotti dal teacher PIMC.

Questo script e' il gate primario dello Stage 0 V-lookahead: non misura ancora un agente
deployabile, ma risponde alla domanda piu' economica e importante:

    "Se uso V per valutare le foglie dopo una mossa, ordino le carte come PIMC?"

La valutazione resta anti-cheat: parte da `ObservationDTO`, campiona stati compatibili con
`determinize_observation`, applica una carta legale, risolve la presa corrente con l'agente
di continuazione dichiarato, poi valuta la foglia con `V` dal punto di vista del giocatore
che deve muovere in quella foglia. Se quel giocatore non e' il root player, il segno viene
invertito per tornare alla prospettiva della decisione originale.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.agents.pimc import determinize_observation, rollout_to_terminal
from briscola_ai.ai.encoding.observation_encoder import encode_observation_2p_with_version
from briscola_ai.ai.models import infer_value_encoder_version, load_value_model_npz
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import GameState


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Configurazione riproducibile della valutazione ranking."""

    data_path: Path
    value_model_path: Path
    continuation_agent: str
    continuation_model_path: Path | None
    determinizations: int
    max_records: int | None
    seed: int
    min_pair_margin: float
    strong_margin_min: float
    reliable_margin_ci_low_min: float


def _iter_jsonl(path: Path):
    """Itera record JSONL non vuoti."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _card_dto_to_domain(card: dict[str, Any]) -> Card:
    """Converte una carta DTO JSON nel modello dominio."""
    suit = card.get("suit")
    number = card.get("number")
    if not isinstance(suit, str) or not isinstance(number, int):
        raise ValueError(f"CardDTO invalido: suit={suit!r} number={number!r}")
    try:
        rank = next(rank for rank in Rank if int(rank.number) == int(number))
    except StopIteration as exc:
        raise ValueError(f"Numero carta fuori range: {number}") from exc
    return Card(suit=Suit(suit), rank=rank)


def _observation_dto_to_player_observation(obs: dict[str, Any]) -> PlayerObservation:
    """
    Ricostruisce `PlayerObservation` da ObservationDTO PIMC.

    Supporta il caso diagnostico atteso: 2-player e `my_turn=true`. Non ricostruisce carte
    nascoste; usa solo mano, tavolo, punteggi, dimensioni e one-hot pubbliche.
    """
    if obs.get("num_players") != 2:
        raise ValueError("evaluate_value_ranking supporta solo observation 2-player")
    if not bool(obs.get("my_turn")):
        raise ValueError("Il dataset diagnostico deve contenere decision-state con my_turn=true")

    player_index = int(obs.get("my_index", 0))
    players = obs.get("players") or []
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("ObservationDTO senza lista players 2-player valida")

    table_cards: list[tuple[Card, int]] = []
    for item in obs.get("table_cards") or []:
        if not isinstance(item, dict) or not isinstance(item.get("card"), dict):
            continue
        table_cards.append((_card_dto_to_domain(item["card"]), int(item["player_index"])))

    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=player_index,
        player_name=str(players[player_index].get("name", f"player_{player_index}")),
        hand=tuple(_card_dto_to_domain(card) for card in (obs.get("my_hand") or [])),
        trump_card=_card_dto_to_domain(obs["trump_card"]) if isinstance(obs.get("trump_card"), dict) else None,
        deck_size=int(obs.get("cards_remaining_in_deck", 0)),
        table_cards=tuple(table_cards),
        current_turn=player_index,
        first_player=table_cards[0][1] if table_cards else player_index,
        game_over=bool(obs.get("game_over", False)),
        winner_index=None,
        winning_team=None,
        players_points=tuple(int(player.get("points", 0)) for player in players),
        players_hand_sizes=tuple(int(player.get("hand_size", 0)) for player in players),
        seen_cards_onehot=tuple(int(v) for v in (obs.get("seen_cards_onehot") or [0] * 40)),
        out_of_play_cards_onehot=tuple(int(v) for v in (obs.get("out_of_play_cards_onehot") or [0] * 40)),
    )


def _build_continuation_agent(config: RankingConfig) -> Agent:
    """Costruisce la policy che risolve la presa corrente dopo la carta candidata."""
    if config.continuation_model_path is not None:
        return build_agent(config.continuation_agent, model_path=config.continuation_model_path)
    return build_agent(config.continuation_agent)


def _score_delta_for_player(state: GameState, player_index: int) -> int:
    """Delta punti dal punto di vista di `player_index` in 2-player."""
    opponent = 1 - int(player_index)
    return int(state.players[player_index].points) - int(state.players[opponent].points)


def _safe_card_index(agent: Agent, observation: PlayerObservation, *, rng: random.Random) -> int:
    """Chiede una mossa alla policy di continuazione e la valida."""
    card_index = int(agent.choose_card_index(observation, rng=rng))
    if not 0 <= card_index < len(observation.hand):
        raise ValueError(f"Agente {agent.name!r} ha prodotto card_index={card_index}, mano={len(observation.hand)}")
    return card_index


def _resolve_current_trick(
    state: GameState,
    *,
    continuation_agent: Agent,
    rng: random.Random,
) -> GameState:
    """
    Porta lo stato al prossimo decision boundary dopo la carta candidata.

    Se la carta candidata ha aperto una presa, facciamo rispondere l'avversario con la policy
    di continuazione. Se invece ha chiuso la presa, `step` ha gia' risolto il trick.
    """
    if state.game_over:
        return state
    if len(state.table_cards) != 1:
        return state

    current = state.current_turn
    observation = make_player_observation(state, current)
    card_index = _safe_card_index(continuation_agent, observation, rng=rng)
    next_state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
    if result.error:
        raise RuntimeError(f"Errore dominio durante risposta di continuazione: {result.error}")
    return next_state


def _value_leaf_score_for_root(
    leaf_state: GameState,
    *,
    root_player: int,
    value_model,
    encoder_version,
    continuation_agent: Agent,
    rng: random.Random,
) -> float:
    """
    Valuta la foglia in punti dal punto di vista del root player.

    Il value model e' allenato su stati in cui il giocatore osservante deve muovere. Per restare
    in distribuzione valutiamo quindi `leaf_state.current_turn`; se non e' il root player,
    cambiamo segno alla predizione.

    Eccezione endgame: a mazzo vuoto l'agente deployato (e i rollout PIMC che generano i
    `mean_score`) usano il solver esatto, non `V`. Per rendere il gate fedele allo Stage 1,
    completiamo la foglia endgame col solver invece di interrogare `V`.
    """
    if leaf_state.game_over:
        return float(_score_delta_for_player(leaf_state, root_player))

    if len(leaf_state.deck) == 0:
        terminal = rollout_to_terminal(
            leaf_state,
            rollout_agent=continuation_agent,
            rng=rng,
            use_endgame_solver=True,
        )
        return float(_score_delta_for_player(terminal, root_player))

    leaf_player = int(leaf_state.current_turn)
    observation_dto = build_observation_dto(leaf_state, player_index=leaf_player, server_version=0).model_dump(
        mode="json"
    )
    encoded = encode_observation_2p_with_version(observation_dto, version=encoder_version)
    current_delta = float(_score_delta_for_player(leaf_state, leaf_player))
    pred_for_leaf_player = float(
        value_model.predict_points(np.asarray(encoded.features, dtype=np.float32), current_score_delta=current_delta)
    )
    return pred_for_leaf_player if leaf_player == root_player else -pred_for_leaf_player


def _candidate_scores_with_value(
    observation: PlayerObservation,
    *,
    legal_indices: list[int],
    continuation_agent: Agent,
    value_model,
    encoder_version,
    determinizations: int,
    rng: random.Random,
) -> dict[int, float]:
    """Stima un valore medio per ogni carta legale usando determinizzazioni compatibili."""
    totals = {card_index: 0.0 for card_index in legal_indices}
    counts = {card_index: 0 for card_index in legal_indices}

    for sample_idx in range(max(1, int(determinizations))):
        sample_rng = random.Random(rng.randrange(0, 2**32) ^ (sample_idx * 0x9E3779B9))
        sampled_state = determinize_observation(observation, rng=sample_rng)
        if sampled_state.current_turn != observation.player_index:
            raise RuntimeError("Determinizzazione incoerente: current_turn diverso dal player osservante")

        for card_index in legal_indices:
            next_state, result = step(
                sampled_state,
                PlayCardAction(player_index=sampled_state.current_turn, card_index=card_index),
            )
            if result.error:
                continue
            rollout_rng = random.Random(sample_rng.randrange(0, 2**32) ^ (card_index * 0x85EBCA6B))
            leaf_state = _resolve_current_trick(next_state, continuation_agent=continuation_agent, rng=rollout_rng)
            score = _value_leaf_score_for_root(
                leaf_state,
                root_player=observation.player_index,
                value_model=value_model,
                encoder_version=encoder_version,
                continuation_agent=continuation_agent,
                rng=rollout_rng,
            )
            totals[card_index] += float(score)
            counts[card_index] += 1

    return {card_index: totals[card_index] / counts[card_index] for card_index in legal_indices if counts[card_index]}


def _pimc_action_values(record: dict[str, Any]) -> dict[int, float]:
    """Estrae mean_score PIMC per card_index dal record diagnostico."""
    teacher = record.get("teacher")
    if not isinstance(teacher, dict) or teacher.get("decision_type") != "search":
        return {}
    diagnostics = teacher.get("search_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    raw_values = diagnostics.get("action_values")
    if not isinstance(raw_values, list):
        return {}
    out: dict[int, float] = {}
    for item in raw_values:
        if not isinstance(item, dict):
            continue
        mean_score = item.get("mean_score")
        rollout_count = item.get("rollout_count", 0)
        if not isinstance(mean_score, (int, float)) or int(rollout_count) <= 0:
            continue
        out[int(item["card_index"])] = float(mean_score)
    return out


def _diagnostic_float(record: dict[str, Any], key: str) -> float | None:
    """Legge un campo float da `teacher.search_diagnostics`."""
    teacher = record.get("teacher")
    diagnostics = teacher.get("search_diagnostics") if isinstance(teacher, dict) else None
    if not isinstance(diagnostics, dict):
        return None
    value = diagnostics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _best_key(values: dict[int, float]) -> int | None:
    """Carta migliore con tie-break stabile uguale a PIMC: valore alto, poi indice piu' basso."""
    if not values:
        return None
    return max(values, key=lambda card_index: (values[card_index], -card_index))


def evaluate_value_ranking(config: RankingConfig) -> dict[str, Any]:
    """Esegue la valutazione ranking-vs-PIMC e ritorna un summary JSON-serializzabile."""
    if config.determinizations <= 0:
        raise ValueError("--determinizations deve essere > 0")

    value_model = load_value_model_npz(config.value_model_path)
    encoder_version = infer_value_encoder_version(value_model)
    continuation_agent = _build_continuation_agent(config)
    rng = random.Random(config.seed)

    counters: dict[str, int] = {
        "records_seen": 0,
        "records_search": 0,
        "records_evaluated": 0,
        "records_strong_reliable": 0,
        "records_skipped": 0,
        "records_failed": 0,
        "top1_total": 0,
        "top1_match": 0,
        "top1_strong_total": 0,
        "top1_strong_match": 0,
        "reference_top1_total": 0,
        "reference_top1_match": 0,
        "reference_top1_strong_total": 0,
        "reference_top1_strong_match": 0,
        "pair_total": 0,
        "pair_match": 0,
        "pair_strong_total": 0,
        "pair_strong_match": 0,
    }
    errors: list[str] = []

    for record in _iter_jsonl(config.data_path):
        counters["records_seen"] += 1
        if config.max_records is not None and counters["records_search"] >= int(config.max_records):
            break

        pimc_values = _pimc_action_values(record)
        if len(pimc_values) < 2:
            counters["records_skipped"] += 1
            continue
        counters["records_search"] += 1

        try:
            observation = _observation_dto_to_player_observation(record["observation"])
            legal_indices = sorted(card for card in pimc_values if 0 <= card < len(observation.hand))
            if len(legal_indices) < 2:
                counters["records_skipped"] += 1
                continue
            value_scores = _candidate_scores_with_value(
                observation,
                legal_indices=legal_indices,
                continuation_agent=continuation_agent,
                value_model=value_model,
                encoder_version=encoder_version,
                determinizations=config.determinizations,
                rng=rng,
            )
        except Exception as exc:
            counters["records_failed"] += 1
            if len(errors) < 10:
                errors.append(f"{record.get('game_id', '?')}#{record.get('event_id', '?')}: {exc}")
            continue

        common_cards = sorted(set(pimc_values) & set(value_scores))
        if len(common_cards) < 2:
            counters["records_skipped"] += 1
            continue

        counters["records_evaluated"] += 1
        margin = _diagnostic_float(record, "margin")
        ci_low = _diagnostic_float(record, "margin_ci95_low")
        is_strong_reliable = (
            margin is not None
            and ci_low is not None
            and margin >= float(config.strong_margin_min)
            and ci_low >= float(config.reliable_margin_ci_low_min)
        )
        if is_strong_reliable:
            counters["records_strong_reliable"] += 1

        best_pimc = _best_key({card: pimc_values[card] for card in common_cards})
        best_value = _best_key({card: value_scores[card] for card in common_cards})
        if best_pimc is not None and best_value is not None:
            counters["top1_total"] += 1
            if best_pimc == best_value:
                counters["top1_match"] += 1
            if is_strong_reliable:
                counters["top1_strong_total"] += 1
                if best_pimc == best_value:
                    counters["top1_strong_match"] += 1

            reference = record.get("reference")
            if isinstance(reference, dict) and isinstance(reference.get("card_index"), int):
                ref_match = int(reference["card_index"]) == best_pimc
                counters["reference_top1_total"] += 1
                if ref_match:
                    counters["reference_top1_match"] += 1
                # Baseline v6 ristretto agli STESSI casi forti, per un confronto onesto con
                # top1_strong_reliable_accuracy (stessa popolazione, non tutti i record).
                if is_strong_reliable:
                    counters["reference_top1_strong_total"] += 1
                    if ref_match:
                        counters["reference_top1_strong_match"] += 1

        for idx, card_a in enumerate(common_cards):
            for card_b in common_cards[idx + 1 :]:
                pimc_diff = float(pimc_values[card_a] - pimc_values[card_b])
                value_diff = float(value_scores[card_a] - value_scores[card_b])
                if abs(pimc_diff) <= float(config.min_pair_margin) or value_diff == 0.0:
                    continue
                same_order = (pimc_diff > 0.0 and value_diff > 0.0) or (pimc_diff < 0.0 and value_diff < 0.0)
                counters["pair_total"] += 1
                if same_order:
                    counters["pair_match"] += 1
                if abs(pimc_diff) >= float(config.strong_margin_min):
                    counters["pair_strong_total"] += 1
                    if same_order:
                        counters["pair_strong_match"] += 1

    def rate(num: int, den: int) -> float | None:
        return (float(num) / float(den)) if den else None

    return {
        "config": {
            "data_path": str(config.data_path),
            "value_model_path": str(config.value_model_path),
            "continuation_agent": config.continuation_agent,
            "continuation_model_path": str(config.continuation_model_path)
            if config.continuation_model_path is not None
            else None,
            "determinizations": int(config.determinizations),
            "max_records": config.max_records,
            "seed": int(config.seed),
            "min_pair_margin": float(config.min_pair_margin),
            "strong_margin_min": float(config.strong_margin_min),
            "reliable_margin_ci_low_min": float(config.reliable_margin_ci_low_min),
        },
        "model": {
            "encoder_version": encoder_version,
            "feature_dim": int(value_model.feature_dim),
            "hidden_dim": int(value_model.hidden_dim),
            "target": value_model.metadata.get("target"),
        },
        "counts": counters,
        "metrics": {
            "pairwise_accuracy": rate(counters["pair_match"], counters["pair_total"]),
            "pairwise_strong_accuracy": rate(counters["pair_strong_match"], counters["pair_strong_total"]),
            "top1_accuracy": rate(counters["top1_match"], counters["top1_total"]),
            "top1_strong_reliable_accuracy": rate(counters["top1_strong_match"], counters["top1_strong_total"]),
            "reference_top1_accuracy": rate(counters["reference_top1_match"], counters["reference_top1_total"]),
            "reference_top1_strong_reliable_accuracy": rate(
                counters["reference_top1_strong_match"], counters["reference_top1_strong_total"]
            ),
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta ranking value-model vs diagnostica PIMC")
    parser.add_argument("--data", required=True, help="JSONL pimc_teacher con teacher.search_diagnostics")
    parser.add_argument("--value-model", required=True, help="Path value model .npz")
    parser.add_argument("--continuation-agent", default="bc_model_hybrid_endgame")
    parser.add_argument("--continuation-model", default="", help="Path modello .npz per la policy di continuazione")
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0, help="0 = tutti i record search disponibili")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pair-margin", type=float, default=0.0)
    parser.add_argument("--strong-margin-min", type=float, default=2.0)
    parser.add_argument("--reliable-margin-ci-low-min", type=float, default=0.0)
    args = parser.parse_args()

    model = str(args.continuation_model).strip()
    config = RankingConfig(
        data_path=Path(args.data),
        value_model_path=Path(args.value_model),
        continuation_agent=str(args.continuation_agent),
        continuation_model_path=Path(model) if model else None,
        determinizations=int(args.determinizations),
        max_records=int(args.max_records) if int(args.max_records) > 0 else None,
        seed=int(args.seed),
        min_pair_margin=float(args.min_pair_margin),
        strong_margin_min=float(args.strong_margin_min),
        reliable_margin_ci_low_min=float(args.reliable_margin_ci_low_min),
    )
    print(json.dumps(evaluate_value_ranking(config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
