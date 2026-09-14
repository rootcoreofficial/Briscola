#!/usr/bin/env python3
"""
Converte diagnostica PIMC root-level in dataset value leaf-level decision-aligned.

Il dataset PIMC teacher salva, per ogni posizione root, il valore medio stimato per ogni
carta giocabile (`search_diagnostics.action_values`). Questo script campiona determinizzazioni
compatibili con la root, applica ogni carta candidata, porta lo stato al decision boundary
successivo e salva l'osservazione della foglia con target pari al valore PIMC della carta.
Ogni foglia conserva anche il `game_id` della root sorgente, così root diverse della stessa
partita non possono essere separate fra train, validation e test.

Differenza rispetto a `generate_value_dataset_numba.py`:
- quel dataset predice l'esito di continuazione della policy base;
- questo dataset insegna a `V` a ordinare foglie come la search PIMC.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents import HybridEndgameAgent
from briscola_ai.ai.agents.pimc import determinize_observation
from briscola_ai.ai.encoding.observation_encoder import encode_player_observation_2p
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import GameState


def _iter_jsonl(path: Path):
    """Itera righe JSONL non vuote."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _card_from_dto(raw: dict[str, Any]) -> Card:
    """Converte CardDTO JSON in Card di dominio."""
    return Card(Suit(str(raw["suit"])), Rank[str(raw["rank"])])


def _observation_from_dto(raw: dict[str, Any]) -> PlayerObservation:
    """Ricostruisce una `PlayerObservation` da ObservationDTO JSON salvata nel dataset."""
    player_index = int(raw["my_index"])
    players = sorted(raw.get("players") or [], key=lambda item: int(item["index"]))
    if len(players) != 2:
        raise ValueError("Dataset leaf PIMC supporta solo osservazioni 2-player")
    table_cards = tuple(
        (_card_from_dto(item["card"]), int(item["player_index"])) for item in raw.get("table_cards") or []
    )
    first_player = int(table_cards[0][1]) if table_cards else player_index
    return PlayerObservation(
        num_players=int(raw["num_players"]),
        is_team_game=bool(raw.get("is_team_game", False)),
        teams=None,
        player_index=player_index,
        player_name=str(players[player_index].get("name", f"P{player_index}")),
        hand=tuple(_card_from_dto(card) for card in raw.get("my_hand") or []),
        trump_card=_card_from_dto(raw["trump_card"]) if raw.get("trump_card") is not None else None,
        deck_size=int(raw["cards_remaining_in_deck"]),
        table_cards=table_cards,
        current_turn=player_index,
        first_player=first_player,
        game_over=bool(raw.get("game_over", False)),
        winner_index=None,
        winning_team=None,
        players_points=tuple(int(player["points"]) for player in players),
        players_hand_sizes=tuple(int(player["hand_size"]) for player in players),
        seen_cards_onehot=tuple(int(value) for value in raw.get("seen_cards_onehot") or [0] * 40),
        out_of_play_cards_onehot=tuple(int(value) for value in raw.get("out_of_play_cards_onehot") or [0] * 40),
    )


def _score_delta_for_player(state: GameState, player_index: int) -> int:
    """Delta punti dal punto di vista del player indicato."""
    return int(state.players[player_index].points) - int(state.players[1 - player_index].points)


def _safe_card_index(agent, observation: PlayerObservation, *, rng: random.Random) -> int:
    """Chiede una mossa alla continuation e valida l'indice mano."""
    card_index = int(agent.choose_card_index(observation, rng=rng))
    if not 0 <= card_index < len(observation.hand):
        raise ValueError(f"Agente {agent.name!r} ha prodotto card_index={card_index}, mano={len(observation.hand)}")
    return card_index


def _resolve_current_trick(state: GameState, *, continuation_agent, rng: random.Random) -> GameState:
    """Replica il boundary usato da ValueLookaheadAgent prima di interrogare `V`."""
    if state.game_over or len(state.table_cards) != 1:
        return state
    current = int(state.current_turn)
    observation = make_player_observation(state, current)
    card_index = _safe_card_index(continuation_agent, observation, rng=rng)
    next_state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
    if result.error:
        raise RuntimeError(f"Errore dominio durante continuation leaf dataset: {result.error}")
    return next_state


def _search_diagnostics(record: dict[str, Any]) -> dict[str, Any] | None:
    teacher = record.get("teacher")
    if not isinstance(teacher, dict) or teacher.get("decision_type") != "search":
        return None
    diagnostics = teacher.get("search_diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else None


def _action_values(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    values = diagnostics.get("action_values")
    if not isinstance(values, list):
        return []
    out: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        if item.get("mean_score") is None:
            continue
        out.append(item)
    return out


def generate_pimc_leaf_value_dataset(
    *,
    data_path: Path,
    policy_model_path: Path,
    out_path: Path,
    max_roots: int,
    samples_per_root: int,
    seed: int,
    min_margin: float,
    min_margin_ci_low: float,
    feature_dtype: str,
) -> dict[str, Any]:
    """Genera dataset `.npz` leaf-value da PIMC action-values."""
    if max_roots <= 0:
        raise ValueError("--max-roots deve essere > 0")
    if samples_per_root <= 0:
        raise ValueError("--samples-per-root deve essere > 0")
    if feature_dtype not in {"float16", "float32"}:
        raise ValueError("--feature-dtype deve essere float16 o float32")

    policy = BCModelAgent.from_npz(policy_model_path)
    continuation = HybridEndgameAgent(fallback=policy, name=f"control_solver({policy_model_path.name})")
    rng = random.Random(int(seed))
    dtype = np.float16 if feature_dtype == "float16" else np.float32

    xs: list[np.ndarray] = []
    target_residual_scaled: list[float] = []
    current_delta: list[float] = []
    target_final_delta: list[float] = []
    root_sign: list[int] = []
    root_id: list[int] = []
    game_ids: list[str] = []
    card_index: list[int] = []
    action_value: list[float] = []
    margin: list[float] = []

    counters: dict[str, Any] = {
        "records_seen": 0,
        "records_search": 0,
        "roots_used": 0,
        "roots_skipped_margin": 0,
        "roots_skipped_invalid": 0,
        "leaf_records_written": 0,
        "leaf_records_skipped_terminal_or_endgame": 0,
        "leaf_records_skipped_error": 0,
    }
    started = time.perf_counter()

    for record in _iter_jsonl(data_path):
        counters["records_seen"] += 1
        diagnostics = _search_diagnostics(record)
        if diagnostics is None:
            continue
        counters["records_search"] += 1
        root_margin = float(diagnostics.get("margin") or 0.0)
        root_ci_low = float(diagnostics.get("margin_ci95_low") or 0.0)
        if root_margin < float(min_margin) or root_ci_low < float(min_margin_ci_low):
            counters["roots_skipped_margin"] += 1
            continue
        values = _action_values(diagnostics)
        if len(values) < 2:
            counters["roots_skipped_invalid"] += 1
            continue
        try:
            observation = _observation_from_dto(record["observation"])
        except KeyError, TypeError, ValueError:
            counters["roots_skipped_invalid"] += 1
            continue
        source_game_id = record.get("game_id")
        if not isinstance(source_game_id, str):
            raise ValueError(
                "Record PIMC valido senza game_id stringa: "
                "rigenera il teacher dataset per consentire lo split per partita."
            )
        if not source_game_id.strip():
            raise ValueError("Record PIMC valido con game_id vuoto")

        root_group_id = int(counters["roots_used"])
        root_rows_before = len(xs)
        for sample_idx in range(int(samples_per_root)):
            sample_rng = random.Random(rng.randrange(0, 2**32) ^ (sample_idx * 0x9E3779B9))
            try:
                sampled_state = determinize_observation(observation, rng=sample_rng)
            except ValueError:
                counters["roots_skipped_invalid"] += 1
                continue
            for item in values:
                local_card_index = int(item["card_index"])
                mean_score_root = float(item["mean_score"])
                try:
                    next_state, result = step(
                        sampled_state,
                        PlayCardAction(player_index=sampled_state.current_turn, card_index=local_card_index),
                    )
                    if result.error:
                        raise RuntimeError(result.error)
                    leaf_rng = random.Random(sample_rng.randrange(0, 2**32) ^ (local_card_index * 0x85EBCA6B))
                    leaf_state = _resolve_current_trick(next_state, continuation_agent=continuation, rng=leaf_rng)
                except RuntimeError, ValueError:
                    counters["leaf_records_skipped_error"] += 1
                    continue

                if leaf_state.game_over or len(leaf_state.deck) == 0:
                    counters["leaf_records_skipped_terminal_or_endgame"] += 1
                    continue

                leaf_player = int(leaf_state.current_turn)
                leaf_observation = make_player_observation(leaf_state, leaf_player)
                encoded = encode_player_observation_2p(leaf_observation, version="v3")
                sign = 1 if leaf_player == observation.player_index else -1
                target_final_leaf = float(sign) * mean_score_root
                current_leaf = float(_score_delta_for_player(leaf_state, leaf_player))

                xs.append(np.asarray(encoded.features, dtype=np.float32))
                current_delta.append(current_leaf)
                target_final_delta.append(target_final_leaf)
                target_residual_scaled.append((target_final_leaf - current_leaf) / 120.0)
                root_sign.append(int(sign))
                root_id.append(root_group_id * int(samples_per_root) + sample_idx)
                game_ids.append(source_game_id)
                card_index.append(local_card_index)
                action_value.append(mean_score_root)
                margin.append(root_margin)

        if len(xs) > root_rows_before:
            counters["roots_used"] += 1
        if int(counters["roots_used"]) >= int(max_roots):
            break

    if not xs:
        raise ValueError("Nessuna leaf valida generata")

    x = np.asarray(xs, dtype=np.float32).astype(dtype, copy=False)
    current_arr = np.asarray(current_delta, dtype=np.float32)
    final_arr = np.asarray(target_final_delta, dtype=np.float32)
    root_sign_arr = np.asarray(root_sign, dtype=np.int8)
    action_value_arr = np.asarray(action_value, dtype=np.float32)
    metadata = {
        "format": "pimc_leaf_value_dataset_v2",
        "schema_version": 2,
        "dataset_kind": "pimc_leaf_value",
        "source_path": str(data_path),
        "policy_model_path": str(policy_model_path),
        "encoder_version": "v3",
        "target": "residual",
        "target_scale": 120.0,
        "max_roots": int(max_roots),
        "samples_per_root": int(samples_per_root),
        "seed": int(seed),
        "min_margin": float(min_margin),
        "min_margin_ci_low": float(min_margin_ci_low),
        "feature_dtype": str(feature_dtype),
        "split_unit": "game",
        "split_group_key": "game_ids",
    }
    counters["leaf_records_written"] = int(x.shape[0])
    counters["elapsed_seconds"] = time.perf_counter() - started

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        x=x,
        y=np.asarray(target_residual_scaled, dtype=np.float32),
        current_delta=current_arr,
        final_delta=final_arr,
        root_sign=root_sign_arr,
        root_id=np.asarray(root_id, dtype=np.int64),
        game_ids=np.asarray(game_ids, dtype=np.str_),
        card_index=np.asarray(card_index, dtype=np.int64),
        action_value=action_value_arr,
        margin=np.asarray(margin, dtype=np.float32),
        metadata_json=json.dumps({**metadata, "summary": counters}, ensure_ascii=False, indent=2),
    )
    return {**metadata, **counters, "out_path": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera dataset leaf-value allineato alle decisioni PIMC.")
    parser.add_argument("--data", required=True, help="JSONL PIMC teacher con search_diagnostics.action_values")
    parser.add_argument("--policy-model", required=True, help="Policy .npz usata come continuation leaf")
    parser.add_argument("--out", required=True, help="Path output .npz")
    parser.add_argument("--max-roots", type=int, default=50000)
    parser.add_argument("--samples-per-root", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-margin", type=float, default=2.0)
    parser.add_argument("--min-margin-ci-low", type=float, default=0.0)
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()
    summary = generate_pimc_leaf_value_dataset(
        data_path=Path(args.data),
        policy_model_path=Path(args.policy_model),
        out_path=Path(args.out),
        max_roots=int(args.max_roots),
        samples_per_root=int(args.samples_per_root),
        seed=int(args.seed),
        min_margin=float(args.min_margin),
        min_margin_ci_low=float(args.min_margin_ci_low),
        feature_dtype=str(args.feature_dtype),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
