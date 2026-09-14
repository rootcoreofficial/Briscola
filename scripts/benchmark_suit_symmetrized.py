#!/usr/bin/env python3
"""Misura il costo di inference della media sulle 24 rinomine dei semi."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents.suit_symmetrized import SuitSymmetrizedBCModelAgent
from briscola_ai.ai.encoding.observation_encoder import encode_player_observation_2p
from briscola_ai.ai.evaluation.suit_symmetry import (
    all_suit_permutations,
    inverse_suit_permutation,
    permute_action_vector,
    permute_player_observation,
)
from briscola_ai.ai.models.bc_model import BCModelAgent
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import new_game_state


def _collect_observations(*, count: int, seed: int) -> list[PlayerObservation]:
    """Raccoglie osservazioni reali distribuite lungo più partite e fasi."""
    observations: list[PlayerObservation] = []
    game_number = 0
    while len(observations) < count:
        state = new_game_state(2, seed=seed + game_number)
        action_number = 0
        while not state.game_over and len(observations) < count:
            observation = make_player_observation(state, state.current_turn)
            observations.append(observation)
            card_index = (action_number * 7 + state.current_turn) % len(observation.hand)
            state, result = step(
                state,
                PlayCardAction(player_index=state.current_turn, card_index=card_index),
            )
            if result.error is not None:
                raise AssertionError(result.error)
            action_number += 1
        game_number += 1
    return observations


def _time_agent(agent: Any, observations: Sequence[PlayerObservation], *, decisions: int) -> dict[str, float]:
    """Cronometra singole decisioni dopo warm-up e restituisce statistiche in millisecondi."""
    rng = random.Random(0)
    warmup_count = min(48, len(observations))
    for observation in observations[:warmup_count]:
        agent.choose_card_index(observation, rng=rng)

    samples_ns = np.empty(decisions, dtype=np.int64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(decisions):
            observation = observations[index % len(observations)]
            start = time.perf_counter_ns()
            agent.choose_card_index(observation, rng=rng)
            samples_ns[index] = time.perf_counter_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()

    samples_ms = samples_ns.astype(np.float64) / 1_000_000.0
    total_seconds = float(np.sum(samples_ns)) / 1_000_000_000.0
    return {
        "mean_ms": float(np.mean(samples_ms)),
        "median_ms": float(np.median(samples_ms)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "decisions_per_second": decisions / total_seconds,
    }


def _check_equivariance(
    agent: SuitSymmetrizedBCModelAgent,
    observations: Sequence[PlayerObservation],
) -> dict[str, float | int]:
    """Misura flip e residuo numerico sulle 23 rinomine non banali di ogni osservazione."""
    comparisons = 0
    action_flips = 0
    max_abs_logits_delta = 0.0
    for observation in observations:
        baseline_logits = agent.symmetrized_logits(observation)
        mask = np.asarray(
            encode_player_observation_2p(observation, version=agent.base_agent.encoder_version).action_mask,
            dtype=bool,
        )
        baseline_action_id = int(np.argmax(np.where(mask, baseline_logits, -np.inf)))
        for permutation in all_suit_permutations()[1:]:
            transformed = permute_player_observation(observation, permutation)
            transformed_logits = agent.symmetrized_logits(transformed)
            remapped_logits = np.asarray(
                permute_action_vector(transformed_logits, inverse_suit_permutation(permutation)),
                dtype=np.float64,
            )
            remapped_action_id = int(np.argmax(np.where(mask, remapped_logits, -np.inf)))
            action_flips += int(remapped_action_id != baseline_action_id)
            max_abs_logits_delta = max(
                max_abs_logits_delta,
                float(np.max(np.abs(remapped_logits - baseline_logits))),
            )
            comparisons += 1
    return {
        "observations": len(observations),
        "nonidentity_comparisons": comparisons,
        "action_flips": action_flips,
        "flip_rate": action_flips / comparisons,
        "max_abs_logits_delta": max_abs_logits_delta,
    }


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path alla policy `.npz` da misurare")
    parser.add_argument("--observations", type=int, default=160, help="Osservazioni reali distinte da raccogliere")
    parser.add_argument("--decisions", type=int, default=2_000, help="Decisioni cronometrate per agente")
    parser.add_argument("--seed", type=int, default=20260711, help="Seed della raccolta osservazioni")
    parser.add_argument("--out-json", default="", help="Path JSON opzionale per il risultato")
    args = parser.parse_args()
    if args.observations <= 0 or args.decisions <= 0:
        raise ValueError("--observations e --decisions devono essere positivi")

    base_agent = BCModelAgent.from_npz(args.model)
    symmetrized_agent = SuitSymmetrizedBCModelAgent(base_agent)
    observations = _collect_observations(count=args.observations, seed=args.seed)

    # Alterniamo l'ordine una volta per limitare l'effetto di clock e cache nella misura breve.
    base_first = _time_agent(base_agent, observations, decisions=args.decisions)
    symmetrized = _time_agent(symmetrized_agent, observations, decisions=args.decisions)
    base_second = _time_agent(base_agent, observations, decisions=args.decisions)
    base = {key: (base_first[key] + base_second[key]) / 2.0 for key in base_first}

    payload = {
        "schema_version": 1,
        "model": str(Path(args.model)),
        "seed": args.seed,
        "observations": len(observations),
        "decisions_per_agent": args.decisions,
        "equivariance": _check_equivariance(symmetrized_agent, observations),
        "base": base,
        "symmetrized_24x_batch": symmetrized,
        "ratios": {
            "mean_latency": symmetrized["mean_ms"] / base["mean_ms"],
            "p95_latency": symmetrized["p95_ms"] / base["p95_ms"],
            "throughput": symmetrized["decisions_per_second"] / base["decisions_per_second"],
        },
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out_json.strip():
        output_path = Path(args.out_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
