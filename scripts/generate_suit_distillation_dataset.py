#!/usr/bin/env python3
"""Genera un dataset numerico per distillare la media simmetrica di una policy v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from briscola_ai.ai.agents import Agent, SuitSymmetrizedBCModelAgent, build_agent
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4, encode_player_observation_2p
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.ai.training.opponent_mix import parse_opponent_mix, sample_opponent_name
from briscola_ai.ai.training.suit_distillation import (
    DATASET_FORMAT,
    SuitDistillationDataset,
    make_game_split_ids,
    masked_softmax_batch,
)
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state

DEFAULT_OPPONENT_MIX = "mirror:0.50,heuristic_trump_saver:0.20,heuristic_v1:0.15,heuristic_v2:0.10,random:0.05"


@dataclass(frozen=True, slots=True)
class SuitDatasetGenerationConfig:
    """Configurazione riproducibile della raccolta teacher."""

    out_path: Path
    num_games: int
    seed: int
    opponent_mix: str = DEFAULT_OPPONENT_MIX
    temperature: float = 1.0
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    progress_every: int = 500


def _sha256_file(path: Path) -> str:
    """SHA-256 a blocchi del modello teacher."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_choice(agent: Agent, observation, *, rng: random.Random) -> int:
    """Chiede una mossa e fallisce se l'agente viola il contratto della mano."""
    card_index = int(agent.choose_card_index(observation, rng=rng))
    if not 0 <= card_index < len(observation.hand):
        raise ValueError(f"{agent.name} ha scelto indice {card_index} con mano {len(observation.hand)}")
    return card_index


def generate_suit_distillation_dataset(
    config: SuitDatasetGenerationConfig,
    *,
    base_agent: BCModelAgent,
    opponents: dict[str, Agent],
    teacher_model_sha256: str | None = None,
    game_id_offset: int = 0,
    split_by_game: np.ndarray | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> tuple[SuitDistillationDataset, dict[str, int | float | dict[str, int]]]:
    """
    Gioca partite con la policy base e un roster, etichettando ogni decisione non forzata col teacher 24x.

    Tutte le osservazioni sono `PlayerObservation`: il generatore non espone mai mazzo o
    mano avversaria al teacher. Una partita produce esattamente 38 esempi utili; le due
    mosse con una sola carta legale non forniscono gradiente e vengono escluse.

    ``game_id_offset`` e ``split_by_game`` permettono al generatore sharded di assegnare
    prima lo split globale e poi produrre shard indipendenti e riprendibili. I default
    preservano la semantica del formato monolitico storico.
    """
    if config.num_games < 3:
        raise ValueError("num_games deve essere >= 3")
    if config.temperature <= 0.0:
        raise ValueError("temperature deve essere > 0")
    if game_id_offset < 0:
        raise ValueError("game_id_offset deve essere >= 0")
    if base_agent.encoder_version != "v4":
        raise ValueError("Lo screening distilla una policy v4")
    teacher = SuitSymmetrizedBCModelAgent(base_agent)
    mix = parse_opponent_mix(config.opponent_mix)
    missing = {item.name for item in mix} - set(opponents)
    if missing:
        raise ValueError(f"Mancano agenti per il mix: {sorted(missing)}")

    max_examples = config.num_games * 38
    features = np.empty((max_examples, FEATURE_DIM_2P_V4), dtype=np.float32)
    masks = np.empty((max_examples, 40), dtype=bool)
    target_probs = np.empty((max_examples, 40), dtype=np.float32)
    target_ids = np.empty(max_examples, dtype=np.int16)
    game_ids = np.empty(max_examples, dtype=np.int32)
    split_ids = np.empty(max_examples, dtype=np.uint8)
    if split_by_game is None:
        assigned_splits = make_game_split_ids(
            config.num_games,
            seed=config.seed ^ 0x51A17,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
        )
    else:
        assigned_splits = np.asarray(split_by_game, dtype=np.uint8)
        if assigned_splits.shape != (config.num_games,):
            raise ValueError(f"split_by_game deve avere shape ({config.num_games},), ottenuto {assigned_splits.shape}")
        if not set(np.unique(assigned_splits).tolist()).issubset({0, 1, 2}):
            raise ValueError("split_by_game contiene valori diversi da train/validation/test")

    rng_games = random.Random(config.seed)
    rng_opponents = np.random.default_rng(config.seed ^ 0x0BADC0DE)
    rng_actions = (random.Random(config.seed ^ 0x13579BDF), random.Random(config.seed ^ 0x2468ACE0))
    opponent_counts: Counter[str] = Counter()
    example_index = 0
    started = time.perf_counter()

    for game_index in range(config.num_games):
        game_seed = rng_games.randrange(0, 2**32)
        opponent_name = sample_opponent_name(mix, rng=rng_opponents)
        opponent = opponents[opponent_name]
        opponent_counts[opponent_name] += 1
        base_seat = game_index % 2
        agents = (base_agent, opponent) if base_seat == 0 else (opponent, base_agent)
        state = new_game_state(2, seed=game_seed)
        game_examples = 0

        while not state.game_over:
            player = state.current_turn
            observation = make_player_observation(state, player)
            if len(observation.hand) >= 2:
                encoded = encode_player_observation_2p(observation, version="v4")
                mask = np.asarray(encoded.action_mask, dtype=bool)
                logits = teacher.symmetrized_logits(observation)
                probs = masked_softmax_batch(
                    logits[None, :],
                    mask[None, :],
                    temperature=config.temperature,
                )[0]
                features[example_index] = np.asarray(encoded.features, dtype=np.float32)
                masks[example_index] = mask
                target_probs[example_index] = probs
                target_ids[example_index] = int(np.argmax(probs))
                game_ids[example_index] = game_id_offset + game_index
                split_ids[example_index] = assigned_splits[game_index]
                example_index += 1
                game_examples += 1

            card_index = _safe_choice(agents[player], observation, rng=rng_actions[player])
            state, result = step(state, PlayCardAction(player_index=player, card_index=card_index))
            if result.error is not None:
                raise RuntimeError(f"Errore motore nella partita {game_index}: {result.error}")

        if game_examples != 38:
            raise AssertionError(f"Partita {game_index}: {game_examples} esempi, attesi 38")
        if config.progress_every > 0 and (game_index + 1) % config.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"games {game_index + 1}/{config.num_games} | examples {example_index} | "
                f"{(game_index + 1) / elapsed:.1f} games/s",
                flush=True,
            )

    if example_index != max_examples:
        raise AssertionError(f"Esempi raccolti {example_index}, attesi {max_examples}")
    split_game_counts = {
        "train": int(np.sum(assigned_splits == 0)),
        "validation": int(np.sum(assigned_splits == 1)),
        "test": int(np.sum(assigned_splits == 2)),
    }
    metadata = {
        "format": DATASET_FORMAT,
        "schema_version": 1,
        "encoder_version": "v4",
        "feature_dim": int(FEATURE_DIM_2P_V4),
        "action_dim": 40,
        "num_games": config.num_games,
        "num_examples": example_index,
        "game_id_start": game_id_offset,
        "game_id_stop": game_id_offset + config.num_games,
        "seed": config.seed,
        "teacher": teacher.name,
        "teacher_model_path": str(base_agent.model_path),
        "teacher_model_sha256": teacher_model_sha256,
        "teacher_temperature": config.temperature,
        "opponent_mix": [{"name": item.name, "prob": item.prob} for item in mix],
        "opponent_game_counts": dict(sorted(opponent_counts.items())),
        "base_seat": "alternating",
        "split_unit": "game",
        "split_game_counts": split_game_counts,
        "forced_decisions_excluded_per_game": 2,
    }
    if metadata_overrides:
        protected = {"format", "encoder_version", "feature_dim", "action_dim", "num_games", "num_examples"}
        overlap = protected & set(metadata_overrides)
        if overlap:
            raise ValueError(f"metadata_overrides non può sostituire campi strutturali: {sorted(overlap)}")
        metadata.update(metadata_overrides)
    dataset = SuitDistillationDataset(
        features=features,
        action_masks=masks,
        target_probs=target_probs,
        target_action_ids=target_ids,
        game_ids=game_ids,
        split_ids=split_ids,
        metadata=metadata,
    )
    dataset.validate()
    elapsed = time.perf_counter() - started
    counters: dict[str, int | float | dict[str, int]] = {
        "games": config.num_games,
        "examples": example_index,
        "elapsed_seconds_before_save": elapsed,
        "games_per_second": config.num_games / elapsed,
        "opponent_game_counts": dict(sorted(opponent_counts.items())),
        "split_game_counts": split_game_counts,
    }
    return dataset, counters


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/best_a2c_v13.npz", help="Policy v4 da simmetrizzare")
    parser.add_argument("--out", required=True, help="Dataset `.npz` output")
    parser.add_argument("--num-games", type=int, default=10_000, help="Partite da raccogliere")
    parser.add_argument("--seed", type=int, default=20260711, help="Seed raccolta e split")
    parser.add_argument("--opponent-mix", default=DEFAULT_OPPONENT_MIX, help="Mix name:weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperatura target soft")
    parser.add_argument("--progress-every", type=int, default=500, help="Frequenza log; 0 disabilita")
    args = parser.parse_args()

    model_path = Path(args.model)
    base_agent = BCModelAgent.from_npz(model_path)
    mix = parse_opponent_mix(args.opponent_mix)
    opponents: dict[str, Agent] = {}
    for item in mix:
        if item.name == "mirror":
            opponents[item.name] = base_agent
        else:
            opponents[item.name] = build_agent(item.name)

    config = SuitDatasetGenerationConfig(
        out_path=Path(args.out),
        num_games=args.num_games,
        seed=args.seed,
        opponent_mix=args.opponent_mix,
        temperature=args.temperature,
        progress_every=args.progress_every,
    )
    dataset, counters = generate_suit_distillation_dataset(
        config,
        base_agent=base_agent,
        opponents=opponents,
        teacher_model_sha256=_sha256_file(model_path),
    )
    save_started = time.perf_counter()
    dataset.save(config.out_path)
    counters["save_seconds"] = time.perf_counter() - save_started
    counters["output_bytes"] = config.out_path.stat().st_size
    print(json.dumps(counters, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
