#!/usr/bin/env python3
"""
Genera un dataset JSONL per allenare una rete di valore `V(observation)`.

Lo script produce righe `(observation -> final_score_delta)` da self-play 2-player. Serve per
lo Stage 0 dell'ipotesi V-lookahead: verificare se una rete scalare riesce a predire l'esito
finale, soprattutto negli stati in cui PIMC agisce.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state

LabelMode = Literal["same-game", "v6-continuation"]


@dataclass(frozen=True, slots=True)
class ValueDatasetConfig:
    """Configurazione riproducibile del dataset di valore."""

    out_path: Path
    agent_name: str
    model_path: Path | None
    num_games: int
    seed: int
    epsilon: float = 0.10
    label_mode: LabelMode = "same-game"
    encoder_version: str = "v3"
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class PendingValueRecord:
    """Record parziale salvato durante una partita, completato quando conosciamo l'esito."""

    record: dict[str, Any]
    player_index: int
    current_score_delta: int


def _build_play_agent(config: ValueDatasetConfig) -> Agent:
    """Costruisce l'agente di self-play."""
    if config.model_path is not None:
        return build_agent(config.agent_name, model_path=config.model_path)
    return build_agent(config.agent_name)


def _score_delta_for_player(state: GameState, player_index: int) -> int:
    """Delta punti dal punto di vista di `player_index` in una partita 2-player."""
    opponent = 1 - int(player_index)
    return int(state.players[player_index].points) - int(state.players[opponent].points)


def _unknown_live_cards(state: GameState, player_index: int) -> int:
    """Carte vive non note al player: mano avversaria + mazzo."""
    return int(len(state.players[1 - player_index].hand) + len(state.deck))


def _phase_for_state(state: GameState, player_index: int) -> str:
    """Bucket grossolano della fase di partita, utile per metriche stratificate."""
    if len(state.deck) == 0:
        return "endgame"
    unknown = _unknown_live_cards(state, player_index)
    if unknown <= 8:
        return "pimc_window"
    if len(state.deck) <= 10:
        return "mid"
    return "early"


def _safe_card_index(agent: Agent, observation, *, rng: random.Random) -> int:
    """Chiede una mossa all'agente e valida che sia un indice nella mano."""
    card_index = int(agent.choose_card_index(observation, rng=rng))
    if not 0 <= card_index < len(observation.hand):
        raise ValueError(f"Agente {agent.name!r} ha prodotto card_index={card_index}, mano={len(observation.hand)}")
    return card_index


def _choose_selfplay_card_index(
    agent: Agent,
    observation,
    *,
    rng: random.Random,
    epsilon: float,
) -> tuple[int, bool]:
    """
    Sceglie una mossa epsilon-greedy.

    Con probabilita' `epsilon` esplora una carta legale uniforme. Il flag booleano indica se
    la mossa e' esplorativa, cosi' il dataset puo' documentare la distribuzione generatrice.
    """
    if not observation.hand:
        raise ValueError("Mano vuota")
    if float(epsilon) > 0.0 and rng.random() < float(epsilon):
        return rng.randrange(len(observation.hand)), True
    return _safe_card_index(agent, observation, rng=rng), False


def _play_to_terminal(
    state: GameState,
    *,
    agent: Agent,
    rng: random.Random,
    max_steps: int = 256,
) -> GameState:
    """Completa una partita senza esplorazione, partendo da uno stato arbitrario."""
    cursor = state
    steps = 0
    while not cursor.game_over:
        if steps >= max_steps:
            raise RuntimeError("Continuazione value dataset non terminata entro il limite di sicurezza")
        steps += 1
        current = cursor.current_turn
        observation = make_player_observation(cursor, current)
        card_index = _safe_card_index(agent, observation, rng=rng)
        cursor, result = step(cursor, PlayCardAction(player_index=current, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante continuazione value dataset: {result.error}")
    return cursor


def _make_pending_record(
    *,
    config: ValueDatasetConfig,
    game_id: str,
    event_id: int,
    game_seed: int,
    ply_index: int,
    state: GameState,
    player_index: int,
    observation_dto: dict[str, Any],
    exploratory_action: bool,
) -> PendingValueRecord:
    """Costruisce una riga parziale; il target finale viene aggiunto a fine partita."""
    current_delta = _score_delta_for_player(state, player_index)
    record = {
        "schema_version": int(config.schema_version),
        "dataset_kind": "value_observation",
        "game_id": game_id,
        "event_id": int(event_id),
        "ply_index": int(ply_index),
        "player_index": int(player_index),
        "is_ai": True,
        "observation": observation_dto,
        "current_score_delta": int(current_delta),
        "phase": _phase_for_state(state, player_index),
        "generation": {
            "seed": int(config.seed),
            "game_seed": int(game_seed),
            "agent": config.agent_name,
            "model_path": str(config.model_path) if config.model_path is not None else None,
            "encoder_version": str(config.encoder_version),
            "epsilon": float(config.epsilon),
            "label_mode": str(config.label_mode),
            "exploratory_action": bool(exploratory_action),
            "unknown_live_cards": _unknown_live_cards(state, player_index),
            "cards_remaining_in_deck": len(state.deck),
        },
    }
    return PendingValueRecord(record=record, player_index=int(player_index), current_score_delta=int(current_delta))


def _finalize_record(
    pending: PendingValueRecord,
    *,
    final_state: GameState,
) -> dict[str, Any]:
    """Aggiunge target finale e residuo normalizzato a una riga parziale."""
    final_delta = _score_delta_for_player(final_state, pending.player_index)
    residual = int(final_delta) - int(pending.current_score_delta)
    record = dict(pending.record)
    record["final_score_delta"] = int(final_delta)
    record["residual_score_delta"] = int(residual)
    record["target_residual_scaled"] = float(residual) / 120.0
    record["target_final_scaled"] = float(final_delta) / 120.0
    return record


def generate_value_dataset(config: ValueDatasetConfig) -> dict[str, int | float]:
    """
    Genera il dataset di valore e ritorna contatori di riepilogo.

    `label_mode=same-game`: il target e' l'esito della stessa partita epsilon-greedy.
    `label_mode=v6-continuation`: ogni stato raccolto viene etichettato completando una copia
    con l'agente base senza esplorazione. Costa di piu', ma isola il valore di continuazione v6.
    """
    if config.num_games <= 0:
        raise ValueError("--num-games deve essere > 0")
    if not 0.0 <= float(config.epsilon) <= 1.0:
        raise ValueError("--epsilon deve essere in [0,1]")
    if config.label_mode not in ("same-game", "v6-continuation"):
        raise ValueError("--label-mode non valido")

    agent = _build_play_agent(config)
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text("", encoding="utf-8")

    rng_game = random.Random(config.seed)
    rng_action = random.Random(config.seed ^ 0x9E3779B9)
    rng_label = random.Random(config.seed ^ 0xD1B54A32)

    counters: dict[str, int | float] = {
        "games_started": 0,
        "games_completed": 0,
        "records_written": 0,
        "exploratory_actions": 0,
        "phase_early": 0,
        "phase_mid": 0,
        "phase_pimc_window": 0,
        "phase_endgame": 0,
    }

    event_id = 0
    with config.out_path.open("a", encoding="utf-8") as out:
        for game_idx in range(config.num_games):
            counters["games_started"] += 1
            game_seed = rng_game.randrange(0, 2**32)
            game_id = f"value_{config.seed}_{game_idx}_{game_seed}"
            state = new_game_state(num_players=2, seed=game_seed)
            pending_records: list[PendingValueRecord] = []
            ply_index = 0

            while not state.game_over:
                current = state.current_turn
                observation = make_player_observation(state, current)
                observation_dto = build_observation_dto(
                    state, player_index=current, server_version=ply_index
                ).model_dump(mode="json")
                card_index, exploratory = _choose_selfplay_card_index(
                    agent,
                    observation,
                    rng=rng_action,
                    epsilon=float(config.epsilon),
                )
                if exploratory:
                    counters["exploratory_actions"] += 1

                pending = _make_pending_record(
                    config=config,
                    game_id=game_id,
                    event_id=event_id,
                    game_seed=game_seed,
                    ply_index=ply_index,
                    state=state,
                    player_index=current,
                    observation_dto=observation_dto,
                    exploratory_action=exploratory,
                )
                phase_key = f"phase_{pending.record['phase']}"
                counters[phase_key] = int(counters.get(phase_key, 0)) + 1

                if config.label_mode == "v6-continuation":
                    label_final = _play_to_terminal(state, agent=agent, rng=rng_label)
                    record = _finalize_record(pending, final_state=label_final)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counters["records_written"] += 1
                else:
                    pending_records.append(pending)

                state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
                if result.error:
                    raise RuntimeError(f"Errore dominio durante value self-play: {result.error}")
                event_id += 1
                ply_index += 1

            counters["games_completed"] += 1
            if config.label_mode == "same-game":
                for pending in pending_records:
                    record = _finalize_record(pending, final_state=state)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counters["records_written"] += 1

    records = int(counters["records_written"])
    counters["exploration_rate_observed"] = (
        float(counters["exploratory_actions"]) / float(records) if records > 0 else 0.0
    )
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera dataset value_observation da self-play 2-player")
    parser.add_argument("--out", required=True, help="Path JSONL output")
    parser.add_argument("--agent", default="bc_model_hybrid_endgame", help="Agente self-play")
    parser.add_argument("--model", default="", help="Path modello .npz se l'agente lo richiede")
    parser.add_argument("--num-games", type=int, required=True, help="Numero partite da generare")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG")
    parser.add_argument("--epsilon", type=float, default=0.10, help="Probabilita' esplorazione uniforme")
    parser.add_argument(
        "--label-mode",
        choices=["same-game", "v6-continuation"],
        default="same-game",
        help="same-game usa il finale della partita epsilon-greedy; v6-continuation completa ogni stato senza epsilon.",
    )
    parser.add_argument("--encoder-version", default="v3", choices=["v1", "v2", "v3"])
    args = parser.parse_args()

    model = str(args.model).strip()
    config = ValueDatasetConfig(
        out_path=Path(args.out),
        agent_name=str(args.agent),
        model_path=Path(model) if model else None,
        num_games=int(args.num_games),
        seed=int(args.seed),
        epsilon=float(args.epsilon),
        label_mode=args.label_mode,
        encoder_version=str(args.encoder_version),
    )
    summary = generate_value_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
