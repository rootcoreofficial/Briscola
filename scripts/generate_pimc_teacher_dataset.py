#!/usr/bin/env python3
"""
Genera un dataset JSONL di mosse teacher PIMC per distillazione BC.

Obiettivo
---------
La pipeline A2C v6 ha quasi saturato lo scaling policy-only, mentre PIMC ha mostrato
un segnale misurabile nel finale/semi-finale. Questo script produce esempi supervised
compatibili con `scripts/train_bc.py`:

- `observation`: ObservationDTO prima della decisione;
- `action.card_index`: mossa scelta dal teacher PIMC;
- metadati extra per audit e filtri futuri.

Per default le partite avanzano con il modello base (es. v6), non con il teacher PIMC,
e salviamo anche le posizioni fuori finestra search: li' il teacher delega al fallback
v6. Il dataset risultante e' quindi "v6 ovunque + correzioni PIMC nel finale", piu'
sicuro per fine-tuning rispetto a sole etichette di finale.

Per gli esempi `decision_type=search`, il record include anche la diagnostica PIMC
(`margin`, SE, CI e delta paired per determinizzazione). Questo permette di contare
prima del training quante correzioni sono davvero affidabili, evitando di allenare
su argmax rumorosi.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from briscola_ai.ai.agents import Agent, PIMCAgent, unknown_live_card_count
from briscola_ai.ai.models import BCModelAgent
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state


@dataclass(frozen=True, slots=True)
class PIMCTeacherDatasetConfig:
    """Configurazione riproducibile per generare esempi teacher PIMC."""

    out_path: Path
    num_examples: int
    max_games: int
    seed: int
    max_unknown_cards: int = 8
    player_index: int | None = None
    include_fallback_examples: bool = True
    advance_with_teacher: bool = False
    strong_margin_min: float = 2.0
    reliable_margin_ci_low_min: float = 0.0
    schema_version: int = 1


def _safe_card_index(agent: Agent, observation, *, rng: random.Random) -> int:
    """
    Chiede una mossa a un agente e valida che sia un indice nella mano.

    Qui preferiamo fallire esplicitamente invece di normalizzare: se il teacher o la
    policy di avanzamento genera mosse invalide, il dataset non deve nascondere il bug.
    """
    card_index = int(agent.choose_card_index(observation, rng=rng))
    if not 0 <= card_index < len(observation.hand):
        raise ValueError(f"Agente {agent.name!r} ha prodotto card_index={card_index}, mano={len(observation.hand)}")
    return card_index


def _pimc_decision_type(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Deduce quale ramo PIMC ha prodotto l'etichetta confrontando i contatori."""
    if int(after["endgame_solver_decisions"]) > int(before["endgame_solver_decisions"]):
        return "endgame_solver"
    if int(after["search_decisions"]) > int(before["search_decisions"]):
        return "search"
    if int(after["fallback_decisions"]) > int(before["fallback_decisions"]):
        return "fallback"
    return "unknown"


def _choose_teacher_action(
    teacher: Agent,
    observation,
    *,
    rng: random.Random,
) -> tuple[int, str, dict[str, Any] | None]:
    """Ritorna `(card_index, decision_type, search_diagnostics)` per l'etichetta teacher."""
    if isinstance(teacher, PIMCAgent):
        before = asdict(teacher.metrics)
        card_index = _safe_card_index(teacher, observation, rng=rng)
        after = asdict(teacher.metrics)
        decision_type = _pimc_decision_type(before, after)
        diagnostics = asdict(teacher.last_search_diagnostics) if teacher.last_search_diagnostics is not None else None
        return card_index, decision_type, diagnostics
    return _safe_card_index(teacher, observation, rng=rng), "agent", None


def _diagnostic_float(diagnostics: dict[str, Any] | None, key: str) -> float | None:
    if diagnostics is None:
        return None
    value = diagnostics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _is_strong_reliable_search(
    *,
    config: PIMCTeacherDatasetConfig,
    decision_type: str,
    disagrees_with_reference: bool,
    diagnostics: dict[str, Any] | None,
) -> bool:
    """Criterio pre-registrato per contare esempi search utili prima del retrain."""
    if decision_type != "search" or not disagrees_with_reference:
        return False
    margin = _diagnostic_float(diagnostics, "margin")
    ci_low = _diagnostic_float(diagnostics, "margin_ci95_low")
    if margin is None or ci_low is None:
        return False
    return margin >= float(config.strong_margin_min) and ci_low >= float(config.reliable_margin_ci_low_min)


def _make_record(
    *,
    config: PIMCTeacherDatasetConfig,
    game_id: str,
    example_index: int,
    game_seed: int,
    server_version: int,
    player_index: int,
    observation_dto: dict[str, Any],
    card_index: int,
    unknown_cards: int,
    teacher: Agent,
    teacher_decision_type: str,
    teacher_search_diagnostics: dict[str, Any] | None,
    reference_agent_name: str,
    reference_card_index: int,
) -> dict[str, Any]:
    """Costruisce una riga JSONL compatibile con `train_bc.py` piu' metadati di audit."""
    disagrees_with_reference = int(card_index) != int(reference_card_index)
    return {
        "schema_version": int(config.schema_version),
        "dataset_kind": "pimc_teacher",
        "game_id": game_id,
        "event_id": example_index,
        "server_version": server_version,
        "player_index": player_index,
        "is_ai": True,
        "observation": observation_dto,
        "action": {"card_index": int(card_index)},
        "reward": 0,
        "next_observation": None,
        "done": None,
        "teacher": {
            "name": teacher.name,
            "decision_type": teacher_decision_type,
            "max_unknown_cards": int(config.max_unknown_cards),
            "search_diagnostics": teacher_search_diagnostics,
        },
        "reference": {
            "agent": reference_agent_name,
            "card_index": int(reference_card_index),
            "disagrees_with_teacher": disagrees_with_reference,
        },
        "generation": {
            "seed": int(config.seed),
            "game_seed": int(game_seed),
            "unknown_live_cards": int(unknown_cards),
            "cards_remaining_in_deck": int(observation_dto["cards_remaining_in_deck"]),
            "include_fallback_examples": bool(config.include_fallback_examples),
            "advance_with_teacher": bool(config.advance_with_teacher),
            "strong_margin_min": float(config.strong_margin_min),
            "reliable_margin_ci_low_min": float(config.reliable_margin_ci_low_min),
        },
    }


def generate_pimc_teacher_dataset(
    config: PIMCTeacherDatasetConfig,
    *,
    teacher: Agent,
    play_agent: Agent,
) -> dict[str, int | float]:
    """
    Genera esempi `(observation -> teacher_action)` in formato JSONL.

    Args:
        config: parametri riproducibili e path output.
        teacher: agente che etichetta le posizioni eleggibili (tipicamente PIMC 16x8).
        play_agent: agente che fa avanzare le partite (tipicamente il modello base v6).

    Ritorna:
        Contatori utili per log/test. Se `teacher` e' PIMC includiamo anche metriche runtime.
    """
    if config.num_examples <= 0:
        raise ValueError("num_examples deve essere > 0")
    if config.max_games <= 0:
        raise ValueError("max_games deve essere > 0")
    if config.max_unknown_cards < 0:
        raise ValueError("max_unknown_cards deve essere >= 0")
    if config.player_index is not None and config.player_index not in (0, 1):
        raise ValueError("player_index deve essere None, 0 o 1")

    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text("", encoding="utf-8")

    rng_games = random.Random(config.seed)
    rng_teacher = random.Random(config.seed ^ 0xA5A5A5A5)
    rng_play = random.Random(config.seed ^ 0x5A5A5A5A)
    rng_reference = random.Random(config.seed ^ 0x6A09E667)

    counters: dict[str, int | float] = {
        "games_started": 0,
        "games_completed": 0,
        "moves_seen": 0,
        "eligible_positions": 0,
        "pimc_window_positions": 0,
        "fallback_window_positions": 0,
        "records_written": 0,
        "records_written_search": 0,
        "records_written_endgame_solver": 0,
        "records_written_fallback": 0,
        "records_written_other": 0,
        "records_written_search_disagree_reference": 0,
        "records_written_search_margin_ge_min": 0,
        "records_written_search_reliable_margin": 0,
        "records_written_search_strong_reliable_disagree": 0,
        "records_skipped_player": 0,
        "records_skipped_outside_pimc_window": 0,
        "records_skipped_invalid_teacher": 0,
        "max_games_reached": 0,
    }

    with config.out_path.open("a", encoding="utf-8") as out:
        for game_no in range(config.max_games):
            if int(counters["records_written"]) >= config.num_examples:
                break

            game_seed = rng_games.randrange(0, 2**32)
            game_id = f"pimc_teacher_{game_no:06d}"
            state = new_game_state(num_players=2, seed=game_seed)
            counters["games_started"] += 1
            server_version = 0
            safety = 200

            while not state.game_over and safety > 0 and int(counters["records_written"]) < config.num_examples:
                safety -= 1
                player = state.current_turn
                observation = make_player_observation(state, player)
                unknown_cards = unknown_live_card_count(observation)
                counters["moves_seen"] += 1

                teacher_action: int | None = None
                player_matches = config.player_index is None or player == config.player_index
                if not player_matches:
                    counters["records_skipped_player"] += 1
                else:
                    inside_pimc_window = unknown_cards <= config.max_unknown_cards
                    if inside_pimc_window:
                        counters["pimc_window_positions"] += 1
                    else:
                        counters["fallback_window_positions"] += 1

                    should_label = inside_pimc_window or config.include_fallback_examples
                    if not should_label:
                        counters["records_skipped_outside_pimc_window"] += 1

                if player_matches and (unknown_cards <= config.max_unknown_cards or config.include_fallback_examples):
                    counters["eligible_positions"] += 1
                    try:
                        teacher_action, decision_type, search_diagnostics = _choose_teacher_action(
                            teacher,
                            observation,
                            rng=rng_teacher,
                        )
                        reference_action = _safe_card_index(play_agent, observation, rng=rng_reference)
                    except ValueError:
                        counters["records_skipped_invalid_teacher"] += 1
                    else:
                        dto = build_observation_dto(state, player_index=player, server_version=server_version)
                        disagrees_with_reference = int(teacher_action) != int(reference_action)
                        record = _make_record(
                            config=config,
                            game_id=game_id,
                            example_index=int(counters["records_written"]),
                            game_seed=game_seed,
                            server_version=server_version,
                            player_index=player,
                            observation_dto=dto.model_dump(mode="json"),
                            card_index=teacher_action,
                            unknown_cards=unknown_cards,
                            teacher=teacher,
                            teacher_decision_type=decision_type,
                            teacher_search_diagnostics=search_diagnostics,
                            reference_agent_name=play_agent.name,
                            reference_card_index=reference_action,
                        )
                        out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        counters["records_written"] += 1
                        if decision_type == "search":
                            counters["records_written_search"] += 1
                            margin = _diagnostic_float(search_diagnostics, "margin")
                            ci_low = _diagnostic_float(search_diagnostics, "margin_ci95_low")
                            if disagrees_with_reference:
                                counters["records_written_search_disagree_reference"] += 1
                            if margin is not None and margin >= float(config.strong_margin_min):
                                counters["records_written_search_margin_ge_min"] += 1
                            if ci_low is not None and ci_low >= float(config.reliable_margin_ci_low_min):
                                counters["records_written_search_reliable_margin"] += 1
                            if _is_strong_reliable_search(
                                config=config,
                                decision_type=decision_type,
                                disagrees_with_reference=disagrees_with_reference,
                                diagnostics=search_diagnostics,
                            ):
                                counters["records_written_search_strong_reliable_disagree"] += 1
                        elif decision_type == "endgame_solver":
                            counters["records_written_endgame_solver"] += 1
                        elif decision_type == "fallback":
                            counters["records_written_fallback"] += 1
                        else:
                            counters["records_written_other"] += 1

                if config.advance_with_teacher and teacher_action is not None:
                    play_action = teacher_action
                else:
                    play_action = _safe_card_index(play_agent, observation, rng=rng_play)

                state, result = step(state, PlayCardAction(player_index=player, card_index=play_action))
                if result.error:
                    raise RuntimeError(f"Errore dominio durante generazione teacher: {result.error}")
                server_version += 1

            if safety <= 0:
                raise RuntimeError("Loop di sicurezza: la partita non termina")
            if state.game_over:
                counters["games_completed"] += 1

        if int(counters["records_written"]) < config.num_examples:
            counters["max_games_reached"] = 1

    if isinstance(teacher, PIMCAgent):
        counters["teacher_search_decisions"] = teacher.metrics.search_decisions
        counters["teacher_endgame_solver_decisions"] = teacher.metrics.endgame_solver_decisions
        counters["teacher_fallback_decisions"] = teacher.metrics.fallback_decisions
        counters["teacher_coerced_moves"] = teacher.metrics.coerced_moves
        counters["teacher_seconds_per_search_decision"] = teacher.metrics.seconds_per_search_decision

    return counters


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera un dataset JSONL teacher PIMC per BC/distillazione.")
    parser.add_argument("--model", default="data/models/best_a2c_v6.npz", help="Modello base `.npz` usato come v6.")
    parser.add_argument("--out", required=True, help="Path JSONL output.")
    parser.add_argument("--num-examples", type=int, default=50000, help="Numero esempi da scrivere. Default: 50000.")
    parser.add_argument("--max-games", type=int, default=20000, help="Massimo partite simulate. Default: 20000.")
    parser.add_argument("--seed", type=int, default=777, help="Seed generazione. Default: 777.")
    parser.add_argument(
        "--determinizations",
        type=int,
        default=16,
        help="Determinizzazioni PIMC per etichetta. Default: 16.",
    )
    parser.add_argument(
        "--max-unknown-cards",
        type=int,
        default=8,
        help=(
            "Soglia carte vive ignote entro cui PIMC fa search; fuori soglia etichetta col fallback "
            "se non usi --only-pimc-window. Default: 8."
        ),
    )
    parser.add_argument(
        "--player-index",
        type=int,
        choices=[0, 1],
        default=None,
        help="Se impostato, salva solo esempi del player indicato. Default: entrambi.",
    )
    parser.add_argument(
        "--only-pimc-window",
        action="store_true",
        help=(
            "Salva solo posizioni con carte vive ignote <= --max-unknown-cards. "
            "Default: include anche posizioni fallback/v6 fuori finestra."
        ),
    )
    parser.add_argument(
        "--advance-with-teacher",
        action="store_true",
        help="Avanza le partite con la mossa teacher nelle posizioni etichettate; default: avanza con il modello base.",
    )
    parser.add_argument(
        "--strong-margin-min",
        type=float,
        default=2.0,
        help="Soglia margine best-second per contare una correzione search forte. Default: 2.0 punti.",
    )
    parser.add_argument(
        "--reliable-margin-ci-low-min",
        type=float,
        default=0.0,
        help="Soglia CI95 lower bound del margine per contare una correzione search affidabile. Default: 0.0.",
    )
    return parser


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()

    model_path = Path(args.model)
    base_agent = BCModelAgent.from_npz(model_path)
    teacher = PIMCAgent(
        rollout_agent=base_agent,
        fallback=base_agent,
        num_determinizations=args.determinizations,
        max_unknown_cards=args.max_unknown_cards,
        use_endgame_solver=True,
        name=f"pimc_teacher({model_path.name},d={args.determinizations},u={args.max_unknown_cards})",
    )
    config = PIMCTeacherDatasetConfig(
        out_path=Path(args.out),
        num_examples=args.num_examples,
        max_games=args.max_games,
        seed=args.seed,
        max_unknown_cards=args.max_unknown_cards,
        player_index=args.player_index,
        include_fallback_examples=not args.only_pimc_window,
        advance_with_teacher=args.advance_with_teacher,
        strong_margin_min=args.strong_margin_min,
        reliable_margin_ci_low_min=args.reliable_margin_ci_low_min,
    )

    counters = generate_pimc_teacher_dataset(config, teacher=teacher, play_agent=base_agent)
    print(json.dumps(counters, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
