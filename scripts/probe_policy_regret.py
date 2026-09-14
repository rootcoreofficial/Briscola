#!/usr/bin/env python3
"""
Individua automaticamente le decisioni costose di una policy Briscola 2-player.

La sonda raccoglie osservazioni non forzate da partite seat-fair, bilanciate nelle
combinazioni fase/posizione, e delega a ``ai.evaluation.policy_regret`` il confronto
controfattuale di tutte le carte legali. Il report contiene soltanto informazione
pubblica o appartenente alla mano osservata; il seed della partita e' metadato di
riproduzione e non viene mai passato allo stimatore.

Esempio breve:

    uv run python scripts/probe_policy_regret.py \\
      --model data/models/best_a2c_v14.npz \\
      --num-observations 72 --determinizations 16 \\
      --out benchmarks/experiments/policy_regret_v14_pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.evaluation.policy_regret import (
    PolicyRegretConfig,
    estimate_policy_regret,
    observation_phase,
)
from briscola_ai.ai.models import PIMC_BELIEF_MODEL_ID, BCModelAgent
from briscola_ai.ai.models.belief_model import MLPBeliefModel, load_belief_model_npz
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import new_game_state
from briscola_ai.versioning import get_code_version, get_rules_version

_PHASES = ("early", "mid", "pimc_window", "endgame")
_POSITIONS = ("lead", "response")
_BUCKETS = tuple((phase, position) for phase in _PHASES for position in _POSITIONS)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """Configurazione della raccolta e della stima automatica."""

    model_path: Path
    belief_model_path: Path | None
    out_path: Path
    num_observations: int = 192
    max_games: int = 400
    seed: int = 20260720
    opponents: tuple[str, ...] = ("mirror", "heuristic_trump_saver", "heuristic_v1")
    determinizations: int = 64
    min_regret_points: float = 1.0
    confidence_z: float = 2.576
    belief_uniform_mix: float = 0.10
    top_cases: int = 30

    def validate(self) -> None:
        """Valida i vincoli che rendono la suite bilanciata e riproducibile."""
        if not self.opponents:
            raise ValueError("Serve almeno un avversario")
        if any(not opponent.strip() for opponent in self.opponents):
            raise ValueError("La lista avversari contiene un nome vuoto")
        balanced_cells = len(_BUCKETS) * len(self.opponents)
        if self.num_observations < balanced_cells or self.num_observations % balanced_cells != 0:
            raise ValueError(
                "num_observations deve bilanciare avversario x fase x posizione: "
                f"atteso un multiplo di {balanced_cells} e >= {balanced_cells}"
            )
        if self.max_games <= 0 or self.max_games % 2 != 0:
            raise ValueError("max_games deve essere positivo e pari per mantenere le coppie seat-fair")
        if self.top_cases < 0:
            raise ValueError("top_cases deve essere >= 0")
        PolicyRegretConfig(
            determinizations=self.determinizations,
            min_regret_points=self.min_regret_points,
            confidence_z=self.confidence_z,
            belief_uniform_mix=self.belief_uniform_mix,
        ).validate()


@dataclass(frozen=True, slots=True)
class CollectedDecision:
    """Osservazione lecita piu' soli metadati necessari a riprodurne la raccolta."""

    observation_id: int
    game_pair_index: int
    game_seed: int
    move_index: int
    opponent: str
    policy_seat: int
    phase: str
    position: str
    observation: PlayerObservation
    chosen_card_index: int


def _safe_choose(agent: Agent, observation: PlayerObservation, *, rng: random.Random) -> int:
    """Chiede una carta e fallisce se un agente produce un indice non valido."""
    card_index = int(agent.choose_card_index(observation, rng=rng))
    if card_index < 0 or card_index >= len(observation.hand):
        raise ValueError(f"Agente {agent.name!r}: card_index={card_index}, hand_size={len(observation.hand)}")
    return card_index


def _build_opponents(names: tuple[str, ...], *, policy: BCModelAgent, model_path: Path) -> dict[str, Agent]:
    """Costruisce una sola istanza per nome; ``mirror`` riusa la policy sotto audit."""
    opponents: dict[str, Agent] = {}
    for name in names:
        if name == "mirror":
            opponents[name] = policy
        elif name == "bc_model":
            opponents[name] = build_agent(name, model_path=model_path)
        else:
            opponents[name] = build_agent(name)
    return opponents


def _bucket_targets(config: ProbeConfig) -> dict[tuple[str, str, str], int]:
    """Quota identica per ogni combinazione avversario, fase e posizione."""
    per_bucket = config.num_observations // (len(_BUCKETS) * len(config.opponents))
    return {(opponent, phase, position): per_bucket for opponent in config.opponents for phase, position in _BUCKETS}


def collect_policy_decisions(
    config: ProbeConfig,
    *,
    policy: BCModelAgent,
) -> tuple[list[CollectedDecision], dict[str, Any]]:
    """
    Raccoglie una suite bilanciata da coppie con stesso mazzo e seat scambiata.

    Lo stato completo serve soltanto al motore per giocare la partita. Appena una
    decisione e' eleggibile, salviamo ``PlayerObservation`` e scartiamo il riferimento
    allo stato: lo stimatore non puo' quindi leggere mano avversaria o deck reale.
    """
    targets = _bucket_targets(config)
    counts: Counter[tuple[str, str, str]] = Counter()
    decisions: list[CollectedDecision] = []
    opponents = _build_opponents(config.opponents, policy=policy, model_path=config.model_path)
    games_rng = random.Random(config.seed ^ 0xC011EC7)
    games_started = 0
    games_completed = 0
    forced_policy_decisions = 0
    eligible_seen = 0
    pair_index = 0

    def suite_complete() -> bool:
        return all(counts[bucket] >= target for bucket, target in targets.items())

    while games_started < config.max_games and not suite_complete():
        opponent_name = config.opponents[pair_index % len(config.opponents)]
        opponent = opponents[opponent_name]
        game_seed = games_rng.randrange(0, 2**32)
        opponent_crc = zlib.crc32(opponent_name.encode("utf-8"))

        for policy_seat in (0, 1):
            state = new_game_state(num_players=2, seed=game_seed)
            play_rng = random.Random(config.seed ^ game_seed ^ opponent_crc ^ (policy_seat * 0x9E3779B9))
            games_started += 1
            move_index = 0
            safety = 200

            while not state.game_over and safety > 0:
                safety -= 1
                current = state.current_turn
                observation = make_player_observation(state, current)
                actor = policy if current == policy_seat else opponent
                chosen_card_index = _safe_choose(actor, observation, rng=play_rng)

                if current == policy_seat:
                    if len(observation.hand) < 2:
                        forced_policy_decisions += 1
                    else:
                        eligible_seen += 1
                        phase = observation_phase(observation)
                        position = "lead" if not observation.table_cards else "response"
                        bucket = (opponent_name, phase, position)
                        if counts[bucket] < targets[bucket]:
                            decisions.append(
                                CollectedDecision(
                                    observation_id=len(decisions),
                                    game_pair_index=pair_index,
                                    game_seed=game_seed,
                                    move_index=move_index,
                                    opponent=opponent_name,
                                    policy_seat=policy_seat,
                                    phase=phase,
                                    position=position,
                                    observation=observation,
                                    chosen_card_index=chosen_card_index,
                                )
                            )
                            counts[bucket] += 1

                state, result = step(
                    state,
                    PlayCardAction(player_index=current, card_index=chosen_card_index),
                )
                if result.error:
                    raise RuntimeError(f"Errore dominio durante la raccolta: {result.error}")
                move_index += 1

            if safety <= 0:
                raise RuntimeError("La partita non termina entro il limite di sicurezza")
            games_completed += 1

        pair_index += 1

    if not suite_complete():
        missing = {
            f"{opponent}:{phase}:{position}": targets[(opponent, phase, position)] - counts[(opponent, phase, position)]
            for opponent in config.opponents
            for phase, position in _BUCKETS
            if counts[(opponent, phase, position)] < targets[(opponent, phase, position)]
        }
        raise RuntimeError(f"max_games raggiunto prima di completare la suite bilanciata: {missing}")

    phase_position_counts = Counter((row.phase, row.position) for row in decisions)
    summary = {
        "games_started": games_started,
        "games_completed": games_completed,
        "game_pairs": games_started // 2,
        "eligible_policy_decisions_seen": eligible_seen,
        "forced_policy_decisions_skipped": forced_policy_decisions,
        "records_collected": len(decisions),
        "bucket_targets": {
            f"{opponent}:{phase}:{position}": targets[(opponent, phase, position)]
            for opponent in config.opponents
            for phase, position in _BUCKETS
        },
        "bucket_counts": {
            f"{opponent}:{phase}:{position}": counts[(opponent, phase, position)]
            for opponent in config.opponents
            for phase, position in _BUCKETS
        },
        "phase_position_counts": {
            f"{phase}:{position}": phase_position_counts[(phase, position)] for phase, position in _BUCKETS
        },
        "opponent_counts": dict(sorted(Counter(row.opponent for row in decisions).items())),
    }
    return decisions, summary


def _distribution(values: list[float]) -> dict[str, int | float | None]:
    """Distribuzione JSON-safe senza NaN per gruppi eventualmente vuoti."""
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Metriche aggregate riutilizzate per totale e sottogruppi."""
    reliable = [record for record in records if bool(record["estimate"]["reliable_error"])]
    exposed = [record for record in reliable if record["context"]["runtime_scope"] == "policy_only"]
    disagreements = [
        record
        for record in records
        if int(record["estimate"]["alternative_card_index"]) != int(record["estimate"]["chosen_card_index"])
    ]
    confirmed = [record for record in records if bool(record["estimate"]["candidate_confirmed_as_evaluation_best"])]
    return {
        "decisions": len(records),
        "candidate_disagreements": len(disagreements),
        "candidate_disagreement_rate": len(disagreements) / len(records) if records else 0.0,
        "candidate_confirmed_as_evaluation_best": len(confirmed),
        "reliable_errors": len(reliable),
        "reliable_error_rate": len(reliable) / len(records) if records else 0.0,
        "reliable_errors_policy_only": len(exposed),
        "reliable_error_policy_only_rate": len(exposed) / len(records) if records else 0.0,
        "regret_all": _distribution([float(record["estimate"]["regret_mean"]) for record in records]),
        "regret_reliable": _distribution([float(record["estimate"]["regret_mean"]) for record in reliable]),
    }


def _group_summaries(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Raggruppa i record su un campo del contesto o della stima."""
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if key in record["context"]:
            value = str(record["context"][key])
        elif key in record["provenance"]:
            value = str(record["provenance"][key])
        else:
            value = str(record["estimate"][key])
        groups[value].append(record)
    return {value: _record_summary(groups[value]) for value in sorted(groups)}


def _tag_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Conta etichette soltanto sugli errori affidabili, che sono l'insieme azionabile."""
    counts: Counter[str] = Counter()
    regret_by_tag: defaultdict[str, list[float]] = defaultdict(list)
    for record in records:
        estimate = record["estimate"]
        if not bool(estimate["reliable_error"]):
            continue
        for tag in estimate["tags"]:
            counts[str(tag)] += 1
            regret_by_tag[str(tag)].append(float(estimate["regret_mean"]))
    return {
        tag: {"count": counts[tag], "regret": _distribution(regret_by_tag[tag])}
        for tag in sorted(counts, key=lambda name: (-counts[name], name))
    }


def _runtime_scope_for_phase(phase: str) -> str:
    """Layer del prodotto che gestisce normalmente la decisione osservata."""
    if phase == "endgame":
        return "exact_solver"
    if phase == "pimc_window":
        return "pimc_search_window"
    return "policy_only"


def _public_context(decision: CollectedDecision) -> dict[str, Any]:
    """Serializza soltanto campi disponibili nell'osservazione del decisore."""
    observation = decision.observation
    return {
        "player_index": observation.player_index,
        "phase": decision.phase,
        "position": decision.position,
        "runtime_scope": _runtime_scope_for_phase(decision.phase),
        "trick_index": len(observation.trick_history),
        "deck_size": observation.deck_size,
        "players_points": list(observation.players_points),
        "hand_card_ids": [card_to_id(card) for card in observation.hand],
        "table_cards": [
            {"card_id": card_to_id(card), "player_index": player_index}
            for card, player_index in observation.table_cards
        ],
        "trump_card_id": card_to_id(observation.trump_card) if observation.trump_card is not None else None,
    }


def _provenance(decision: CollectedDecision) -> dict[str, Any]:
    """Metadati di riproduzione mai passati allo stimatore di regret."""
    return {
        "observation_id": decision.observation_id,
        "game_pair_index": decision.game_pair_index,
        "game_seed": decision.game_seed,
        "move_index": decision.move_index,
        "opponent": decision.opponent,
        "policy_seat": decision.policy_seat,
    }


def _file_ref(path: Path) -> dict[str, Any]:
    """Path, dimensione e SHA-256 per rendere il report auto-auditabile."""
    payload = path.read_bytes()
    return {"path": str(path), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _git_commit() -> str | None:
    """Commit corrente best-effort, senza rendere Git un requisito della sonda."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except OSError, subprocess.CalledProcessError:
        return None


def _routing_decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Applica il gate preregistrato per evitare training basati su casi isolati."""
    exposed = [
        record
        for record in records
        if bool(record["estimate"]["reliable_error"]) and record["context"]["runtime_scope"] == "policy_only"
    ]
    runtime_covered = [
        record
        for record in records
        if bool(record["estimate"]["reliable_error"]) and record["context"]["runtime_scope"] != "policy_only"
    ]
    tag_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in exposed:
        for raw_tag in record["estimate"]["tags"]:
            tag_records[str(raw_tag)].append(record)

    clusters: list[dict[str, Any]] = []
    for tag, tagged in tag_records.items():
        opponents = {str(record["provenance"]["opponent"]) for record in tagged}
        game_pairs = {int(record["provenance"]["game_pair_index"]) for record in tagged}
        clusters.append(
            {
                "tag": tag,
                "count": len(tagged),
                "opponent_count": len(opponents),
                "game_pair_count": len(game_pairs),
                "actionable": tag != "other" and len(tagged) >= 3 and len(opponents) >= 2 and len(game_pairs) >= 2,
            }
        )
    clusters.sort(key=lambda item: (-int(item["count"]), str(item["tag"])))
    actionable = [cluster for cluster in clusters if bool(cluster["actionable"])]
    other_cluster = next((cluster for cluster in clusters if cluster["tag"] == "other"), None)

    if actionable:
        verdict = "actionable_policy_error_cluster"
        explanation = (
            "Una categoria policy-only supera ripetizione e diversita': "
            "autorizzato progettare un solo intervento mirato."
        )
    elif other_cluster is not None and int(other_cluster["count"]) >= 3:
        verdict = "unclassified_policy_error_cluster"
        explanation = (
            "Gli errori policy-only si ripetono ma la tassonomia non li spiega: "
            "migliorare la diagnosi prima del training."
        )
    elif exposed:
        verdict = "sparse_policy_error_signal"
        explanation = (
            "Esistono casi policy-only affidabili, ma nessuna categoria si ripete abbastanza "
            "da giustificare un training."
        )
    else:
        verdict = "no_policy_error_signal"
        explanation = "La suite non trova errori affidabili fuori dalle finestre gia' gestite da PIMC e solver."

    return {
        "verdict": verdict,
        "explanation_it": explanation,
        "reliable_policy_only_errors": len(exposed),
        "reliable_runtime_scope_errors": len(runtime_covered),
        "cluster_thresholds": {"count_min": 3, "opponent_count_min": 2, "game_pair_count_min": 2},
        "clusters": clusters,
    }


def run_policy_regret_probe(config: ProbeConfig) -> dict[str, Any]:
    """Esegue raccolta, controfattuali, classificazione e aggregazione."""
    config.validate()
    policy = BCModelAgent.from_npz(config.model_path)
    belief_model: MLPBeliefModel | None = None
    if config.belief_model_path is not None:
        belief_model = load_belief_model_npz(config.belief_model_path)

    decisions, collection = collect_policy_decisions(config, policy=policy)
    regret_config = PolicyRegretConfig(
        determinizations=config.determinizations,
        min_regret_points=config.min_regret_points,
        confidence_z=config.confidence_z,
        belief_uniform_mix=config.belief_uniform_mix,
    )
    records: list[dict[str, Any]] = []
    for decision in decisions:
        estimate_rng = random.Random(config.seed ^ 0xA11D17 ^ (decision.observation_id * 0x9E3779B9))
        estimate = estimate_policy_regret(
            decision.observation,
            chosen_card_index=decision.chosen_card_index,
            rollout_agent=policy,
            rng=estimate_rng,
            config=regret_config,
            belief_model=belief_model,
        )
        records.append(
            {
                "provenance": _provenance(decision),
                "context": _public_context(decision),
                "estimate": asdict(estimate),
            }
        )

    top_cases = sorted(
        (record for record in records if bool(record["estimate"]["reliable_error"])),
        key=lambda record: (
            -float(record["estimate"]["regret_mean"]),
            int(record["provenance"]["observation_id"]),
        ),
    )[: config.top_cases]
    aggregate = {
        "overall": _record_summary(records),
        "by_phase": _group_summaries(records, "phase"),
        "by_position": _group_summaries(records, "position"),
        "by_opponent": _group_summaries(records, "opponent"),
        "by_method": _group_summaries(records, "method"),
        "by_runtime_scope": _group_summaries(records, "runtime_scope"),
        "reliable_error_tags": _tag_summary(records),
    }
    inputs: dict[str, Any] = {"policy_model": _file_ref(config.model_path)}
    inputs["belief_model"] = _file_ref(config.belief_model_path) if config.belief_model_path is not None else None

    return {
        "schema": "briscola.policy_regret_probe.v1",
        "versions": {
            "code": get_code_version(),
            "rules": get_rules_version(),
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "inputs": inputs,
        "protocol": {
            "purpose": "diagnose_residual_policy_errors_not_promote_model",
            "seed": config.seed,
            "num_observations": config.num_observations,
            "opponents": list(config.opponents),
            "seat_fair_collection": True,
            "balanced_cells": "opponent_x_phase_x_position",
            "phase_position_buckets": [f"{phase}:{position}" for phase, position in _BUCKETS],
            "determinizations": config.determinizations,
            "sample_split": "first_half_select_candidate_second_half_estimate_paired_regret",
            "min_regret_points": config.min_regret_points,
            "confidence_z": config.confidence_z,
            "belief_sampling": belief_model is not None,
            "belief_uniform_mix": config.belief_uniform_mix if belief_model is not None else None,
            "endgame": "exact_minimax_from_player_observation",
            "rollout_policy": policy.name,
        },
        "anti_cheat": {
            "estimator_input": "PlayerObservation",
            "actual_hidden_state_used_by_estimator": False,
            "provenance_is_never_passed_to_estimator": True,
            "saved_context_contains_only_public_information_and_observer_hand": True,
        },
        "collection": collection,
        "decision": _routing_decision(records),
        "aggregate": aggregate,
        "top_reliable_errors": top_cases,
        "decisions": records,
        "caveats": [
            "Fuori dall'endgame il regret e' una stima PIMC rispetto alla continuation policy, non verita' assoluta.",
            "Il belief pesa mondi plausibili ma non legge la mano avversaria reale.",
            "La suite e' bilanciata per fase/posizione e non stima direttamente punti persi per partita live.",
            "Un'etichetta descrive il cambio di carta; non e' automaticamente la causa interna dell'errore.",
        ],
    }


def _parse_opponents(raw: str) -> tuple[str, ...]:
    """Lista CSV ordinata e senza duplicati accidentali."""
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(set(names)) != len(names):
        raise ValueError("--opponents contiene duplicati")
    return names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="data/models/best_a2c_v14.npz", help="Policy `.npz` da diagnosticare.")
    parser.add_argument(
        "--belief-model",
        default=f"data/models/{PIMC_BELIEF_MODEL_ID}",
        help="Belief `.npz` per pesare le determinizzazioni.",
    )
    parser.add_argument(
        "--uniform-determinizations",
        action="store_true",
        help="Ignora --belief-model e campiona uniformemente le carte ignote.",
    )
    parser.add_argument("--out", required=True, help="Report JSON di output.")
    parser.add_argument(
        "--num-observations",
        type=int,
        default=192,
        help="Multiplo di 8 x numero avversari; default 192.",
    )
    parser.add_argument("--max-games", type=int, default=400, help="Tetto pari di partite di raccolta; default 400.")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--opponents",
        default="mirror,heuristic_trump_saver,heuristic_v1",
        help="Roster CSV della raccolta.",
    )
    parser.add_argument("--determinizations", type=int, default=64, help="Pari e >=4; default 64.")
    parser.add_argument("--min-regret-points", type=float, default=1.0)
    parser.add_argument("--confidence-z", type=float, default=2.576, help="Default 2.576 (intervallo circa 99%%).")
    parser.add_argument("--belief-uniform-mix", type=float, default=0.10)
    parser.add_argument("--top-cases", type=int, default=30)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    belief_path = None if args.uniform_determinizations else Path(args.belief_model)
    config = ProbeConfig(
        model_path=Path(args.model),
        belief_model_path=belief_path,
        out_path=Path(args.out),
        num_observations=int(args.num_observations),
        max_games=int(args.max_games),
        seed=int(args.seed),
        opponents=_parse_opponents(str(args.opponents)),
        determinizations=int(args.determinizations),
        min_regret_points=float(args.min_regret_points),
        confidence_z=float(args.confidence_z),
        belief_uniform_mix=float(args.belief_uniform_mix),
        top_cases=int(args.top_cases),
    )
    report = run_policy_regret_probe(config)
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    overall = report["aggregate"]["overall"]
    print(f"Saved policy-regret probe: {config.out_path}")
    print(
        json.dumps(
            {
                "decisions": overall["decisions"],
                "candidate_disagreements": overall["candidate_disagreements"],
                "reliable_errors": overall["reliable_errors"],
                "reliable_error_rate": overall["reliable_error_rate"],
                "top_tags": report["aggregate"]["reliable_error_tags"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
