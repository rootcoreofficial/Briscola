#!/usr/bin/env python3
"""Valida su holdout l'ablation congiunta delle unità ReLU dormienti di v14.

L'elenco delle unità è congelato nell'evidenza della diagnostica precedente. Questo
script usa seed indipendenti, confronta policy originale e ablated sugli stessi stati,
misura la simmetria su tutte le 24 rinomine e crea un `.npz` sperimentale per il direct
match standard. Un match JSON può essere incorporato con ``--match-json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from diagnose_hidden_units import _encode_batch
from probe_suit_symmetry import (
    DEFAULT_OPPONENTS,
    CandidateObservation,
    _collect_candidates,
    _display_path,
    _git_commit,
    _git_worktree_dirty,
    _normalize_opponents,
    _select_balanced,
    _sha256_file,
)

from briscola_ai.ai.evaluation.hidden_units import ablate_mlp_hidden_units
from briscola_ai.ai.evaluation.match import SeatFairStats
from briscola_ai.ai.evaluation.round_robin import seat_fair_avg_point_diff_ci, seat_fair_score_rate_ci
from briscola_ai.ai.evaluation.suit_symmetry import (
    all_suit_permutations,
    inverse_suit_permutation,
    permute_card_id,
    permute_player_observation,
)
from briscola_ai.ai.models.bc_model import BCModelAgent, MLPBCModel
from briscola_ai.domain.observation import PlayerObservation
from briscola_ai.versioning import get_code_version, get_rules_version

SCHEMA = "briscola.dormant_unit_ablation.v1"
ACTION_AGREEMENT_GATE = 0.999
MAX_SUIT_FLIP_DELTA = 0.005
MAX_ABS_MATCH_POINT_DIFF = 0.20
MIN_MATCH_CI_LOW = -0.30


def _repo_root() -> Path:
    """Ritorna la root del checkout."""
    return Path(__file__).resolve().parents[1]


def _load_source_evidence(path: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Carica e valida il contratto minimo che congela le unità selezionate in-sample."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "briscola.hidden_unit_diagnostic.v1":
        raise ValueError(f"Schema evidenza sorgente inatteso: {payload.get('schema')!r}")
    units = payload.get("models", {}).get("primary", {}).get("utilization", {}).get("dead_units")
    if not isinstance(units, list) or not units or any(not isinstance(index, int) for index in units):
        raise ValueError("L'evidenza non contiene models.primary.utilization.dead_units validi")
    normalized = tuple(units)
    if len(set(normalized)) != len(normalized):
        raise ValueError("L'evidenza contiene unità dormienti duplicate")
    return payload, normalized


def _masked_probabilities(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Softmax batch float64 sulle sole azioni legali."""
    masked = np.where(masks, np.asarray(logits, dtype=np.float64), -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exponentials = np.where(masks, np.exp(shifted), 0.0)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    """Statistiche finite compatte per delta e divergenze."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not bool(np.all(np.isfinite(flat))):
        raise ValueError("Distribuzione vuota o non finita")
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
        "max": float(np.max(flat)),
    }


def _policy_agreement(
    original: MLPBCModel,
    ablated: MLPBCModel,
    inputs: np.ndarray,
    masks: np.ndarray,
    unit_indices: tuple[int, ...],
    *,
    activation_epsilon: float,
) -> dict[str, Any]:
    """Confronta action, distribuzioni e attività holdout del gruppo congelato."""
    original_logits = original.logits(inputs)
    ablated_logits = ablated.logits(inputs)
    original_actions = np.argmax(np.where(masks, original_logits, -np.inf), axis=1)
    ablated_actions = np.argmax(np.where(masks, ablated_logits, -np.inf), axis=1)
    original_probabilities = _masked_probabilities(original_logits, masks)
    ablated_probabilities = _masked_probabilities(ablated_logits, masks)
    midpoint = 0.5 * (original_probabilities + ablated_probabilities)

    def kl_rows(source: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(source)
        positive = source > 0.0
        terms[positive] = source[positive] * np.log2(source[positive] / midpoint[positive])
        return np.sum(terms, axis=1)

    js_bits = np.clip(0.5 * (kl_rows(original_probabilities) + kl_rows(ablated_probabilities)), 0.0, 1.0)
    hidden = np.maximum(inputs @ original.w1 + original.b1, 0.0)[:, unit_indices]
    activation_rates = np.mean(hidden > activation_epsilon, axis=0)
    changed = original_actions != ablated_actions
    probability_delta = np.max(np.abs(original_probabilities - ablated_probabilities), axis=1)
    logit_delta = np.max(np.abs(original_logits - ablated_logits), axis=1)
    return {
        "observations": int(inputs.shape[0]),
        "action_agreement_rate": float(1.0 - np.mean(changed)),
        "changed_action_count": int(np.sum(changed)),
        "states_with_any_selected_unit_active": int(np.sum(np.any(hidden > activation_epsilon, axis=1))),
        "selected_units_never_active": [int(unit_indices[index]) for index in np.flatnonzero(activation_rates == 0.0)],
        "selected_units_still_below_dead_threshold": int(np.sum(activation_rates <= 0.001)),
        "max_abs_probability_delta": _distribution(probability_delta),
        "max_abs_logit_delta": _distribution(logit_delta),
        "js_divergence_bits": _distribution(js_bits),
    }


def _remap_table() -> np.ndarray:
    """Costruisce `permutation x action` verso l'orientamento dell'identità."""
    return np.asarray(
        [
            [permute_card_id(action_id, inverse_suit_permutation(permutation)) for action_id in range(40)]
            for permutation in all_suit_permutations()
        ],
        dtype=np.int16,
    )


def _suit_comparison(
    agent: BCModelAgent,
    ablated: MLPBCModel,
    candidates: list[CandidateObservation],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    """Confronta la simmetria sull'intero holdout senza materializzare 98k feature insieme."""
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("La policy originale deve essere MLP")
    permutations = all_suit_permutations()
    remap = _remap_table()
    original_flip_count = 0
    ablated_flip_count = 0
    original_any_flip_count = 0
    ablated_any_flip_count = 0
    policy_disagreement_count = 0
    comparisons = 0

    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        permuted = [
            permute_player_observation(candidate.observation, permutation)
            for candidate in chunk
            for permutation in permutations
        ]
        inputs, masks = _encode_batch(agent, permuted)
        original_logits = agent.model.logits(inputs)
        ablated_logits = ablated.logits(inputs)
        original_raw = np.argmax(np.where(masks, original_logits, -np.inf), axis=1)
        ablated_raw = np.argmax(np.where(masks, ablated_logits, -np.inf), axis=1)
        permutation_indices = np.tile(np.arange(24), len(chunk))
        original_actions = remap[permutation_indices, original_raw].reshape(len(chunk), 24)
        ablated_actions = remap[permutation_indices, ablated_raw].reshape(len(chunk), 24)
        original_flips = original_actions[:, 1:] != original_actions[:, [0]]
        ablated_flips = ablated_actions[:, 1:] != ablated_actions[:, [0]]

        original_flip_count += int(np.sum(original_flips))
        ablated_flip_count += int(np.sum(ablated_flips))
        original_any_flip_count += int(np.sum(np.any(original_flips, axis=1)))
        ablated_any_flip_count += int(np.sum(np.any(ablated_flips, axis=1)))
        policy_disagreement_count += int(np.sum(original_actions != ablated_actions))
        comparisons += len(chunk) * 23

    original_rate = original_flip_count / comparisons
    ablated_rate = ablated_flip_count / comparisons
    orbit_decisions = len(candidates) * 24
    return {
        "observations": len(candidates),
        "nonidentity_comparisons": comparisons,
        "original_flip_rate": original_rate,
        "ablated_flip_rate": ablated_rate,
        "flip_rate_delta": ablated_rate - original_rate,
        "original_any_flip_observation_rate": original_any_flip_count / len(candidates),
        "ablated_any_flip_observation_rate": ablated_any_flip_count / len(candidates),
        "policy_agreement_across_orbit": 1.0 - policy_disagreement_count / orbit_decisions,
        "policy_disagreement_count_across_orbit": policy_disagreement_count,
    }


def _candidate_metadata(
    source_model: MLPBCModel,
    *,
    source_path: Path,
    source_sha256: str,
    evidence_path: Path,
    evidence_sha256: str,
    units: tuple[int, ...],
    root: Path,
) -> dict[str, Any]:
    """Crea metadati sintetici: il candidato è diagnostico e non va esposto come ufficiale."""
    metadata = dict(source_model.metadata)
    metadata.update(
        {
            "label": "v14 ablation congiunta unità dormienti (holdout)",
            "description_it": (
                "Candidato diagnostico non promosso: pesi v14 invariati salvo le righe w2 delle unità quasi "
                "inattive, azzerate congiuntamente per un controllo causale su seed indipendenti."
            ),
            "inference_overkill_guard": False,
            "dormant_unit_ablation": {
                "format": SCHEMA,
                "source_model": _display_path(source_path, root=root),
                "source_model_sha256": source_sha256,
                "source_evidence": _display_path(evidence_path, root=root),
                "source_evidence_sha256": evidence_sha256,
                "unit_count": len(units),
                "unit_indices": list(units),
            },
        }
    )
    return metadata


def _save_candidate(path: Path, model: MLPBCModel) -> None:
    """Salva la MLP e gli eventuali array belief embedded senza introdurre metriche lunghe."""
    arrays: dict[str, Any] = {
        "w1": model.w1,
        "b1": model.b1,
        "w2": model.w2,
        "b2": model.b2,
        "metadata_json": np.asarray(json.dumps(model.metadata, ensure_ascii=True, sort_keys=True)),
    }
    for name in ("belief_w1", "belief_b1", "belief_w2", "belief_b2"):
        value = getattr(model, name)
        if value is not None:
            arrays[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _load_match(
    path: Path,
    *,
    root: Path,
    expected_agent_a: str,
    expected_agent_b: str,
) -> dict[str, Any]:
    """Valida il direct match standard e aggiunge CI corrette sulle coppie seat-fair."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("engine") != "numba" or payload.get("mode") != "seat_fair":
        raise ValueError("Il match deve usare engine=numba e mode=seat_fair")
    seed_suite = payload.get("seed_suite", {})
    if seed_suite.get("range_start") != 1_000_000 or seed_suite.get("range_step") != 1:
        raise ValueError("Il match deve usare seed range holdout da 1.000.000 con step 1")
    stats = SeatFairStats(**payload["stats"])
    if stats.num_games != 10_000:
        raise ValueError(f"Il match deve contenere 10.000 partite, ottenute {stats.num_games}")
    agents = payload.get("agents")
    expected_agents = {"agent0": expected_agent_a, "agent1": expected_agent_b}
    if agents != expected_agents:
        raise ValueError(f"Agenti del match inattesi: {agents!r}; attesi {expected_agents!r}")
    if (stats.agent_a_name, stats.agent_b_name) != (expected_agent_a, expected_agent_b):
        raise ValueError("I nomi aggregati del match non coincidono con gli agenti richiesti")
    diff_ci = seat_fair_avg_point_diff_ci(stats, confidence=0.95)
    score_ci = seat_fair_score_rate_ci(stats, confidence=0.95)
    score_rate = (stats.wins_agent_a + 0.5 * stats.draws) / stats.num_games
    return {
        "source": _display_path(path, root=root),
        "source_sha256": _sha256_file(path),
        "stats": asdict(stats),
        "avg_point_diff_ci95": asdict(diff_ci) if diff_ci is not None else None,
        "score_rate": score_rate,
        "score_rate_ci95": asdict(score_ci),
    }


def _decision(agreement: dict[str, Any], suit: dict[str, Any], match: dict[str, Any] | None) -> dict[str, Any]:
    """Applica i gate preregistrati senza promuovere automaticamente alcun modello."""
    checks: dict[str, bool | None] = {
        "action_agreement": agreement["action_agreement_rate"] >= ACTION_AGREEMENT_GATE,
        "suit_flip_delta": abs(suit["flip_rate_delta"]) <= MAX_SUIT_FLIP_DELTA,
        "match_point_estimate": None,
        "match_ci_lower_bound": None,
    }
    if match is not None:
        avg_diff = float(match["stats"]["avg_point_diff_agent_a_minus_agent_b"])
        ci_low = float(match["avg_point_diff_ci95"]["low"])
        checks["match_point_estimate"] = abs(avg_diff) <= MAX_ABS_MATCH_POINT_DIFF
        checks["match_ci_lower_bound"] = ci_low >= MIN_MATCH_CI_LOW
    if match is None:
        verdict = "pending_direct_match"
    elif all(value is True for value in checks.values()):
        verdict = "go_reinitialize_small_dormant_subset"
    else:
        verdict = "stop_dormant_capacity_reuse"
    return {
        "verdict": verdict,
        "checks": checks,
        "note_it": (
            "Un GO autorizza solo un piccolo screening di reinizializzazione e nuova distillazione; "
            "non autorizza potatura, promozione o modifica di v14 live."
        ),
    }


def _parse_args() -> argparse.Namespace:
    """Definisce il protocollo ufficiale e i path degli artefatti."""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Holdout causale delle unità ReLU dormienti di v14")
    parser.add_argument("--model", type=Path, default=root / "data/models/best_a2c_v14.npz")
    parser.add_argument(
        "--source-evidence",
        type=Path,
        default=root / "docs/reports/evidence/hidden_units_v14.v1.json",
    )
    parser.add_argument("--holdout-seed-start", type=int, default=1_000_000)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--samples-per-cell", type=int, default=256)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--suit-chunk-size", type=int, default=64)
    parser.add_argument(
        "--candidate-out",
        type=Path,
        default=root / "data/models/v14_dormant123_ablated_holdout_v0.npz",
    )
    parser.add_argument("--match-json", type=Path)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=root / "docs/reports/evidence/dormant_unit_ablation_v14.v1.json",
    )
    return parser.parse_args()


def main() -> int:
    """Esegue il controllo holdout, salva candidato ed evidenza deterministica."""
    args = _parse_args()
    root = _repo_root()
    model_path = args.model.resolve()
    evidence_path = args.source_evidence.resolve()
    if args.holdout_seed_start != 1_000_000:
        raise ValueError("Il protocollo v0 congela --holdout-seed-start a 1.000.000")
    if args.seed_count <= 0 or args.samples_per_cell <= 0 or args.suit_chunk_size <= 0:
        raise ValueError("seed-count, samples-per-cell e suit-chunk-size devono essere > 0")
    for path in (model_path, evidence_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_evidence, units = _load_source_evidence(evidence_path)
    source_sha256 = _sha256_file(model_path)
    expected_sha256 = source_evidence.get("artifacts", {}).get("primary", {}).get("sha256")
    if source_sha256 != expected_sha256:
        raise ValueError(f"SHA modello {source_sha256} diverso dall'evidenza {expected_sha256}")
    agent = BCModelAgent.from_npz(model_path)
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("Il modello deve essere una MLP")

    evidence_sha256 = _sha256_file(evidence_path)
    metadata = _candidate_metadata(
        agent.model,
        source_path=model_path,
        source_sha256=source_sha256,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        units=units,
        root=root,
    )
    ablated = ablate_mlp_hidden_units(agent.model, units, metadata=metadata)
    candidate_path = args.candidate_out.resolve()
    _save_candidate(candidate_path, ablated)

    seeds = list(range(args.holdout_seed_start, args.holdout_seed_start + args.seed_count))
    opponents = _normalize_opponents(args.opponents)
    print(
        f"Raccolta holdout: seed {seeds[0]}..{seeds[-1]}, 2 seat, {len(opponents)} avversari; "
        f"quota {args.samples_per_cell} per cella..."
    )
    candidates, coverage = _collect_candidates(agent, opponents=opponents, seeds=seeds)
    selected = _select_balanced(
        candidates,
        opponents=opponents,
        samples_per_cell=args.samples_per_cell,
        coverage=coverage,
    )
    observations: list[PlayerObservation] = [candidate.observation for candidate in selected]
    inputs, masks = _encode_batch(agent, observations)
    activation_epsilon = float(source_evidence["thresholds"]["activation_epsilon"])
    agreement = _policy_agreement(
        agent.model,
        ablated,
        inputs,
        masks,
        units,
        activation_epsilon=activation_epsilon,
    )
    print(f"Agreement holdout: {agreement['action_agreement_rate']:.6%}")
    print(f"Simmetria completa: {len(selected)} osservazioni x 24 rinomine...")
    suit = _suit_comparison(agent, ablated, selected, chunk_size=args.suit_chunk_size)
    match = None
    if args.match_json is not None:
        candidate_agent_name = BCModelAgent.from_npz(candidate_path).name
        match = _load_match(
            args.match_json.resolve(),
            root=root,
            expected_agent_a=candidate_agent_name,
            expected_agent_b=agent.name,
        )

    report = {
        "schema": SCHEMA,
        "method": {
            "anti_cheat": "raccolta e policy ricevono solo PlayerObservation; GameState resta nel motore",
            "unit_selection": "congelata dall'evidenza in-sample prima del holdout",
            "joint_ablation": "righe w2 selezionate poste a zero; tutti gli altri pesi identici a v14",
            "holdout_seed_range": [seeds[0], seeds[-1]],
            "same_observations_for_original_and_ablated": True,
            "suit_permutations": 24,
        },
        "gates": {
            "action_agreement_rate_min": ACTION_AGREEMENT_GATE,
            "max_abs_suit_flip_rate_delta": MAX_SUIT_FLIP_DELTA,
            "max_abs_match_avg_point_diff": MAX_ABS_MATCH_POINT_DIFF,
            "match_avg_point_diff_ci95_low_min": MIN_MATCH_CI_LOW,
        },
        "versions": {
            "code": get_code_version(),
            "rules": get_rules_version(),
            "git_commit": _git_commit(root),
            "git_worktree_dirty": _git_worktree_dirty(root),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "artifacts": {
            "source_model": {
                "path": _display_path(model_path, root=root),
                "sha256": source_sha256,
                "size_bytes": model_path.stat().st_size,
            },
            "source_evidence": {
                "path": _display_path(evidence_path, root=root),
                "sha256": evidence_sha256,
            },
            "candidate": {
                "path_local": _display_path(candidate_path, root=root),
                "sha256": _sha256_file(candidate_path),
                "size_bytes": candidate_path.stat().st_size,
            },
        },
        "ablated_units": {"count": len(units), "indices": list(units)},
        "coverage": coverage,
        "holdout_seed_manifest_sha256": hashlib.sha256(
            "\n".join(str(seed) for seed in seeds).encode("ascii")
        ).hexdigest(),
        "policy_agreement": agreement,
        "suit_symmetry": suit,
        "direct_match": match,
        "decision": _decision(agreement, suit, match),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Candidato: {_display_path(candidate_path, root=root)}")
    print(f"Report: {_display_path(args.out_json, root=root)}")
    print(f"Verdetto: {report['decision']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
