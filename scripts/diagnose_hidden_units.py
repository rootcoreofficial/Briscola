#!/usr/bin/env python3
"""Confronta utilizzo, ridondanza e fragilità delle unità ReLU di due policy MLP.

La raccolta riusa esattamente il campionamento bilanciato della sonda di simmetria:
stessi seed, due seat, quattro fasi e solo ``PlayerObservation``. Le traiettorie sono
generate dal modello primario (v14 nel protocollo ufficiale); primario e riferimento
vengono poi valutati sugli stessi stati, evitando che una differenza di distribuzione
delle partite venga scambiata per una differenza interna delle reti.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from probe_suit_symmetry import (
    DEFAULT_OPPONENTS,
    PHASES,
    CandidateObservation,
    _collect_candidates,
    _display_path,
    _git_commit,
    _git_worktree_dirty,
    _load_seed_suite,
    _normalize_opponents,
    _select_balanced,
    _sha256_file,
)

from briscola_ai.ai.encoding.observation_encoder import encode_player_observation_2p
from briscola_ai.ai.evaluation.hidden_units import (
    HiddenUnitThresholds,
    analyze_hidden_unit_arrays,
    analyze_suit_ablation_arrays,
)
from briscola_ai.ai.evaluation.suit_symmetry import (
    all_suit_permutations,
    inverse_suit_permutation,
    permute_card_id,
    permute_player_observation,
)
from briscola_ai.ai.models.bc_model import BCModelAgent, MLPBCModel
from briscola_ai.domain.observation import PlayerObservation
from briscola_ai.versioning import get_code_version, get_rules_version

SCHEMA = "briscola.hidden_unit_diagnostic.v1"


def _repo_root() -> Path:
    """Ritorna la root del checkout che contiene lo script."""
    return Path(__file__).resolve().parents[1]


def _encode_batch(agent: BCModelAgent, observations: list[PlayerObservation]) -> tuple[np.ndarray, np.ndarray]:
    """Codifica osservazioni per l'input effettivo della MLP e conserva le action mask."""
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError(f"La diagnostica richiede una MLP, ottenuto {type(agent.model).__name__}")
    model = agent.model
    encoder_dim = model.feature_dim - 40 if model.has_belief_input else model.feature_dim
    inputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for observation in observations:
        encoded = encode_player_observation_2p(observation, version=agent.encoder_version)
        encoder_features = np.asarray(encoded.features, dtype=np.float32)
        if encoder_features.shape != (encoder_dim,):
            raise ValueError(f"Feature encoder {encoder_features.shape}, attese {(encoder_dim,)}")
        policy_features = model.policy_input(encoder_features)
        inputs.append(np.asarray(policy_features, dtype=np.float32))
        masks.append(np.asarray(encoded.action_mask, dtype=bool))
    return np.stack(inputs), np.stack(masks)


def _suit_subset(selected: list[CandidateObservation], *, per_cell: int) -> list[CandidateObservation]:
    """Prende deterministicamente i primi campioni già ordinati di ogni cella opponent/fase."""
    grouped: defaultdict[tuple[str, str], list[CandidateObservation]] = defaultdict(list)
    for candidate in selected:
        grouped[(candidate.opponent, candidate.phase)].append(candidate)
    subset: list[CandidateObservation] = []
    for opponent in sorted({candidate.opponent for candidate in selected}):
        for phase in PHASES:
            cell = grouped[(opponent, phase)]
            if len(cell) < per_cell:
                raise RuntimeError(f"Campioni simmetria insufficienti per {opponent}/{phase}: {len(cell)}/{per_cell}")
            subset.extend(cell[:per_cell])
    return subset


def _suit_orbits(
    agent: BCModelAgent,
    candidates: list[CandidateObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Codifica le 24 rinomine semantiche e costruisce la mappa verso gli action id originali."""
    permutations = all_suit_permutations()
    flattened = [
        permute_player_observation(candidate.observation, permutation)
        for candidate in candidates
        for permutation in permutations
    ]
    inputs, masks = _encode_batch(agent, flattened)
    remap = np.asarray(
        [
            [permute_card_id(action_id, inverse_suit_permutation(permutation)) for action_id in range(40)]
            for permutation in permutations
        ],
        dtype=np.int16,
    )
    return (
        inputs.reshape(len(candidates), 24, -1),
        masks.reshape(len(candidates), 24, 40),
        remap,
    )


def _analyze_agent(
    agent: BCModelAgent,
    observations: list[PlayerObservation],
    suit_candidates: list[CandidateObservation],
    *,
    thresholds: HiddenUnitThresholds,
) -> dict[str, Any]:
    """Esegue diagnostica base e ablation di simmetria sullo stesso agente."""
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("Sono supportate soltanto policy MLP a un livello nascosto")
    inputs, masks = _encode_batch(agent, observations)
    result = analyze_hidden_unit_arrays(agent.model, inputs, masks, thresholds=thresholds)
    orbit_inputs, orbit_masks, remap = _suit_orbits(agent, suit_candidates)
    result["suit_ablation"] = analyze_suit_ablation_arrays(agent.model, orbit_inputs, orbit_masks, remap)
    return result


def _artifact(path: Path, agent: BCModelAgent, *, root: Path) -> dict[str, Any]:
    """Serializza identità e architettura senza copiare metadati di training voluminosi."""
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("Artefatto non MLP")
    return {
        "path": _display_path(path, root=root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "label": agent.model.metadata.get("label"),
        "format": agent.model.metadata.get("format"),
        "encoder_version": agent.encoder_version,
        "feature_dim": agent.model.feature_dim,
        "hidden_dim": int(agent.model.b1.shape[0]),
        "overkill_guard_enabled": agent.overkill_guard_enabled,
    }


def _assessment(analysis: dict[str, Any]) -> dict[str, Any]:
    """Traduce i segnali preregistrati in una decisione prudente sul widening."""
    utilization = analysis["utilization"]
    geometry = analysis["geometry"]
    influence = analysis["influence"]
    spare_capacity = utilization["dead_unit_fraction"] >= 0.10 or geometry["redundant_unit_fraction"] >= 0.25
    fragile = influence["dominant_unit_count"] > 0
    densely_used = utilization["dead_unit_fraction"] <= 0.02 and geometry["redundant_unit_fraction"] <= 0.10

    if spare_capacity:
        recommendation = "do_not_widen_spare_capacity_signal"
        reasons: list[str] = []
        if utilization["dead_unit_fraction"] >= 0.10:
            reasons.append("molte unità quasi inattive")
        if geometry["redundant_unit_fraction"] >= 0.25:
            reasons.append("molte unità quasi duplicate")
        explanation = f"La rete mostra {' e '.join(reasons)}: aumentare la larghezza non è giustificato."
    elif fragile:
        recommendation = "do_not_widen_single_unit_fragility"
        explanation = "Almeno una singola unità cambia troppe decisioni: prima va studiata la robustezza."
    elif densely_used:
        recommendation = "widening_screen_eligible_not_proven"
        explanation = (
            "Non emerge capacità palesemente inutilizzata né fragilità singola; un piccolo A/B di widening è "
            "ammissibile, ma questa diagnostica da sola non dimostra che migliori il gioco."
        )
    else:
        recommendation = "no_clear_capacity_bottleneck"
        explanation = "I segnali sono misti: non c'è evidenza sufficiente per spendere un run su una rete più larga."
    return {
        "recommendation": recommendation,
        "explanation_it": explanation,
        "signals": {
            "spare_capacity": spare_capacity,
            "single_unit_fragility": fragile,
            "dense_utilization": densely_used,
        },
    }


def _comparison(primary: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    """Espone differenze v14-v13 sulle metriche che rispondono direttamente alla decisione."""
    return {
        "dead_unit_fraction_delta": (
            primary["utilization"]["dead_unit_fraction"] - reference["utilization"]["dead_unit_fraction"]
        ),
        "redundant_unit_fraction_delta": (
            primary["geometry"]["redundant_unit_fraction"] - reference["geometry"]["redundant_unit_fraction"]
        ),
        "effective_rank_fraction_delta": (
            primary["geometry"]["effective_rank_fraction"] - reference["geometry"]["effective_rank_fraction"]
        ),
        "max_single_ablation_flip_rate_delta": (
            primary["influence"]["max_single_ablation_flip_rate"]
            - reference["influence"]["max_single_ablation_flip_rate"]
        ),
        "suit_flip_rate_delta": (
            primary["suit_ablation"]["baseline_flip_rate"] - reference["suit_ablation"]["baseline_flip_rate"]
        ),
    }


def _parse_args() -> argparse.Namespace:
    """Definisce la CLI della diagnostica."""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Diagnostica delle 256 unità ReLU di v14 con controllo v13")
    parser.add_argument("--model", type=Path, default=root / "data/models/best_a2c_v14.npz")
    parser.add_argument("--reference-model", type=Path, default=root / "data/models/best_a2c_v13.npz")
    parser.add_argument("--seed-suite", type=Path, default=root / "seed_suites/small_1000.txt")
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--samples-per-cell", type=int, default=256)
    parser.add_argument("--suit-samples-per-cell", type=int, default=16)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument(
        "--out-json",
        type=Path,
        default=root / "docs/reports/evidence/hidden_units_v14.v1.json",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, *, suite_size: int) -> None:
    """Valida quote e path prima della simulazione."""
    if args.seed_count <= 0 or args.seed_count > suite_size:
        raise ValueError(f"--seed-count deve essere tra 1 e {suite_size}")
    if args.samples_per_cell <= 0:
        raise ValueError("--samples-per-cell deve essere > 0")
    if args.suit_samples_per_cell <= 0 or args.suit_samples_per_cell > args.samples_per_cell:
        raise ValueError("--suit-samples-per-cell deve essere in 1..samples-per-cell")


def main() -> int:
    """Raccoglie gli stati, confronta i modelli e scrive JSON deterministico."""
    args = _parse_args()
    root = _repo_root()
    model_path = args.model.resolve()
    reference_path = args.reference_model.resolve()
    suite_path = args.seed_suite.resolve()
    for path in (model_path, reference_path, suite_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    all_seeds = _load_seed_suite(suite_path)
    _validate_args(args, suite_size=len(all_seeds))
    opponents = _normalize_opponents(args.opponents)
    seeds = all_seeds[: args.seed_count]
    primary_agent = BCModelAgent.from_npz(model_path)
    reference_agent = BCModelAgent.from_npz(reference_path)
    thresholds = HiddenUnitThresholds()

    print(
        f"Raccolta v14: {len(seeds)} seed x 2 seat x {len(opponents)} avversari; "
        f"quota {args.samples_per_cell} per cella..."
    )
    candidates, coverage = _collect_candidates(primary_agent, opponents=opponents, seeds=seeds)
    selected = _select_balanced(
        candidates,
        opponents=opponents,
        samples_per_cell=args.samples_per_cell,
        coverage=coverage,
    )
    observations = [candidate.observation for candidate in selected]
    suit_candidates = _suit_subset(selected, per_cell=args.suit_samples_per_cell)

    print(f"Diagnostica primaria su {len(observations)} stati; simmetria su {len(suit_candidates)} stati...")
    primary = _analyze_agent(
        primary_agent,
        observations,
        suit_candidates,
        thresholds=thresholds,
    )
    print("Diagnostica del riferimento sugli stessi stati...")
    reference = _analyze_agent(
        reference_agent,
        observations,
        suit_candidates,
        thresholds=thresholds,
    )

    report = {
        "schema": SCHEMA,
        "method": {
            "anti_cheat": "raccolta e analisi usano soltanto PlayerObservation; GameState resta nel motore",
            "trajectory_policy": _display_path(model_path, root=root),
            "same_observations_for_both_models": True,
            "forward": "relu(x @ w1 + b1) @ w2 + b2; argmax sulle sole azioni legali",
            "single_unit_ablation": "sottrazione esatta hidden[j] * w2[j] dai logits; nessun peso modificato",
            "redundancy": "correlazione Pearson assoluta sulle attivazioni centrate e non costanti",
            "effective_rank": "entropia dello spettro della varianza delle attivazioni centrate",
            "suit_ablation": "24 rinomine semantiche; action id rimappati all'orientamento originale",
        },
        "thresholds": {
            "activation_epsilon": thresholds.activation_epsilon,
            "dead_activation_rate_max": thresholds.dead_activation_rate_max,
            "always_active_rate_min": thresholds.always_active_rate_min,
            "redundant_abs_correlation_min": thresholds.redundant_abs_correlation_min,
            "dominant_ablation_flip_rate_min": thresholds.dominant_ablation_flip_rate_min,
            "spare_capacity_dead_fraction_min": 0.10,
            "spare_capacity_redundant_fraction_min": 0.25,
            "dense_utilization_dead_fraction_max": 0.02,
            "dense_utilization_redundant_fraction_max": 0.10,
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
            "primary": _artifact(model_path, primary_agent, root=root),
            "reference": _artifact(reference_path, reference_agent, root=root),
            "seed_suite": {
                "path": _display_path(suite_path, root=root),
                "sha256": _sha256_file(suite_path),
                "selected_seed_count": len(seeds),
                "selected_seed_sha256": hashlib.sha256(
                    "\n".join(str(seed) for seed in seeds).encode("ascii")
                ).hexdigest(),
            },
        },
        "coverage": coverage,
        "suit_ablation_sample": {
            "selected_observations": len(suit_candidates),
            "samples_per_opponent_phase": args.suit_samples_per_cell,
            "selection_manifest_sha256": hashlib.sha256(
                "\n".join(candidate.selection_sha256 for candidate in suit_candidates).encode("ascii")
            ).hexdigest(),
        },
        "models": {
            "primary": primary,
            "reference": reference,
        },
        "primary_minus_reference": _comparison(primary, reference),
        "decision": _assessment(primary),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report scritto: {_display_path(args.out_json, root=root)}")
    print(f"Decisione: {report['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
