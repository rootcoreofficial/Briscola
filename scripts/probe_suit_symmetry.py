#!/usr/bin/env python3
"""
Misura quanto una policy dipende dai nomi arbitrari dei quattro semi.

La sonda gioca una suite seat-fair contro piu' avversari, conserva soltanto
``PlayerObservation`` lecite e seleziona una quota bilanciata per avversario e fase
tramite un ranking SHA-256 indipendente dall'ordine di raccolta. Ogni osservazione viene
poi valutata in batch sulle 24 rinomine dei semi.

Il report distingue due domande:

* identita' vs le altre 23 rinomine: quanto cambia la policy rispetto allo stato reale;
* tutte le 276 coppie tra le 24 rinomine: quanto e' stabile l'intera orbita di simmetria.

Il bootstrap ricampiona osservazioni complete, non le 23/276 comparazioni correlate.
Durante la simulazione il ``GameState`` serve soltanto al motore e non viene mai passato
alla policy, salvato o serializzato. Seed, seat e ordinale sono metadati di riproduzione
mai forniti al modello; il report non include mani avversarie o ordine del mazzo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents import Agent, build_agent, unknown_live_card_count
from briscola_ai.ai.evaluation.suit_symmetry import (
    all_suit_permutations,
    evaluate_observation_suit_symmetry,
)
from briscola_ai.ai.models.bc_model import BCModelAgent
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.state import new_game_state
from briscola_ai.versioning import get_code_version, get_rules_version

DEFAULT_OPPONENTS = ("mirror", "heuristic_trump_saver", "heuristic_v1", "random")
PHASES = ("early", "mid", "pimc_window", "endgame")
PAIR_LEFT, PAIR_RIGHT = np.triu_indices(24, k=1)
PAIR_COUNT = int(PAIR_LEFT.size)


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """Campione lecito e contesto minimo di raccolta utile per breakdown e riproduzione."""

    observation: PlayerObservation
    opponent: str
    phase: str
    game_seed: int
    policy_seat: int
    decision_ordinal: int
    position: str
    legal_count: int
    unknown_live_cards: int
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class ProbedObservation:
    """Metriche numeriche di una osservazione, non destinate alla serializzazione diretta."""

    candidate: CandidateObservation
    baseline_action_id: int
    baseline_top2_gap: float
    permutation_actions: np.ndarray
    permutation_top2_gaps: np.ndarray
    identity_flips: np.ndarray
    identity_near_ties: np.ndarray
    identity_js_bits: np.ndarray
    identity_max_abs_delta: np.ndarray
    pairwise_flips: np.ndarray
    pairwise_near_ties: np.ndarray
    pairwise_js_bits: np.ndarray
    pairwise_max_abs_delta: np.ndarray


def _repo_root() -> Path:
    """Ritorna la root del checkout che contiene questo script."""
    return Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    """Calcola SHA-256 a blocchi, senza caricare interamente gli asset grandi."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(*parts: object) -> str:
    """Hash stabile di parti scalari, con JSON canonico per evitare concatenazioni ambigue."""
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_seed(*parts: object) -> int:
    """Deriva un seed a 64 bit da SHA-256, stabile tra processi Python."""
    return int(_stable_digest(*parts)[:16], 16)


def _display_path(path: Path, *, root: Path) -> str:
    """Preferisce un path relativo al repo, piu' portabile nel report."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_commit(root: Path) -> str | None:
    """Legge il commit del checkout best-effort, senza rendere Git un requisito runtime."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    value = completed.stdout.strip()
    return value or None


def _git_worktree_dirty(root: Path) -> bool | None:
    """Indica best-effort se il checkout contiene modifiche o file non tracciati."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return bool(completed.stdout.strip())


def _load_seed_suite(path: Path) -> list[int]:
    """Carica interi base 10, ignorando commenti e righe vuote."""
    seeds: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            seeds.append(int(line, 10))
        except ValueError as exc:
            raise ValueError(f"Seed non valido in {path}:{line_number}: {line!r}") from exc
    if not seeds:
        raise ValueError(f"Seed suite vuota: {path}")
    return seeds


def _normalize_opponents(raw_values: list[str]) -> tuple[str, ...]:
    """Accetta sia argomenti separati sia una lista separata da virgole."""
    names = tuple(part.strip() for raw in raw_values for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("Specificare almeno un avversario")
    if len(set(names)) != len(names):
        raise ValueError(f"Avversari duplicati: {names}")
    return names


def _phase(observation: PlayerObservation) -> str:
    """Assegna una fase usando solo mazzo pubblico e numero di carte vive ignote."""
    unknown = unknown_live_card_count(observation)
    if observation.deck_size == 0:
        return "endgame"
    if unknown <= 8:
        return "pimc_window"
    if observation.deck_size <= 10:
        return "mid"
    return "early"


def _build_opponent(name: str, model_agent: BCModelAgent) -> Agent:
    """Costruisce il roster; ``mirror`` riusa esattamente la policy sotto esame."""
    if name == "mirror":
        return model_agent
    return build_agent(name)


def _collect_candidates(
    model_agent: BCModelAgent,
    *,
    opponents: tuple[str, ...],
    seeds: list[int],
) -> tuple[list[CandidateObservation], dict[str, Any]]:
    """
    Gioca due seat per seed e conserva soltanto osservazioni non forzate del modello.

    Gli RNG sono separati per giocatore e derivati dal contesto completo. In questo modo
    aggiungere un avversario o cambiare l'ordine del roster non altera le traiettorie gia'
    esistenti.
    """
    candidates: list[CandidateObservation] = []
    candidate_counts: Counter[tuple[str, str]] = Counter()
    forced_counts: Counter[tuple[str, str]] = Counter()
    model_decisions: Counter[str] = Counter()

    for opponent_name in opponents:
        opponent_agent = _build_opponent(opponent_name, model_agent)
        for game_seed in seeds:
            for policy_seat in (0, 1):
                # Lo stato completo resta confinato al loop del motore. Non entra mai in un campione.
                player_names = ["policy" if index == policy_seat else "opponent" for index in (0, 1)]
                state = new_game_state(2, player_names, seed=game_seed)
                agents = (
                    (model_agent if policy_seat == 0 else opponent_agent),
                    (model_agent if policy_seat == 1 else opponent_agent),
                )
                rngs = tuple(
                    random.Random(
                        _stable_seed(
                            "suit-symmetry-agent-rng-v1",
                            opponent_name,
                            game_seed,
                            policy_seat,
                            player_index,
                            "policy" if player_index == policy_seat else "opponent",
                        )
                    )
                    for player_index in (0, 1)
                )
                policy_decision_ordinal = 0

                while not state.game_over:
                    player_index = state.current_turn
                    observation = make_player_observation(state, player_index)
                    if player_index == policy_seat:
                        model_decisions[opponent_name] += 1
                        phase = _phase(observation)
                        if len(observation.hand) >= 2:
                            unknown = unknown_live_card_count(observation)
                            selection_hash = _stable_digest(
                                "suit-symmetry-sample-v1",
                                opponent_name,
                                game_seed,
                                policy_seat,
                                policy_decision_ordinal,
                            )
                            candidates.append(
                                CandidateObservation(
                                    observation=observation,
                                    opponent=opponent_name,
                                    phase=phase,
                                    game_seed=game_seed,
                                    policy_seat=policy_seat,
                                    decision_ordinal=policy_decision_ordinal,
                                    position="lead" if not observation.table_cards else "response",
                                    legal_count=len(observation.hand),
                                    unknown_live_cards=unknown,
                                    selection_sha256=selection_hash,
                                )
                            )
                            candidate_counts[(opponent_name, phase)] += 1
                        else:
                            forced_counts[(opponent_name, phase)] += 1
                        policy_decision_ordinal += 1

                    card_index = int(agents[player_index].choose_card_index(observation, rng=rngs[player_index]))
                    if card_index < 0 or card_index >= len(observation.hand):
                        raise ValueError(
                            f"{agents[player_index].name} ha restituito card_index={card_index} "
                            f"con mano di {len(observation.hand)} carte"
                        )
                    state, result = step(
                        state,
                        PlayCardAction(player_index=player_index, card_index=card_index),
                    )
                    if result.error is not None:
                        raise RuntimeError(f"Errore del motore durante la raccolta: {result.error}")

    coverage = {
        "games": len(opponents) * len(seeds) * 2,
        "eligible_nonforced_observations": len(candidates),
        "forced_observations_skipped": int(sum(forced_counts.values())),
        "model_decisions": {name: model_decisions[name] for name in opponents},
        "cells": {
            f"{name}/{phase}": {
                "candidates": candidate_counts[(name, phase)],
                "forced_skipped": forced_counts[(name, phase)],
            }
            for name in opponents
            for phase in PHASES
        },
    }
    return candidates, coverage


def _select_balanced(
    candidates: list[CandidateObservation],
    *,
    opponents: tuple[str, ...],
    samples_per_cell: int,
    coverage: dict[str, Any],
) -> list[CandidateObservation]:
    """Seleziona i digest piu' bassi in ogni cella e fallisce se una quota non e' coperta."""
    selected: list[CandidateObservation] = []
    shortages: list[str] = []
    for opponent in opponents:
        for phase in PHASES:
            pool = [item for item in candidates if item.opponent == opponent and item.phase == phase]
            pool.sort(key=lambda item: (item.selection_sha256, item.game_seed, item.policy_seat, item.decision_ordinal))
            cell_key = f"{opponent}/{phase}"
            if len(pool) < samples_per_cell:
                shortages.append(f"{cell_key}: {len(pool)}/{samples_per_cell}")
                coverage["cells"][cell_key]["selected"] = len(pool)
                continue
            chosen = pool[:samples_per_cell]
            selected.extend(chosen)
            coverage["cells"][cell_key]["selected"] = len(chosen)
    if shortages:
        details = ", ".join(shortages)
        raise RuntimeError(f"Quota di campionamento insufficiente ({details})")

    selected.sort(key=lambda item: (item.opponent, PHASES.index(item.phase), item.selection_sha256))
    coverage["samples_per_opponent_phase"] = samples_per_cell
    coverage["selected_observations"] = len(selected)
    coverage["selection_manifest_sha256"] = hashlib.sha256(
        "\n".join(item.selection_sha256 for item in selected).encode("ascii")
    ).hexdigest()
    return selected


def _pairwise_metrics(
    probabilities: np.ndarray,
    permutation_top2_gaps: np.ndarray,
    *,
    near_tie_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcola flip, near-tie, JS e delta sulle 276 coppie."""
    if probabilities.shape != (24, 40):
        raise ValueError(f"Probabilita' rimappate con shape {probabilities.shape}, attesa (24, 40)")
    if permutation_top2_gaps.shape != (24,):
        raise ValueError(f"Gap top-2 con shape {permutation_top2_gaps.shape}, attesa (24,)")
    left = probabilities[PAIR_LEFT]
    right = probabilities[PAIR_RIGHT]
    middle = 0.5 * (left + right)

    def kl_rows(source: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(source, dtype=np.float64)
        positive = source > 0.0
        terms[positive] = source[positive] * np.log2(source[positive] / middle[positive])
        return np.sum(terms, axis=1)

    js_bits = np.clip(0.5 * (kl_rows(left) + kl_rows(right)), 0.0, 1.0)
    actions = np.argmax(probabilities, axis=1)
    flips = actions[PAIR_LEFT] != actions[PAIR_RIGHT]
    near_ties = (permutation_top2_gaps[PAIR_LEFT] <= near_tie_threshold) | (
        permutation_top2_gaps[PAIR_RIGHT] <= near_tie_threshold
    )
    max_delta = np.max(np.abs(left - right), axis=1)
    return (
        flips.astype(bool),
        near_ties.astype(bool),
        js_bits.astype(np.float64),
        max_delta.astype(np.float64),
    )


def _probe_one(
    model_agent: BCModelAgent, candidate: CandidateObservation, *, near_tie_threshold: float
) -> ProbedObservation:
    """Esegue il batch di 24 permutazioni e valida il controllo identita'."""
    result = evaluate_observation_suit_symmetry(
        model_agent,
        candidate.observation,
        near_tie_threshold=near_tie_threshold,
    )
    comparisons = result.comparisons
    if len(comparisons) != 24 or not comparisons[0].is_identity:
        raise AssertionError("Ordine delle permutazioni inatteso: l'identita' deve essere l'indice 0")
    identity = comparisons[0]
    if (
        not identity.agreement
        or identity.remapped_action_id != result.baseline_action_id
        or identity.js_divergence_bits > 1e-15
        or identity.max_abs_probability_delta > 1e-15
    ):
        raise AssertionError(f"Controllo identita' fallito: {identity}")

    raw_probabilities = getattr(result, "remapped_probabilities", None)
    if raw_probabilities is None:
        raise RuntimeError(
            "Il modulo suit_symmetry non espone remapped_probabilities: impossibile calcolare le 276 coppie"
        )
    permutation_actions = np.asarray([comparison.remapped_action_id for comparison in comparisons], dtype=np.int16)
    permutation_top2_gaps = np.asarray(
        [comparison.remapped_top2_gap for comparison in comparisons],
        dtype=np.float64,
    )
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    pair_flips, pair_near_ties, pair_js, pair_delta = _pairwise_metrics(
        probabilities,
        permutation_top2_gaps,
        near_tie_threshold=near_tie_threshold,
    )
    identity_flips = np.asarray([not comparison.agreement for comparison in comparisons[1:]], dtype=bool)
    identity_near_ties = np.asarray([comparison.near_tie for comparison in comparisons[1:]], dtype=bool)
    expected_identity_near_ties = (permutation_top2_gaps[0] <= near_tie_threshold) | (
        permutation_top2_gaps[1:] <= near_tie_threshold
    )
    if not np.array_equal(identity_near_ties, expected_identity_near_ties):
        raise AssertionError("Flag near-tie incoerenti con i gap top-2 delle permutazioni")
    identity_js = np.asarray([comparison.js_divergence_bits for comparison in comparisons[1:]], dtype=np.float64)
    identity_delta = np.asarray(
        [comparison.max_abs_probability_delta for comparison in comparisons[1:]],
        dtype=np.float64,
    )
    return ProbedObservation(
        candidate=candidate,
        baseline_action_id=result.baseline_action_id,
        baseline_top2_gap=result.baseline_top2_gap,
        permutation_actions=permutation_actions,
        permutation_top2_gaps=permutation_top2_gaps,
        identity_flips=identity_flips,
        identity_near_ties=identity_near_ties,
        identity_js_bits=identity_js,
        identity_max_abs_delta=identity_delta,
        pairwise_flips=pair_flips,
        pairwise_near_ties=pair_near_ties,
        pairwise_js_bits=pair_js,
        pairwise_max_abs_delta=pair_delta,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    """Statistiche descrittive finite con quantili deterministici."""
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


OBSERVATION_METRIC_NAMES = (
    "identity_flip_rate",
    "identity_any_flip_rate",
    "identity_mean_js_bits",
    "identity_max_js_bits",
    "identity_mean_max_abs_delta",
    "pairwise_flip_rate",
    "pairwise_mean_js_bits",
    "pairwise_mean_max_abs_delta",
    "baseline_near_tie_rate",
    "any_permutation_near_tie_rate",
    "identity_near_tie_comparison_rate",
    "pairwise_near_tie_comparison_rate",
)


def _observation_metric_matrix(
    records: list[ProbedObservation],
    *,
    near_tie_threshold: float,
) -> np.ndarray:
    """Una riga per osservazione: e' questa la matrice ricampionata dal bootstrap."""
    return np.asarray(
        [
            (
                float(np.mean(record.identity_flips)),
                float(np.any(record.identity_flips)),
                float(np.mean(record.identity_js_bits)),
                float(np.max(record.identity_js_bits)),
                float(np.mean(record.identity_max_abs_delta)),
                float(np.mean(record.pairwise_flips)),
                float(np.mean(record.pairwise_js_bits)),
                float(np.mean(record.pairwise_max_abs_delta)),
                float(record.permutation_top2_gaps[0] <= near_tie_threshold),
                float(np.any(record.permutation_top2_gaps <= near_tie_threshold)),
                float(np.mean(record.identity_near_ties)),
                float(np.mean(record.pairwise_near_ties)),
            )
            for record in records
        ],
        dtype=np.float64,
    )


def _bootstrap_columns(
    matrix: np.ndarray,
    *,
    metric_names: tuple[str, ...],
    repetitions: int,
    seed: int,
    namespace: str,
) -> dict[str, dict[str, float]]:
    """CI percentile 95% ricampionando righe con rimpiazzo, in chunk a memoria limitata."""
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"Matrice bootstrap invalida: {matrix.shape}")
    if matrix.shape[1] != len(metric_names):
        raise ValueError(f"Nomi metriche incoerenti: {len(metric_names)} per {matrix.shape[1]} colonne")
    estimates = np.mean(matrix, axis=0)
    if matrix.shape[0] == 1:
        lows = highs = estimates
    else:
        rng = np.random.default_rng(_stable_seed("suit-symmetry-bootstrap-v1", seed, namespace))
        replicas = np.empty((repetitions, matrix.shape[1]), dtype=np.float64)
        chunk_size = 32
        for start in range(0, repetitions, chunk_size):
            stop = min(start + chunk_size, repetitions)
            indices = rng.integers(0, matrix.shape[0], size=(stop - start, matrix.shape[0]))
            replicas[start:stop] = np.mean(matrix[indices], axis=1)
        lows = np.quantile(replicas, 0.025, axis=0)
        highs = np.quantile(replicas, 0.975, axis=0)
    return {
        name: {
            "estimate": float(estimates[index]),
            "ci95_low": float(lows[index]),
            "ci95_high": float(highs[index]),
        }
        for index, name in enumerate(metric_names)
    }


def _excluding_near_ties(
    records: list[ProbedObservation],
    *,
    pairwise: bool,
) -> dict[str, float | int | None]:
    """Filtra singoli confronti fragili, conservando quelli validi della stessa osservazione."""
    valid_flip_chunks: list[np.ndarray] = []
    any_flip_by_observation: list[bool] = []
    for record in records:
        flips = record.pairwise_flips if pairwise else record.identity_flips
        near_ties = record.pairwise_near_ties if pairwise else record.identity_near_ties
        valid = ~near_ties
        if bool(np.any(valid)):
            valid_flip_chunks.append(flips[valid])
            any_flip_by_observation.append(bool(np.any(flips[valid])))

    if not valid_flip_chunks:
        return {
            "observations": 0,
            "total_observations": len(records),
            "observations_without_valid_comparisons": len(records),
            "comparisons": 0,
            "agreement_rate": None,
            "flip_rate": None,
            "any_flip_observation_rate": None,
        }
    flips = np.concatenate(valid_flip_chunks)
    return {
        "observations": len(any_flip_by_observation),
        "total_observations": len(records),
        "observations_without_valid_comparisons": len(records) - len(any_flip_by_observation),
        "comparisons": int(flips.size),
        "agreement_rate": float(1.0 - np.mean(flips)),
        "flip_rate": float(np.mean(flips)),
        "any_flip_observation_rate": float(np.mean(any_flip_by_observation)),
    }


def _aggregate(
    records: list[ProbedObservation],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
    namespace: str,
    near_tie_threshold: float,
) -> dict[str, Any]:
    """Aggrega un gruppo mantenendo separati confronti e unita' statistiche."""
    if not records:
        raise ValueError(f"Gruppo vuoto: {namespace}")
    identity_flips = np.concatenate([record.identity_flips for record in records])
    identity_near_ties = np.concatenate([record.identity_near_ties for record in records])
    identity_js = np.concatenate([record.identity_js_bits for record in records])
    identity_delta = np.concatenate([record.identity_max_abs_delta for record in records])
    pair_flips = np.concatenate([record.pairwise_flips for record in records])
    pair_near_ties = np.concatenate([record.pairwise_near_ties for record in records])
    pair_js = np.concatenate([record.pairwise_js_bits for record in records])
    pair_delta = np.concatenate([record.pairwise_max_abs_delta for record in records])
    baseline_top2_gaps = np.asarray([record.baseline_top2_gap for record in records], dtype=np.float64)
    permutation_top2_gaps = np.concatenate([record.permutation_top2_gaps for record in records])
    metric_matrix = _observation_metric_matrix(records, near_tie_threshold=near_tie_threshold)
    return {
        "observations": len(records),
        "identity_vs_23": {
            "comparisons_per_observation": 23,
            "comparisons": int(identity_flips.size),
            "agreement_rate": float(1.0 - np.mean(identity_flips)),
            "flip_rate": float(np.mean(identity_flips)),
            "any_flip_observation_rate": float(np.mean(np.any(np.stack([r.identity_flips for r in records]), axis=1))),
            "near_tie_comparison_rate": float(np.mean(identity_near_ties)),
            "js_divergence_bits": _distribution(identity_js),
            "max_abs_probability_delta": _distribution(identity_delta),
            "excluding_near_ties": _excluding_near_ties(
                records,
                pairwise=False,
            ),
        },
        "all_276_pairs": {
            "pairs_per_observation": PAIR_COUNT,
            "comparisons": int(pair_flips.size),
            "agreement_rate": float(1.0 - np.mean(pair_flips)),
            "flip_rate": float(np.mean(pair_flips)),
            "near_tie_comparison_rate": float(np.mean(pair_near_ties)),
            "js_divergence_bits": _distribution(pair_js),
            "max_abs_probability_delta": _distribution(pair_delta),
            "excluding_near_ties": _excluding_near_ties(
                records,
                pairwise=True,
            ),
        },
        "baseline_top2_gap": _distribution(baseline_top2_gaps),
        "all_permutations_top2_gap": _distribution(permutation_top2_gaps),
        "near_tie_threshold": near_tie_threshold,
        # Chiave mantenuta per compatibilita': indica esclusivamente la distribuzione baseline.
        "near_tie_observation_rate": float(np.mean(baseline_top2_gaps <= near_tie_threshold)),
        "baseline_near_tie_observation_rate": float(np.mean(baseline_top2_gaps <= near_tie_threshold)),
        "any_permutation_near_tie_observation_rate": float(
            np.mean([np.any(record.permutation_top2_gaps <= near_tie_threshold) for record in records])
        ),
        "permutation_near_tie_rate": float(np.mean(permutation_top2_gaps <= near_tie_threshold)),
        "bootstrap_observation_ci95": _bootstrap_columns(
            metric_matrix,
            metric_names=OBSERVATION_METRIC_NAMES,
            repetitions=bootstrap_reps,
            seed=bootstrap_seed,
            namespace=namespace,
        ),
    }


def _grouped_breakdown(
    records: list[ProbedObservation],
    *,
    key,
    namespace: str,
    bootstrap_reps: int,
    bootstrap_seed: int,
    near_tie_threshold: float,
) -> dict[str, Any]:
    """Costruisce breakdown ordinati senza generare gruppi vuoti."""
    groups: dict[str, list[ProbedObservation]] = {}
    for record in records:
        groups.setdefault(str(key(record)), []).append(record)
    return {
        group_name: _aggregate(
            groups[group_name],
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
            namespace=f"{namespace}/{group_name}",
            near_tie_threshold=near_tie_threshold,
        )
        for group_name in sorted(groups)
    }


def _permutation_label(target_suits: tuple[str, ...]) -> str:
    """Etichetta compatta source->target nell'ordine canonico dei semi sorgente."""
    sources = ("clubs", "cups", "coins", "swords")
    return ",".join(f"{source}>{target}" for source, target in zip(sources, target_suits, strict=True))


def _permutation_breakdown(
    records: list[ProbedObservation],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
    near_tie_threshold: float,
) -> list[dict[str, Any]]:
    """Riporta ciascuna delle 24 rinomine; l'identita' funge da controllo negativo."""
    permutations = all_suit_permutations()
    actions = np.stack([record.permutation_actions for record in records])
    baseline = actions[:, [0]]
    flips = actions != baseline
    gaps = np.stack([record.permutation_top2_gaps for record in records])
    own_near_ties = gaps <= near_tie_threshold
    comparison_near_ties = own_near_ties | own_near_ties[:, [0]]

    # Cinque colonne per permutazione in un solo bootstrap condiviso.
    js = np.zeros((len(records), 24), dtype=np.float64)
    delta = np.zeros((len(records), 24), dtype=np.float64)
    for row, record in enumerate(records):
        js[row, 1:] = record.identity_js_bits
        delta[row, 1:] = record.identity_max_abs_delta
    matrix = np.concatenate(
        [
            flips.astype(np.float64),
            js,
            delta,
            own_near_ties.astype(np.float64),
            comparison_near_ties.astype(np.float64),
        ],
        axis=1,
    )
    metric_names = tuple(
        [f"flip/{index}" for index in range(24)]
        + [f"js/{index}" for index in range(24)]
        + [f"delta/{index}" for index in range(24)]
        + [f"own-near/{index}" for index in range(24)]
        + [f"comparison-near/{index}" for index in range(24)]
    )
    bootstrap = _bootstrap_columns(
        matrix,
        metric_names=metric_names,
        repetitions=bootstrap_reps,
        seed=bootstrap_seed,
        namespace="permutation",
    )

    output: list[dict[str, Any]] = []
    for index, permutation in enumerate(permutations):
        targets = tuple(suit.value for suit in permutation)
        output.append(
            {
                "index": index,
                "mapping": targets,
                "label": _permutation_label(targets),
                "is_identity": index == 0,
                "observations": len(records),
                "agreement_rate": float(1.0 - np.mean(flips[:, index])),
                "flip_rate": float(np.mean(flips[:, index])),
                "remapped_top2_gap": _distribution(gaps[:, index]),
                "own_near_tie_rate": float(np.mean(own_near_ties[:, index])),
                "identity_comparison_near_tie_rate": float(np.mean(comparison_near_ties[:, index])),
                "js_divergence_bits": _distribution(js[:, index]),
                "max_abs_probability_delta": _distribution(delta[:, index]),
                "bootstrap_observation_ci95": {
                    "flip_rate": bootstrap[f"flip/{index}"],
                    "mean_js_bits": bootstrap[f"js/{index}"],
                    "mean_max_abs_delta": bootstrap[f"delta/{index}"],
                    "own_near_tie_rate": bootstrap[f"own-near/{index}"],
                    "identity_comparison_near_tie_rate": bootstrap[f"comparison-near/{index}"],
                },
            }
        )
    return output


def _worst_cases(records: list[ProbedObservation], *, count: int) -> list[dict[str, Any]]:
    """Serializza contesto osservabile, metadati di audit e permutazioni piu' divergenti."""
    permutations = [tuple(suit.value for suit in permutation) for permutation in all_suit_permutations()]

    def severity(record: ProbedObservation) -> tuple[float, str]:
        maximum = max(float(np.max(record.identity_js_bits)), float(np.max(record.pairwise_js_bits)))
        return (-maximum, record.candidate.selection_sha256)

    output: list[dict[str, Any]] = []
    for record in sorted(records, key=severity)[:count]:
        candidate = record.candidate
        observation = candidate.observation
        identity_index = int(np.argmax(record.identity_js_bits)) + 1
        pair_index = int(np.argmax(record.pairwise_js_bits))
        left_index = int(PAIR_LEFT[pair_index])
        right_index = int(PAIR_RIGHT[pair_index])
        output.append(
            {
                "selection_sha256": candidate.selection_sha256,
                "opponent": candidate.opponent,
                "phase": candidate.phase,
                "game_seed": candidate.game_seed,
                "policy_seat": candidate.policy_seat,
                "decision_ordinal": candidate.decision_ordinal,
                "position": candidate.position,
                "deck_size": observation.deck_size,
                "unknown_live_cards": candidate.unknown_live_cards,
                "legal_count": candidate.legal_count,
                "legal_action_ids": sorted(card_to_id(card) for card in observation.hand),
                "table_cards": [
                    {"action_id": card_to_id(card), "player_index": player_index}
                    for card, player_index in observation.table_cards
                ],
                "players_points": list(observation.players_points),
                "completed_tricks": len(observation.trick_history),
                "trump_suit": observation.trump_card.suit.value if observation.trump_card is not None else None,
                "baseline_action_id": record.baseline_action_id,
                "baseline_top2_gap": record.baseline_top2_gap,
                "identity_worst": {
                    "permutation_index": identity_index,
                    "mapping": permutations[identity_index],
                    "remapped_action_id": int(record.permutation_actions[identity_index]),
                    "agreement": not bool(record.identity_flips[identity_index - 1]),
                    "baseline_top2_gap": record.baseline_top2_gap,
                    "remapped_top2_gap": float(record.permutation_top2_gaps[identity_index]),
                    "near_tie": bool(record.identity_near_ties[identity_index - 1]),
                    "js_divergence_bits": float(record.identity_js_bits[identity_index - 1]),
                    "max_abs_probability_delta": float(record.identity_max_abs_delta[identity_index - 1]),
                },
                "pairwise_worst": {
                    "left_permutation_index": left_index,
                    "left_mapping": permutations[left_index],
                    "left_action_id": int(record.permutation_actions[left_index]),
                    "left_top2_gap": float(record.permutation_top2_gaps[left_index]),
                    "right_permutation_index": right_index,
                    "right_mapping": permutations[right_index],
                    "right_action_id": int(record.permutation_actions[right_index]),
                    "right_top2_gap": float(record.permutation_top2_gaps[right_index]),
                    "agreement": not bool(record.pairwise_flips[pair_index]),
                    "near_tie": bool(record.pairwise_near_ties[pair_index]),
                    "js_divergence_bits": float(record.pairwise_js_bits[pair_index]),
                    "max_abs_probability_delta": float(record.pairwise_max_abs_delta[pair_index]),
                },
            }
        )
    return output


def _identity_control(records: list[ProbedObservation]) -> dict[str, Any]:
    """Riassume il controllo negativo gia' verificato durante ogni probe."""
    return {
        "observations": len(records),
        "expected_agreement_rate": 1.0,
        "agreement_rate": 1.0,
        "max_js_divergence_bits": 0.0,
        "max_abs_probability_delta": 0.0,
        "passed": True,
    }


def _parse_args() -> argparse.Namespace:
    """Definisce la CLI della sonda."""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Diagnostica di simmetria dei semi della policy")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "data" / "models" / "best_a2c_v14.npz",
        help="Policy .npz da analizzare (default: best_a2c_v14.npz)",
    )
    parser.add_argument(
        "--seed-suite",
        type=Path,
        default=root / "seed_suites" / "small_1000.txt",
        help="File di seed, uno per riga",
    )
    parser.add_argument("--seed-count", type=int, default=64, help="Numero di seed iniziali della suite")
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=256,
        help="Quota per ogni combinazione avversario x fase",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=list(DEFAULT_OPPONENTS),
        help="Roster (spazi o virgole): mirror heuristic_trump_saver heuristic_v1 random",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2_000, help="Repliche bootstrap per osservazione")
    parser.add_argument("--bootstrap-seed", type=int, default=20260711, help="Seed radice del bootstrap")
    parser.add_argument(
        "--near-tie-threshold",
        type=float,
        default=1e-4,
        help="Gap top-2 sotto cui una decisione e' classificata quasi-pari",
    )
    parser.add_argument("--worst-cases", type=int, default=20, help="Numero di casi estremi nel report")
    parser.add_argument("--out-json", type=Path, required=True, help="Path del report JSON")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, *, suite_size: int) -> None:
    """Fallisce presto su configurazioni prive di significato statistico."""
    if args.seed_count <= 0 or args.seed_count > suite_size:
        raise ValueError(f"--seed-count deve essere tra 1 e {suite_size}, ottenuto {args.seed_count}")
    if args.samples_per_cell <= 0:
        raise ValueError("--samples-per-cell deve essere > 0")
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps deve essere > 0")
    if args.near_tie_threshold < 0.0:
        raise ValueError("--near-tie-threshold deve essere >= 0")
    if args.worst_cases < 0:
        raise ValueError("--worst-cases deve essere >= 0")


def main() -> int:
    """Esegue raccolta, quota, probe e scrittura deterministica del report."""
    args = _parse_args()
    root = _repo_root()
    model_path = args.model.resolve()
    seed_suite_path = args.seed_suite.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    if not seed_suite_path.is_file():
        raise FileNotFoundError(f"Seed suite non trovata: {seed_suite_path}")

    all_seeds = _load_seed_suite(seed_suite_path)
    _validate_args(args, suite_size=len(all_seeds))
    opponents = _normalize_opponents(args.opponents)
    seeds = all_seeds[: args.seed_count]
    model_agent = BCModelAgent.from_npz(model_path)

    print(
        f"Raccolta: {len(seeds)} seed x 2 seat x {len(opponents)} avversari; quota {args.samples_per_cell} per cella..."
    )
    candidates, coverage = _collect_candidates(model_agent, opponents=opponents, seeds=seeds)
    selected = _select_balanced(
        candidates,
        opponents=opponents,
        samples_per_cell=args.samples_per_cell,
        coverage=coverage,
    )

    print(f"Probe di {len(selected)} osservazioni x 24 permutazioni...")
    records: list[ProbedObservation] = []
    progress_step = max(1, len(selected) // 8)
    for index, candidate in enumerate(selected, start=1):
        records.append(
            _probe_one(
                model_agent,
                candidate,
                near_tie_threshold=args.near_tie_threshold,
            )
        )
        if index % progress_step == 0 or index == len(selected):
            print(f"  {index}/{len(selected)}")

    overall = _aggregate(
        records,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        namespace="overall",
        near_tie_threshold=args.near_tie_threshold,
    )
    breakdown = {
        "phase": _grouped_breakdown(
            records,
            key=lambda record: record.candidate.phase,
            namespace="phase",
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
            near_tie_threshold=args.near_tie_threshold,
        ),
        "opponent": _grouped_breakdown(
            records,
            key=lambda record: record.candidate.opponent,
            namespace="opponent",
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
            near_tie_threshold=args.near_tie_threshold,
        ),
        "opponent_phase": _grouped_breakdown(
            records,
            key=lambda record: f"{record.candidate.opponent}/{record.candidate.phase}",
            namespace="opponent-phase",
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
            near_tie_threshold=args.near_tie_threshold,
        ),
        "position": _grouped_breakdown(
            records,
            key=lambda record: record.candidate.position,
            namespace="position",
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
            near_tie_threshold=args.near_tie_threshold,
        ),
        "legal_count": _grouped_breakdown(
            records,
            key=lambda record: record.candidate.legal_count,
            namespace="legal-count",
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
            near_tie_threshold=args.near_tie_threshold,
        ),
        "permutation": _permutation_breakdown(
            records,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
            near_tie_threshold=args.near_tie_threshold,
        ),
    }

    symmetry_module_path = root / "src" / "briscola_ai" / "ai" / "evaluation" / "suit_symmetry.py"
    report = {
        "schema": "briscola.suit_symmetry_probe.v1",
        "method": {
            "anti_cheat": (
                "policy e metriche ricevono solo PlayerObservation; GameState non conservato ne' serializzato; "
                "seed/seat/ordinale sono metadati di riproduzione mai passati al modello"
            ),
            "policy_stage": "logits grezzi mascherati, prima del post-processing runtime",
            "action_selection": "argmax dopo la mask delle azioni legali",
            "softmax_temperature": 1.0,
            "suit_permutations": 24,
            "identity_comparisons_per_observation": 23,
            "all_pairs_per_observation": PAIR_COUNT,
            "near_tie_semantics": (
                "un confronto e' near-tie se almeno una delle due distribuzioni ha gap top-2 <= soglia; "
                "near_tie_observation_rate resta il solo baseline per compatibilita'"
            ),
            "phase_definition": {
                "early": "deck_size > 10, unknown_live_cards > 8",
                "mid": "0 < deck_size <= 10, unknown_live_cards > 8",
                "pimc_window": "deck_size > 0, unknown_live_cards <= 8",
                "endgame": "deck_size == 0",
            },
            "sample_selection": (f"i {args.samples_per_cell} digest SHA-256 minori per ogni cella opponent/phase"),
            "bootstrap_unit": "PlayerObservation",
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
            "model": {
                "path": _display_path(model_path, root=root),
                "sha256": _sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
                "architecture": type(model_agent.model).__name__,
                "feature_dim": int(model_agent.model.feature_dim),
                "encoder_version": model_agent.encoder_version,
                "overkill_guard_enabled": model_agent.overkill_guard_enabled,
                "metadata": model_agent.model.metadata,
            },
            "seed_suite": {
                "path": _display_path(seed_suite_path, root=root),
                "sha256": _sha256_file(seed_suite_path),
                "suite_size": len(all_seeds),
                "selected_seed_count": len(seeds),
                "selected_seeds": seeds,
            },
            "probe_script_sha256": _sha256_file(Path(__file__).resolve()),
            "symmetry_module_sha256": _sha256_file(symmetry_module_path),
        },
        "config": {
            "opponents": list(opponents),
            "seed_count": args.seed_count,
            "samples_per_opponent_phase": args.samples_per_cell,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "near_tie_threshold": args.near_tie_threshold,
            "worst_cases": args.worst_cases,
        },
        "coverage": coverage,
        "identity_control": _identity_control(records),
        "overall": overall,
        "breakdown": breakdown,
        "worst_cases": _worst_cases(records, count=args.worst_cases),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out_json.write_text(serialized, encoding="utf-8")

    identity_metrics = overall["identity_vs_23"]
    pairwise_metrics = overall["all_276_pairs"]
    print(f"Report: {args.out_json}")
    print(
        "Identity vs 23: "
        f"flip={identity_metrics['flip_rate']:.4%}, "
        f"JS media={identity_metrics['js_divergence_bits']['mean']:.6f} bit"
    )
    print(
        "Tutte le 276 coppie: "
        f"flip={pairwise_metrics['flip_rate']:.4%}, "
        f"JS media={pairwise_metrics['js_divergence_bits']['mean']:.6f} bit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
