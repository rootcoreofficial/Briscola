#!/usr/bin/env python3
"""
Build the significant-model Excel report.

The report is intentionally curated: it tracks official best models, one teacher
model, and only the rejected candidates that explain an important decision. It
does not try to dump every experiment under `benchmarks/experiments/`.

Reproducibility note (important): normal report builds read the compact, versioned
evidence snapshot in `docs/reports/evidence/`. The original `.npz` models and benchmark
JSONs remain useful audit inputs; many historical inputs are gitignored and are required
only when a maintainer explicitly refreshes that snapshot with `--refresh-evidence`.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "reports" / "model_progress.xlsx"
EVIDENCE_PATH = ROOT / "docs" / "reports" / "evidence" / "model_progress.v1.json"
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_SNAPSHOT_DATE = "2026-07-17"
_XLSX_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def project_version() -> str:
    """Read the canonical package version used in the Dashboard conclusion."""
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = payload.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml does not contain project.version")
    return version


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Curated model included in the report."""

    model_id: str
    path: Path
    role: str
    status: str
    order: int
    progress_source: str
    progress_score: float | None
    h2h_source: str
    h2h_score: float | None
    decision: str
    notes: str
    data_quality: str = "exact"


def _rel(path: str) -> Path:
    """Return a repository-relative path as an absolute Path."""
    return ROOT / path


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        model_id="bc_v3",
        path=_rel("data/models/bc_v3.npz"),
        role="teacher/anchor",
        status="teacher",
        order=0,
        progress_source="Not plotted: supervised teacher, not a playing baseline.",
        progress_score=None,
        h2h_source="",
        h2h_score=None,
        decision="Use as BC teacher/anchor for v3 A2C runs.",
        notes="Behavior cloning MLP v3; important as init/anchor, not as official best.",
    ),
    ModelSpec(
        model_id="best_a2c",
        path=_rel("data/models/best_a2c.npz"),
        role="official best",
        status="promoted",
        order=1,
        progress_source="benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/matrix_big.json",
        progress_score=16.77358,
        h2h_source="benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/head_to_head_best_a2c_v2_big_numba.json",
        h2h_score=0.76442,
        decision="Promoted as v2 best with overkill guard.",
        notes="Strong v2 baseline; remains useful for regression comparisons.",
    ),
    ModelSpec(
        model_id="best_a2c_v3",
        path=_rel("data/models/best_a2c_v3.npz"),
        role="official best",
        status="promoted",
        order=2,
        progress_source="benchmarks/experiments/a2c_v3_league_seed301_1m_numba/baseline_best_a2c_v3_big_vs_heuristic_v1_numba.json",
        progress_score=17.28786,
        h2h_source="benchmarks/experiments/best_a2c_v3_vs_best_a2c_2026-06-28_big_numba.json",
        h2h_score=0.18258,
        decision="Promoted as recommended v3 baseline.",
        notes="Encoder v3, BC/A2C v3 pipeline, guard enabled for runtime/UI.",
    ),
    ModelSpec(
        model_id="best_a2c_v4",
        path=_rel("data/models/best_a2c_v4.npz"),
        role="official best",
        status="promoted",
        order=3,
        progress_source="benchmarks/experiments/a2c_v3_league_seed301_1m_numba/eval_big_vs_heuristic_v1_numba.json",
        progress_score=17.50188,
        h2h_source="benchmarks/experiments/a2c_v3_league_seed301_1m_numba/head_to_head_best_a2c_v3_big_numba.json",
        h2h_score=0.35628,
        decision="Promoted as recommended local/webapp/cloud model.",
        notes="League v3 1M run warm-started from best_a2c_v3 with best_a2c_v3 in the opponent mix.",
    ),
    ModelSpec(
        model_id="best_a2c_v5",
        path=_rel("data/models/best_a2c_v5.npz"),
        role="official best",
        status="promoted",
        order=4,
        progress_source="benchmarks/experiments/a2c_v5_seed401_1m_numba/eval_big_vs_heuristic_v1_numba.json",
        progress_score=17.832,
        h2h_source="benchmarks/experiments/a2c_v5_seed401_1m_numba/head_to_head_best_a2c_v4_big_numba.json",
        h2h_score=0.33972,
        decision="Promoted as recommended model for the v0.11.0 release.",
        notes="League v5 1M run warm-started from best_a2c_v4 with best_a2c_v4 in the opponent mix.",
    ),
    ModelSpec(
        model_id="best_a2c_v6",
        path=_rel("data/models/best_a2c_v6.npz"),
        role="official best",
        status="promoted",
        order=5,
        progress_source="benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_heuristic_v1_big_numba.json",
        progress_score=18.40148,
        h2h_source="benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_best_a2c_v5_big_holdout_numba.json",
        h2h_score=0.45866,
        decision="Promoted as recommended model for the v0.12.0 release.",
        notes="Scaling v6 5M run warm-started from best_a2c_v5 with best_a2c_v5 in the opponent mix.",
    ),
    ModelSpec(
        model_id="best_a2c_v7",
        path=_rel("data/models/best_a2c_v7.npz"),
        role="official best",
        status="promoted",
        order=6,
        progress_source="data/eval_best_a2c_v7_vs_heuristic_v1_big_holdout_seedrange1000000.json",
        progress_score=18.7314,
        h2h_source="data/eval_a2c_vs_value_lookahead_5M_vs_v6_big_holdout_seedrange1000000.json",
        h2h_score=2.27394,
        decision="Promoted as recommended model for the v0.19.0 release.",
        notes=(
            "5M A2C run warm-started from best_a2c_v6 against the fast Numba value-lookahead opponent. "
            "It becomes the default .npz policy; value-lookahead remains the stronger runtime option."
        ),
    ),
    ModelSpec(
        model_id="best_a2c_v8",
        path=_rel("data/models/best_a2c_v8.npz"),
        role="official best",
        status="promoted",
        order=7,
        progress_source="benchmarks/experiments/fase3/iter2_h256_vs_heuristic_v1_big.json",
        progress_score=17.60724,
        h2h_source="benchmarks/experiments/fase3/iter2_h256_vs_v7_big.json",
        h2h_score=0.89104,
        decision="Promoted as recommended model for the v0.22.0 release.",
        notes=(
            "Encoder v4 (trick-history features) + Net2Net widening to hidden 256. Chain: warm-start from "
            "best_a2c_v7 (v4 zero-pad), 5M A2C games vs value-lookahead(v7), widening 128->256, 5M more games. "
            "Beats best_a2c_v7 head-to-head (+0.89, CI +0.74..+1.05, 100k paired); the lower score vs "
            "heuristic_v1 relative to v7 (17.61 vs 18.73) reflects style non-transitivity, not regression."
        ),
    ),
    ModelSpec(
        model_id="best_a2c_v9",
        path=_rel("data/models/best_a2c_v9.npz"),
        role="official best",
        status="promoted",
        order=8,
        progress_source="benchmarks/experiments/fase3/superA_vs_h1_big.json",
        progress_score=18.78196,
        h2h_source="benchmarks/experiments/fase3/superA_vs_v8_big.json",
        h2h_score=0.96962,
        decision="Promoted as recommended model for the v0.26.0 release.",
        notes=(
            "20M-game 'super training' vs a mixed panel: value-lookahead(v8) 65%, v8 mirror 15%, "
            "heuristics+random 20% (the 'bar' share, maintainer's recipe). Beats v8 +0.97 head-to-head "
            "AND sets the heuristic_v1 record (+18.78): first model that improves on BOTH meters. "
            "Also beats the PIMC-teacher arm (5M vs the strongest master) by +0.63: volume+variety "
            "won over elite teaching in this regime."
        ),
    ),
    ModelSpec(
        model_id="best_a2c_v10",
        path=_rel("data/models/best_a2c_v10.npz"),
        role="official best",
        status="promoted",
        order=9,
        progress_source="benchmarks/experiments/fase3/definitivo_vs_h1_big.json",
        progress_score=20.5235,
        h2h_source="benchmarks/experiments/fase3/definitivo_vs_v9_big.json",
        h2h_score=0.66076,
        decision="Promoted as recommended model for the v0.27.0 release.",
        notes=(
            "The 'definitive' 30M-game run vs the complete panel: PIMC-belief teacher 25%, "
            "value-lookahead 35%, mirror 15%, heuristics+random 25%, all on v9 base (panel doses by "
            "the maintainer). Beats v9 +0.66 head-to-head and sets a new all-time heuristic_v1 record "
            "(+20.52, up from 18.78): both progression meters at their maximum."
        ),
    ),
    ModelSpec(
        model_id="best_a2c_v11",
        path=_rel("data/models/best_a2c_v11.npz"),
        role="official best",
        status="promoted",
        order=10,
        progress_source="benchmarks/experiments/fase3/v11_vs_h1_big.json",
        progress_score=20.79522,
        h2h_source="benchmarks/experiments/fase3/v11_vs_v10_big.json",
        h2h_score=0.85186,
        decision="Promoted as recommended model for the v0.31.0 release.",
        notes=(
            "Dose-shift hypothesis validated: 40% of the panel to the PIMC 16x8 belief teacher "
            "(dose moved from the fading value-lookahead), base and teachers on v10, only 5M games "
            "(6x fewer than v10). Beats v10 +0.85 head-to-head (above the +0.3..+0.5 success band: "
            "the diminishing-returns curve bent UP) and sets a new heuristic_v1 record (+20.80). "
            "Runs WITHOUT the overkill guard, measured harmful on this model (-0.5 with guard on)."
        ),
    ),
    ModelSpec(
        model_id="best_a2c_v13",
        path=_rel("data/models/best_a2c_v13.npz"),
        role="official best",
        status="promoted",
        order=11,
        progress_source="benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/eval_v13_vs_heuristic_v1_medium.json",
        progress_score=21.5182,
        h2h_source="benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/eval_v13_vs_v11_medium.json",
        h2h_score=-0.0272,
        decision="Promoted as recommended model for the v0.34.0 release.",
        notes=(
            "Overkill-shaping beta=0.3, warm-started from best_a2c_v11 for 5M games with a BC anchor "
            "toward v11. This is a behavior-cleanup promotion, not a strength-jump claim: policy-only "
            "v13 vs v11 is neutral (-0.03, CI -0.38..+0.32), and the default PIMC 16x8 gate is also "
            "neutral-slightly-positive (+0.14, CI -0.20..+0.47). The value is that low-lead trump "
            "overkill drops sharply (about 28-31% on v11 gates to 6-8% on v13) without breaking "
            "trump_saver or heuristic_v1 performance. Summary: same strength, better behavior."
        ),
    ),
    ModelSpec(
        model_id="best_a2c_v14",
        path=_rel("data/models/best_a2c_v14.npz"),
        role="official best",
        status="promoted",
        order=12,
        progress_source=(
            "benchmarks/experiments/suit_distillation_v0_50k_seed20260712/eval_v14_vs_heuristic_v1_big_numba.json"
        ),
        progress_score=21.75808,
        h2h_source="benchmarks/experiments/suit_distillation_v0_50k_seed20260712/eval_vs_v13_medium.json",
        h2h_score=0.6626,
        decision="Promoted as recommended model for the v0.36.0 release.",
        notes=(
            "A single MLP distilled from the exact average of v13 logits over all 24 suit renamings, "
            "using 50k games and 1.9M labelled decisions. It reduces suit-dependent argmax flips from "
            "18.19% to 6.04% while preserving decisiveness. It beats v13 policy-only by +0.66 "
            "(CI +0.24..+1.09) and in the real PIMC belief 16x8 stack by +0.43 "
            "(CI +0.03..+0.84). The homogeneous big 100k control vs heuristic_v1 reaches +21.76."
        ),
    ),
    ModelSpec(
        model_id="best_a2c_v15",
        path=_rel("data/models/best_a2c_v15.npz"),
        role="official best",
        status="promoted",
        order=13,
        progress_source=(
            "benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/"
            "eval_v15_vs_heuristic_v1_big_numba.json"
        ),
        progress_score=21.86688,
        h2h_source=(
            "benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/confirm_student_vs_v14_100k.json"
        ),
        h2h_score=0.18046,
        decision="Promoted as the recommended model and 12x8 runtime for the v0.38.0 release.",
        notes=(
            "A 250k-game, 9.5M-decision student distilled from the exact 24-view suit average of the strongest "
            "20M checkpoint. Policy-only it beats v14 by +0.18 on 100k paired games and reaches +21.87 in the "
            "homogeneous big control vs heuristic_v1. The release promotion is an efficiency decision: the real "
            "PIMC belief 12x8 stack is non-inferior to v14 16x8 (+0.11, CI -0.11..+0.32 on 20k games) while its "
            "mean and p95 search latency are about 25% lower. v14 16x8 remains selectable."
        ),
    ),
]


# The reference metrics in `MODEL_SPECS` are intentionally not treated as one
# mathematical series. Historical evaluations used different seed ranges and, for
# v13, a smaller gate. Keeping the protocol beside every value prevents accidental
# comparisons such as medium/PIMC v13 against big/policy-only predecessors.
METRIC_PROTOCOLS: dict[str, dict[str, str]] = {
    "bc_v3": {"progress": "not applicable", "h2h": "not applicable"},
    "best_a2c": {
        "progress": "policy + promoted guard; big 100k; holdout seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; standard seeds; Numba; vs previous v2 best",
    },
    "best_a2c_v3": {
        "progress": "policy + promoted guard; big 100k; holdout seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; holdout seeds; Numba; vs best_a2c",
    },
    "best_a2c_v4": {
        "progress": "policy + promoted guard; big 100k; holdout seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; holdout seeds; Numba; vs best_a2c_v3",
    },
    "best_a2c_v5": {
        "progress": "policy + promoted guard; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; standard seeds; Numba; vs best_a2c_v4",
    },
    "best_a2c_v6": {
        "progress": "policy + promoted guard; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; holdout seeds; Numba; vs best_a2c_v5",
    },
    "best_a2c_v7": {
        "progress": "policy + promoted guard; big 100k; holdout seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; holdout seeds; Numba; vs best_a2c_v6",
    },
    "best_a2c_v8": {
        "progress": "policy + promoted guard; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; standard seeds; Numba; vs best_a2c_v7",
    },
    "best_a2c_v9": {
        "progress": "policy + promoted guard; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; standard seeds; Numba; vs best_a2c_v8",
    },
    "best_a2c_v10": {
        "progress": "policy + promoted guard; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy + promoted guard; big 100k; standard seeds; Numba; vs best_a2c_v9",
    },
    "best_a2c_v11": {
        "progress": "policy, guard off; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy, guard off; big 100k; standard seeds; Numba; vs best_a2c_v10 with guard",
    },
    "best_a2c_v13": {
        "progress": "policy, guard off; medium 10k; suite medium; Numba; vs heuristic_v1",
        "h2h": "policy, guard off; medium 10k; suite medium; Numba; vs best_a2c_v11",
    },
    "best_a2c_v14": {
        "progress": "policy, guard off; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy, guard off; medium 10k; suite medium; domain; vs best_a2c_v13",
    },
    "best_a2c_v15": {
        "progress": "policy, guard off; big 100k; standard seeds; Numba; vs heuristic_v1",
        "h2h": "policy, guard off; 100k; seed range 11,000,000; domain; vs best_a2c_v14",
    },
}

# Only these rows share the same benchmark/engine/seed protocol and therefore form the
# chart. They use each model's promoted runtime configuration: v8-v10 include the guard,
# v11 and v14 do not, so the series is not presented as pure architectural progress.
# v13 is deliberately absent because its available promotion gate is medium 10k; its
# direct policy and PIMC gates remain separately labelled in the evidence sheets. The
# dedicated v14 big control lets the latest promoted model rejoin the comparable series.
HOMOGENEOUS_CHART_MODEL_IDS = (
    "best_a2c_v8",
    "best_a2c_v9",
    "best_a2c_v10",
    "best_a2c_v11",
    "best_a2c_v14",
    "best_a2c_v15",
)


REJECTED_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate": "seed301_200k",
        "path": "benchmarks/experiments/a2c_v3_league_seed301_200k_numba/model.npz",
        "training_games": 200000,
        "decision": "not promoted",
        "reason": "Positive big head-to-head, but too small and quality vs heuristic_v1 did not clearly improve.",
        "evidence": "big vs best_a2c_v3 +0.12/+0.13; decision-quality vs heuristic_v1 +16.91.",
    },
    {
        "candidate": "seed302_200k_conservative",
        "path": "benchmarks/experiments/a2c_v3_league_seed302_200k_conservative_numba/model.npz",
        "training_games": 200000,
        "decision": "not promoted",
        "reason": "Did not pass the medium filter against best_a2c_v3.",
        "evidence": "medium vs best_a2c_v3 -0.12/+0.05; vs heuristic_v1 +16.94/+16.95.",
    },
    {
        "candidate": "pimc_distillation_v7",
        "path": "data/models/pimc_distill_v7_*.npz",
        "training_games": "dataset: 50k records",
        "decision": "branch closed",
        "reason": (
            "One-shot BC on PIMC actions was lossy: fitting expert argmax labels erased more of the RL policy "
            "than the sparse search corrections added."
        ),
        "evidence": (
            "Expert-iteration probe: hard warm-start -2.33 vs v7; anchored variant -1.52; soft targets -9.19."
        ),
    },
    {
        "candidate": "exit_iter1b_belief_input",
        "path": "data/models/exit_iter1b_a2c_v4belief_vs_vl_v7_5M_seed20260716.npz",
        "training_games": 5_000_000,
        "decision": "not promoted",
        "reason": (
            "The frozen belief vector is a deterministic function of the same public v4 features; as policy input "
            "it added redundancy rather than information."
        ),
        "evidence": "Medium 10k: -0.56 vs its iter1 control (CI -1.05..-0.06); +0.82 vs v7.",
    },
    {
        "candidate": "pimc_teacher_v8_5m",
        "path": "data/models/pimc_teacher_v8_5M_32x10_seed20260721.npz",
        "training_games": 5_000_000,
        "decision": "not promoted",
        "reason": (
            "An elite-teacher-only diet improved the mirror matchup but regressed anti-weak style; the larger "
            "mixed-panel arm won on every promotion axis."
        ),
        "evidence": "Big 100k: +0.40 vs v8, +16.71 vs heuristic_v1, and -0.63 vs the promoted mixed-panel arm.",
    },
    {
        "candidate": "v11_guard_on",
        "path": "benchmarks/experiments/fase3/v11guard_vs_v10_big.json",
        "training_games": "eval-only wrapper",
        "decision": "wrapper rejected",
        "reason": "The historical overkill guard suppressed choices that the stronger v11 policy made deliberately.",
        "evidence": "Big 100k vs v10: +0.32 with guard, compared with +0.85 without it (about -0.53).",
    },
    {
        "candidate": "best_a2c_v12",
        "path": "data/models/a2c_v12_saver12_pimc40_10M_seed20260708.npz",
        "training_games": 10_000_000,
        "decision": "not promoted",
        "reason": (
            "Adding trump_saver to the opponent mix neither produced a significant strength gain nor changed the "
            "target behavior."
        ),
        "evidence": (
            "Big 100k: +0.11 vs v11 (CI -0.03..+0.25), +20.74 vs heuristic_v1; lead-load and trump-use "
            "profiles remained effectively unchanged."
        ),
    },
]


MILESTONES: list[dict[str, Any]] = [
    {
        "order": 1,
        "date": "2026-06-08",
        "model_id": "best_a2c",
        "type": "promoted",
        "decision": "Make v2 A2C with overkill guard the official best.",
        "why": "Good big benchmark strength and guard mitigated poor trump overkill behavior.",
        "evidence": "Big holdout vs heuristic_v1 +16.77; promotion H2H source kept in benchmarks.",
        "impact": "Stable v2 baseline for UI and later v3 comparisons.",
        "source": "data/models/best_a2c.npz + a2c_v2_best_overkill_gap001_1m_seed50_numba",
    },
    {
        "order": 2,
        "date": "2026-06-23",
        "model_id": "bc_v3",
        "type": "teacher",
        "decision": "Use encoder v3 BC model as teacher/anchor.",
        "why": "Encoder v3 adds public-history and strategic aggregate features without hidden information.",
        "evidence": "BC v3 metadata: feature_dim 310, 20 epochs.",
        "impact": "Provided a stronger and safer anchor for v3 A2C training.",
        "source": "data/models/bc_v3.npz",
    },
    {
        "order": 3,
        "date": "2026-06-23",
        "model_id": "best_a2c_v3",
        "type": "promoted",
        "decision": "Promote v3 A2C as recommended baseline.",
        "why": "Improved holdout strength and kept overkill under control with runtime guard.",
        "evidence": "Consolidation: big H2H roughly non-regressive vs best_a2c; holdout vs heuristic_v1 improved.",
        "impact": "New default recommended model, v2 best retained for regression.",
        "source": "data/models/best_a2c_v3.npz + PLAN.md",
    },
    {
        "order": 4,
        "date": "2026-06-28",
        "model_id": "seed301_200k",
        "type": "rejected",
        "decision": "Do not promote the 200k league candidate.",
        "why": "The signal was positive but too small, and heuristic_v1 quality was weaker than the v3 best.",
        "evidence": "Big vs best_a2c_v3 +0.12/+0.13; decision-quality +16.91 vs heuristic_v1.",
        "impact": "Confirmed that 200k is only a screening run.",
        "source": "benchmarks/experiments/a2c_v3_league_seed301_200k_numba/",
    },
    {
        "order": 5,
        "date": "2026-06-28",
        "model_id": "seed302_200k_conservative",
        "type": "rejected",
        "decision": "Do not promote the conservative 200k variant.",
        "why": "It did not beat best_a2c_v3 on the medium screen.",
        "evidence": "Medium vs best_a2c_v3 -0.12/+0.05.",
        "impact": "Avoided spending on a big benchmark for a weak candidate.",
        "source": "benchmarks/experiments/a2c_v3_league_seed302_200k_conservative_numba/",
    },
    {
        "order": 6,
        "date": "2026-06-28",
        "model_id": "best_a2c_v4",
        "type": "promoted",
        "decision": "Promote v4 as the recommended local/webapp/cloud model.",
        "why": "It beats best_a2c_v3 head-to-head and does not regress against heuristic_v1.",
        "evidence": "Big vs best_a2c_v3 +0.45/+0.36; big vs heuristic_v1 +17.43/+17.50.",
        "impact": "Frontend/server/cloud default now points to v4 via release asset provisioning.",
        "source": "data/models/best_a2c_v4.npz + a2c_v3_league_seed301_1m_numba",
    },
    {
        "order": 7,
        "date": "2026-06-28",
        "model_id": "best_a2c_v5",
        "type": "promoted",
        "decision": "Promote v5 as the recommended model for v0.11.0.",
        "why": (
            "It beats best_a2c_v4 head-to-head and improves the heuristic_v1 holdout "
            "without material quality regressions."
        ),
        "evidence": (
            "Big vs best_a2c_v4 +0.34; big vs heuristic_v1 +17.83; decision-quality +18.00, overkill 0.0%, waste 0.07%."
        ),
        "impact": "Frontend/server default now points to v5; cloud rollout needs the v0.11.0 asset URL in env.",
        "source": "data/models/best_a2c_v5.npz + a2c_v5_seed401_1m_numba",
    },
    {
        "order": 8,
        "date": "2026-06-28",
        "model_id": "best_a2c_v6",
        "type": "promoted",
        "decision": "Promote v6 as the recommended model for v0.12.0.",
        "why": (
            "The 1M/3M/5M scaling curve improved monotonically, and the 5M checkpoint beat best_a2c_v5 "
            "on both standard and holdout suites without quality regressions."
        ),
        "evidence": (
            "5M big vs best_a2c_v5 +0.46; holdout big vs best_a2c_v5 +0.46; "
            "big vs heuristic_v1 +18.40; decision-quality +18.58, overkill 0.0%, waste 0.07%."
        ),
        "impact": "Frontend/server default now points to v6; cloud rollout needs the v0.12.0 asset URL in env.",
        "source": "data/models/best_a2c_v6.npz + a2c_v6_scaling_seed501_5m_numba",
    },
    {
        "order": 9,
        "date": "2026-06-29",
        "model_id": "best_a2c_v6",
        "type": "runtime_default",
        "decision": "Make v6 + exact endgame solver the default UI opponent for v0.13.0.",
        "why": (
            "The solver is exact, anti-cheat, and effectively free at runtime; it preserves v6 during the game "
            "and improves the fully-known endgame instead of trying to distill solver behavior into the network."
        ),
        "evidence": (
            "control_solver(v6) / bc_model_hybrid_endgame equivalence; "
            "PIMC max_unknown=0 sweep showed roughly +1.3..+1.6 avg diff vs v6."
        ),
        "impact": (
            "Runtime default changes to bc_model_hybrid_endgame(best_a2c_v6); "
            "model progression chart remains unchanged."
        ),
        "source": "src/briscola_ai/ai/agents/registry.py + PLAN.md",
    },
    {
        "order": 10,
        "date": "2026-06-29",
        "model_id": "best_a2c_v6",
        "type": "runtime_option",
        "decision": "Expose PIMC(v6, 16x8) as an advanced selectable UI opponent for v0.14.0.",
        "why": (
            "PIMC adds measurable value over v6 + solver, but costs CPU, so it should be testable by humans "
            "without replacing the lower-risk default."
        ),
        "evidence": (
            "PIMC(v6,16x10) beat control_solver(v6) at 2000 games; direct Pareto run found no evidence "
            "that 16x10 was stronger than 16x8."
        ),
        "impact": (
            "UI can select bc_model_pimc_16x8(best_a2c_v6); default remains bc_model_hybrid_endgame(best_a2c_v6)."
        ),
        "source": "src/briscola_ai/ai/agents/registry.py + scripts/evaluate_pimc.py + PLAN.md",
    },
    {
        "order": 11,
        "date": "2026-07-01",
        "model_id": "best_a2c_v7",
        "type": "promoted",
        "decision": "Promote v7 as the recommended .npz policy for v0.19.0.",
        "why": (
            "A 5M A2C run against the fast Numba value-lookahead opponent produced a policy that beats v6 "
            "head-to-head and also improves the v6+solver runtime baseline."
        ),
        "evidence": (
            "Big holdout vs best_a2c_v6 +2.27; medium candidate+solver vs v6+solver +2.27; "
            "big holdout vs heuristic_v1 +18.73. It remains slightly below v6 value-lookahead runtime "
            "(-0.64 on 10k), so value-lookahead stays selectable as the advanced option."
        ),
        "impact": "Frontend/server/cloud default model moves to best_a2c_v7; value model asset remains unchanged.",
        "source": "data/models/best_a2c_v7.npz + train_a2c fast_numba_determinized value-lookahead screen",
    },
    {
        "order": 12,
        "date": "2026-07-03",
        "model_id": "best_a2c_v8",
        "type": "promoted",
        "decision": "Promote v8 (encoder v4 + hidden 256) as the recommended .npz policy for v0.22.0.",
        "why": (
            "The ExIt iteration-1/2 arms measured every student-side lever with paired controls: the v4 "
            "trick-history features add +0.27 net (first positive runtime evidence of the encoder program), "
            "Net2Net capacity adds a marginal +0.18, and the combined chain beats v7 by +0.89."
        ),
        "evidence": (
            "Big holdout 100k paired CIs: vs best_a2c_v7 +0.89 (CI +0.74..+1.05); vs iter1-v4 +0.18 "
            "(CI +0.03..+0.33); vs heuristic_v1 +17.61. Belief-as-policy-input was killed (-0.56 vs its "
            "own init): the belief is a deterministic function of the same v4 features, so as policy input "
            "it is redundant and gradient-chasing it erodes tuned instinct."
        ),
        "impact": (
            "Frontend/server/cloud default model moves to best_a2c_v8 (first v4-encoder, first hidden-256 "
            "default); value model asset unchanged. The initially empty v4 history inside determinized "
            "value-lookahead simulations was later fixed by carrying real public trick history into the "
            "simulation state; it is no longer a runtime limitation."
        ),
        "source": "data/models/best_a2c_v8.npz + ExIt iteration-1/2 (docs/plans/belief-expert-iteration.md)",
    },
    {
        "order": 13,
        "date": "2026-07-04",
        "model_id": "best_a2c_v9",
        "type": "promoted",
        "decision": "Promote v9 (20M mixed-panel super training) as the recommended .npz policy for v0.26.0.",
        "why": (
            "Two-arm experiment: 20M games vs mixed panel (VL(v8) 65%, mirror 15%, heuristics+random 20%) "
            "vs 5M games vs the strongest teacher (PIMC belief 32x10). Volume+variety won on every axis."
        ),
        "evidence": (
            "Arm A vs v8: +0.97 (CI +0.82..+1.12, 100k paired); vs heuristic_v1: +18.78 (all-time record); "
            "head-to-head vs arm B: +0.63 (CI +0.48..+0.78). Arm B (elite teacher) reached only +0.40 vs v8 "
            "and regressed vs heuristic_v1 (16.71): monotone diet erodes anti-weak style."
        ),
        "impact": (
            "Default model moves to best_a2c_v9; first promotion that improves BOTH progression meters "
            "simultaneously (no style non-transitivity to explain). Mix recipe credited to the maintainer."
        ),
        "source": "data/models/best_a2c_v9.npz + two-arm super training (data/exit/due_bracci.log)",
    },
    {
        "order": 14,
        "date": "2026-07-05",
        "model_id": "best_a2c_v10",
        "type": "promoted",
        "decision": "Promote v10 (30M complete-panel run) as the recommended .npz policy for v0.27.0.",
        "why": (
            "Combines every lesson from the two-arm experiment: volume, variety, the elite PIMC-belief "
            "teacher at the per-game-efficient dose (25%), and a 25% heuristics share to keep anti-weak style."
        ),
        "evidence": (
            "Big holdout 100k paired CIs: vs best_a2c_v9 +0.66 (CI +0.51..+0.81); vs heuristic_v1 +20.52 "
            "(record, previous 18.78). Diminishing returns visible (+0.97 -> +0.66 per generation despite "
            "more games and a better teacher): the asymptote is near."
        ),
        "impact": "Default model moves to best_a2c_v10; both progression meters at all-time highs.",
        "source": "data/models/best_a2c_v10.npz + 30M complete-panel run (data/exit/definitivo.log)",
    },
    {
        "order": 15,
        "date": "2026-07-07",
        "model_id": "best_a2c_v11",
        "type": "promoted",
        "decision": "Promote v11 (5M PIMC dose-shift) as the recommended .npz policy for v0.31.0.",
        "why": (
            "Moving teacher dose from value-lookahead to PIMC belief restored the improvement slope with "
            "one sixth of v10's training volume."
        ),
        "evidence": (
            "Big 100k: +0.85 vs v10 (CI +0.71..+0.99), +20.80 vs heuristic_v1. The matched guard-on "
            "ablation reached only +0.32 vs v10."
        ),
        "impact": (
            "Default model moves to best_a2c_v11 and the inference overkill guard is disabled; PIMC belief "
            "16x8 becomes the efficient runtime default."
        ),
        "source": "docs/diario/14-sonda-e-dose-shift.md + benchmarks/experiments/fase3/v11_vs_*.json",
    },
    {
        "order": 16,
        "date": "2026-07-07",
        "model_id": "best_a2c_v11",
        "type": "runtime_ablation",
        "decision": "Retire the inference overkill guard for v11 and later promoted policies.",
        "why": "The wrapper had become harmful and obscured what behavior belonged to the policy itself.",
        "evidence": "Big 100k vs v10: +0.32 with guard, +0.85 without it; wrapper cost about 0.53 points/game.",
        "impact": "Later quality tables must expose wrapper and guard state explicitly.",
        "source": "docs/diario/14-sonda-e-dose-shift.md + benchmarks/experiments/fase3/v11guard_vs_v10_big.json",
    },
    {
        "order": 17,
        "date": "2026-07-08",
        "model_id": "best_a2c_v12",
        "type": "rejected",
        "decision": "Do not promote v12 after the 10M trump-saver opponent-mix run.",
        "why": (
            "The candidate was statistically flat against v11 and did not change the lead-load or trump-use "
            "behaviors it was intended to teach."
        ),
        "evidence": (
            "Big 100k: +0.11 vs v11 (CI -0.03..+0.25), +20.74 vs heuristic_v1; behavioral counters "
            "remained effectively identical to v11."
        ),
        "impact": "v11 remained the official best; v12 stayed a local rejected artifact.",
        "source": "docs/diario/14-sonda-e-dose-shift.md + benchmarks/experiments/fase3/v12_vs_*.json",
    },
    {
        "order": 18,
        "date": "2026-07-09",
        "model_id": "best_a2c_v13",
        "type": "promoted",
        "decision": "Promote v13 beta=0.3 as the recommended default model for v0.34.0.",
        "why": (
            "The overkill shaping changed the target behavior without measurable strength loss: same "
            "force as v11 within confidence intervals, much less low-lead trump overkill."
        ),
        "evidence": (
            "Policy-only v13 vs v11 -0.03 (CI -0.38..+0.32); default PIMC 16x8 v13 vs v11 +0.14 "
            "(CI -0.20..+0.47). Low-lead overkill drops from 27.8% to 6.1% vs v11 and from 31.4% "
            "to 7.8% vs heuristic_v1."
        ),
        "impact": (
            "Default remains bc_model_pimc_belief_16x8, but now backed by best_a2c_v13.npz; no runtime "
            "overkill guard is reintroduced."
        ),
        "source": "data/models/best_a2c_v13.npz + v13_overkill_gap_beta0300_5M_seed20260709 gates",
    },
    {
        "order": 19,
        "date": "2026-07-12",
        "model_id": "best_a2c_v14",
        "type": "promoted",
        "decision": "Promote the 50k suit-symmetry distillation as the recommended model for v0.36.0.",
        "why": (
            "The distilled policy keeps one normal forward pass while transferring most of the exact 24-view "
            "teacher's suit invariance and its playing-strength benefit."
        ),
        "evidence": (
            "Suit flips 18.19% -> 6.04%; policy-only +0.66 vs v13 (CI +0.24..+1.09); default PIMC belief "
            "16x8 +0.43 vs v13 (CI +0.03..+0.84); big 100k Numba +21.76 vs heuristic_v1."
        ),
        "impact": (
            "Default remains bc_model_pimc_belief_16x8, now backed by best_a2c_v14.npz; inference cost and "
            "the no-guard runtime configuration remain unchanged."
        ),
        "source": "docs/plans/suit-distillation-v0-2026-07-11.md + suit_distillation_v0_50k_seed20260712 gates",
    },
    {
        "order": 20,
        "date": "2026-07-17",
        "model_id": "best_a2c_v15",
        "type": "promoted",
        "decision": "Promote the teacher-20M student with PIMC belief 12x8 as the v0.38.0 default.",
        "why": (
            "The larger suit-distillation transfers the 20M teacher into one compact policy, while 12x8 preserves "
            "the measured runtime strength of v14 16x8 at lower search cost."
        ),
        "evidence": (
            "Policy-only +0.18 vs v14 on 100k and +21.87 vs heuristic_v1 on the homogeneous big gate; "
            "runtime 12x8 vs v14 16x8 +0.11 (CI -0.11..+0.32) on 20k, with mean/p95 latency about 25% lower."
        ),
        "impact": (
            "The default becomes bc_model_pimc_belief_12x8(best_a2c_v15.npz). v14 16x8 and the older official "
            "models remain selectable; 64x10 remains the maximum-search option."
        ),
        "source": "docs/diario/23-dodici-mondi-bastano.md + teacher20M 250k distillation and 12x8 gates",
    },
]


def repo_path(path: Path | str) -> str:
    """Format a path relative to the repository root when possible."""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    return str(p)


def load_npz_metadata(path: Path) -> dict[str, Any]:
    """Load `metadata_json` from a model file, returning a small error record if unavailable."""
    if not path.exists():
        return {"_missing": True}
    with np.load(path, allow_pickle=False) as data:
        raw = data.get("metadata_json")
        if raw is None:
            return {"_missing": False, "_metadata_missing": True}
        return json.loads(str(raw))


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON file using a repo-relative path."""
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_evidence_manifest() -> dict[str, Any]:
    """Load and minimally validate the committed evidence snapshot.

    A normal build must work in a clean clone, where many historical model weights and
    raw benchmark outputs are intentionally absent. The snapshot is therefore the
    canonical build input; raw paths preserved inside it are audit references, not
    runtime dependencies.
    """
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evidence schema {payload.get('schema_version')!r}; expected {EVIDENCE_SCHEMA_VERSION}"
        )
    required_sections = {"models", "promotion_evidence", "decision_quality"}
    missing = required_sections.difference(payload)
    if missing:
        raise ValueError(f"Evidence manifest is missing sections: {sorted(missing)}")
    return payload


def manifest_model_metadata(model_id: str) -> dict[str, Any]:
    """Return the versioned metadata snapshot for one curated model."""
    models = load_evidence_manifest()["models"]
    try:
        return dict(models[model_id])
    except KeyError as exc:
        raise ValueError(f"Evidence manifest has no metadata for {model_id!r}") from exc


def manifest_rows(section: str) -> list[dict[str, Any]]:
    """Return normalized rows and attach a stable JSON-pointer source."""
    rows = []
    for index, source_row in enumerate(load_evidence_manifest()[section]):
        row = dict(source_row)
        row["source"] = f"{repo_path(EVIDENCE_PATH)}#/{section}/{index}"
        rows.append(row)
    return rows


def short_opponent(name: str) -> str:
    """Normalize verbose bc_model labels to stable report names."""
    match = re.search(r"bc_model\(([^,]+)", name)
    if match:
        return match.group(1)
    return name


def matrix_rows(source: str, *, model_id: str, label: str) -> list[dict[str, Any]]:
    """Flatten an evaluation matrix JSON into report rows."""
    payload = load_json(source)
    out: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        stats = row["stats"]
        suite = row["suite"]
        out.append(
            {
                "model_id": model_id,
                "label": label,
                "benchmark": payload.get("benchmark"),
                "engine": payload.get("engine"),
                "suite": suite.get("name"),
                "seed_range_start": suite.get("range_start"),
                "seat_fair": True,
                "opponent": short_opponent(row.get("opponent") or stats.get("agent_b_name", "")),
                "avg_diff": stats.get("avg_point_diff_agent_a_minus_agent_b"),
                "wins_model": stats.get("wins_agent_a"),
                "wins_opponent": stats.get("wins_agent_b"),
                "draws": stats.get("draws"),
                "eval_games": stats.get("num_games") or payload.get("num_games"),
                "source": source,
                "data_quality": "exact",
            }
        )
    return out


def h2h_rows(source: str, *, model_id: str, label: str, opponent: str) -> list[dict[str, Any]]:
    """Flatten a head-to-head JSON that may be either matrix-shaped or stats-shaped."""
    payload = load_json(source)
    if "rows" in payload:
        return matrix_rows(source, model_id=model_id, label=label)
    stats = payload["stats"]
    seed_suite = payload.get("seed_suite", {})
    suite_name = seed_suite.get("name")
    if not suite_name:
        suite_name = "holdout" if seed_suite.get("range_start") == 1_000_000 else "standard"
    return [
        {
            "model_id": model_id,
            "label": label,
            "benchmark": payload.get("benchmark"),
            "engine": payload.get("engine"),
            "suite": suite_name,
            "seed_range_start": seed_suite.get("range_start"),
            "seat_fair": True,
            "opponent": opponent,
            "avg_diff": stats.get("avg_point_diff_agent_a_minus_agent_b"),
            "wins_model": stats.get("wins_agent_a"),
            "wins_opponent": stats.get("wins_agent_b"),
            "draws": stats.get("draws"),
            "eval_games": stats.get("num_games") or payload.get("num_games"),
            "source": source,
            "data_quality": "exact",
        }
    ]


def _decision_quality_rows_from_local_sources() -> list[dict[str, Any]]:
    """Read decision-quality rows from local raw artifacts for a snapshot refresh."""
    sources = [
        (
            "best_a2c_v3",
            "Best A2C v3",
            "benchmarks/experiments/best_a2c_v3_decision_quality_vs_heuristic_v1_2026-06-28_medium_numba.json",
        ),
        (
            "best_a2c_v4",
            "Best A2C v4",
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/decision_quality_vs_heuristic_v1_medium_numba.json",
        ),
        (
            "best_a2c_v5",
            "Best A2C v5",
            "benchmarks/experiments/a2c_v5_seed401_1m_numba/decision_quality_vs_heuristic_v1_big_numba.json",
        ),
        (
            "best_a2c_v6",
            "Best A2C v6",
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/quality_5m_vs_heuristic_v1_big_numba.json",
        ),
        (
            "best_a2c_v11",
            "Best A2C v11",
            "benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/quality_v11_vs_heuristic_v1_medium.json",
        ),
        (
            "best_a2c_v13",
            "Best A2C v13",
            "benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/quality_v13_vs_heuristic_v1_medium.json",
        ),
        (
            "best_a2c_v14",
            "Best A2C v14",
            "benchmarks/experiments/suit_distillation_v0_50k_seed20260712/quality_vs_heuristic_v1_medium.json",
        ),
        (
            "best_a2c_v15",
            "Best A2C v15",
            "benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/"
            "decision_quality_student_vs_heuristic_v1_medium.json",
        ),
    ]
    rows = []
    for model_id, label, source in sources:
        payload = load_json(source)
        quality = payload.get("quality", {})
        match = payload.get("match", {})
        rows.append(
            {
                "model_id": model_id,
                "label": label,
                "benchmark": payload.get("benchmark"),
                "engine": payload.get("engine"),
                "suite": "seat_fair",
                "seed_range_start": None,
                "seat_fair": True,
                "opponent": "heuristic_v1",
                "avg_diff": match.get("avg_point_diff_agent_a_minus_agent_b"),
                "trump_waste_rate": quality.get("trump_waste_rate"),
                "trump_overkill_rate": quality.get("trump_overkill_rate"),
                "trump_overkill_low_rate": quality.get("trump_overkill_rate_low_lead_points"),
                "eval_games": payload.get("num_games"),
                "source": source,
                "data_quality": "exact",
            }
        )
    return rows


_PROMOTED_GUARD_BY_MODEL = {
    "best_a2c": True,
    "best_a2c_v3": True,
    "best_a2c_v4": True,
    "best_a2c_v5": True,
    "best_a2c_v6": True,
    "best_a2c_v7": True,
    "best_a2c_v8": True,
    "best_a2c_v9": True,
    "best_a2c_v10": True,
    "best_a2c_v11": False,
    "best_a2c_v11_guard_on": True,
    "best_a2c_v12": False,
    "best_a2c_v13": False,
    "best_a2c_v14": False,
    "best_a2c_v15": False,
}


def _runtime_fields(row: dict[str, Any]) -> tuple[str, bool]:
    """Describe the evaluated inference stack, including post-policy wrappers."""
    label = str(row.get("label", ""))
    if "PIMC 12x8" in label:
        return "bc_model_pimc_belief_12x8", False
    if "PIMC 16x8" in label:
        return "bc_model_pimc_belief_16x8", False
    model_id = str(row["model_id"])
    return "bc_model (direct policy)", _PROMOTED_GUARD_BY_MODEL[model_id]


def _snapshot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare locally extracted rows for the compact committed snapshot."""
    snapshot = []
    for source_row in rows:
        row = dict(source_row)
        row["raw_source"] = row.pop("source")
        row["wrapper"], row["guard"] = _runtime_fields(row)
        snapshot.append(row)
    return snapshot


def _snapshot_model_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep only report-relevant metadata, excluding large training histories."""
    keys = (
        "label",
        "format",
        "encoder_version",
        "feature_dim",
        "seed",
        "init",
        "opponent_mix",
        "bc_anchor_path",
        "bc_anchor_beta",
        "inference_overkill_guard",
    )
    snapshot = {key: meta.get(key) for key in keys}
    train = meta.get("train") or {}
    snapshot["train"] = {"num_games": train.get("num_games")} if train.get("num_games") is not None else {}
    return snapshot


def model_rows() -> list[dict[str, Any]]:
    """Build one summary row per significant model."""
    rows = []
    for spec in MODEL_SPECS:
        meta = manifest_model_metadata(spec.model_id)
        train = meta.get("train") or {}
        rows.append(
            {
                "order": spec.order,
                "model_id": spec.model_id,
                "role": spec.role,
                "status": spec.status,
                "path": repo_path(spec.path),
                "label": meta.get("label", ""),
                "format": meta.get("format", ""),
                "encoder": meta.get("encoder_version", ""),
                "feature_dim": meta.get("feature_dim", ""),
                "training_games": train.get("num_games", ""),
                "seed": meta.get("seed", ""),
                "init": meta.get("init", ""),
                "opponent_mix": json.dumps(meta.get("opponent_mix"), ensure_ascii=False)
                if meta.get("opponent_mix") is not None
                else "",
                "bc_anchor": meta.get("bc_anchor_path", ""),
                "bc_anchor_beta": meta.get("bc_anchor_beta", ""),
                "guard": meta.get("inference_overkill_guard", ""),
                "reference_vs_h1": spec.progress_score,
                "reference_vs_h1_protocol": METRIC_PROTOCOLS[spec.model_id]["progress"],
                "reference_h2h": spec.h2h_score,
                "reference_h2h_protocol": METRIC_PROTOCOLS[spec.model_id]["h2h"],
                "decision": spec.decision,
                "notes": spec.notes,
                "data_quality": spec.data_quality,
            }
        )
    return rows


def _promotion_rows_from_local_sources() -> list[dict[str, Any]]:
    """Read normalized promotion rows from local raw artifacts for a snapshot refresh."""
    rows: list[dict[str, Any]] = []
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/matrix_big.json",
            model_id="best_a2c",
            label="Best A2C v2",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/head_to_head_best_a2c_v2_big_numba.json",
            model_id="best_a2c",
            label="Best A2C v2",
            opponent="previous_best_v2",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/baseline_best_a2c_v3_big_vs_heuristic_v1_numba.json",
            model_id="best_a2c_v3",
            label="Best A2C v3",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/best_a2c_v3_vs_best_a2c_2026-06-28_big_numba.json",
            model_id="best_a2c_v3",
            label="Best A2C v3",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/eval_big_vs_heuristic_v1_numba.json",
            model_id="best_a2c_v4",
            label="Best A2C v4",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/head_to_head_best_a2c_v3_big_numba.json",
            model_id="best_a2c_v4",
            label="Best A2C v4",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v5_seed401_1m_numba/eval_big_vs_heuristic_v1_numba.json",
            model_id="best_a2c_v5",
            label="Best A2C v5",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v5_seed401_1m_numba/head_to_head_best_a2c_v4_big_numba.json",
            model_id="best_a2c_v5",
            label="Best A2C v5",
            opponent="best_a2c_v4",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_heuristic_v1_big_numba.json",
            model_id="best_a2c_v6",
            label="Best A2C v6",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_best_a2c_v5_big_numba.json",
            model_id="best_a2c_v6",
            label="Best A2C v6",
            opponent="best_a2c_v5",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_best_a2c_v5_big_holdout_numba.json",
            model_id="best_a2c_v6",
            label="Best A2C v6",
            opponent="best_a2c_v5_holdout",
        )
    )
    rows.extend(
        h2h_rows(
            "data/eval_best_a2c_v7_vs_heuristic_v1_big_holdout_seedrange1000000.json",
            model_id="best_a2c_v7",
            label="Best A2C v7",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "data/eval_a2c_vs_value_lookahead_5M_vs_v6_big_holdout_seedrange1000000.json",
            model_id="best_a2c_v7",
            label="Best A2C v7",
            opponent="best_a2c_v6",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/iter2_h256_vs_heuristic_v1_big.json",
            model_id="best_a2c_v8",
            label="Best A2C v8",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/iter2_h256_vs_v7_big.json",
            model_id="best_a2c_v8",
            label="Best A2C v8",
            opponent="best_a2c_v7",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/superA_vs_h1_big.json",
            model_id="best_a2c_v9",
            label="Best A2C v9",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/superA_vs_v8_big.json",
            model_id="best_a2c_v9",
            label="Best A2C v9",
            opponent="best_a2c_v8",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/definitivo_vs_h1_big.json",
            model_id="best_a2c_v10",
            label="Best A2C v10",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/definitivo_vs_v9_big.json",
            model_id="best_a2c_v10",
            label="Best A2C v10",
            opponent="best_a2c_v9",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/v11_vs_h1_big.json",
            model_id="best_a2c_v11",
            label="Best A2C v11",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/v11_vs_v10_big.json",
            model_id="best_a2c_v11",
            label="Best A2C v11",
            opponent="best_a2c_v10",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/v11guard_vs_v10_big.json",
            model_id="best_a2c_v11_guard_on",
            label="Best A2C v11 + overkill guard",
            opponent="best_a2c_v10",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/v12_vs_h1_big.json",
            model_id="best_a2c_v12",
            label="Rejected A2C v12",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/v12_vs_v11_big.json",
            model_id="best_a2c_v12",
            label="Rejected A2C v12",
            opponent="best_a2c_v11",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/fase3/v12_vs_trump_saver_big.json",
            model_id="best_a2c_v12",
            label="Rejected A2C v12",
            opponent="heuristic_trump_saver",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/eval_v13_vs_v11_medium.json",
            model_id="best_a2c_v13",
            label="Best A2C v13 policy",
            opponent="best_a2c_v11_policy",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/eval_v13_vs_heuristic_v1_medium.json",
            model_id="best_a2c_v13",
            label="Best A2C v13 policy",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            (
                "benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/"
                "pimc16x8_medium/v13_vs_v11_pimc16x8_medium.json"
            ),
            model_id="best_a2c_v13",
            label="Best A2C v13 PIMC 16x8",
            opponent="best_a2c_v11_pimc16x8",
        )
    )
    rows.extend(
        h2h_rows(
            (
                "benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/"
                "pimc16x8_medium/v13_vs_trumpsaver_pimc16x8_medium.json"
            ),
            model_id="best_a2c_v13",
            label="Best A2C v13 PIMC 16x8",
            opponent="heuristic_trump_saver",
        )
    )
    rows.extend(
        h2h_rows(
            (
                "benchmarks/experiments/v13_overkill_gap_beta0300_5M_seed20260709/"
                "pimc16x8_medium/v13_vs_heuristic_v1_pimc16x8_medium.json"
            ),
            model_id="best_a2c_v13",
            label="Best A2C v13 PIMC 16x8",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            ("benchmarks/experiments/suit_distillation_v0_50k_seed20260712/eval_v14_vs_heuristic_v1_big_numba.json"),
            model_id="best_a2c_v14",
            label="Best A2C v14 policy",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/suit_distillation_v0_50k_seed20260712/eval_vs_v13_medium.json",
            model_id="best_a2c_v14",
            label="Best A2C v14 policy",
            opponent="best_a2c_v13_policy",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/suit_distillation_v0_50k_seed20260712/pimc16x8_vs_v13_medium.json",
            model_id="best_a2c_v14",
            label="Best A2C v14 PIMC 16x8",
            opponent="best_a2c_v13_pimc16x8",
        )
    )
    rows.extend(
        h2h_rows(
            (
                "benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/"
                "eval_v15_vs_heuristic_v1_big_numba.json"
            ),
            model_id="best_a2c_v15",
            label="Best A2C v15 policy",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            (
                "benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/"
                "confirm_student_vs_v14_100k.json"
            ),
            model_id="best_a2c_v15",
            label="Best A2C v15 policy",
            opponent="best_a2c_v14_policy",
        )
    )
    rows.extend(
        h2h_rows(
            (
                "benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724/"
                "efficiency_12x8/confirm_student12_vs_v14_16_20k.json"
            ),
            model_id="best_a2c_v15",
            label="Best A2C v15 PIMC 12x8",
            opponent="best_a2c_v14_pimc16x8",
        )
    )
    return rows


def promotion_rows() -> list[dict[str, Any]]:
    """Return promotion evidence from the committed canonical snapshot."""
    return manifest_rows("promotion_evidence")


def decision_quality_rows() -> list[dict[str, Any]]:
    """Return decision-quality evidence from the committed canonical snapshot."""
    return manifest_rows("decision_quality")


def refresh_evidence_manifest() -> None:
    """Refresh the canonical snapshot from local model and benchmark artifacts.

    This is deliberately opt-in because the raw inputs are large and gitignored. A
    maintainer with the historical artifacts can refresh and review the compact JSON;
    everyone else, including CI, can reproduce the workbook directly from that JSON.
    """
    raw_metadata = {spec.model_id: load_npz_metadata(spec.path) for spec in MODEL_SPECS}
    missing_models = [model_id for model_id, metadata in raw_metadata.items() if metadata.get("_missing")]
    if missing_models:
        raise FileNotFoundError(f"Cannot refresh evidence; missing model files: {missing_models}")
    models = {model_id: _snapshot_model_metadata(metadata) for model_id, metadata in raw_metadata.items()}
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "snapshot_date": EVIDENCE_SNAPSHOT_DATE,
        "description": (
            "Canonical compact inputs for docs/reports/model_progress.xlsx. raw_source paths are optional "
            "audit references; normal report builds use only this file."
        ),
        "models": models,
        "promotion_evidence": _snapshot_rows(_promotion_rows_from_local_sources()),
        "decision_quality": _snapshot_rows(_decision_quality_rows_from_local_sources()),
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sources_rows() -> list[dict[str, Any]]:
    """List sources used by the report."""
    rows = [
        {
            "kind": "canonical_evidence",
            "id": "model_progress.v1",
            "path": repo_path(EVIDENCE_PATH),
            "data_quality": "versioned_snapshot",
            "note": "Complete build input; sufficient to regenerate the workbook in a clean clone.",
        },
        {
            "kind": "stable_narrative",
            "id": "model_chain",
            "path": "docs/diario/05-catena-campioni.md",
            "data_quality": "curated_narrative",
            "note": "Historical model chain and promotion context.",
        },
        {
            "kind": "stable_narrative",
            "id": "belief_exit",
            "path": "docs/plans/belief-expert-iteration.md",
            "data_quality": "curated_experiment_log",
            "note": "ExIt, belief-input, distillation, and real-history conclusions.",
        },
        {
            "kind": "stable_narrative",
            "id": "v11_v12",
            "path": "docs/diario/14-sonda-e-dose-shift.md",
            "data_quality": "curated_experiment_log",
            "note": "v11 promotion, guard ablation, and v12 rejection.",
        },
        {
            "kind": "stable_narrative",
            "id": "v13",
            "path": "docs/diario/17-stessa-forza-comportamento-migliore.md",
            "data_quality": "curated_experiment_log",
            "note": "v13 policy/PIMC strength gates and behavior-quality evidence.",
        },
        {
            "kind": "stable_narrative",
            "id": "v14",
            "path": "docs/plans/suit-distillation-v0-2026-07-11.md",
            "data_quality": "curated_experiment_log",
            "note": "v14 suit-distillation pipeline, policy/PIMC gates, and promotion decision.",
        },
        {
            "kind": "stable_narrative",
            "id": "v15",
            "path": "docs/diario/23-dodici-mondi-bastano.md",
            "data_quality": "curated_experiment_log",
            "note": "v15 teacher-20M distillation, 12x8 efficiency gate, release audit, and promotion decision.",
        },
    ]
    for spec in MODEL_SPECS:
        rows.append(
            {
                "kind": "optional_raw_model",
                "id": spec.model_id,
                "path": repo_path(spec.path),
                "data_quality": spec.data_quality,
                "note": "Original metadata audit input; normalized fields are preserved in canonical_evidence.",
            }
        )
        if spec.progress_source:
            rows.append(
                {
                    "kind": "optional_raw_metric",
                    "id": spec.model_id,
                    "path": spec.progress_source,
                    "data_quality": "exact" if spec.progress_score is not None else "not_applicable",
                    "note": "Original metric audit input; not required to build the workbook.",
                }
            )
        if spec.h2h_source:
            rows.append(
                {
                    "kind": "optional_raw_metric",
                    "id": spec.model_id,
                    "path": spec.h2h_source,
                    "data_quality": "exact",
                    "note": "Original metric audit input; not required to build the workbook.",
                }
            )
    for row in REJECTED_CANDIDATES:
        rows.append(
            {
                "kind": "rejected_candidate_reference",
                "id": row["candidate"],
                "path": row["path"],
                "data_quality": "manual_summary",
                "note": row["evidence"],
            }
        )
    return rows


def detail_rows(spec: ModelSpec) -> list[list[Any]]:
    """Build a model detail sheet."""
    meta = manifest_model_metadata(spec.model_id)
    train = meta.get("train") or {}
    rows: list[list[Any]] = [
        [f"Detail: {spec.model_id}"],
        [],
        ["Key", "Value"],
        ["Role", spec.role],
        ["Status", spec.status],
        ["Path", repo_path(spec.path)],
        ["Label", meta.get("label", "")],
        ["Format", meta.get("format", "")],
        ["Encoder", meta.get("encoder_version", "")],
        ["Feature dim", meta.get("feature_dim", "")],
        ["Training games", train.get("num_games", "")],
        ["Seed", meta.get("seed", "")],
        ["Init", meta.get("init", "")],
        ["Opponent mix", json.dumps(meta.get("opponent_mix"), ensure_ascii=False) if meta.get("opponent_mix") else ""],
        ["BC anchor", meta.get("bc_anchor_path", "")],
        ["BC anchor beta", meta.get("bc_anchor_beta", "")],
        ["Guard anti-overkill", meta.get("inference_overkill_guard", "")],
        ["Progress score", spec.progress_score if spec.progress_score is not None else ""],
        ["Progress protocol", METRIC_PROTOCOLS[spec.model_id]["progress"]],
        ["H2H score", spec.h2h_score if spec.h2h_score is not None else ""],
        ["H2H protocol", METRIC_PROTOCOLS[spec.model_id]["h2h"]],
        ["Decision", spec.decision],
        ["Notes", spec.notes],
        [],
        ["Decision milestones"],
        ["Date", "Type", "Decision", "Why", "Evidence"],
    ]
    for milestone in MILESTONES:
        if milestone["model_id"] == spec.model_id:
            rows.append(
                [
                    milestone["date"],
                    milestone["type"],
                    milestone["decision"],
                    milestone["why"],
                    milestone["evidence"],
                ]
            )
    return rows


def sheet_from_dicts(rows: list[dict[str, Any]], columns: list[str]) -> list[list[Any]]:
    """Convert dict rows to a worksheet-like matrix."""
    return [columns] + [[row.get(column, "") for column in columns] for row in rows]


def build_workbook_data() -> dict[str, list[list[Any]]]:
    """Build all report sheets."""
    models = model_rows()
    promotion = promotion_rows()
    quality = decision_quality_rows()

    # This deliberately narrow series is the only charted comparison. It shares
    # benchmark size, engine, opponent, seed range, and seat-fair protocol. Wrapper
    # state is still shown per row because v8-v10 used the promoted guard while v11
    # did not; the chart is product-stack evidence, not pure architecture progress.
    chart_rows = []
    for model_id in HOMOGENEOUS_CHART_MODEL_IDS:
        candidates = [
            row
            for row in promotion
            if row["model_id"] == model_id
            and row["opponent"] == "heuristic_v1"
            and row["benchmark"] == "big"
            and row["engine"] == "numba"
            and row["suite"] == "standard"
            and row["eval_games"] == 100_000
            and row["wrapper"] == "bc_model (direct policy)"
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one homogeneous chart row for {model_id}, found {len(candidates)}")
        chart_rows.append(candidates[0])

    dashboard: list[list[Any]] = [
        ["Briscola AI - Model Progress Report"],
        ["Curated report for significant models only. Canonical input: docs/reports/evidence/model_progress.v1.json."],
        [],
        ["Homogeneous recent comparison"],
        [
            "Model",
            "Avg point diff vs heuristic_v1",
            "Evaluation games",
            "Protocol",
            "Wrapper",
            "Guard",
        ],
    ]
    for row in chart_rows:
        dashboard.append(
            [
                row["model_id"],
                row["avg_diff"],
                row["eval_games"],
                "big 100k; standard seeds 0..49,999; seat-fair; Numba; promoted runtime config",
                row["wrapper"],
                row["guard"],
            ]
        )
    dashboard.extend(
        [
            [],
            ["Current conclusion"],
            [
                "best_a2c_v15 is the recommended .npz policy in current "
                f"v{project_version()}; it backs the default bc_model_pimc_belief_12x8 stack without an "
                "overkill guard. The 250k-game distillation transfers the exact 24-view average of the strongest "
                "20M teacher into one compact policy. Policy-only v15 is +0.18 vs v14 on 100k paired games and "
                "+21.87 vs heuristic_v1 on the homogeneous big gate, so v15 is the last chart row. The runtime "
                "12x8 gate is +0.11 vs v14 16x8 (CI -0.11..+0.32) on 20k games while mean and p95 search latency "
                "fall by about 25%. This is an efficiency/non-inferiority promotion, not a claim that 12x8 is "
                "statistically stronger. v13 remains outside the chart because it has no big 100k result."
            ],
            [],
            ["Quick comparison"],
            ["Model", "Status", "Encoder", "Training games", "Decision"],
        ]
    )
    for row in models:
        dashboard.append([row["model_id"], row["status"], row["encoder"], row["training_games"], row["decision"]])

    sheets: dict[str, list[list[Any]]] = {
        "Dashboard": dashboard,
        "Milestones": sheet_from_dicts(
            MILESTONES,
            ["order", "date", "model_id", "type", "decision", "why", "evidence", "impact", "source"],
        ),
        "Best Models": sheet_from_dicts(
            models,
            [
                "order",
                "model_id",
                "role",
                "status",
                "path",
                "label",
                "encoder",
                "feature_dim",
                "training_games",
                "seed",
                "init",
                "opponent_mix",
                "bc_anchor",
                "bc_anchor_beta",
                "guard",
                "reference_vs_h1",
                "reference_vs_h1_protocol",
                "reference_h2h",
                "reference_h2h_protocol",
                "decision",
                "notes",
                "data_quality",
            ],
        ),
        "Promotion Evidence": sheet_from_dicts(
            promotion,
            [
                "model_id",
                "label",
                "benchmark",
                "engine",
                "suite",
                "seed_range_start",
                "seat_fair",
                "wrapper",
                "guard",
                "opponent",
                "avg_diff",
                "wins_model",
                "wins_opponent",
                "draws",
                "eval_games",
                "source",
                "raw_source",
                "data_quality",
            ],
        ),
        "Decision Quality": sheet_from_dicts(
            quality,
            [
                "model_id",
                "label",
                "benchmark",
                "engine",
                "suite",
                "seed_range_start",
                "seat_fair",
                "wrapper",
                "guard",
                "opponent",
                "avg_diff",
                "trump_waste_rate",
                "trump_overkill_rate",
                "trump_overkill_low_rate",
                "eval_games",
                "source",
                "raw_source",
                "data_quality",
            ],
        ),
        "Rejected Candidates": sheet_from_dicts(
            REJECTED_CANDIDATES,
            ["candidate", "path", "training_games", "decision", "reason", "evidence"],
        ),
        "Sources": sheet_from_dicts(sources_rows(), ["kind", "id", "path", "data_quality", "note"]),
    }
    for spec in MODEL_SPECS:
        sheets[f"Detail {spec.model_id}"[:31]] = detail_rows(spec)
    return sheets


def col_name(index: int) -> str:
    """Return Excel column name for 1-based index."""
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_ref(row: int, col: int, *, absolute: bool = False) -> str:
    """Return an Excel cell reference."""
    c = col_name(col)
    if absolute:
        return f"${c}${row}"
    return f"{c}{row}"


def sheet_xml(rows: list[list[Any]], *, sheet_name: str, drawing_rel: bool = False) -> str:
    """Serialize one worksheet."""
    max_col = max((len(row) for row in rows), default=1)
    max_row = max(len(rows), 1)
    dimension = f"A1:{cell_ref(max_row, max_col)}"
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        f'<dimension ref="{dimension}"/>',
    ]
    if sheet_name != "Dashboard":
        parts.append(
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
            'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        )
    else:
        parts.append('<sheetViews><sheetView workbookViewId="0"/></sheetViews>')
    widths = [18, 18, 18, 18, 28, 28, 24, 24, 18, 18, 18, 28, 24, 18, 18, 18, 18, 36, 40]
    parts.append("<cols>")
    for idx in range(1, max_col + 1):
        width = widths[idx - 1] if idx <= len(widths) else 22
        parts.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')
    parts.append("</cols><sheetData>")
    for r_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            if value is None:
                continue
            ref = cell_ref(r_idx, c_idx)
            style = "1" if r_idx == 1 or (r_idx == 5 and sheet_name == "Dashboard") else "0"
            if isinstance(value, bool):
                parts.append(f'<c r="{ref}" s="{style}" t="b"><v>{1 if value else 0}</v></c>')
            elif isinstance(value, int | float) and not isinstance(value, bool):
                number = f"{float(value):.10g}" if isinstance(value, float) else str(value)
                num_style = "2" if isinstance(value, float) else style
                parts.append(f'<c r="{ref}" s="{num_style}"><v>{number}</v></c>')
            else:
                text = escape(str(value))
                wrap_style = "3" if len(str(value)) > 60 else style
                parts.append(f'<c r="{ref}" s="{wrap_style}" t="inlineStr"><is><t>{text}</t></is></c>')
        parts.append("</row>")
    parts.append("</sheetData>")
    if sheet_name != "Dashboard" and max_row > 1 and max_col > 1:
        parts.append(f'<autoFilter ref="A1:{cell_ref(max_row, max_col)}"/>')
    if drawing_rel:
        parts.append('<drawing r:id="rId1"/>')
    parts.append("</worksheet>")
    return "".join(parts)


def workbook_xml(sheet_names: list[str]) -> str:
    """Serialize workbook.xml."""
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<workbookPr/>"
        '<bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" windowHeight="14000"/></bookViews>'
        f"<sheets>{''.join(sheets)}</sheets>"
        "</workbook>"
    )


def workbook_rels(sheet_names: list[str]) -> str:
    """Serialize workbook relationships."""
    rels = []
    for idx, _ in enumerate(sheet_names, start=1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    """Serialize [Content_Types].xml."""
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/xl/drawings/drawing1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>',
        '<Override PartName="/xl/charts/chart1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}</Types>"
    )


def root_rels_xml() -> str:
    """Serialize package root relationships."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    """Serialize a compact styles.xml."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><color rgb="FF1F2937"/><name val="Aptos"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Aptos"/></font>'
        "</fonts>"
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF2563EB"/><bgColor indexed="64"/></patternFill></fill>'
        "</fills>"
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="4">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="4" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">'
        '<alignment wrapText="1" vertical="top"/></xf>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def sheet_drawing_rels_xml() -> str:
    """Worksheet relationship to the dashboard drawing."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
        'Target="../drawings/drawing1.xml"/>'
        "</Relationships>"
    )


def drawing_xml() -> str:
    """Drawing anchor for the dashboard chart."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<xdr:twoCellAnchor>"
        "<xdr:from><xdr:col>5</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>3</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>"
        "<xdr:to><xdr:col>16</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>26</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>"
        '<xdr:graphicFrame macro="">'
        '<xdr:nvGraphicFramePr><xdr:cNvPr id="2" name="Progression Chart"/>'
        "<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr>"
        '<xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        '<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId1"/>'
        "</a:graphicData></a:graphic>"
        "</xdr:graphicFrame>"
        "<xdr:clientData/>"
        "</xdr:twoCellAnchor>"
        "</xdr:wsDr>"
    )


def drawing_rels_xml() -> str:
    """Drawing relationship to the chart."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
        'Target="../charts/chart1.xml"/>'
        "</Relationships>"
    )


def dashboard_progress_row_count(dashboard_rows: list[list[Any]]) -> int:
    """Count homogeneous comparison rows in the Dashboard sheet.

    The Dashboard section is deliberately simple: title row, header row, then one
    comparable row per selected runtime configuration until the blank separator.
    """
    for idx, row in enumerate(dashboard_rows):
        if row[:1] == ["Homogeneous recent comparison"]:
            data_start = idx + 2
            count = 0
            for data_row in dashboard_rows[data_start:]:
                if not data_row or data_row[0] == "":
                    break
                count += 1
            return count
    return 0


def chart_xml(progress_row_count: int) -> str:
    """Chart XML for the homogeneous recent runtime-stack comparison."""
    first_row = 6
    last_row = max(first_row, first_row + progress_row_count - 1)
    cats_ref = f"Dashboard!$A${first_row}:$A${last_row}"
    vals_ref = f"Dashboard!$B${first_row}:$B${last_row}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<c:chart>"
        '<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="it-IT" sz="1400" b="1"/>'
        "<a:t>Confronto omogeneo recente vs heuristic_v1</a:t></a:r></a:p></c:rich></c:tx></c:title>"
        "<c:plotArea><c:layout/>"
        '<c:lineChart><c:grouping val="standard"/>'
        '<c:ser><c:idx val="0"/><c:order val="0"/>'
        "<c:tx><c:v>big 100k, seed 0..49,999, configurazione promossa</c:v></c:tx>"
        '<c:marker><c:symbol val="circle"/><c:size val="7"/></c:marker>'
        f"<c:cat><c:strRef><c:f>{cats_ref}</c:f></c:strRef></c:cat>"
        f"<c:val><c:numRef><c:f>{vals_ref}</c:f></c:numRef></c:val>"
        "</c:ser>"
        '<c:axId val="100"/><c:axId val="101"/>'
        "</c:lineChart>"
        '<c:catAx><c:axId val="100"/><c:scaling><c:orientation val="minMax"/></c:scaling>'
        '<c:delete val="0"/><c:axPos val="b"/><c:tickLblPos val="nextTo"/><c:crossAx val="101"/>'
        '<c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/></c:catAx>'
        '<c:valAx><c:axId val="101"/><c:scaling><c:orientation val="minMax"/></c:scaling>'
        '<c:delete val="0"/><c:axPos val="l"/><c:majorGridlines/><c:numFmt formatCode="0.00" sourceLinked="0"/>'
        '<c:tickLblPos val="nextTo"/><c:crossAx val="100"/><c:crosses val="autoZero"/>'
        '<c:crossBetween val="between"/></c:valAx>'
        "</c:plotArea>"
        '<c:legend><c:legendPos val="r"/><c:layout/><c:overlay val="0"/>'
        '<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="1100"/></a:pPr>'
        '<a:endParaRPr lang="it-IT"/></a:p></c:txPr></c:legend>'
        '<c:plotVisOnly val="1"/>'
        "</c:chart>"
        "</c:chartSpace>"
    )


def write_xlsx(sheets: dict[str, list[list[Any]]], out_path: Path) -> None:
    """Write the report as an .xlsx file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets)
    progress_row_count = dashboard_progress_row_count(sheets["Dashboard"])

    def write_text(zf: zipfile.ZipFile, filename: str, content: str) -> None:
        """Write one XML part with stable ZIP metadata so regeneration is deterministic."""
        info = zipfile.ZipInfo(filename=filename, date_time=_XLSX_ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, content.encode("utf-8"))

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        write_text(zf, "[Content_Types].xml", content_types_xml(len(sheet_names)))
        write_text(zf, "_rels/.rels", root_rels_xml())
        write_text(zf, "xl/workbook.xml", workbook_xml(sheet_names))
        write_text(zf, "xl/_rels/workbook.xml.rels", workbook_rels(sheet_names))
        write_text(zf, "xl/styles.xml", styles_xml())
        for idx, name in enumerate(sheet_names, start=1):
            write_text(
                zf,
                f"xl/worksheets/sheet{idx}.xml",
                sheet_xml(sheets[name], sheet_name=name, drawing_rel=(name == "Dashboard")),
            )
            if name == "Dashboard":
                write_text(zf, f"xl/worksheets/_rels/sheet{idx}.xml.rels", sheet_drawing_rels_xml())
        write_text(zf, "xl/drawings/drawing1.xml", drawing_xml())
        write_text(zf, "xl/drawings/_rels/drawing1.xml.rels", drawing_rels_xml())
        write_text(zf, "xl/charts/chart1.xml", chart_xml(progress_row_count))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the significant-model Excel report.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output .xlsx path.")
    parser.add_argument(
        "--refresh-evidence",
        action="store_true",
        help="Refresh the committed evidence snapshot from local gitignored raw artifacts before building.",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    if args.refresh_evidence:
        refresh_evidence_manifest()
    sheets = build_workbook_data()
    write_xlsx(sheets, out_path)
    print(f"Wrote {repo_path(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
