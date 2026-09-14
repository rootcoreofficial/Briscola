"""Regression tests for the reproducible model-progress Excel report.

The generator writes XLSX XML directly, so these tests cover both the curated data
contract and the chart ranges. Normal builds must use only the committed evidence
manifest: many historical model weights and benchmark outputs are intentionally absent
from a clean clone.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def _load_build_model_report_module():
    """Load the report script as a module without making `scripts/` a package."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_model_report.py"
    spec = importlib.util.spec_from_file_location("build_model_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_model_report = _load_build_model_report_module()


def test_evidence_manifest_is_versioned_and_complete() -> None:
    """The committed snapshot must cover every curated model and normalized evidence table."""
    manifest = json.loads(build_model_report.EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == build_model_report.EVIDENCE_SCHEMA_VERSION
    assert set(manifest["models"]) == {spec.model_id for spec in build_model_report.MODEL_SPECS}
    assert manifest["promotion_evidence"]
    assert manifest["decision_quality"]
    assert all("raw_source" in row for row in manifest["promotion_evidence"])
    assert all("wrapper" in row and "guard" in row for row in manifest["decision_quality"])


def test_normal_build_does_not_read_gitignored_artifacts(monkeypatch) -> None:
    """A clean clone can build the workbook from the canonical snapshot alone."""

    def fail_local_read(*_args, **_kwargs):
        raise AssertionError("normal report build tried to read a gitignored raw artifact")

    monkeypatch.setattr(build_model_report, "load_npz_metadata", fail_local_read)
    monkeypatch.setattr(build_model_report, "load_json", fail_local_read)

    sheets = build_model_report.build_workbook_data()

    assert sheets["Dashboard"]
    assert sheets["Promotion Evidence"]
    assert sheets["Decision Quality"]


def test_summary_reference_scores_match_canonical_evidence() -> None:
    """Every exact summary number must occur in the raw source captured by the manifest."""
    manifest = json.loads(build_model_report.EVIDENCE_PATH.read_text(encoding="utf-8"))
    evidence = manifest["promotion_evidence"]

    for spec in build_model_report.MODEL_SPECS:
        for source, score in (
            (spec.progress_source, spec.progress_score),
            (spec.h2h_source, spec.h2h_score),
        ):
            if score is None:
                continue
            source_values = {row["avg_diff"] for row in evidence if row["raw_source"] == source}
            assert score in source_values, f"{spec.model_id}: {score} is not present in {source}"


def test_dashboard_chart_range_tracks_homogeneous_rows_only() -> None:
    """The chart grows with its comparable series and never restores the path-dependent H2H sum."""
    dashboard_rows = [
        ["Briscola AI - Model Progress Report"],
        ["Canonical evidence snapshot."],
        [],
        ["Homogeneous recent comparison"],
        ["Model", "Avg point diff", "Games", "Protocol", "Wrapper", "Guard"],
        ["best_a2c_v8", 17.61, 100_000, "same", "bc_model", True],
        ["best_a2c_v9", 18.78, 100_000, "same", "bc_model", True],
        ["best_a2c_v10", 20.52, 100_000, "same", "bc_model", True],
        ["best_a2c_v11", 20.80, 100_000, "same", "bc_model", False],
        ["best_a2c_v14", 21.76, 100_000, "same", "bc_model", False],
        ["best_a2c_v15", 21.87, 100_000, "same", "bc_model", False],
        [],
        ["Current conclusion"],
    ]

    count = build_model_report.dashboard_progress_row_count(dashboard_rows)
    chart = build_model_report.chart_xml(count)

    assert count == 6
    assert "Dashboard!$A$6:$A$11" in chart
    assert "Dashboard!$B$6:$B$11" in chart
    assert "cumul" not in chart.lower()
    assert "Dashboard!$E$" not in chart


def test_dashboard_uses_one_explicit_protocol_and_ends_with_v15() -> None:
    """The chart includes v15's matching big gate while leaving v13's medium gates separate."""
    sheets = build_model_report.build_workbook_data()
    dashboard = sheets["Dashboard"]
    count = build_model_report.dashboard_progress_row_count(dashboard)
    chart_rows = dashboard[5 : 5 + count]

    assert [row[0] for row in chart_rows] == list(build_model_report.HOMOGENEOUS_CHART_MODEL_IDS)
    assert all(row[2] == 100_000 for row in chart_rows)
    assert all("standard seeds 0..49,999" in row[3] for row in chart_rows)
    assert [row[5] for row in chart_rows] == [True, True, True, False, False, False]
    assert "best_a2c_v13" not in {row[0] for row in chart_rows}
    assert chart_rows[-1][0] == "best_a2c_v15"

    conclusion_index = dashboard.index(["Current conclusion"])
    conclusion = dashboard[conclusion_index + 1][0]
    assert "best_a2c_v15" in conclusion
    assert f"current v{build_model_report.project_version()}" in conclusion
    assert "last chart row" in conclusion
    assert "Policy-only v15 is +0.18" in conclusion
    assert "12x8 gate is +0.11" in conclusion


def test_v13_policy_and_pimc_evidence_are_separate() -> None:
    """The summary reference is policy-only; PIMC remains a separately labelled runtime gate."""
    model = next(row for row in build_model_report.model_rows() if row["model_id"] == "best_a2c_v13")
    evidence = [row for row in build_model_report.promotion_rows() if row["model_id"] == "best_a2c_v13"]

    assert model["reference_h2h"] == -0.0272
    assert "medium 10k" in model["reference_h2h_protocol"]
    assert "policy" in model["reference_h2h_protocol"]
    assert {row["wrapper"] for row in evidence} == {
        "bc_model (direct policy)",
        "bc_model_pimc_belief_16x8",
    }
    assert {row["engine"] for row in evidence} == {"numba", "domain"}


def test_v14_policy_and_pimc_evidence_are_separate() -> None:
    """v14's reference score is direct policy; the real runtime gate stays explicitly labelled."""
    model = next(row for row in build_model_report.model_rows() if row["model_id"] == "best_a2c_v14")
    evidence = [row for row in build_model_report.promotion_rows() if row["model_id"] == "best_a2c_v14"]

    assert model["reference_h2h"] == 0.6626
    assert "medium 10k" in model["reference_h2h_protocol"]
    assert "policy" in model["reference_h2h_protocol"]
    assert {row["wrapper"] for row in evidence} == {
        "bc_model (direct policy)",
        "bc_model_pimc_belief_16x8",
    }
    assert {row["engine"] for row in evidence} == {"numba", "domain"}


def test_v15_policy_and_runtime_evidence_are_separate() -> None:
    """v15's policy gain and the cheaper 12x8 runtime gate must remain distinct facts."""
    model = next(row for row in build_model_report.model_rows() if row["model_id"] == "best_a2c_v15")
    evidence = [row for row in build_model_report.promotion_rows() if row["model_id"] == "best_a2c_v15"]

    assert model["reference_h2h"] == 0.18046
    assert "100k" in model["reference_h2h_protocol"]
    assert "policy" in model["reference_h2h_protocol"]
    assert {row["wrapper"] for row in evidence} == {
        "bc_model (direct policy)",
        "bc_model_pimc_belief_12x8",
    }
    assert {row["engine"] for row in evidence} == {"numba", "domain", None}


def test_quality_rows_expose_wrapper_and_guard() -> None:
    """Historical zero-overkill rows must disclose that the runtime guard was active."""
    rows = {row["model_id"]: row for row in build_model_report.decision_quality_rows()}

    assert rows["best_a2c_v3"]["wrapper"] == "bc_model (direct policy)"
    assert rows["best_a2c_v3"]["guard"] is True
    assert rows["best_a2c_v11"]["guard"] is False
    assert rows["best_a2c_v13"]["guard"] is False
    assert rows["best_a2c_v14"]["guard"] is False
    assert rows["best_a2c_v15"]["guard"] is False


def test_recent_decisions_and_stable_sources_are_present() -> None:
    """The report records the v11/v12 decisions and important killed branches."""
    milestones = {(row["model_id"], row["type"]) for row in build_model_report.MILESTONES}
    rejected = {row["candidate"] for row in build_model_report.REJECTED_CANDIDATES}
    sources = build_model_report.sources_rows()

    assert ("best_a2c_v11", "promoted") in milestones
    assert ("best_a2c_v12", "rejected") in milestones
    assert ("best_a2c_v14", "promoted") in milestones
    assert ("best_a2c_v15", "promoted") in milestones
    assert {
        "pimc_distillation_v7",
        "exit_iter1b_belief_input",
        "pimc_teacher_v8_5m",
        "v11_guard_on",
        "best_a2c_v12",
    }.issubset(rejected)
    assert any(row["kind"] == "canonical_evidence" for row in sources)
    assert any(row["kind"] == "stable_narrative" for row in sources)


def test_generated_xlsx_chart_matches_homogeneous_series(tmp_path: Path) -> None:
    """The embedded chart references every and only comparable Dashboard row."""
    out_path = tmp_path / "model_progress.xlsx"
    sheets = build_model_report.build_workbook_data()
    build_model_report.write_xlsx(sheets, out_path)

    progress_count = build_model_report.dashboard_progress_row_count(sheets["Dashboard"])
    first_progress_row = 6
    last_progress_row = first_progress_row + progress_count - 1

    with zipfile.ZipFile(out_path) as zf:
        chart_root = ET.fromstring(zf.read("xl/charts/chart1.xml"))
    ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
    chart_ranges = {node.text for node in chart_root.findall(".//c:f", ns)}

    assert chart_ranges == {
        f"Dashboard!$A${first_progress_row}:$A${last_progress_row}",
        f"Dashboard!$B${first_progress_row}:$B${last_progress_row}",
    }


def test_xlsx_output_is_deterministic(tmp_path: Path) -> None:
    """Identical evidence produces byte-identical workbooks."""
    sheets = build_model_report.build_workbook_data()
    first = tmp_path / "first.xlsx"
    second = tmp_path / "second.xlsx"

    build_model_report.write_xlsx(sheets, first)
    build_model_report.write_xlsx(sheets, second)

    assert first.read_bytes() == second.read_bytes()
