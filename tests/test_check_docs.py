"""Test del quality gate ermetico per documentazione ed evidenze."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _load_check_docs_module():
    """Carica lo script senza trasformare `scripts/` in un package applicativo."""
    module_path = Path(__file__).resolve().parents[1] / "scripts/check_docs.py"
    spec = importlib.util.spec_from_file_location("check_docs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_docs = _load_check_docs_module()


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(tmp_path: Path) -> None:
    """Il parser non deve accettare estensioni che altri consumer JSON potrebbero interpretare diversamente."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "value": 1, "value": 2}\n', encoding="utf-8")
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"schema": 1, "value": NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicata"):
        check_docs.load_strict_json(duplicate)
    with pytest.raises(ValueError, match="costante non JSON"):
        check_docs.load_strict_json(nonfinite)


def test_local_links_ignore_code_and_external_urls_but_check_repository_urls(tmp_path: Path) -> None:
    """Solo riferimenti verificabili offline devono influire sul gate."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "present.md").write_text("# Presente\n", encoding="utf-8")
    (tmp_path / "PLAN.md").write_text("# Piano\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "[locale](docs/present.md)",
                "[interno GitHub](https://github.com/ilCapo77/briscola.ai/blob/master/PLAN.md)",
                "[esterno](https://example.com/non-controllato)",
                "```markdown",
                "[esempio non reale](docs/assente-nel-codice.md)",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    outcome = check_docs.check_local_links(tmp_path)

    assert outcome.errors == ()
    assert outcome.detail.startswith("2 link verificati")

    (tmp_path / "README.md").write_text("[rotto](docs/missing.md)\n", encoding="utf-8")
    broken = check_docs.check_local_links(tmp_path)
    assert len(broken.errors) == 1
    assert "docs/missing.md" in broken.errors[0]


def _write_consistent_release_fixture(root: Path) -> None:
    """Crea il contratto minimo letto dal controllo cross-file."""
    (root / "pyproject.toml").write_text('[project]\nname = "briscola-ai"\nversion = "0.36.0"\n', encoding="utf-8")
    (root / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "briscola-ai"\nversion = "0.36.0"\n',
        encoding="utf-8",
    )
    provisioning = root / "src/briscola_ai/ai/models/provisioning.py"
    provisioning.parent.mkdir(parents=True)
    provisioning.write_text('DEFAULT_MODEL_ID = "best_a2c_v14.npz"\n', encoding="utf-8")
    model = root / "data/models/best_a2c_v14.npz"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    (root / "README.md").write_text(
        "La release corrente del repository è `0.36.0`\nLa policy ufficiale è **`data/models/best_a2c_v14.npz`**\n",
        encoding="utf-8",
    )
    (root / "PLAN.md").write_text(
        "Release corrente del repository: `0.36.0`\nIl default è su `best_a2c_v14.npz`\n",
        encoding="utf-8",
    )


def test_release_consistency_catches_lockfile_and_documentation_drift(tmp_path: Path) -> None:
    """Il gate deve coprire i due errori di release già avvenuti nel progetto."""
    _write_consistent_release_fixture(tmp_path)
    assert check_docs.check_release_consistency(tmp_path).errors == ()

    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "briscola-ai"\nversion = "0.35.1"\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("documentazione vecchia\n", encoding="utf-8")
    outcome = check_docs.check_release_consistency(tmp_path)

    assert any("0.35.1" in error and "0.36.0" in error for error in outcome.errors)
    assert any("README.md" in error for error in outcome.errors)


def test_diary_chapters_accept_markdown_and_raw_h2_but_reject_gaps(tmp_path: Path) -> None:
    """Il capitolo 2 usa volutamente HTML per l'anchor storico e deve essere contato."""
    diary = tmp_path / "src/briscola_ai/frontend/static/diario.md"
    diary.parent.mkdir(parents=True)
    diary.write_text(
        '## Capitolo 1 — Uno\n\n<h2 id="due">Capitolo 2 — Due</h2>\n\n## Capitolo 3 — Tre\n',
        encoding="utf-8",
    )
    assert check_docs.check_diary_chapters(tmp_path).errors == ()

    diary.write_text("## Capitolo 1 — Uno\n\n## Capitolo 3 — Tre\n", encoding="utf-8")
    outcome = check_docs.check_diary_chapters(tmp_path)
    assert outcome.errors
    assert "attesa [1, 2, 3]" in outcome.errors[0]


def test_model_report_check_detects_stale_bytes_without_touching_repository(tmp_path: Path) -> None:
    """Il confronto usa una destinazione temporanea e segnala un workbook non rigenerato."""
    report = tmp_path / "docs/reports/model_progress.xlsx"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"current")

    def matching_builder(output_path: Path) -> None:
        output_path.write_bytes(b"current")

    def stale_builder(output_path: Path) -> None:
        output_path.write_bytes(b"new")

    assert check_docs.check_model_report(tmp_path, builder=matching_builder).errors == ()
    stale = check_docs.check_model_report(tmp_path, builder=stale_builder)
    assert stale.errors
    assert "non è aggiornato" in stale.errors[0]
    assert report.read_bytes() == b"current"


def test_repository_docs_check_passes_end_to_end() -> None:
    """Il comando pubblico deve attraversare il checkout reale senza accedere alla rete."""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/check_docs.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "[OK]   report modelli" in completed.stdout
