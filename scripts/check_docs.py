#!/usr/bin/env python3
"""Quality gate ermetico per documentazione, evidenze e report versionati.

Il controllo non accede alla rete. I link relativi e i link GitHub che puntano a file
dello stesso repository vengono risolti sul checkout locale; gli altri URL esterni sono
ignorati per non rendere la CI dipendente dalla disponibilità di servizi terzi.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import markdown

REPOSITORY_WEB_PREFIXES = (
    "https://github.com/ilCapo77/briscola.ai/blob/master/",
    "https://github.com/ilCapo77/briscola.ai/raw/master/",
    "https://raw.githubusercontent.com/ilCapo77/briscola.ai/master/",
)


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """Esito leggibile di un singolo gruppo di controlli."""

    name: str
    detail: str
    errors: tuple[str, ...] = ()


class ReportBuilder(Protocol):
    """Callback iniettabile usata dai test del controllo di freschezza."""

    def __call__(self, output_path: Path) -> None: ...


class _LinkHTMLParser(HTMLParser):
    """Estrae href dal documento HTML prodotto dal parser Markdown."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


def _markdown_files(root: Path) -> list[Path]:
    """Elenca documenti mantenuti nel repository, includendo file nuovi non ancora staged."""
    candidates = set(root.glob("*.md"))
    for directory in ("docs", "seed_suites", "src", "scripts", "tests", ".github"):
        base = root / directory
        if base.is_dir():
            candidates.update(base.rglob("*.md"))
    return sorted(path for path in candidates if path.is_file())


def _extract_markdown_links(path: Path) -> list[str]:
    """Renderizza Markdown e raccoglie link reali, escludendo esempi nei code fence."""
    rendered = markdown.markdown(path.read_text(encoding="utf-8"), extensions=["fenced_code"])
    parser = _LinkHTMLParser()
    parser.feed(rendered)
    parser.close()
    return parser.hrefs


def _repository_path_from_web_url(target: str) -> str | None:
    """Mappa i link blob/raw del repository in path locali verificabili."""
    for prefix in REPOSITORY_WEB_PREFIXES:
        if target.startswith(prefix):
            return unquote(target.removeprefix(prefix).split("#", 1)[0].split("?", 1)[0])
    return None


def _resolve_local_link(root: Path, source: Path, target: str) -> tuple[Path | None, str | None]:
    """Ritorna il path locale di un href o ``None`` per URL/anchor non verificabili offline."""
    repository_path = _repository_path_from_web_url(target)
    if repository_path is not None:
        candidate = root / repository_path
    else:
        try:
            parsed = urlsplit(target)
        except ValueError as exc:
            return None, f"href non valido {target!r}: {exc}"
        if parsed.scheme or parsed.netloc or target.startswith("//") or target.startswith("/"):
            return None, None
        relative_path = unquote(parsed.path)
        if not relative_path:
            return None, None
        candidate = source.parent / relative_path

    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None, f"link locale fuori dal repository: {target!r}"
    return resolved, None


def check_local_links(root: Path) -> CheckOutcome:
    """Verifica esistenza e contenimento di link Markdown locali e blob/raw GitHub interni."""
    documents = _markdown_files(root)
    checked = 0
    errors: list[str] = []
    for source in documents:
        for target in _extract_markdown_links(source):
            resolved, error = _resolve_local_link(root, source, target)
            if error is not None:
                errors.append(f"{source.relative_to(root)}: {error}")
                continue
            if resolved is None:
                continue
            checked += 1
            if not resolved.exists():
                errors.append(f"{source.relative_to(root)}: link inesistente {target!r}")
    return CheckOutcome(
        name="link locali",
        detail=f"{checked} link verificati in {len(documents)} file Markdown",
        errors=tuple(sorted(set(errors))),
    )


def _reject_json_constant(value: str) -> Any:
    """Rifiuta NaN/Infinity, accettati dal decoder standard ma non da JSON rigoroso."""
    raise ValueError(f"costante non JSON: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Costruisce un oggetto fallendo su chiavi duplicate silenziosamente sovrascritte."""
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"chiave duplicata: {key!r}")
        output[key] = value
    return output


def load_strict_json(path: Path) -> Any:
    """Carica JSON RFC-compatible, senza costanti non finite né chiavi duplicate."""
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_object_without_duplicate_keys,
    )


def check_evidence_json(root: Path) -> CheckOutcome:
    """Valida sintassi rigorosa, root object e identificatore di schema delle evidenze."""
    evidence_dir = root / "docs/reports/evidence"
    paths = sorted(evidence_dir.glob("*.json")) if evidence_dir.is_dir() else []
    errors: list[str] = []
    if not paths:
        errors.append("docs/reports/evidence non contiene file JSON")
    for path in paths:
        try:
            payload = load_strict_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path.relative_to(root)}: JSON non valido ({exc})")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path.relative_to(root)}: la root deve essere un oggetto")
            continue
        schema = payload.get("schema", payload.get("schema_version"))
        if not isinstance(schema, str | int) or schema == "":
            errors.append(f"{path.relative_to(root)}: manca schema/schema_version")
    return CheckOutcome(
        name="evidenze JSON",
        detail=f"{len(paths)} file validati in modalità rigorosa",
        errors=tuple(errors),
    )


def _project_version(root: Path) -> str:
    """Legge la versione canonica da pyproject.toml."""
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = payload.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml non contiene project.version")
    return version


def _locked_project_version(root: Path) -> str:
    """Trova la versione del package editable `briscola-ai` nel lockfile uv."""
    payload = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    packages = payload.get("package", [])
    matches = [package for package in packages if package.get("name") == "briscola-ai"]
    if len(matches) != 1 or not isinstance(matches[0].get("version"), str):
        raise ValueError("uv.lock deve contenere esattamente un package briscola-ai versionato")
    return str(matches[0]["version"])


def _default_model_id(root: Path) -> str:
    """Estrae la costante letterale senza importare backend o leggere variabili d'ambiente."""
    path = root / "src/briscola_ai/ai/models/provisioning.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "DEFAULT_MODEL_ID":
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise ValueError("DEFAULT_MODEL_ID letterale non trovato in provisioning.py")


def check_release_consistency(root: Path) -> CheckOutcome:
    """Allinea versione e modello corrente fra sorgenti operative e documentazione principale."""
    errors: list[str] = []
    try:
        version = _project_version(root)
        locked_version = _locked_project_version(root)
        model_id = _default_model_id(root)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, SyntaxError, ValueError) as exc:
        return CheckOutcome("coerenza release", "lettura configurazione fallita", (str(exc),))

    if locked_version != version:
        errors.append(f"uv.lock contiene briscola-ai {locked_version}, pyproject.toml dichiara {version}")
    model_path = root / "data/models" / model_id
    if not model_path.is_file():
        errors.append(f"modello predefinito assente: {model_path.relative_to(root)}")

    expected_fragments = {
        root / "README.md": (
            f"La release corrente del repository è `{version}`",
            f"La policy ufficiale è **`data/models/{model_id}`**",
        ),
        root / "PLAN.md": (
            f"Release corrente del repository: `{version}`",
            f"su `{model_id}`",
        ),
    }
    for path, fragments in expected_fragments.items():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)} non leggibile: {exc}")
            continue
        for fragment in fragments:
            if fragment not in text:
                errors.append(f"{path.relative_to(root)} non contiene il riferimento corrente {fragment!r}")

    return CheckOutcome(
        name="coerenza release",
        detail=f"versione {version}, modello {model_id}",
        errors=tuple(errors),
    )


def check_diary_chapters(root: Path) -> CheckOutcome:
    """Verifica che i capitoli numerati del diario pubblico siano un intervallo continuo da 1."""
    path = root / "src/briscola_ai/frontend/static/diario.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return CheckOutcome("capitoli diario", "file non leggibile", (str(exc),))
    chapter_numbers = [
        int(match.group(1))
        for match in re.finditer(r"(?:^##\s+|<h2\b[^>]*>)Capitolo\s+(\d+)\b", text, flags=re.MULTILINE)
    ]
    expected = list(range(1, max(chapter_numbers, default=0) + 1))
    errors: list[str] = []
    if chapter_numbers != expected:
        errors.append(f"sequenza capitoli {chapter_numbers!r}, attesa {expected!r}")
    return CheckOutcome(
        name="capitoli diario",
        detail=f"sequenza continua 1..{chapter_numbers[-1] if chapter_numbers else 0}",
        errors=tuple(errors),
    )


def _default_report_builder(root: Path) -> ReportBuilder:
    """Costruisce il callback che attraversa la CLI reale del generatore Excel."""

    def build(output_path: Path) -> None:
        completed = subprocess.run(
            [sys.executable, str(root / "scripts/build_model_report.py"), "--out", str(output_path)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
            raise RuntimeError(detail)

    return build


def check_model_report(root: Path, *, builder: ReportBuilder | None = None) -> CheckOutcome:
    """Rigenera l'XLSX fuori dal repository e richiede uguaglianza byte-per-byte."""
    committed = root / "docs/reports/model_progress.xlsx"
    errors: list[str] = []
    if not committed.is_file():
        return CheckOutcome("report modelli", "file assente", ("docs/reports/model_progress.xlsx non esiste",))
    selected_builder = builder or _default_report_builder(root)
    with tempfile.TemporaryDirectory(prefix="briscola-docs-check-") as tmp_dir:
        generated = Path(tmp_dir) / "model_progress.xlsx"
        try:
            selected_builder(generated)
        except Exception as exc:  # il confine subprocess/callback deve diventare un errore leggibile del gate
            errors.append(f"rigenerazione fallita: {exc}")
        else:
            if not generated.is_file():
                errors.append("il generatore non ha prodotto model_progress.xlsx")
            elif committed.read_bytes() != generated.read_bytes():
                errors.append(
                    "docs/reports/model_progress.xlsx non è aggiornato; eseguire "
                    "`uv run python scripts/build_model_report.py`"
                )
    return CheckOutcome(
        name="report modelli",
        detail="rigenerazione deterministica uguale al file versionato",
        errors=tuple(errors),
    )


def run_checks(root: Path, *, report_builder: ReportBuilder | None = None) -> list[CheckOutcome]:
    """Esegue tutti i controlli in ordine economico, senza fermarsi al primo errore."""
    return [
        check_evidence_json(root),
        check_local_links(root),
        check_release_consistency(root),
        check_diary_chapters(root),
        check_model_report(root, builder=report_builder),
    ]


def _parse_args() -> argparse.Namespace:
    """Definisce la CLI, con root esplicita utile anche a checkout temporanei."""
    parser = argparse.ArgumentParser(description="Controlla documentazione ed evidenze senza accedere alla rete")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    """Stampa un riepilogo compatto e ritorna non-zero se almeno un controllo fallisce."""
    args = _parse_args()
    root = args.root.resolve()
    outcomes = run_checks(root)
    failed = False
    for outcome in outcomes:
        if outcome.errors:
            failed = True
            print(f"[FAIL] {outcome.name}: {outcome.detail}")
            for error in outcome.errors:
                print(f"  - {error}")
        else:
            print(f"[OK]   {outcome.name}: {outcome.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
