"""
Test del provisioning modello allo startup (`ensure_model_available`).

Usiamo URL `file://` per evitare dipendenze di rete: la logica di download/verifica/scrittura
atomica è la stessa di un URL http(s).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from briscola_ai.ai.models.provisioning import ensure_model_available
from briscola_ai.main import _provision_startup_models


def _make_source(tmp_path: Path, content: bytes = b"fake-model-bytes") -> tuple[Path, str]:
    src = tmp_path / "source.npz"
    src.write_bytes(content)
    return src, src.as_uri()


def test_returns_true_if_already_present(tmp_path: Path) -> None:
    """Se il modello esiste già localmente (e nessuno sha da verificare), deve ritornare True
    senza scaricare ("già presente")."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "best_a2c_v3.npz").write_bytes(b"x")

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="best_a2c_v3.npz", url=None)
    assert ok is True
    assert "già presente" in msg


def test_returns_false_if_missing_and_no_url(tmp_path: Path) -> None:
    """Se il modello manca e non c'è alcun URL da cui scaricarlo, deve ritornare False
    con messaggio esplicativo (nessun BRISCOLA_MODEL_URL)."""
    models_dir = tmp_path / "models"
    ok, msg = ensure_model_available(models_dir=models_dir, model_id="best_a2c_v3.npz", url=None)
    assert ok is False
    assert "nessun BRISCOLA_MODEL_URL" in msg


def test_downloads_when_missing(tmp_path: Path) -> None:
    """Se il modello manca ma c'è un URL, deve creare la dir di destinazione, scaricare
    e scrivere il file con il contenuto sorgente identico."""
    src, url = _make_source(tmp_path)
    models_dir = tmp_path / "models"  # non esiste ancora: deve crearla

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="best_a2c_v3.npz", url=url)

    assert ok is True
    target = models_dir / "best_a2c_v3.npz"
    assert target.exists()
    assert target.read_bytes() == src.read_bytes()


def test_sha256_match_installs(tmp_path: Path) -> None:
    """Quando lo sha256 atteso coincide con quello scaricato, il file deve essere installato (download accettato)."""
    content = b"model-with-hash"
    src, url = _make_source(tmp_path, content)
    digest = hashlib.sha256(content).hexdigest()
    models_dir = tmp_path / "models"

    ok, _ = ensure_model_available(models_dir=models_dir, model_id="m.npz", url=url, sha256=digest)
    assert ok is True
    assert (models_dir / "m.npz").exists()


def test_sha256_mismatch_does_not_install(tmp_path: Path) -> None:
    """Se lo sha256 scaricato non corrisponde al pin atteso, deve fallire e NON lasciare
    alcun file parziale/sbagliato a destinazione."""
    src, url = _make_source(tmp_path, b"good-bytes")
    models_dir = tmp_path / "models"

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="m.npz", url=url, sha256="deadbeef")
    assert ok is False
    assert "sha256 non corrispondente" in msg
    assert not (models_dir / "m.npz").exists()  # niente file parziale/sbagliato


def test_download_failure_is_non_fatal(tmp_path: Path) -> None:
    """Un download fallito (URL inesistente) non deve essere fatale: ritorna False
    con messaggio e non lascia file a destinazione."""
    models_dir = tmp_path / "models"
    ok, msg = ensure_model_available(
        models_dir=models_dir,
        model_id="m.npz",
        url=(tmp_path / "does_not_exist.npz").as_uri(),
    )
    assert ok is False
    assert "download fallito" in msg
    assert not (models_dir / "m.npz").exists()


def test_existing_file_with_matching_sha_is_verified(tmp_path: Path) -> None:
    """Se il file locale esiste e il suo sha256 coincide con il pin atteso, deve essere accettato
    senza riscaricare ("verificato")."""
    content = b"pinned-model"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.npz").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="m.npz", url=None, sha256=digest)
    assert ok is True
    assert "verificato" in msg


def test_existing_file_sha_mismatch_redownloads_when_url(tmp_path: Path) -> None:
    """Se il file locale ha sha diverso e c'e' un URL, viene riscaricato (sha = pin di versione)."""
    new_content = b"new-version-bytes"
    src, url = _make_source(tmp_path, new_content)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.npz").write_bytes(b"old-stale-bytes")
    digest = hashlib.sha256(new_content).hexdigest()

    ok, _ = ensure_model_available(models_dir=models_dir, model_id="m.npz", url=url, sha256=digest)
    assert ok is True
    assert (models_dir / "m.npz").read_bytes() == new_content


def test_existing_file_sha_mismatch_no_url_fails(tmp_path: Path) -> None:
    """Se il file locale ha sha diverso dal pin e non c'è URL per riscaricarlo, deve fallire
    ("non corrisponde") invece di accettare un modello stale."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.npz").write_bytes(b"stale")

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="m.npz", url=None, sha256="abc123")
    assert ok is False
    assert "non corrisponde" in msg


def test_disallowed_url_scheme_is_rejected(tmp_path: Path) -> None:
    """Uno schema URL non ammesso (es. ftp://) deve essere rifiutato per sicurezza, senza tentare il download."""
    models_dir = tmp_path / "models"
    ok, msg = ensure_model_available(models_dir=models_dir, model_id="m.npz", url="ftp://example.com/m.npz")
    assert ok is False
    assert "schema URL non ammesso" in msg


def test_download_passes_timeout_and_accepts_https(monkeypatch, tmp_path: Path) -> None:
    """Il download passa il `timeout` a urlopen; lo schema https è ammesso (senza rete via monkeypatch)."""
    import urllib.request

    captured: dict[str, object] = {}

    class _FakeResp:
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"downloaded-bytes"

    def _fake_urlopen(url: object, timeout: object = None):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    models_dir = tmp_path / "models"
    ok, _ = ensure_model_available(
        models_dir=models_dir,
        model_id="m.npz",
        url="https://example.com/best_a2c_v3.npz",
        timeout=12.5,
    )
    assert ok is True
    assert captured["timeout"] == 12.5
    assert (models_dir / "m.npz").read_bytes() == b"downloaded-bytes"


def test_startup_provisioning_downloads_policy_and_value_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Lo startup deve poter scaricare sia la policy consigliata sia il value model del lookahead."""
    policy_bytes = b"policy-v7"
    value_bytes = b"value-lookahead"
    policy_src = tmp_path / "policy-source.npz"
    value_src = tmp_path / "value-source.npz"
    policy_src.write_bytes(policy_bytes)
    value_src.write_bytes(value_bytes)

    models_dir = tmp_path / "models"
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("BRISCOLA_MODEL_URL", policy_src.as_uri())
    monkeypatch.setenv("BRISCOLA_MODEL_SHA256", hashlib.sha256(policy_bytes).hexdigest())
    monkeypatch.setenv("BRISCOLA_VALUE_MODEL_URL", value_src.as_uri())
    monkeypatch.setenv("BRISCOLA_VALUE_MODEL_SHA256", hashlib.sha256(value_bytes).hexdigest())

    messages = _provision_startup_models()

    assert (models_dir / "best_a2c_v15.npz").read_bytes() == policy_bytes
    assert (models_dir / "value_v0_h128_clean50k_seed20260701.npz").read_bytes() == value_bytes
    assert any(msg.startswith("Model provisioning:") for msg in messages)
    assert any(msg.startswith("Value model provisioning:") for msg in messages)
