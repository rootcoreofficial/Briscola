"""
Net2Net widening: la funzione della rete DEVE essere preservata esattamente a rumore zero.

È la proprietà che rende il widening un warm start legittimo (Fase 0.c: from-scratch
costa ~5 punti): se la rete allargata non partisse identica, l'iterazione-2 confonderebbe
"capacità extra" con "perdita di istinto".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_widen_module():
    """Carica lo script come modulo per testare `widen_net2net`."""
    spec = importlib.util.spec_from_file_location("widen_net2net_test", _ROOT / "scripts" / "widen_mlp_net2net.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


widen_mod = _load_widen_module()


def _forward(x, w1, b1, w2, b2):
    h = np.maximum(x @ w1 + b1, 0.0)
    return h @ w2 + b2


@pytest.mark.parametrize("new_hidden", [130, 256, 512])
def test_widening_preserves_function_exactly_without_noise(new_hidden: int) -> None:
    """A noise=0, logits e value head della rete allargata coincidono con l'originale."""
    rng = np.random.default_rng(1)
    d, h = 369, 128
    w1 = rng.normal(0, 0.1, size=(d, h)).astype(np.float32)
    b1 = rng.normal(0, 0.1, size=h).astype(np.float32)
    w2 = rng.normal(0, 0.1, size=(h, 40)).astype(np.float32)
    b2 = rng.normal(0, 0.1, size=40).astype(np.float32)
    wv = rng.normal(0, 0.1, size=h).astype(np.float32)

    widened = widen_mod.widen_net2net(
        w1=w1, b1=b1, w2=w2, b2=b2, new_hidden=new_hidden, rng=np.random.default_rng(2), noise=0.0, wv=wv
    )
    assert widened["w1"].shape == (d, new_hidden)
    assert widened["w2"].shape == (new_hidden, 40)

    for trial in range(20):
        x = np.random.default_rng(trial).normal(0, 1, size=d).astype(np.float32)
        original = _forward(x, w1, b1, w2, b2)
        wide = _forward(x, widened["w1"], widened["b1"], widened["w2"], widened["b2"])
        assert wide == pytest.approx(original, abs=1e-4)

        h_orig = np.maximum(x @ w1 + b1, 0.0)
        h_wide = np.maximum(x @ widened["w1"] + widened["b1"], 0.0)
        assert float(h_wide @ widened["wv"]) == pytest.approx(float(h_orig @ wv), abs=1e-4)


def test_widening_with_noise_stays_close_and_breaks_symmetry() -> None:
    """Col rumore la funzione resta vicina all'originale ma le copie divergono tra loro."""
    rng = np.random.default_rng(3)
    d, h = 369, 128
    w1 = rng.normal(0, 0.1, size=(d, h)).astype(np.float32)
    b1 = np.zeros(h, dtype=np.float32)
    w2 = rng.normal(0, 0.1, size=(h, 40)).astype(np.float32)
    b2 = np.zeros(40, dtype=np.float32)

    widened = widen_mod.widen_net2net(
        w1=w1, b1=b1, w2=w2, b2=b2, new_hidden=256, rng=np.random.default_rng(4), noise=1e-3
    )
    x = np.random.default_rng(9).normal(0, 1, size=d).astype(np.float32)
    delta = np.abs(
        _forward(x, widened["w1"], widened["b1"], widened["w2"], widened["b2"]) - _forward(x, w1, b1, w2, b2)
    )
    assert float(delta.max()) < 0.05  # vicino all'originale
    # Le colonne copia non sono identiche alle sorgenti (simmetria rotta).
    assert not np.allclose(widened["w1"][:, h:], w1[:, :1].repeat(256 - h, axis=1))


def test_widening_rejects_shrinking() -> None:
    """Restringere non è widening: errore esplicito."""
    w1 = np.zeros((10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="new_hidden"):
        widen_mod.widen_net2net(
            w1=w1,
            b1=np.zeros(8, dtype=np.float32),
            w2=np.zeros((8, 40), dtype=np.float32),
            b2=np.zeros(40, dtype=np.float32),
            new_hidden=4,
            rng=np.random.default_rng(0),
        )
