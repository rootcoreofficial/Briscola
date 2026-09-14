"""
Smoke test per `scripts/style_feature_probe.py`.

È volutamente minuscolo (poche partite, modello sintetico): verifica che la pipeline giri
end-to-end e produca un JSON ben formato, NON i valori numerici della diagnosi (che dipendono
dal modello reale). Marcato `slow` perché lancia lo script in subprocess.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4


def _write_synthetic_v4_model(path: Path) -> None:
    """MLP BC sintetico con encoder v4 (feature_dim=369), pesi a zero: basta per lo smoke."""
    d, h = int(FEATURE_DIM_2P_V4), 8
    np.savez(
        path,
        w1=np.zeros((d, h), dtype=np.float32),
        b1=np.zeros((h,), dtype=np.float32),
        w2=np.zeros((h, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps({"format": "mlp_bc_v1", "feature_dim": d}, ensure_ascii=False),
    )


@pytest.mark.slow
def test_style_feature_probe_smoke(tmp_path: Path) -> None:
    model_path = tmp_path / "synthetic_v4.npz"
    _write_synthetic_v4_model(model_path)
    out_json = tmp_path / "probe.json"

    script = Path(__file__).resolve().parent.parent / "scripts" / "style_feature_probe.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "both",
            "--num-games",
            "4",
            "--opponents",
            "mirror,heuristic_trump_saver",
            "--model",
            str(model_path),
            "--seed",
            "0",
            "--out",
            str(out_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    # Metadati stabili richiesti dal contratto dell'output.
    assert payload["meta"]["feature_dim"] == int(FEATURE_DIM_2P_V4)
    assert payload["meta"]["seed"] == 0
    assert payload["meta"]["n_pool_states"] >= 1
    # Blocco controfattuale: chiavi presenti e CI ben formata (lista di 2 numeri).
    cf = payload["counterfactual"]["empirical_mean"]
    assert set(cf["p_by_bucket_mirror"]) == {"carico_nb", "carico_br", "liscio_nb", "briscola_bassa"}
    ci = cf["delta_p_carico_nb_ci95"]
    assert len(ci) == 2 and ci[0] <= ci[1]
    # Blocco temporale: percentuali argmax presenti.
    assert "argmax_live_pct" in payload["temporal"]
