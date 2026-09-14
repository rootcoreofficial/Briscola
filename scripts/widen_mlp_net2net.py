#!/usr/bin/env python3
"""
Allarga l'hidden layer di una MLP `.npz` preservandone ESATTAMENTE la funzione (Net2Net).

Perché (iterazione 2 del piano belief/expert-iteration):
la Fase 0.c mostra che ripartire da zero costa ~5 punti: la capacità va aggiunta SENZA
buttare l'istinto accumulato. Con il widening Net2WiderNet (Chen, Goodfellow, Shlens 2015)
ogni neurone nuovo è la copia di uno esistente e i pesi USCENTI dell'originale vengono
divisi tra le copie: a rumore zero, la rete allargata calcola esattamente la stessa
funzione di quella di partenza (ReLU inclusa, perché copie identiche → attivazioni
identiche → somma pesata invariata).

Il rumore opzionale (`--noise`) rompe la simmetria tra le copie: senza, i gradienti delle
copie resterebbero identici e la capacità extra non verrebbe mai usata.

Supporta sia il formato A2C (`w1,b1,w2,b2,wv,bv`) sia il formato BC (senza critic head).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from briscola_ai.versioning import get_code_version


def widen_net2net(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    new_hidden: int,
    rng: np.random.Generator,
    noise: float = 0.0,
    wv: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Ritorna i pesi allargati a `new_hidden` neuroni (funzione preservata a noise=0).

    Schema: per ogni slot nuovo j >= H si sceglie una sorgente m(j) uniforme in [0,H);
    `count[i]` = quante volte il neurone i compare in totale (originale incluso).
    - entranti: `w1[:, j] = w1[:, m(j)] (+ rumore)`, `b1[j] = b1[m(j)]`;
    - uscenti: OGNI copia di i (originale incluso) riceve `w2[i, :] / count[i]`
      (idem per la critic head `wv`), così la somma resta invariata.
    """
    hidden = int(w1.shape[1])
    if new_hidden <= hidden:
        raise ValueError(f"new_hidden={new_hidden} deve essere > hidden attuale ({hidden})")

    mapping = np.concatenate([np.arange(hidden), rng.integers(0, hidden, size=new_hidden - hidden)])
    counts = np.bincount(mapping, minlength=hidden).astype(np.float32)

    new_w1 = w1[:, mapping].astype(np.float32)
    new_b1 = b1[mapping].astype(np.float32)
    if noise > 0.0:
        # Rumore SOLO sulle copie (slot >= hidden): gli originali restano intatti, quindi
        # la deviazione dalla funzione originale è limitata dalle sole copie perturbate.
        new_w1[:, hidden:] += rng.normal(0.0, noise, size=new_w1[:, hidden:].shape).astype(np.float32)

    scale = (1.0 / counts)[mapping]  # peso uscente di ogni slot = originale / molteplicità
    new_w2 = (w2[mapping, :] * scale[:, None]).astype(np.float32)

    out: dict[str, np.ndarray] = {
        "w1": new_w1,
        "b1": new_b1,
        "w2": new_w2,
        "b2": b2.astype(np.float32),
    }
    if wv is not None:
        out["wv"] = (wv[mapping] * scale).astype(np.float32)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Widening Net2Net di una MLP .npz (funzione preservata)")
    parser.add_argument("--model", required=True, help="Modello .npz di partenza (formato A2C o BC)")
    parser.add_argument("--out", required=True, help="Path del modello allargato")
    parser.add_argument("--new-hidden", type=int, required=True, help="Nuova dimensione hidden (> attuale)")
    parser.add_argument("--noise", type=float, default=1e-3, help="Std del rumore sulle copie (default 1e-3)")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG per mapping/rumore (riproducibilita')")
    args = parser.parse_args()

    with np.load(args.model) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files if k != "metadata_json"}
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}

    rng = np.random.default_rng(args.seed)
    widened = widen_net2net(
        w1=arrays["w1"],
        b1=arrays["b1"],
        w2=arrays["w2"],
        b2=arrays["b2"],
        new_hidden=int(args.new_hidden),
        rng=rng,
        noise=float(args.noise),
        wv=arrays.get("wv"),
    )
    # Chiavi non toccate dal widening (bv, belief_* embedded, ecc.) passano invariate.
    passthrough = {k: v for k, v in arrays.items() if k not in widened}

    metadata = dict(metadata)
    metadata["hidden_dim"] = int(args.new_hidden)
    metadata["net2net"] = {
        "source": str(args.model),
        "source_hidden": int(arrays["w1"].shape[1]),
        "noise": float(args.noise),
        "seed": int(args.seed),
        "code_version": get_code_version(),
    }
    label = metadata.get("label")
    metadata["label"] = f"{label} [net2net {args.new_hidden}]" if label else f"net2net {args.new_hidden}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **widened, **passthrough, metadata_json=json.dumps(metadata))
    print(
        f"Salvato {out_path} | hidden {arrays['w1'].shape[1]} -> {args.new_hidden} "
        f"| noise={args.noise} seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
