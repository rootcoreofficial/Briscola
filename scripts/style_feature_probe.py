"""
Sonda diagnostica: la policy usa le feature di STILE dell'encoder v4?

Strumento di laboratorio riproducibile (non una macchina per promuovere modelli): misura
QUANTO e in QUALE VERSO una policy BC/A2C (encoder v4) condiziona la mossa sulle feature di
stile dell'avversario, e la distribuzione temporale degli stati in cui la scelta conta.

Nasce dall'analisi 2026-07-08 (nota `docs/plans/sonda-stile-finestra-2026-07-08.md`):
serviva ricomputabilità dei numeri R2/R3 citati in PLAN.md, non solo scratchpad.

Metodo
------
Raccoglie stati REALI da self-play del dominio (v11 nel seggio IA vs una lista di avversari),
filtrati a "carico-non-briscola vs liscia" DA LEADER: tavolo vuoto, in mano >=1 carico
non-briscola (asso/tre fuori briscola) E >=1 liscia non-briscola -> vera scelta "espongo il
carico o no".

- mode=counterfactual: congela lo stato, sovrascrive l'intero blocco encoder v4
  (`V4_EXTRA_SLICE`) con un PROFILO EMPIRICO (media/mediana del blocco osservato contro
  ciascun avversario: valori che co-occorrono davvero, non inventati), e misura lo
  spostamento della massa softmax per bucket (carico_nb / carico_br / liscio_nb /
  briscola_bassa). Bootstrap CI 95% sul mean ΔP_carico_nb = P(saver) - P(mirror).
  Controllo positivo: swap del blocco fase (inizio<->finale) -> DEVE muoversi.
- mode=temporal: distribuzione di trick_index / deck_size / contatori di stile, separata
  per categoria dell'argmax live.

Caveat (dichiarato nell'output): il controfattuale congela il resto dello stato e scambia un
blocco che nella realta' co-varia con esso -> indizio empirico, non prova causale pulita.

Esempi
------
    uv run python scripts/style_feature_probe.py --mode both --num-games 500 \\
        --model data/models/best_a2c_v11.npz \\
        --out benchmarks/experiments/style_feature_probe/v11_both.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents.registry import build_agent
from briscola_ai.ai.encoding.observation_encoder import (
    V3_IDX_DECK_SIZE,
    V3_IDX_HAND_SIZE,
    V3_IDX_IS_ENDGAME,
    V4_EXTRA_SLICE,
    V4_IDX_OPP_LEAD_LOAD,
    V4_IDX_OPP_TRUMP_RESPONSE,
    encode_player_observation_2p,
)
from briscola_ai.domain.card_id import card_to_id, id_to_card
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state

# Punti per action id (Asso=11, Tre=10, ...), da API pubblica card_id/models.
POINTS_BY_ACTION_ID = [int(id_to_card(a).rank.points) for a in range(40)]
BUCKETS = ("carico_nb", "carico_br", "liscio_nb", "briscola_bassa")
# Profili di fase per il controllo positivo (inizio vs finale).
PHASE_EARLY = {V3_IDX_DECK_SIZE: 0.85, V3_IDX_HAND_SIZE: 1.0, V3_IDX_IS_ENDGAME: 0.0}
PHASE_ENDGAME = {V3_IDX_DECK_SIZE: 0.0, V3_IDX_HAND_SIZE: 0.33, V3_IDX_IS_ENDGAME: 1.0}


def category(action_id: int, trump_idx: int) -> str:
    """Categoria della carta-azione: carico/liscia x briscola/non-briscola."""
    is_trump = (action_id // 10) == trump_idx
    carico = POINTS_BY_ACTION_ID[action_id] >= 10
    if carico and not is_trump:
        return "carico_nb"
    if carico and is_trump:
        return "carico_br"
    if is_trump:
        return "briscola_bassa"
    return "liscio_nb"


def softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64).copy()
    z[mask <= 0] = -1e18
    z -= z.max()
    e = np.exp(z)
    e[mask <= 0] = 0.0
    total = e.sum()
    return e / total if total > 0 else e


def collect_states(ai: Any, model: Any, opp_name: str, model_path: str, n_games: int, seed: int) -> list[dict]:
    """Self-play del dominio; raccoglie stati carico_nb-vs-liscio da leader nel seggio IA."""
    opp = ai if opp_name == "mirror" else build_agent(opp_name, model_path=model_path)
    # crc32 (deterministico tra processi, a differenza di hash()) per un seed stabile per avversario.
    rng = random.Random(seed ^ zlib.crc32(opp_name.encode()))
    states: list[dict] = []
    for g in range(n_games):
        ai_seat = g % 2  # alterno il posto per diversificare
        agents = {ai_seat: ai, 1 - ai_seat: opp}
        state = new_game_state(num_players=2, seed=seed * 1_000_003 + g)
        safety = 5000
        while not state.game_over and safety > 0:
            safety -= 1
            cur = state.current_turn
            obs = make_player_observation(state, cur)
            if cur == ai_seat and len(obs.table_cards) == 0 and obs.trump_card is not None:
                trump_idx = card_to_id(obs.trump_card) // 10
                enc = encode_player_observation_2p(obs, version="v4")
                mask = np.asarray(enc.action_mask, dtype=np.float64)
                feats = np.asarray(enc.features, dtype=np.float64)
                legal = [a for a in range(40) if mask[a] > 0]
                cats = [category(a, trump_idx) for a in legal]
                if "carico_nb" in cats and "liscio_nb" in cats and len(legal) >= 2:
                    z = model.logits(feats.astype(np.float32)).astype(np.float64)
                    z[mask <= 0] = -1e18
                    states.append(
                        {
                            "f": feats,
                            "mask": mask,
                            "trump": trump_idx,
                            "trick": len(obs.trick_history),
                            "deck": int(obs.deck_size),
                            "cuts": float(feats[V4_IDX_OPP_TRUMP_RESPONSE]),
                            "load": float(feats[V4_IDX_OPP_LEAD_LOAD]),
                            "amax": category(int(np.argmax(z)), trump_idx),
                        }
                    )
            idx = agents[cur].choose_card_index(obs, rng=rng)
            state, res = step(state, PlayCardAction(player_index=cur, card_index=idx))
            if res.error:
                raise RuntimeError(f"Errore dominio nella simulazione: {res.error}")
    return states


def bucket_probs(model: Any, feats: np.ndarray, mask: np.ndarray, trump_idx: int) -> dict[str, float]:
    p = softmax_masked(model.logits(feats.astype(np.float32)), mask)
    out = dict.fromkeys(BUCKETS, 0.0)
    for a in range(40):
        if mask[a] > 0:
            out[category(a, trump_idx)] += float(p[a])
    return out


def argmax_id(model: Any, feats: np.ndarray, mask: np.ndarray) -> int:
    z = model.logits(feats.astype(np.float32)).astype(np.float64)
    z[mask <= 0] = -1e18
    return int(np.argmax(z))


def _with_block(feats: np.ndarray, profile: np.ndarray) -> np.ndarray:
    v = feats.copy()
    v[V4_EXTRA_SLICE] = profile
    return v


def _with_phase(feats: np.ndarray, phase: dict[int, float]) -> np.ndarray:
    v = feats.copy()
    for k, val in phase.items():
        v[k] = val
    return v


def run_counterfactual(model: Any, pool: list[dict], profiles_mean: dict, profiles_med: dict, boot_seed: int) -> dict:
    """Contrasto saver vs mirror sul blocco v4, con profili empirici (media e mediana)."""

    def contrast(profiles: dict) -> dict:
        rows_mir = {k: [] for k in BUCKETS}
        rows_sav = {k: [] for k in BUCKETS}
        dcar_nb: list[float] = []
        flips = 0
        for s in pool:
            bm = bucket_probs(model, _with_block(s["f"], profiles["mirror"]), s["mask"], s["trump"])
            bs = bucket_probs(model, _with_block(s["f"], profiles["heuristic_trump_saver"]), s["mask"], s["trump"])
            for k in BUCKETS:
                rows_mir[k].append(bm[k])
                rows_sav[k].append(bs[k])
            dcar_nb.append(bs["carico_nb"] - bm["carico_nb"])
            flips += argmax_id(model, _with_block(s["f"], profiles["mirror"]), s["mask"]) != argmax_id(
                model, _with_block(s["f"], profiles["heuristic_trump_saver"]), s["mask"]
            )
        d = np.asarray(dcar_nb)
        rng = np.random.default_rng(boot_seed)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)])
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        return {
            "n": len(pool),
            "argmax_flip_pct": round(100.0 * flips / max(len(pool), 1), 2),
            "delta_p_carico_nb_mean": round(float(d.mean()), 5),
            "delta_p_carico_nb_ci95": [round(lo, 5), round(hi, 5)],
            "p_by_bucket_mirror": {k: round(float(np.mean(rows_mir[k])), 5) for k in BUCKETS},
            "p_by_bucket_saver": {k: round(float(np.mean(rows_sav[k])), 5) for k in BUCKETS},
        }

    # Controllo positivo (fase): deve muoversi.
    dcar_nb = [
        bucket_probs(model, _with_phase(s["f"], PHASE_ENDGAME), s["mask"], s["trump"])["carico_nb"]
        - bucket_probs(model, _with_phase(s["f"], PHASE_EARLY), s["mask"], s["trump"])["carico_nb"]
        for s in pool
    ]
    flips = sum(
        argmax_id(model, _with_phase(s["f"], PHASE_EARLY), s["mask"])
        != argmax_id(model, _with_phase(s["f"], PHASE_ENDGAME), s["mask"])
        for s in pool
    )
    return {
        "empirical_mean": contrast(profiles_mean),
        "empirical_median": contrast(profiles_med),
        "positive_control_phase": {
            "delta_p_carico_nb_mean": round(float(np.mean(dcar_nb)), 5),
            "argmax_flip_pct": round(100.0 * flips / max(len(pool), 1), 2),
        },
        "note": "Controfattuale empirico, non prova causale: il blocco v4 co-varia col resto dello stato congelato.",
    }


def _dist(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "p25": round(float(np.percentile(a, 25)), 3),
        "median": round(float(np.median(a)), 3),
        "p75": round(float(np.percentile(a, 75)), 3),
    }


def run_temporal(pool: list[dict]) -> dict:
    from collections import Counter

    amax = Counter(s["amax"] for s in pool)

    def subset(cat: str | None) -> dict:
        rows = pool if cat is None else [s for s in pool if s["amax"] == cat]
        if not rows:
            return {"n": 0}
        return {k: _dist([s[k] for s in rows]) for k in ("trick", "deck", "cuts", "load")}

    return {
        "argmax_live_counts": {k: int(amax.get(k, 0)) for k in BUCKETS},
        "argmax_live_pct": {k: round(100.0 * amax.get(k, 0) / max(len(pool), 1), 2) for k in BUCKETS},
        "pool": subset(None),
        "when_leads_carico_nb": subset("carico_nb"),
        "when_leads_liscio_nb": subset("liscio_nb"),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["counterfactual", "temporal", "both"], default="both")
    ap.add_argument("--model", default="data/models/best_a2c_v11.npz")
    ap.add_argument("--opponents", default="mirror,heuristic_trump_saver,heuristic_v1")
    ap.add_argument("--num-games", type=int, default=500, help="Partite per avversario.")
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--out", default=None, help="Path JSON di output (opzionale).")
    args = ap.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    ai = build_agent("bc_model", model_path=args.model)
    model = ai.model

    per_opp: dict[str, list[dict]] = {}
    for opp in opponents:
        st = collect_states(ai, model, opp, args.model, args.num_games, args.seed)
        per_opp[opp] = st
        print(f"[collect] {opp:24s} stati: {len(st)}")
    pool = [s for opp in opponents for s in per_opp[opp]]

    result: dict[str, Any] = {
        "meta": {
            "git_commit": _git_commit(),
            "model": args.model,
            "feature_dim": int(model.feature_dim),
            "seed": args.seed,
            "num_games_per_opponent": args.num_games,
            "opponents": opponents,
            "n_pool_states": len(pool),
        },
    }

    if args.mode in ("counterfactual", "both"):
        if "mirror" not in per_opp or "heuristic_trump_saver" not in per_opp:
            raise SystemExit("counterfactual richiede almeno gli avversari 'mirror' e 'heuristic_trump_saver'.")
        prof_mean = {o: np.mean([s["f"][V4_EXTRA_SLICE] for s in per_opp[o]], axis=0) for o in per_opp}
        prof_med = {o: np.median([s["f"][V4_EXTRA_SLICE] for s in per_opp[o]], axis=0) for o in per_opp}
        result["counterfactual"] = run_counterfactual(model, pool, prof_mean, prof_med, args.seed)
        result["profiles_style_counters"] = {
            o: {
                "cuts": round(float(prof_mean[o][V4_IDX_OPP_TRUMP_RESPONSE - V4_EXTRA_SLICE.start]), 4),
                "lead_load": round(float(prof_mean[o][V4_IDX_OPP_LEAD_LOAD - V4_EXTRA_SLICE.start]), 4),
            }
            for o in per_opp
        }

    if args.mode in ("temporal", "both"):
        result["temporal"] = run_temporal(pool)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n[out] scritto {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
