#!/usr/bin/env python3
"""
Training RL didattico: Policy Gradient (REINFORCE) per battere un avversario fissato.

Obiettivo
---------
Il Behavior Cloning (BC) imita un teacher: al massimo lo eguaglia.
Per *superare* `heuristic_v1` dobbiamo ottimizzare direttamente una metrica di gioco
(es. punti / vittorie) tramite Reinforcement Learning.

Questo script implementa una baseline semplice e didattica:
- policy MLP (1 hidden layer + ReLU) con spazio azioni 40 carte + action mask
- update con REINFORCE (Monte Carlo) + baseline (media mobile del return)
- ottimizzazione con Adam

Anti-cheat
----------
La policy vede solo `PlayerObservation` (osservazione parziale lecita).

Nota importante (stabilità)
---------------------------
RL è più rumoroso del supervised. Per partire bene, è consigliato:
1) pre-addestrare con BC (teacher-only) → un modello vicino a `heuristic_v1`
2) fare fine-tuning RL contro `heuristic_v1` con questo script

Esempio (warm-start da BC MLP):
  python scripts/train_pg.py \\
    --init ./data/bc_model_teacher_mlp.npz \\
    --out ./data/rl_vs_heuristic_v1.npz \\
    --opponent heuristic_v1 \\
    --num-games 20000 \\
    --seed 0

Poi valuta:
  python scripts/evaluate_agents.py --benchmark small \\
    --agent0 bc_model --agent0-model ./data/rl_vs_heuristic_v1.npz --agent1 heuristic_v1
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.encoding.card_action_space import action_id_from_suit_number
from briscola_ai.ai.encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    EncoderVersion,
    encode_player_observation_2p,
    feature_dim_for_encoder_version,
)
from briscola_ai.ai.models import MLPBCModel, load_bc_model_npz
from briscola_ai.ai.training.opponent_mix import OpponentMixItem, parse_opponent_mix, sample_opponent_name
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state


def _masked_logits_1d(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maschera logits 1D: azioni non valide -> numero molto negativo."""
    very_negative = -1e9
    out = logits.copy()
    out[~mask] = very_negative
    return out


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    """Softmax 1D numericamente stabile."""
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


def _entropy(probs: np.ndarray) -> float:
    """Entropia (Shannon) per una distribuzione discreta."""
    p = probs + 1e-12
    return float(-np.sum(p * np.log(p)))


@dataclass
class AdamState:
    """Stato Adam per un singolo tensore."""

    m: np.ndarray
    v: np.ndarray


def _adam_init(param: np.ndarray) -> AdamState:
    """Inizializza stato Adam (m,v) con zeri, stessa shape del parametro."""
    return AdamState(m=np.zeros_like(param), v=np.zeros_like(param))


def _adam_update(
    param: np.ndarray,
    grad: np.ndarray,
    *,
    state: AdamState,
    lr: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Aggiornamento Adam in-place."""
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * (grad * grad)
    m_hat = state.m / (1.0 - beta1**t)
    v_hat = state.v / (1.0 - beta2**t)
    param -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)


@dataclass
class PolicyStep:
    """Dati salvati per un passo della policy (per REINFORCE)."""

    x: np.ndarray  # (D,)
    z1: np.ndarray  # (H,)
    h: np.ndarray  # (H,)
    probs: np.ndarray  # (40,)
    action_id: int


@dataclass
class MLPPolicy:
    """
    Policy MLP minimale (1 hidden layer + ReLU).

    Nota:
    Usiamo lo stesso formato di parametri del loader (`w1/b1/w2/b2`).
    """

    w1: np.ndarray  # (D, H)
    b1: np.ndarray  # (H,)
    w2: np.ndarray  # (H, 40)
    b2: np.ndarray  # (40,)

    @property
    def feature_dim(self) -> int:
        return int(self.w1.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.w1.shape[1])

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward: ritorna (z1, h, logits)."""
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        logits = h @ self.w2 + self.b2
        return z1, h, logits

    def sample_action(self, *, x: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> PolicyStep:
        """Campiona un'azione valida secondo softmax mascherata."""
        z1, h, logits = self.forward(x)
        masked = _masked_logits_1d(logits, mask)
        probs = _softmax_1d(masked)
        action_id = int(rng.choice(40, p=probs))
        return PolicyStep(x=x, z1=z1, h=h, probs=probs, action_id=action_id)


@dataclass(frozen=True, slots=True)
class OpponentPool:
    """Pool di avversari campionabili (opponent mix)."""

    items: list[OpponentMixItem]
    agents_by_name: dict[str, Agent]

    def sample(self, *, rng: np.random.Generator) -> Agent:
        """Campiona un avversario secondo la distribuzione."""
        name = sample_opponent_name(self.items, rng=rng)
        return self.agents_by_name[name]

    def to_metadata(self) -> list[dict[str, float | str]]:
        """Rappresentazione serializzabile (ordine stabile) per `metadata_json`."""
        return [{"name": item.name, "prob": float(item.prob)} for item in self.items]


def _action_id_to_card_index(*, action_id: int, hand) -> int:
    """Converte action_id (carta canonica) in indice nella mano corrente."""
    for i, card in enumerate(hand):
        cid = action_id_from_suit_number(suit=card.suit.value, number=card.rank.number)
        if cid == action_id:
            return i
    raise ValueError(f"action_id {action_id} non corrisponde a nessuna carta nella mano (hand_size={len(hand)})")


def _play_one_game_2p_collect(
    *,
    policy: MLPPolicy,
    opponent: Agent,
    rng_opponent: random.Random,
    rng_action: np.random.Generator,
    game_seed: int,
    policy_seat: int,
    encoder_version: EncoderVersion,
) -> tuple[GameState, list[PolicyStep]]:
    """
    Simula una partita 2-player.

    - `policy_seat`: 0 o 1, indica quale player è controllato dalla policy RL.
    - colleziona solo i passi in cui gioca la policy.
    """
    state = new_game_state(num_players=2, seed=game_seed)
    steps: list[PolicyStep] = []

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        obs = make_player_observation(state, current)

        if current == policy_seat:
            encoded = encode_player_observation_2p(obs, version=encoder_version)
            x = np.asarray(encoded.features, dtype=np.float32)
            mask = np.asarray(encoded.action_mask, dtype=bool)
            if x.shape[0] != policy.feature_dim:
                raise ValueError(f"Feature dim mismatch: got={x.shape[0]} expected={policy.feature_dim}")
            step_rec = policy.sample_action(x=x, mask=mask, rng=rng_action)
            card_index = _action_id_to_card_index(action_id=step_rec.action_id, hand=obs.hand)
            steps.append(step_rec)
        else:
            # Opponent policy (deterministica o stochastic).
            card_index = opponent.choose_card_index(obs, rng=rng_opponent)

        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

    return state, steps


def _return_for_policy(*, final_state: GameState, policy_seat: int) -> float:
    """
    Return (reward) normalizzato in [-1, 1].

    Usiamo la differenza punti finale, normalizzata su 120:
      R = (points_policy - points_opp) / 120
    """
    p0 = final_state.players[0].points
    p1 = final_state.players[1].points
    if policy_seat == 0:
        return float(p0 - p1) / 120.0
    return float(p1 - p0) / 120.0


@dataclass
class TrainMetrics:
    """Metriche aggregate (per logging)."""

    iter: int
    games: int
    avg_return: float
    win_rate: float
    draw_rate: float
    avg_entropy: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RL policy gradient (MLP, 40 carte + action mask)")
    parser.add_argument("--out", required=True, help="Path output modello (.npz)")
    parser.add_argument("--init", default="", help="Warm-start da un modello `.npz` MLP (es. BC).")
    parser.add_argument(
        "--encoder-version",
        choices=["v1", "v2", "v3"],
        default="v1",
        help=(
            "Versione encoder per observation 2-player. "
            "v1=istantaneo (248 dim), v2=v1 + seen_cards_onehot[40] (288 dim, storia pubblica), "
            "v3=v2 + feature strategiche aggregate (310 dim, solo engine domain)."
        ),
    )
    parser.add_argument(
        "--upgrade-init-v1-to-v2",
        action="store_true",
        help=(
            "Se usi `--encoder-version v2` e `--init` è un modello v1, "
            "espande `w1` aggiungendo 40 righe a zero (warm-start compatibile)."
        ),
    )
    parser.add_argument(
        "--opponent",
        default="heuristic_v1",
        help=(
            "Nome avversario (se non usi --opponent-mix). "
            "Esempi: heuristic_v1, random, greedy_points, best_a2c "
            "(alias che carica `best_a2c.npz` dalla directory modelli)."
        ),
    )
    parser.add_argument(
        "--opponent-mix",
        default="",
        help=(
            "Miscela avversari (robustezza): `name:weight,name:weight,...` "
            "(es. `heuristic_v1:0.7,random:0.2,greedy_points:0.1`). "
            "Se presente, sovrascrive `--opponent`."
        ),
    )
    parser.add_argument("--num-games", type=int, default=20000, help="Numero partite di training (2-player).")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità).")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dim (se non si usa --init).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay (solo pesi).")
    parser.add_argument("--entropy-beta", type=float, default=0.0, help="Entropia bonus (>=0).")
    parser.add_argument(
        "--inference-overkill-guard",
        action="store_true",
        help=(
            "Salva nei metadati del modello un flag per abilitare, a inference-time, "
            "un post-processing anti-overkill: se stiamo per vincere con una briscola da secondi di mano, "
            "giochiamo automaticamente la briscola vincente minima disponibile."
        ),
    )
    parser.add_argument("--update-every", type=int, default=10, help="Aggiorna i pesi ogni N partite (grad batch).")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Stampa metriche ogni N update (default: 10).",
    )
    parser.add_argument("--seat-fair", action="store_true", help="Alterna la seat della policy (riduce bias player0).")
    args = parser.parse_args()

    out_path = Path(args.out)
    encoder_version: EncoderVersion = str(args.encoder_version)
    rng = np.random.default_rng(args.seed)
    rng_game = np.random.default_rng(args.seed ^ 0x9E3779B9)
    rng_opponent_select = np.random.default_rng(args.seed ^ 0xA5A5A5A5)

    opponent_pool: OpponentPool | None = None
    opponent_mix_raw = args.opponent_mix.strip()
    if opponent_mix_raw:
        items = parse_opponent_mix(opponent_mix_raw)
        agents_by_name = {item.name: build_agent(item.name) for item in items}
        opponent_pool = OpponentPool(items=items, agents_by_name=agents_by_name)
        # Per compatibilità, lasciamo comunque valorizzato `opponent` (ma non lo usiamo).
        opponent = build_agent(items[0].name)
    else:
        opponent = build_agent(args.opponent)
    rng_opponent = random.Random(args.seed ^ 0xC0FFEE)

    # Inizializzazione policy.
    target_feature_dim = int(feature_dim_for_encoder_version(encoder_version))
    if args.init.strip():
        loaded = load_bc_model_npz(Path(args.init))
        if not isinstance(loaded, MLPBCModel):
            raise ValueError("--init deve puntare a un modello MLP (w1/b1/w2/b2).")
        w1 = loaded.w1.copy()
        b1 = loaded.b1.copy()
        w2 = loaded.w2.copy()
        b2 = loaded.b2.copy()
        hdim = int(w1.shape[1])
        init_dim = int(w1.shape[0])
        if init_dim != target_feature_dim:
            if (
                bool(args.upgrade_init_v1_to_v2)
                and init_dim == int(FEATURE_DIM_2P_V1)
                and target_feature_dim == int(FEATURE_DIM_2P_V2)
            ):
                pad = np.zeros((target_feature_dim - init_dim, hdim), dtype=np.float32)
                w1 = np.vstack([w1, pad])
            else:
                raise ValueError(
                    "Feature dim mismatch tra `--init` e encoder scelto: "
                    f"init={init_dim} target={target_feature_dim} (encoder={encoder_version}). "
                    "Soluzioni: usa `--encoder-version` coerente, oppure abilita `--upgrade-init-v1-to-v2`."
                )
        policy = MLPPolicy(w1=w1, b1=b1, w2=w2, b2=b2)
    else:
        if args.hidden_dim <= 0:
            raise ValueError("--hidden-dim deve essere > 0")
        hdim = int(args.hidden_dim)
        w1 = rng.normal(loc=0.0, scale=0.02, size=(target_feature_dim, hdim)).astype(np.float32)
        b1 = np.zeros((hdim,), dtype=np.float32)
        w2 = rng.normal(loc=0.0, scale=0.02, size=(hdim, 40)).astype(np.float32)
        b2 = np.zeros((40,), dtype=np.float32)
        policy = MLPPolicy(w1=w1, b1=b1, w2=w2, b2=b2)

    # Adam state.
    st_w1 = _adam_init(policy.w1)
    st_b1 = _adam_init(policy.b1)
    st_w2 = _adam_init(policy.w2)
    st_b2 = _adam_init(policy.b2)
    t = 0

    baseline = 0.0
    baseline_momentum = 0.9

    update_every = int(args.update_every)
    if update_every <= 0:
        raise ValueError("--update-every deve essere > 0")
    log_every = int(args.log_every)
    if log_every <= 0:
        raise ValueError("--log-every deve essere > 0")

    # Accumulo grad.
    gw1 = np.zeros_like(policy.w1)
    gb1 = np.zeros_like(policy.b1)
    gw2 = np.zeros_like(policy.w2)
    gb2 = np.zeros_like(policy.b2)

    returns_buf: list[float] = []
    wins = 0
    draws = 0
    entropies: list[float] = []

    metrics: list[TrainMetrics] = []

    for game_idx in range(1, int(args.num_games) + 1):
        game_seed = int(rng_game.integers(0, 2**32))
        policy_seat = (game_idx % 2) if args.seat_fair else 0
        current_opponent = opponent_pool.sample(rng=rng_opponent_select) if opponent_pool is not None else opponent

        final_state, traj = _play_one_game_2p_collect(
            policy=policy,
            opponent=current_opponent,
            rng_opponent=rng_opponent,
            rng_action=rng,
            game_seed=game_seed,
            policy_seat=policy_seat,
            encoder_version=encoder_version,
        )

        ret = _return_for_policy(final_state=final_state, policy_seat=policy_seat)
        returns_buf.append(ret)
        adv = ret - baseline
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * ret

        # Win/draw tracking (in termini di punti).
        p0 = final_state.players[0].points
        p1 = final_state.players[1].points
        policy_points = p0 if policy_seat == 0 else p1
        opp_points = p1 if policy_seat == 0 else p0
        if policy_points > opp_points:
            wins += 1
        elif policy_points == opp_points:
            draws += 1

        # REINFORCE: per ogni passo, grad = adv * grad(-log pi(a|s)).
        for step_rec in traj:
            probs = step_rec.probs
            entropies.append(_entropy(probs))

            # grad wrt logits: adv * (probs - onehot(a))
            dlogits = probs.copy()
            dlogits[step_rec.action_id] -= 1.0
            dlogits *= float(adv)

            beta = float(args.entropy_beta)
            if beta > 0.0:
                # Loss include `-beta * H(pi)` per incoraggiare esplorazione.
                # Gradiente esatto: d(-H)/dlogits = p * (log p + 1 - E[log p + 1]).
                logp = np.log(probs + 1e-12)
                s = float(np.sum(probs * (logp + 1.0)))
                dent = probs * (logp + 1.0 - s)
                dlogits += beta * dent

            # Backprop MLP.
            # logits = h W2 + b2
            gw2 += np.outer(step_rec.h, dlogits).astype(np.float32)
            gb2 += dlogits.astype(np.float32)

            dh = policy.w2 @ dlogits  # (H,)
            dz1 = dh * (step_rec.z1 > 0.0)
            gw1 += np.outer(step_rec.x, dz1).astype(np.float32)
            gb1 += dz1.astype(np.float32)

        # Aggiorniamo ogni `update_every` partite.
        if game_idx % update_every == 0:
            t += 1
            # Normalizziamo per numero partite del batch per ridurre dipendenza da update_every.
            scale = 1.0 / float(update_every)
            gw1 *= scale
            gb1 *= scale
            gw2 *= scale
            gb2 *= scale

            wd = float(args.weight_decay)
            if wd > 0.0:
                gw1 += wd * policy.w1
                gw2 += wd * policy.w2

            _adam_update(policy.w1, gw1, state=st_w1, lr=float(args.lr), t=t)
            _adam_update(policy.b1, gb1, state=st_b1, lr=float(args.lr), t=t)
            _adam_update(policy.w2, gw2, state=st_w2, lr=float(args.lr), t=t)
            _adam_update(policy.b2, gb2, state=st_b2, lr=float(args.lr), t=t)

            gw1.fill(0.0)
            gb1.fill(0.0)
            gw2.fill(0.0)
            gb2.fill(0.0)

            avg_ret = float(np.mean(returns_buf)) if returns_buf else 0.0
            win_rate = float(wins) / float(update_every)
            draw_rate = float(draws) / float(update_every)
            avg_ent = float(np.mean(entropies)) if entropies else 0.0

            row = TrainMetrics(
                iter=t,
                games=game_idx,
                avg_return=avg_ret,
                win_rate=win_rate,
                draw_rate=draw_rate,
                avg_entropy=avg_ent,
            )
            metrics.append(row)
            if t % log_every == 0 or game_idx == update_every:
                print(
                    f"iter {t:04d} | games {game_idx:06d} | "
                    f"avg_return {row.avg_return:+.3f} | win {row.win_rate:.3f} draw {row.draw_rate:.3f} | "
                    f"entropy {row.avg_entropy:.3f}"
                )
            returns_buf.clear()
            wins = 0
            draws = 0
            entropies.clear()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadati UI (opzionali ma utili):
    # - se presenti, la UI può mostrare un label/descrizione "belli" senza dover inferire tutto dal filename
    # - restano best-effort: il modello funziona anche se questi campi non esistono
    def _format_num_games(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}k"
        return str(n)

    def _format_opponent_label() -> str:
        if opponent_pool is not None:
            parts = [f"{item.name} {float(item.prob):.2f}" for item in opponent_pool.items]
            return "mix(" + ", ".join(parts) + ")"
        return str(args.opponent).strip()

    ui_label = f"PG (REINFORCE) {_format_num_games(int(args.num_games))} game"
    ui_description_it = (
        "Policy addestrata con REINFORCE (policy gradient) su spazio azioni 40 carte + action mask, "
        f"contro {_format_opponent_label()}. "
        "Osservazione anti-cheat: PlayerObservation (vista parziale lecita)."
    )
    payload = {
        "format": "mlp_pg_v1",
        "label": ui_label,
        "description_it": ui_description_it,
        "feature_dim": int(policy.feature_dim),
        "hidden_dim": int(policy.hidden_dim),
        "action_dim": 40,
        "seed": int(args.seed),
        "opponent": str(args.opponent) if not opponent_mix_raw else None,
        "opponent_mix": opponent_pool.to_metadata() if opponent_pool is not None else None,
        "init": args.init.strip() or None,
        "encoder": f"encode_observation_2p:{encoder_version}",
        "encoder_version": encoder_version,
        "inference_overkill_guard": bool(args.inference_overkill_guard),
        "train": {
            "algorithm": "reinforce",
            "optimizer": "adam",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "entropy_beta": float(args.entropy_beta),
            "update_every": int(args.update_every),
            "seat_fair": bool(args.seat_fair),
            "num_games": int(args.num_games),
        },
        "metrics": [asdict(m) for m in metrics],
    }
    np.savez(
        out_path,
        w1=policy.w1,
        b1=policy.b1,
        w2=policy.w2,
        b2=policy.b2,
        metadata_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
