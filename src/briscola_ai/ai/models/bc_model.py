"""
Agente che gioca usando un modello Behavior Cloning salvato in `.npz`.

Contesto didattico
------------------
Nel progetto alleniamo un primo modello supervisionato con `scripts/train_bc.py`,
usando uno spazio azioni fisso "40 carte + action mask".

Questo modulo integra quel modello come un `Agent` del progetto, così possiamo:
- valutarlo con `scripts/evaluate_agents.py` su seed suite riproducibili;
- confrontarlo con baseline (random, heuristic_v1, ...).

Anti-cheat
----------
L'agente riceve solo una `PlayerObservation` (vista parziale lecita) e NON ha
accesso allo stato completo (`GameState`, `deck`, mani avversarie).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ...domain.observation import PlayerObservation
from ...domain.rules import who_wins_trick
from ..encoding.card_action_space import action_id_from_suit_number
from ..encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    FEATURE_DIM_2P_V3,
    FEATURE_DIM_2P_V4,
    EncoderVersion,
    encode_player_observation_2p,
    feature_dim_for_encoder_version,
)
from ..training.reward_shaping import card_conservation_cost


class LoadedBCModel(Protocol):
    """Interfaccia minima per un modello BC caricato da `.npz`."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadati del training (best effort, da `metadata_json`)."""

    @property
    def feature_dim(self) -> int:
        """Dimensione feature attesa dall'encoder."""

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Ritorna logits ``(..., 40)`` da feature ``(..., D)``, anche in batch."""


@dataclass(frozen=True, slots=True)
class LinearBCModel:
    """
    Modello BC lineare (softmax) salvato in `.npz`.

    Convenzione (vedi `scripts/train_bc.py`):
    - `w`: (D, 40) float32
    - `b`: (40,) float32
    - `metadata_json`: stringa JSON (opzionale, informativa)
    """

    w: np.ndarray
    b: np.ndarray
    metadata: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        """Dimensione feature attesa dall'encoder."""
        return int(self.w.shape[0])

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Logits lineari, anche in batch: ``xW + b``."""
        return x @ self.w + self.b


@dataclass(frozen=True, slots=True)
class MLPBCModel:
    """
    Modello BC MLP (1 hidden layer + ReLU) salvato in `.npz`.

    Convenzione (vedi `scripts/train_bc.py`):
    - `w1`: (D, H) float32
    - `b1`: (H,) float32
    - `w2`: (H, 40) float32
    - `b2`: (40,) float32
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    metadata: dict[str, Any]
    # Iterazione 1b (piano belief/ExIt): belief network CONGELATA embedded nell'artefatto.
    # Se presenti, l'input della policy è `encoder v4 (369) + 40 probabilità belief` = 409:
    # l'inferenza sulla mano avversaria diventa parte dell'osservazione della policy.
    belief_w1: np.ndarray | None = None
    belief_b1: np.ndarray | None = None
    belief_w2: np.ndarray | None = None
    belief_b2: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        """Dimensione dell'INPUT della policy (409 se la belief è embedded, altrimenti l'encoder)."""
        return int(self.w1.shape[0])

    @property
    def has_belief_input(self) -> bool:
        """True se l'artefatto include la belief network come input aggiuntivo."""
        return self.belief_w1 is not None

    def belief_probs(self, encoder_features: np.ndarray) -> np.ndarray:
        """Probabilità belief (40,) dalle feature encoder (sigmoid, rete congelata)."""
        assert self.belief_w1 is not None and self.belief_b1 is not None
        assert self.belief_w2 is not None and self.belief_b2 is not None
        h = np.maximum(encoder_features @ self.belief_w1 + self.belief_b1, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ self.belief_w2 + self.belief_b2)))

    def policy_input(self, encoder_features: np.ndarray) -> np.ndarray:
        """Input completo della policy: feature encoder, più le probabilità belief se presenti."""
        if not self.has_belief_input:
            return encoder_features
        return np.concatenate([encoder_features, self.belief_probs(encoder_features)]).astype(np.float32)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Forward MLP, anche in batch: ``relu(xW1 + b1)W2 + b2`` sull'input completo della policy."""
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        return h @ self.w2 + self.b2


def _parse_metadata_json(raw: Any) -> dict[str, Any]:
    """Parsa `metadata_json` salvato in npz (best effort)."""
    try:
        text = str(raw.item())
    except Exception:
        text = str(raw)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_metadata_json": text}
    return parsed if isinstance(parsed, dict) else {"metadata": parsed}


def _validate_declared_feature_dim(metadata: dict[str, Any], actual: int) -> None:
    """Se `metadata.feature_dim` è presente, deve coincidere."""
    declared_dim = metadata.get("feature_dim")
    if isinstance(declared_dim, int) and declared_dim != actual:
        raise ValueError(f"Feature dim mismatch: metadata={declared_dim} actual={actual}")


def _infer_encoder_version_from_metadata(metadata: dict[str, Any]) -> EncoderVersion | None:
    """
    Estrae (best effort) la versione encoder dai metadati.

    Convenzioni supportate:
    - `metadata.encoder_version`: "v1" / "v2" / "v3"
    - `metadata.encoder`: stringa tipo "encode_observation_2p:v1" / "...:v2" / "...:v3"
    """
    raw_version = metadata.get("encoder_version")
    if isinstance(raw_version, str):
        v = raw_version.strip().lower()
        if v in ("v1", "v2", "v3", "v4"):
            return v  # type: ignore[return-value]

    raw_encoder = metadata.get("encoder")
    if isinstance(raw_encoder, str):
        enc = raw_encoder.strip().lower()
        for candidate in ("v1", "v2", "v3", "v4"):
            if enc.endswith(f":{candidate}") or enc == candidate:
                return candidate

    return None


def _infer_encoder_version_for_model(*, metadata: dict[str, Any], feature_dim: int) -> EncoderVersion:
    """
    Decide quale encoder usare per un modello.

    Regola (in ordine):
    1) se `metadata.encoder[_version]` è presente, lo rispettiamo (e validiamo `feature_dim`);
    2) altrimenti facciamo fallback su `feature_dim` (248=v1, 288=v2, 310=v3, 369=v4).
    """
    declared = _infer_encoder_version_from_metadata(metadata)
    if declared is not None:
        expected = feature_dim_for_encoder_version(declared)
        # Policy con belief embedded (iterazione 1b): encoder v4 dichiarato, ma input
        # della policy = 369 + 40 probabilita' belief. Le shape della belief sono gia'
        # validate dal loader; qui accettiamo la coerenza "v4 + 40".
        if declared == "v4" and int(feature_dim) == expected + 40:
            return declared
        if int(feature_dim) != expected:
            raise ValueError(
                "Modello incoerente: "
                f"metadata encoder={declared} ma feature_dim={int(feature_dim)} (atteso {expected})."
            )
        return declared

    if int(feature_dim) == int(FEATURE_DIM_2P_V1):
        return "v1"
    if int(feature_dim) == int(FEATURE_DIM_2P_V2):
        return "v2"
    if int(feature_dim) == int(FEATURE_DIM_2P_V3):
        return "v3"
    if int(feature_dim) == int(FEATURE_DIM_2P_V4):
        return "v4"
    if int(feature_dim) == int(FEATURE_DIM_2P_V4) + 40:
        # Policy con belief embedded (iterazione 1b): l'ENCODER resta v4 (369);
        # le 40 feature extra sono le probabilità belief calcolate a inference.
        return "v4"

    raise ValueError(
        "Impossibile inferire l'encoder: "
        f"feature_dim={int(feature_dim)} non è v1({int(FEATURE_DIM_2P_V1)}), "
        f"v2({int(FEATURE_DIM_2P_V2)}), v3({int(FEATURE_DIM_2P_V3)}) o v4({int(FEATURE_DIM_2P_V4)})."
    )


def _infer_overkill_guard_enabled(metadata: dict[str, Any]) -> bool:
    """
    Decide se abilitare un post-processing anti-overkill (secondo di mano).

    Motivazione:
    alcune policy possono vincere spesso ma "sprecare" briscole alte (es. Asso di briscola)
    quando una briscola bassa avrebbe comunque vinto. Questo comportamento è facilmente
    correggibile a inference-time senza retrain.

    Precedenza:
    1) `metadata.inference_overkill_guard` (bool) oppure `metadata.inference.overkill_guard`
    2) env var `BRISCOLA_BC_OVERKILL_GUARD` in {"1","true","yes","on"} (fallback, utile per A/B test)
    """
    raw = metadata.get("inference_overkill_guard")
    if isinstance(raw, bool):
        return bool(raw)

    inf = metadata.get("inference")
    if isinstance(inf, dict):
        raw2 = inf.get("overkill_guard")
        if isinstance(raw2, bool):
            return bool(raw2)

    env = os.getenv("BRISCOLA_BC_OVERKILL_GUARD", "").strip().lower()
    return env in {"1", "true", "yes", "on"}


def _card_cost_tuple_trump_only(card) -> tuple[int, int]:
    """
    Ordine "economico" tra briscole per post-processing.

    Delega alla definizione canonica condivisa (`card_conservation_cost`): guard runtime,
    reward shaping e metriche decision-quality devono usare lo STESSO ordinamento.
    """
    return card_conservation_cost(card)


def _apply_overkill_guard_second_hand(observation: PlayerObservation, *, chosen_card_index: int) -> int:
    """
    Post-processing anti-overkill (2-player, secondo di mano).

    Regola:
    - se stiamo per vincere una presa giocando una briscola,
      scegliamo la briscola vincente "minima" disponibile in mano (per conservare risorse).

    Anti-cheat:
    - usa solo `PlayerObservation` (mano + carta sul tavolo + briscola pubblica).
    """
    if observation.num_players != 2:
        return chosen_card_index
    if observation.game_over:
        return chosen_card_index
    if observation.trump_card is None:
        return chosen_card_index
    if observation.current_turn != observation.player_index:
        return chosen_card_index
    if len(observation.table_cards) != 1:
        return chosen_card_index
    if chosen_card_index < 0 or chosen_card_index >= len(observation.hand):
        return chosen_card_index

    trump_suit = observation.trump_card.suit
    lead_card, lead_player = observation.table_cards[0]
    chosen = observation.hand[chosen_card_index]
    if chosen.suit != trump_suit:
        return chosen_card_index

    trick_cards = ((lead_card, lead_player), (chosen, observation.player_index))
    if who_wins_trick(trick_cards, trump_suit) != observation.player_index:
        return chosen_card_index

    winning_trumps: list[tuple[tuple[int, int], int]] = []
    for idx, card in enumerate(observation.hand):
        if card.suit != trump_suit:
            continue
        trick_cards = ((lead_card, lead_player), (card, observation.player_index))
        if who_wins_trick(trick_cards, trump_suit) == observation.player_index:
            winning_trumps.append((_card_cost_tuple_trump_only(card), idx))

    if not winning_trumps:
        return chosen_card_index

    _min_cost, min_idx = min(winning_trumps)
    return int(min_idx)


def apply_overkill_guard_second_hand(observation: PlayerObservation, *, chosen_card_index: int) -> int:
    """
    Applica il post-processing anti-overkill generico su una scelta già calcolata.

    È pubblico perché lo stesso guard può proteggere policy non-BC (es. V-lookahead) senza duplicare
    logica. La funzione usa solo `PlayerObservation`, quindi mantiene il confine anti-cheat.
    """
    return _apply_overkill_guard_second_hand(observation, chosen_card_index=chosen_card_index)


# Cache in-process dei modelli `.npz` caricati, chiave (path assoluto, mtime_ns, size).
# `load_bc_model_npz` e' l'unico choke point della lettura `.npz`: lo usano sia `BCModelAgent.from_npz`
# sia `model_catalog.validate_model_compatible_for_ui` (chiamata a ogni `POST /api/games` con bc_model).
# Senza cache, ogni creazione partita rileggerebbe il file (anche decine di MB). La chiave su
# (mtime, size) invalida la cache se il file cambia (es. promozione di un nuovo best). Il modello
# caricato e' read-only per l'inferenza, quindi condividerlo tra richieste e' sicuro.
_BC_MODEL_NPZ_CACHE: dict[tuple[str, int, int], LoadedBCModel] = {}


def load_bc_model_npz(path: Path) -> LoadedBCModel:
    """
    Carica un modello BC da `.npz`, riusando la cache in-process quando il file non e' cambiato.

    Argomenti:
        path: path al file `.npz` salvato da `scripts/train_bc.py`.
    """
    try:
        st = os.stat(path)
    except OSError:
        # Path non statabile (es. inesistente): deleghiamo all'uncached, che dara' l'errore appropriato.
        return _load_bc_model_npz_uncached(path)
    key = (os.path.abspath(str(path)), st.st_mtime_ns, int(st.st_size))
    cached = _BC_MODEL_NPZ_CACHE.get(key)
    if cached is not None:
        return cached
    model = _load_bc_model_npz_uncached(path)
    _BC_MODEL_NPZ_CACHE[key] = model
    return model


def _load_bc_model_npz_uncached(path: Path) -> LoadedBCModel:
    """
    Carica un modello BC da `.npz` (senza cache).

    Argomenti:
        path: path al file `.npz` salvato da `scripts/train_bc.py`.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Formato non supportato: {path} (atteso .npz)")

    with np.load(path) as data:
        keys = set(data.keys())
        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = _parse_metadata_json(data["metadata_json"])

        raw_fmt = metadata.get("format")
        fmt = raw_fmt.strip() if isinstance(raw_fmt, str) else ""

        # Preferiamo esplicitamente il formato dichiarato nel metadata, ma supportiamo anche inferenza.
        is_mlp = fmt in {"mlp_bc_v1", "mlp_pg_v1"} or {"w1", "b1", "w2", "b2"}.issubset(keys)
        if is_mlp:
            missing = {"w1", "b1", "w2", "b2"} - keys
            if missing:
                raise ValueError(
                    f"File modello invalido: mancano chiavi MLP {sorted(missing)} (trovate: {sorted(keys)})"
                )

            w1 = np.asarray(data["w1"], dtype=np.float32)
            b1 = np.asarray(data["b1"], dtype=np.float32)
            w2 = np.asarray(data["w2"], dtype=np.float32)
            b2 = np.asarray(data["b2"], dtype=np.float32)

            if w1.ndim != 2 or b1.ndim != 1 or w2.ndim != 2 or b2.ndim != 1:
                raise ValueError("Shape invalide per MLP (attesi w1/w2 2D e b1/b2 1D)")
            if w2.shape[1] != 40 or b2.shape[0] != 40:
                raise ValueError(f"Action dim invalida: w2={w2.shape} b2={b2.shape} (atteso (*,40) e (40,))")
            if w1.shape[1] != b1.shape[0]:
                raise ValueError(f"Hidden dim mismatch: w1={w1.shape} b1={b1.shape}")
            if w2.shape[0] != b1.shape[0]:
                raise ValueError(f"Hidden dim mismatch: w2={w2.shape} b1={b1.shape}")

            _validate_declared_feature_dim(metadata, int(w1.shape[0]))

            belief_arrays: dict[str, np.ndarray | None] = {
                "belief_w1": None,
                "belief_b1": None,
                "belief_w2": None,
                "belief_b2": None,
            }
            if "belief_w1" in keys:
                missing_belief = {"belief_w1", "belief_b1", "belief_w2", "belief_b2"} - keys
                if missing_belief:
                    raise ValueError(f"Belief embedded incompleta: mancano {sorted(missing_belief)}")
                bw1 = np.asarray(data["belief_w1"], dtype=np.float32)
                bb1 = np.asarray(data["belief_b1"], dtype=np.float32)
                bw2 = np.asarray(data["belief_w2"], dtype=np.float32)
                bb2 = np.asarray(data["belief_b2"], dtype=np.float32)
                if bw2.shape[1] != 40 or bb2.shape != (40,):
                    raise ValueError(f"Belief embedded: shape uscita invalida {bw2.shape}/{bb2.shape}")
                if int(w1.shape[0]) != int(bw1.shape[0]) + 40:
                    raise ValueError(
                        "Policy con belief embedded: attesa feature_dim = encoder + 40 "
                        f"(policy={int(w1.shape[0])}, encoder belief={int(bw1.shape[0])})"
                    )
                belief_arrays = {"belief_w1": bw1, "belief_b1": bb1, "belief_w2": bw2, "belief_b2": bb2}
            return MLPBCModel(w1=w1, b1=b1, w2=w2, b2=b2, metadata=metadata, **belief_arrays)

        if "w" not in data or "b" not in data:
            raise ValueError(f"File modello invalido: chiavi attese w/b, trovate: {sorted(keys)}")

        w = np.asarray(data["w"], dtype=np.float32)
        b = np.asarray(data["b"], dtype=np.float32)
        if w.ndim != 2 or b.ndim != 1:
            raise ValueError(f"Shape invalide: w.ndim={w.ndim} b.ndim={b.ndim} (attesi 2 e 1)")
        if w.shape[1] != 40 or b.shape[0] != 40:
            raise ValueError(f"Action dim invalida: w={w.shape} b={b.shape} (atteso (*,40) e (40,))")

        _validate_declared_feature_dim(metadata, int(w.shape[0]))
        return LinearBCModel(w=w, b=b, metadata=metadata)


@dataclass(frozen=True, slots=True)
class BCModelAgent:
    """
    Agente che seleziona la carta con logit massimo tra quelle in mano.

    Implementazione:
    1) `PlayerObservation` -> `EncodedObservation` (feature + action_mask)
    2) logits = model(x)  (lineare o MLP, stesso output 40)
    3) mask: azioni non valide -> logit molto negativo
    4) argmax -> action_id (0..39)
    5) action_id -> indice nella mano corrente (card_index)
    """

    model: LoadedBCModel
    model_path: Path
    encoder_version: EncoderVersion
    overkill_guard_enabled: bool

    @property
    def name(self) -> str:
        """Nome leggibile dell'agente (includiamo solo il basename per evitare path lunghi)."""
        suffix = "" if self.encoder_version == "v1" else f",encoder={self.encoder_version}"
        if self.overkill_guard_enabled:
            suffix += ",overkill_guard=on"
        return f"bc_model({self.model_path.name}{suffix})"

    @classmethod
    def from_npz(cls, path: str | Path) -> BCModelAgent:
        """Costruisce un agente caricando un `.npz` (output di `scripts/train_bc.py`)."""
        model_path = Path(path)
        model = load_bc_model_npz(model_path)
        encoder_version = _infer_encoder_version_for_model(metadata=model.metadata, feature_dim=int(model.feature_dim))
        overkill_guard_enabled = _infer_overkill_guard_enabled(model.metadata)
        return cls(
            model=model,
            model_path=model_path,
            encoder_version=encoder_version,
            overkill_guard_enabled=overkill_guard_enabled,
        )

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        hand = observation.hand
        if not hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        encoded = encode_player_observation_2p(observation, version=self.encoder_version)
        has_belief = bool(getattr(self.model, "has_belief_input", False))
        expected_encoder_dim = self.model.feature_dim - 40 if has_belief else self.model.feature_dim
        if len(encoded.features) != expected_encoder_dim:
            raise ValueError(
                "Feature dim mismatch: "
                f"encoder={len(encoded.features)} model={self.model.feature_dim} ({self.model_path})"
            )

        encoder_x = np.asarray(encoded.features, dtype=np.float32)
        # Narrowing esplicito per mypy: la belief esiste solo sul modello MLP.
        x = self.model.policy_input(encoder_x) if has_belief and isinstance(self.model, MLPBCModel) else encoder_x
        logits = self.model.logits(x)  # (40,)

        mask = np.asarray(encoded.action_mask, dtype=bool)
        if mask.shape != (40,):
            raise ValueError(f"Mask shape invalida: {mask.shape} (attesa (40,))")
        if not bool(np.any(mask)):
            raise ValueError("Action mask vuota: nessuna azione valida")

        very_negative = -1e9
        masked_logits = logits.copy()
        masked_logits[~mask] = very_negative

        action_id = int(np.argmax(masked_logits))

        # Convertiamo action_id -> card_index nella mano corrente.
        for i, card in enumerate(hand):
            cid = action_id_from_suit_number(suit=card.suit.value, number=card.rank.number)
            if cid == action_id:
                card_index = i
                break
        else:
            card_index = -1

        # Fallback difensivo: se per qualche motivo non troviamo la carta corrispondente,
        # scegliamo una carta valida in mano in modo riproducibile.
        if card_index < 0:
            valid: list[int] = []
            for i, card in enumerate(hand):
                cid = action_id_from_suit_number(suit=card.suit.value, number=card.rank.number)
                if bool(mask[cid]):
                    valid.append(i)
            card_index = valid[rng.randrange(len(valid))] if valid else rng.randrange(len(hand))

        # Post-processing opzionale: anti-overkill (secondo di mano).
        if self.overkill_guard_enabled:
            card_index = _apply_overkill_guard_second_hand(observation, chosen_card_index=int(card_index))

        return int(card_index)
