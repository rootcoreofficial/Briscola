"""
Registry e factory degli agenti.

Qui vive il catalogo stabile usato da CLI, backend e UI. Le implementazioni concrete
stanno nei moduli specializzati; questa factory e' l'unico punto che decide come
costruirle a partire dal nome canonico.
"""

from __future__ import annotations

from pathlib import Path

from ..encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, FEATURE_DIM_2P_V3
from ..models.bc_model import BCModelAgent
from ..models.belief_model import load_belief_model_npz
from ..models.catalog import get_models_dir_from_env, resolve_model_path
from ..models.provisioning import PIMC_BELIEF_MODEL_ID, VALUE_LOOKAHEAD_MODEL_ID
from ..models.value_model import load_value_model_npz
from .base import Agent, AgentSpec
from .hybrid_endgame import HybridEndgameAgent
from .pimc import PIMCAgent
from .rule_based import (
    GreedyPointsAgent,
    HeuristicAgentV1,
    HeuristicAgentV2,
    HeuristicTrumpSaverAgent,
    RandomAgent,
)
from .value_lookahead import ValueLookaheadAgent

_AGENT_BUILDERS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "greedy_points": GreedyPointsAgent,
    "heuristic_v1": HeuristicAgentV1,
    "heuristic_v2": HeuristicAgentV2,
    # Sonda di exploitability (audit produzione 2026-07-07): disponibile per valutazioni
    # e training, ma NON in `list_agent_specs()` — non è un avversario offerto dalla UI.
    "heuristic_trump_saver": HeuristicTrumpSaverAgent,
    "hybrid_endgame": HybridEndgameAgent,
}

BC_MODEL_SPEC = AgentSpec(
    name="bc_model",
    label="Modello locale (.npz)",
    description_it=(
        "Usa un modello addestrato e salvato in un file `.npz` (Behavior Cloning / RL). "
        "Il file è scelto dalla UI tra quelli disponibili sul server."
    ),
)

BC_MODEL_HYBRID_ENDGAME_SPEC = AgentSpec(
    name="bc_model_hybrid_endgame",
    label="Modello locale + solver finale",
    description_it=(
        "Usa il modello `.npz` scelto dalla UI durante la partita e, a mazzo vuoto, passa al solver "
        "esatto del finale ricostruito dalla sola osservazione pubblica. È la variante runtime "
        "consigliata per testare il modello corrente con finale esatto."
    ),
)

BC_MODEL_VALUE_LOOKAHEAD_8X8_SPEC = AgentSpec(
    name="bc_model_value_lookahead_8x8",
    label="Modello locale + value lookahead",
    description_it=(
        "Usa il modello `.npz` scelto dalla UI come policy base, il solver esatto a mazzo vuoto e una lookahead "
        "depth-1 guidata da una rete di valore quando restano al massimo 8 carte vive ignote. Resta una variante "
        "storica utile per confronti e ablation: è più forte del solo modello `.npz`, ma il default corrente usa "
        "PIMC belief 12×8."
    ),
    requires_model_id=VALUE_LOOKAHEAD_MODEL_ID,
)

BC_MODEL_PIMC_16X8_SPEC = AgentSpec(
    name="bc_model_pimc_16x8",
    label="Modello locale + PIMC finale",
    description_it=(
        "Usa il modello `.npz` scelto dalla UI come fallback e policy di simulazione, il solver esatto a mazzo vuoto "
        "e una search PIMC con 16 determinizzazioni quando restano al massimo 8 carte vive ignote. "
        "È più forte ma più costoso lato CPU: usalo come avversario avanzato selezionabile."
    ),
)

BC_MODEL_PIMC_BELIEF_64X10_SPEC = AgentSpec(
    name="bc_model_pimc_belief_64x10",
    label="Modello locale + PIMC belief (max)",
    description_it=(
        "La variante a massima forza: usa il modello `.npz` scelto dalla UI come policy di simulazione, il solver "
        "esatto a mazzo vuoto e una search PIMC con 64 determinizzazioni PESATE dalla belief network "
        "(stima di quali carte ha in mano l'avversario, dedotta dal suo comportamento) quando restano al "
        "massimo 10 carte vive ignote. Nei benchmark storici batte il modello puro di circa 4 punti/partita; "
        "è più costosa del default 12×8 e resta selezionabile per chi privilegia la forza alla capacità server."
    ),
    requires_model_id=PIMC_BELIEF_MODEL_ID,
)

BC_MODEL_PIMC_BELIEF_16X8_SPEC = AgentSpec(
    name="bc_model_pimc_belief_16x8",
    label="Modello locale + PIMC belief 16x8",
    description_it=(
        "La precedente configurazione consigliata: 16 determinizzazioni pesate dalla belief network, finestra 8 "
        "e solver esatto a mazzo vuoto. Resta selezionabile per confrontare v15 col runtime di v14 o per usare "
        "quattro campioni in più rispetto al nuovo default 12×8."
    ),
    requires_model_id=PIMC_BELIEF_MODEL_ID,
)

BC_MODEL_PIMC_BELIEF_12X8_SPEC = AgentSpec(
    name="bc_model_pimc_belief_12x8",
    label="IA consigliata (PIMC belief 12x8)",
    description_it=(
        "L'avversario consigliato per Briscola AI v15: 12 determinizzazioni pesate dalla belief network, "
        "finestra 8 e solver esatto a mazzo vuoto. Nel gate appaiato da 20.000 partite non perde forza "
        "misurabile rispetto al precedente 16×8 e riduce di circa il 25% il tempo della search."
    ),
    requires_model_id=PIMC_BELIEF_MODEL_ID,
)

BEST_A2C_SPEC = AgentSpec(
    name="best_a2c",
    label="Best A2C (locale)",
    description_it=(
        "Carica un modello “campione” A2C da un file locale `best_a2c.npz` nella directory modelli. "
        "È pensato per training in stile league (avversario congelato) e per confronti riproducibili."
    ),
    requires_model_id="best_a2c.npz",
)

_BEST_A2C_DEFAULT_MODEL_ID = "best_a2c.npz"
_VALUE_LOOKAHEAD_8X8_DETERMINIZATIONS = 8
_VALUE_LOOKAHEAD_8X8_MAX_UNKNOWN_CARDS = 8
_PIMC_16X8_DETERMINIZATIONS = 16
_PIMC_16X8_MAX_UNKNOWN_CARDS = 8
_PIMC_BELIEF_64X10_DETERMINIZATIONS = 64
_PIMC_BELIEF_64X10_MAX_UNKNOWN_CARDS = 10
_PIMC_BELIEF_16X8_DETERMINIZATIONS = 16
_PIMC_BELIEF_16X8_MAX_UNKNOWN_CARDS = 8
_PIMC_BELIEF_12X8_DETERMINIZATIONS = 12
_PIMC_BELIEF_12X8_MAX_UNKNOWN_CARDS = 8
# Variante eval-only per testare il confine della finestra PIMC senza esporre un'altra
# scelta in UI: stessa dose agile della 16x8, ma search attiva fino a 10 carte vive ignote.
BC_MODEL_PIMC_BELIEF_16X10_EVAL_NAME = "bc_model_pimc_belief_16x10"
_SELECTED_MODEL_AGENT_NAMES = frozenset(
    {
        BC_MODEL_SPEC.name,
        BC_MODEL_HYBRID_ENDGAME_SPEC.name,
        BC_MODEL_VALUE_LOOKAHEAD_8X8_SPEC.name,
        BC_MODEL_PIMC_16X8_SPEC.name,
        BC_MODEL_PIMC_BELIEF_64X10_SPEC.name,
        BC_MODEL_PIMC_BELIEF_16X8_SPEC.name,
        BC_MODEL_PIMC_BELIEF_12X8_SPEC.name,
        BC_MODEL_PIMC_BELIEF_16X10_EVAL_NAME,
    }
)

HYBRID_ENDGAME_BEST_A2C_SPEC = AgentSpec(
    name="hybrid_endgame_best_a2c",
    label="Hybrid Endgame (Best A2C)",
    description_it=(
        "Come Hybrid Endgame, ma usa il modello `best_a2c.npz` come policy in mid-game e il solver "
        "esatto a mazzo vuoto. Richiede il file `best_a2c.npz` nella directory modelli (non sempre "
        "presente: in tal caso l'opzione è non disponibile)."
    ),
    requires_model_id="best_a2c.npz",
)

AI_AGENTS_COMMON_NOTE_IT = (
    "Nota anti-cheat: tutte le IA ricevono solo un’osservazione parziale (PlayerObservation). "
    "Non possono leggere l’ordine del mazzo né le carte specifiche in mano all’avversario."
)


def list_agent_specs() -> list[AgentSpec]:
    """Ritorna la lista di agenti disponibili con metadati in ordine stabile."""
    return [
        RandomAgent.spec,
        GreedyPointsAgent.spec,
        HeuristicAgentV1.spec,
        HeuristicAgentV2.spec,
        HybridEndgameAgent.spec,
        HYBRID_ENDGAME_BEST_A2C_SPEC,
        BC_MODEL_SPEC,
        BC_MODEL_VALUE_LOOKAHEAD_8X8_SPEC,
        BC_MODEL_HYBRID_ENDGAME_SPEC,
        BC_MODEL_PIMC_16X8_SPEC,
        BC_MODEL_PIMC_BELIEF_12X8_SPEC,
        BC_MODEL_PIMC_BELIEF_16X8_SPEC,
        BC_MODEL_PIMC_BELIEF_64X10_SPEC,
    ]


def agent_uses_selected_model(name: str) -> bool:
    """Ritorna True se l'agente richiede un `model_id` scelto dal catalogo `.npz`."""
    return name in _SELECTED_MODEL_AGENT_NAMES


def _load_best_a2c_agent() -> BCModelAgent:
    """
    Carica il modello campione `best_a2c.npz` dalla directory modelli e ne valida la compatibilità.

    Estratto come helper perché serve sia all'agente `best_a2c` sia a `hybrid_endgame_best_a2c`
    (che lo usa come policy mid-game), così la logica di risoluzione path/validazione resta unica.
    """
    models_dir = get_models_dir_from_env()
    try:
        path = resolve_model_path(models_dir=models_dir, model_id=_BEST_A2C_DEFAULT_MODEL_ID)
    except FileNotFoundError as exc:
        raise ValueError(
            "Modello 'best_a2c' non disponibile: file non trovato. "
            "Convenzione: salva (o copia) un modello `.npz` compatibile in "
            f"{models_dir.resolve()!s}/{_BEST_A2C_DEFAULT_MODEL_ID}. "
            "Puoi cambiare directory impostando `BRISCOLA_MODELS_DIR`."
        ) from exc

    agent = BCModelAgent.from_npz(path)
    supported = {int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2), int(FEATURE_DIM_2P_V3)}
    if int(agent.model.feature_dim) not in supported:
        expected = f"{int(FEATURE_DIM_2P_V1)} (v1), {int(FEATURE_DIM_2P_V2)} (v2) or {int(FEATURE_DIM_2P_V3)} (v3)"
        raise ValueError(
            "Modello 'best_a2c' non compatibile: feature_dim non coerente con un encoder 2-player supportato. "
            f"model={int(agent.model.feature_dim)} expected={expected} ({path})."
        )
    return agent


def build_agent(name: str, *, model_path: Path | None = None) -> Agent:
    """
    Costruisce un agente a partire dal nome canonico.

    Nota:
    usiamo una mappa esplicita (no import dinamici) per semplicità e riproducibilità.
    """
    if name == "best_a2c":
        return _load_best_a2c_agent()

    if name == "hybrid_endgame_best_a2c":
        # Variante esplicita di hybrid_endgame con policy mid-game = best_a2c.
        # `hybrid_endgame` resta invariato (fallback heuristic_v2) per stabilità dei benchmark.
        return HybridEndgameAgent(fallback=_load_best_a2c_agent(), name="hybrid_endgame_best_a2c")

    if name == "bc_model":
        if model_path is None:
            raise ValueError("Agente 'bc_model' richiede `model_path` (file .npz)")
        return BCModelAgent.from_npz(model_path)

    if name == "bc_model_hybrid_endgame":
        if model_path is None:
            raise ValueError("Agente 'bc_model_hybrid_endgame' richiede `model_path` (file .npz)")
        return HybridEndgameAgent(
            fallback=BCModelAgent.from_npz(model_path),
            name="bc_model_hybrid_endgame",
        )

    if name == "bc_model_value_lookahead_8x8":
        if model_path is None:
            raise ValueError("Agente 'bc_model_value_lookahead_8x8' richiede `model_path` (file .npz)")
        models_dir = get_models_dir_from_env()
        try:
            value_model_path = resolve_model_path(
                models_dir=models_dir,
                model_id=VALUE_LOOKAHEAD_MODEL_ID,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "Agente 'bc_model_value_lookahead_8x8' non disponibile: manca il value model "
                f"`{VALUE_LOOKAHEAD_MODEL_ID}` nella directory modelli."
            ) from exc
        value_model = load_value_model_npz(value_model_path)
        policy_agent = BCModelAgent.from_npz(model_path)
        control = HybridEndgameAgent(
            fallback=policy_agent,
            name="bc_model_value_lookahead_8x8_control",
        )
        return ValueLookaheadAgent(
            value_model=value_model,
            fallback=control,
            continuation_agent=control,
            num_determinizations=_VALUE_LOOKAHEAD_8X8_DETERMINIZATIONS,
            max_unknown_cards=_VALUE_LOOKAHEAD_8X8_MAX_UNKNOWN_CARDS,
            overkill_guard_enabled=True,
            name="bc_model_value_lookahead_8x8",
        )

    if name == "bc_model_pimc_16x8":
        if model_path is None:
            raise ValueError("Agente 'bc_model_pimc_16x8' richiede `model_path` (file .npz)")
        model_agent = BCModelAgent.from_npz(model_path)
        return PIMCAgent(
            rollout_agent=model_agent,
            fallback=model_agent,
            num_determinizations=_PIMC_16X8_DETERMINIZATIONS,
            max_unknown_cards=_PIMC_16X8_MAX_UNKNOWN_CARDS,
            use_endgame_solver=True,
            name="bc_model_pimc_16x8",
        )

    if name in (
        "bc_model_pimc_belief_64x10",
        "bc_model_pimc_belief_16x8",
        BC_MODEL_PIMC_BELIEF_12X8_SPEC.name,
        BC_MODEL_PIMC_BELIEF_16X10_EVAL_NAME,
    ):
        if model_path is None:
            raise ValueError(f"Agente {name!r} richiede `model_path` (file .npz)")
        models_dir = get_models_dir_from_env()
        try:
            belief_model_path = resolve_model_path(models_dir=models_dir, model_id=PIMC_BELIEF_MODEL_ID)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Agente {name!r} non disponibile: manca la belief network "
                f"`{PIMC_BELIEF_MODEL_ID}` nella directory modelli."
            ) from exc
        belief_model = load_belief_model_npz(belief_model_path)
        model_agent = BCModelAgent.from_npz(model_path)
        is_max = name == "bc_model_pimc_belief_64x10"
        is_12x8 = name == BC_MODEL_PIMC_BELIEF_12X8_SPEC.name
        is_eval_16x10 = name == BC_MODEL_PIMC_BELIEF_16X10_EVAL_NAME
        max_unknown_cards = (
            _PIMC_BELIEF_64X10_MAX_UNKNOWN_CARDS
            if is_max or is_eval_16x10
            else _PIMC_BELIEF_12X8_MAX_UNKNOWN_CARDS
            if is_12x8
            else _PIMC_BELIEF_16X8_MAX_UNKNOWN_CARDS
        )
        return PIMCAgent(
            rollout_agent=model_agent,
            fallback=model_agent,
            num_determinizations=(
                _PIMC_BELIEF_64X10_DETERMINIZATIONS
                if is_max
                else _PIMC_BELIEF_12X8_DETERMINIZATIONS
                if is_12x8
                else _PIMC_BELIEF_16X8_DETERMINIZATIONS
            ),
            max_unknown_cards=max_unknown_cards,
            use_endgame_solver=True,
            belief_model=belief_model,
            # Search Python per tutte le config pubbliche: la search JIT valeva ~2x di CPU
            # per mossa ma costava ~8s di compilazione a ogni cold start delle repliche
            # (scale-to-zero dopo ~90s!). Forza equivalente verificata nei due versi
            # (numba vs python: CI sovrapposte; 16x8 python su v11: +3.36, CI +3.05..+3.66);
            # il risparmio CPU oggi arriva dalla configurazione 12x8, non dal JIT.
            # I kernel numba della search restano per training/benchmark offline.
            use_numba_search=False,
            name=name,
        )

    try:
        return _AGENT_BUILDERS[name]()
    except KeyError as exc:
        raise ValueError(f"Agente non supportato: {name!r}") from exc
