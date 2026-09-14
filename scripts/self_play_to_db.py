#!/usr/bin/env python3
"""
Self-play → SQLite (event log).

Obiettivo didattico
-------------------
Per addestrare o valutare un agente (anche prima delle reti neurali) ci serve un modo
per generare molte partite senza UI e senza HTTP/WS.

Questo script:
- simula partite usando direttamente il dominio (`GameState + step`);
- salva gli eventi nel DB SQLite con lo stesso schema/event_type usato dal backend;
- rende l'export JSONL ripetibile e veloce.

Nota:
- Le azioni generate qui sono “IA” per definizione (self-play).
  Se vuoi dataset “umano”, usa il backend e colleziona partite reali.

Anti-cheat (importante)
-----------------------
Le policy degli agenti sono calcolate *solo* a partire da una `PlayerObservation` (vista parziale lecita),
non dal `GameState` completo. Questo evita leak informativi accidentali (es. ordine del mazzo, mano avversaria).

Nota metadati:
questa garanzia è documentata nel `README.md` (e nel codice), ma non viene ripetuta nel DB per ogni partita:
nel DB salviamo invece i metadati davvero “operativi” (es. quali agenti hanno giocato e i seed).
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from briscola_ai.ai.agents import Agent, build_agent, list_agent_specs
from briscola_ai.backend.dto import CardDTO, PlayActionResultDTO, TableCardDTO, TrickResultDTO
from briscola_ai.backend.event_log import EventLog, EventLogConfig
from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.engine import PlayCardAction, StepResult, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state
from briscola_ai.versioning import get_code_version, get_rules_version


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configurazione self-play."""

    db_path: Path
    num_games: int
    seed: int
    num_players: int
    agent_names: tuple[str, ...]


def _build_play_action_result(step_result: StepResult, *, server_version: int) -> PlayActionResultDTO:
    """
    Converte `StepResult` (dominio) in `PlayActionResultDTO` (JSON-friendly).

    Questo mantiene coerenza con il payload salvato dal backend nell'event log.
    """
    if step_result.played_card is None or step_result.player is None:
        raise ValueError("StepResult incompleto: played_card/player mancante")

    trick_cards_dto: list[TableCardDTO] | None = None
    captured_cards_dto: list[CardDTO] = []
    if step_result.trick_completed:
        trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in step_result.trick_cards]
        captured_cards_dto = [CardDTO.from_domain(card) for card, _ in step_result.trick_cards]

    return PlayActionResultDTO(
        server_version=server_version,
        played_card=CardDTO.from_domain(step_result.played_card),
        player=step_result.player,
        trick_completed=step_result.trick_completed,
        trick_winner=step_result.trick_winner,
        trick_size=len(step_result.trick_cards),
        cards_dealt=step_result.cards_dealt,
        trick_cards=trick_cards_dto,
        captured_cards=captured_cards_dto,
    )


def _log_observations_for_all_players(
    event_log: EventLog,
    *,
    game_id: str,
    state: GameState,
    server_version: int,
) -> None:
    """
    Logga `observation_sent` per tutti i player.

    Questo replica il comportamento del backend (broadcast WS) e rende semplice
    costruire transizioni `(s, a, r, s', done)` nell'exporter.
    """
    for player_index in range(state.num_players):
        obs = build_observation_dto(state, player_index, server_version)
        event_log.log_event(
            game_id,
            "observation_sent",
            obs.model_dump(),
            server_version=server_version,
            player_index=player_index,
        )


def simulate_self_play_to_db(config: SelfPlayConfig) -> dict[str, int]:
    """
    Simula `config.num_games` e scrive nel DB SQLite.

    Ritorna contatori per debug/monitoring.
    """
    rng_game = random.Random(config.seed)
    rng_action_master = random.Random(config.seed ^ 0x9E3779B9)

    agents: tuple[Agent, ...] = tuple(build_agent(name) for name in config.agent_names)

    event_log = EventLog(EventLogConfig(path=str(config.db_path)))
    code_version = get_code_version()
    rules_version = get_rules_version()

    counters = {
        "games_written": 0,
        "actions_written": 0,
        "trick_results_written": 0,
        "observations_written": 0,
        "errors": 0,
    }

    try:
        for _ in range(config.num_games):
            game_seed = rng_game.randrange(0, 2**32)
            action_seed = rng_action_master.randrange(0, 2**32)
            rng_action = random.Random(action_seed)
            state = new_game_state(num_players=config.num_players, seed=game_seed)

            game_id = str(uuid.uuid4())
            server_version = 0

            # Anchor/metadata partita.
            event_log.ensure_game(
                game_id,
                num_players=state.num_players,
                seed=game_seed,
                code_version=code_version,
                rules_version=rules_version,
            )
            event_log.log_event(
                game_id,
                "game_created",
                {
                    "seed": game_seed,
                    "action_seed": action_seed,
                    "code_version": code_version,
                    "rules_version": rules_version,
                    "num_players": state.num_players,
                    "is_team_game": state.is_team_game,
                    "agents": {str(i): agents[i].name for i in range(state.num_players)},
                },
                server_version=server_version,
            )

            # Stato iniziale.
            _log_observations_for_all_players(event_log, game_id=game_id, state=state, server_version=server_version)
            counters["observations_written"] += state.num_players

            safety = 5000
            while not state.game_over and safety > 0:
                safety -= 1
                current = state.current_turn
                hand_size = len(state.players[current].hand)
                if hand_size <= 0:
                    counters["errors"] += 1
                    break

                # Policy dell'agente: decisione basata su osservazione lecita (anti-cheat).
                observation = make_player_observation(state, current)
                card_index = agents[current].choose_card_index(observation, rng=rng_action)

                # Guard rail: qualunque carta in mano è valida, quindi l'indice deve essere in range.
                if card_index < 0 or card_index >= hand_size:
                    counters["errors"] += 1
                    break

                new_state, step_result = step(state, PlayCardAction(player_index=current, card_index=card_index))
                if step_result.error:
                    counters["errors"] += 1
                    break

                server_version += 1
                action_result = _build_play_action_result(step_result, server_version=server_version)

                event_log.log_event(
                    game_id,
                    "action_play_card",
                    {
                        "is_ai": True,
                        "player_index": current,
                        "card_index": card_index,
                        "result": action_result.model_dump(exclude_none=True),
                    },
                    server_version=server_version,
                    player_index=current,
                )
                counters["actions_written"] += 1

                if step_result.trick_completed:
                    points = sum(card.rank.points for card, _ in step_result.trick_cards)
                    winner_index = step_result.trick_winner if step_result.trick_winner is not None else 0
                    winner_name = new_state.players[winner_index].name
                    trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in step_result.trick_cards]
                    trick_result = TrickResultDTO(
                        trick_cards=trick_cards_dto,
                        winner_index=winner_index,
                        winner_name=winner_name,
                        points=points,
                        server_version=server_version,
                    )
                    event_log.log_event(
                        game_id, "trick_result", trick_result.model_dump(), server_version=server_version
                    )
                    counters["trick_results_written"] += 1

                state = new_state

                _log_observations_for_all_players(
                    event_log, game_id=game_id, state=state, server_version=server_version
                )
                counters["observations_written"] += state.num_players

            # Marker esplicito: partita completa (`game_over=true`).
            #
            # Serve soprattutto per:
            # - export “solo complete games” (dataset più pulito);
            # - statistiche offline senza dover inferire la fine partita da altri eventi.
            if state.game_over:
                with contextlib.suppress(Exception):
                    event_log.try_mark_game_finished(game_id)
                event_log.log_event(
                    game_id,
                    "game_finished",
                    {
                        "game_over": True,
                        "num_players": state.num_players,
                        "is_team_game": state.is_team_game,
                        "final_points_by_player_index": [p.points for p in state.players],
                    },
                    server_version=server_version,
                )

            counters["games_written"] += 1

        return counters
    finally:
        event_log.close()


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Self-play Briscola (dominio) → SQLite (event log)")
    parser.add_argument("--db", default="./data/briscola_events.sqlite3", help="Path DB SQLite (event log)")
    parser.add_argument("--num-games", type=int, default=10, help="Numero di partite da simulare")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità)")
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 4], help="Numero giocatori (2 o 4)")
    parser.add_argument(
        "--agents",
        default="",
        help=(
            "Agenti per player (CSV). Esempi:\n"
            "- 2-player: --agents heuristic_v1,random\n"
            "- 4-player: --agents random,random,random,random\n"
            "Se vuoto, usa `random` per tutti."
        ),
    )
    args = parser.parse_args()

    available = {spec.name for spec in list_agent_specs()}
    if args.agents.strip():
        names = tuple(n.strip() for n in args.agents.split(",") if n.strip())
    else:
        names = tuple("random" for _ in range(args.num_players))

    if len(names) != args.num_players:
        raise SystemExit(
            f"Errore: `--agents` deve contenere esattamente {args.num_players} nomi (uno per player). Ottenuti: {names}"
        )
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"Errore: agenti non supportati: {unknown}. Disponibili: {sorted(available)}")

    cfg = SelfPlayConfig(
        db_path=Path(args.db),
        num_games=args.num_games,
        seed=args.seed,
        num_players=args.num_players,
        agent_names=names,
    )

    start = time.time()
    counters = simulate_self_play_to_db(cfg)
    elapsed = time.time() - start

    print(f"Self-play completato in {elapsed:.2f}s.")
    print(f"- agents: {list(cfg.agent_names)}")
    for k, v in counters.items():
        print(f"- {k}: {v}")
    return 0 if counters["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
