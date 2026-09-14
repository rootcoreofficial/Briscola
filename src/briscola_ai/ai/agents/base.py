"""
Contratti comuni per gli agenti giocabili.

Questo modulo contiene solo il vocabolario condiviso: l'interfaccia minima di una
policy e i metadati esposti a CLI/UI. Tenerlo piccolo rende piu' facile capire
che ogni agente del progetto riceve una `PlayerObservation`, non lo stato completo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from ...domain.observation import PlayerObservation


class Agent(Protocol):
    """
    Interfaccia minima di un agente.

    Un agente deve scegliere un `card_index` valido (indice nella mano del player).
    In Briscola qualunque carta in mano e' giocabile, quindi l'insieme delle azioni valide
    e' sempre `range(len(hand))` (finche' la partita non e' finita).
    """

    @property
    def name(self) -> str:
        """Nome leggibile dell'agente (usato in CLI/log/metriche)."""

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        """Sceglie l'indice della carta da giocare per il giocatore osservante."""


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """
    Metadati didattici di un agente.

    Scopo:
    - avere una sorgente unica di verita' per nome/descrizione dell'agente;
    - poter esporre questi metadati a UI/CLI senza duplicare stringhe nel frontend.
    """

    name: str
    label: str
    description_it: str
    requires_model_id: str | None = None
