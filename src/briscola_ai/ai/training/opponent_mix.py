"""
Opponent mix per training RL (didattico).

Perché serve
------------
Allenare una policy RL contro *un solo* avversario (es. `heuristic_v1`) può portare a:
- overfitting: la policy impara exploit specifici, poco robusti;
- scarsa generalizzazione: performance buone vs un avversario, peggiori altrove.

Una tecnica semplice e molto efficace è l'**opponent mix**:
ad ogni partita scegliamo l'avversario da una distribuzione (es. 70% heuristic, 20% random, 10% greedy).

Questo modulo:
- parsa una stringa user-friendly `name:weight,name:weight,...`
- valida e normalizza i pesi
- offre una piccola utility per campionare l'avversario in modo riproducibile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OpponentMixItem:
    """Un elemento della miscela: nome agente + probabilità normalizzata."""

    name: str
    prob: float


def parse_opponent_mix(spec: str) -> list[OpponentMixItem]:
    """
    Converte una stringa di mix in una lista normalizzata.

    Formato accettato:
    - `name:weight,name:weight,...` (es. `heuristic_v1:0.7,random:0.2,greedy_points:0.1`)
    - opzionalmente `name` senza `:weight` (peso implicito = 1.0)

    Regole:
    - i weight devono essere > 0
    - spazi ignorati
    - nomi duplicati vengono sommati

    Ritorna:
        Lista di `OpponentMixItem` con `prob` che somma a 1.0 (entro errore numerico).
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("spec vuota")

    weights_by_name: dict[str, float] = {}
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue

        if ":" in token:
            name_raw, weight_raw = token.split(":", 1)
            name = name_raw.strip()
            if not name:
                raise ValueError(f"Nome agente vuoto in: {token!r}")
            try:
                weight = float(weight_raw.strip())
            except ValueError as exc:
                raise ValueError(f"Peso non numerico per {name!r}: {weight_raw!r}") from exc
        else:
            name = token
            weight = 1.0

        if weight <= 0.0:
            raise ValueError(f"Peso deve essere > 0 per {name!r}, ottenuto {weight}")
        weights_by_name[name] = weights_by_name.get(name, 0.0) + weight

    if not weights_by_name:
        raise ValueError("spec non contiene nessun elemento valido")

    total = float(sum(weights_by_name.values()))
    if total <= 0.0:
        raise ValueError("Somma pesi non valida (<=0)")

    items = [OpponentMixItem(name=k, prob=float(v) / total) for k, v in sorted(weights_by_name.items())]
    # Normalizzazione finale (stabile) per evitare drift numerici su somme lunghe.
    s = float(sum(i.prob for i in items))
    if s <= 0.0:
        raise ValueError("Somma probabilità non valida (<=0)")
    items = [OpponentMixItem(name=i.name, prob=i.prob / s) for i in items]
    return items


def sample_opponent_name(items: list[OpponentMixItem], *, rng: np.random.Generator) -> str:
    """
    Campiona il nome dell'avversario secondo la distribuzione in `items`.

    Nota:
    `items` deve essere già normalizzato (output di `parse_opponent_mix`).
    """
    if not items:
        raise ValueError("items vuota")
    probs = np.asarray([i.prob for i in items], dtype=np.float64)
    if probs.shape != (len(items),):
        raise ValueError("probs shape mismatch")
    probs = probs / float(np.sum(probs))
    idx = int(rng.choice(len(items), p=probs))
    return items[idx].name
