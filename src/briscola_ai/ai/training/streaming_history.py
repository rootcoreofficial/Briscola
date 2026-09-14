"""Storico completo o summary O(1) per training di durata arbitraria.

I trainer brevi conservano ogni riga per compatibilita' e analisi dettagliata. Nei run
multi-milione servono invece soltanto conteggio, prima e ultima riga: continuare ad
accumulare oggetti in memoria vanificherebbe il significato di ``metrics-mode=summary``.
La rappresentazione di resume coincide con quella in memoria e non ricostruisce righe
mai conservate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, cast

HistoryMode = Literal["full", "summary"]


@dataclass(slots=True)
class StreamingHistory[T]:
    """Conserva tutte le righe oppure solo statistiche sufficienti O(1)."""

    mode: HistoryMode
    count: int = 0
    first: T | None = None
    last: T | None = None
    _items: list[T] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in ("full", "summary"):
            raise ValueError(f"Modalità storico non supportata: {self.mode!r}")
        if self.count < 0:
            raise ValueError("count deve essere >= 0")
        if self.mode == "summary" and self._items:
            raise ValueError("Uno storico summary non può contenere la lista completa")
        if self.mode == "full" and self.count != len(self._items):
            raise ValueError("count incoerente con le righe dello storico full")
        if self.count == 0 and (self.first is not None or self.last is not None):
            raise ValueError("Uno storico vuoto non può avere prima/ultima riga")

    @property
    def retained_rows(self) -> int:
        """Numero di oggetti trattenuti, utile per provare il limite di memoria."""
        if self.mode == "full":
            return len(self._items)
        return int(self.first is not None) + int(self.last is not None and self.count > 1)

    def append(self, item: T) -> None:
        """Registra una riga senza alias mutabili provenienti dal chiamante."""
        value = deepcopy(item)
        if self.count == 0:
            self.first = deepcopy(value)
        self.last = deepcopy(value)
        self.count += 1
        if self.mode == "full":
            self._items.append(value)

    def metadata(self, *, full_key: str, summary_key: str) -> dict[str, object]:
        """Restituisce il contratto metadata usato storicamente dai trainer."""
        if self.mode == "full":
            return {full_key: deepcopy(self._items)}
        return {
            summary_key: {
                "mode": "summary",
                "count": self.count,
                "first": deepcopy(self.first),
                "last": deepcopy(self.last),
            }
        }

    def resume_state(self) -> dict[str, object]:
        """Serializza soltanto ciò che è stato intenzionalmente mantenuto in memoria."""
        state: dict[str, object] = {
            "mode": self.mode,
            "count": self.count,
            "first": deepcopy(self.first),
            "last": deepcopy(self.last),
        }
        if self.mode == "full":
            state["items"] = deepcopy(self._items)
        return state

    @classmethod
    def from_resume_state(cls, state: dict[str, object], *, expected_mode: HistoryMode) -> StreamingHistory[T]:
        """Ripristina lo storico rifiutando cambi di modalità tra due segmenti."""
        mode = str(state.get("mode", ""))
        if mode != expected_mode:
            raise ValueError(f"Modalità storico del resume {mode!r}, attesa {expected_mode!r}")
        raw_items = state.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("items dello storico resume deve essere una lista")
        raw_count = state.get("count", 0)
        if not isinstance(raw_count, int):
            raise ValueError("count dello storico resume deve essere intero")
        return cls(
            mode=expected_mode,
            count=raw_count,
            first=cast("T | None", deepcopy(state.get("first"))),
            last=cast("T | None", deepcopy(state.get("last"))),
            _items=deepcopy(raw_items),
        )
