"""Schedule riproducibili e streaming per il training RL a due giocatori.

La schedule separa il campionamento dell'ambiente dall'esecuzione del rollout. In
particolare, la modalità ``paired`` genera due partite adiacenti con lo stesso seed di
mazzo e lo stesso tipo di avversario, assegnando la policy prima alla seat 0 e poi alla
seat 1. Poiché la coppia resta dentro un singolo optimizer update, entrambe le partite
vedono esattamente gli stessi pesi.

I run brevi possono ancora materializzare l'intera schedule con
``build_training_game_schedule``. I run multi-milione usano invece
``TrainingGameScheduleStream``: conserva soltanto RNG, contatore e digest del prefisso,
quindi la memoria non dipende dal numero totale di partite.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from briscola_ai.ai.training.opponent_mix import OpponentMixItem, sample_opponent_name

TrainingScheduleMode = Literal["serial", "paired"]

_DIGEST_DOMAIN = b"briscola.training_game_schedule.v2\n"


@dataclass(frozen=True, slots=True)
class ScheduledTrainingGame:
    """Identità dell'ambiente assegnato a una singola partita di training."""

    ordinal: int
    game_seed: int
    policy_seat: int
    opponent_name: str
    pair_index: int | None


def _digest_record(game: ScheduledTrainingGame) -> bytes:
    """Serializza una riga in modo stabile e indipendente da JSON/Python."""
    pair_index = -1 if game.pair_index is None else game.pair_index
    return f"{game.ordinal}\t{game.game_seed}\t{game.policy_seat}\t{game.opponent_name}\t{pair_index}\n".encode()


def _initial_digest() -> bytes:
    """Stato iniziale domain-separated della catena SHA-256."""
    return hashlib.sha256(_DIGEST_DOMAIN).digest()


def _extend_digest(previous: bytes, game: ScheduledTrainingGame) -> bytes:
    """Estende la catena; a differenza di ``hashlib`` lo stato e' serializzabile."""
    return hashlib.sha256(previous + _digest_record(game)).digest()


def _sample_opponent(
    *,
    default_opponent_name: str,
    opponent_mix: Sequence[OpponentMixItem] | None,
    rng: np.random.Generator,
) -> str:
    """Campiona dal mix oppure restituisce l'unico opponent configurato."""
    if opponent_mix is None:
        return default_opponent_name
    return sample_opponent_name(list(opponent_mix), rng=rng)


class TrainingGameScheduleStream:
    """Generatore O(1) di ambienti con stato e digest riprendibili.

    Il digest e' una catena SHA-256 anziche' l'hash di un buffer crescente: il solo
    valore di 32 byte basta per proseguire dopo un checkpoint. I checkpoint del trainer
    sono sempre allineati agli update; in modalita' ``paired`` questo implica che non
    esiste una mezza coppia da salvare.
    """

    def __init__(
        self,
        *,
        mode: TrainingScheduleMode,
        seat_fair: bool,
        default_opponent_name: str,
        opponent_mix: Sequence[OpponentMixItem] | None,
        rng_game: np.random.Generator,
        rng_opponent: np.random.Generator,
        consumed_games: int = 0,
        digest_hex: str | None = None,
    ) -> None:
        if mode not in ("serial", "paired"):
            raise ValueError(f"Training schedule non supportata: {mode!r}")
        if not default_opponent_name.strip():
            raise ValueError("default_opponent_name non può essere vuoto")
        if consumed_games < 0:
            raise ValueError("consumed_games deve essere >= 0")
        if mode == "paired" and consumed_games % 2 != 0:
            raise ValueError("Una schedule paired può riprendere solo dopo una coppia completa")

        self.mode = mode
        self.seat_fair = bool(seat_fair)
        self.default_opponent_name = default_opponent_name
        self.opponent_mix = opponent_mix
        self.rng_game = rng_game
        self.rng_opponent = rng_opponent
        self.consumed_games = int(consumed_games)
        if digest_hex is None:
            if consumed_games != 0:
                raise ValueError("Un prefisso già consumato richiede digest_hex")
            self._digest = _initial_digest()
        else:
            try:
                digest = bytes.fromhex(digest_hex)
            except ValueError as exc:
                raise ValueError("digest_hex della schedule non valido") from exc
            if len(digest) != hashlib.sha256().digest_size:
                raise ValueError("digest_hex della schedule deve contenere uno SHA-256")
            self._digest = digest

    @property
    def sha256(self) -> str:
        """Digest del prefisso gia' emesso, estendibile dopo il resume."""
        return self._digest.hex()

    def _next_serial(self) -> ScheduledTrainingGame:
        ordinal = self.consumed_games + 1
        return ScheduledTrainingGame(
            ordinal=ordinal,
            game_seed=int(self.rng_game.integers(0, 2**32)),
            policy_seat=(ordinal % 2) if self.seat_fair else 0,
            opponent_name=_sample_opponent(
                default_opponent_name=self.default_opponent_name,
                opponent_mix=self.opponent_mix,
                rng=self.rng_opponent,
            ),
            pair_index=None,
        )

    def _next_pair(self) -> tuple[ScheduledTrainingGame, ScheduledTrainingGame]:
        pair_index = self.consumed_games // 2
        game_seed = int(self.rng_game.integers(0, 2**32))
        opponent_name = _sample_opponent(
            default_opponent_name=self.default_opponent_name,
            opponent_mix=self.opponent_mix,
            rng=self.rng_opponent,
        )
        first_ordinal = self.consumed_games + 1
        return (
            ScheduledTrainingGame(first_ordinal, game_seed, 0, opponent_name, pair_index),
            ScheduledTrainingGame(first_ordinal + 1, game_seed, 1, opponent_name, pair_index),
        )

    def take(self, count: int) -> tuple[ScheduledTrainingGame, ...]:
        """Emette ``count`` righe senza conservare quelle dei batch precedenti."""
        if count <= 0:
            raise ValueError("count deve essere > 0")
        if self.mode == "paired" and count % 2 != 0:
            raise ValueError("Una schedule paired deve essere consumata per coppie complete")

        games: list[ScheduledTrainingGame] = []
        while len(games) < count:
            batch = (self._next_serial(),) if self.mode == "serial" else self._next_pair()
            for game in batch:
                self._digest = _extend_digest(self._digest, game)
                self.consumed_games += 1
                games.append(game)
        return tuple(games)


def build_training_game_schedule(
    *,
    num_games: int,
    update_every: int,
    mode: TrainingScheduleMode,
    seat_fair: bool,
    default_opponent_name: str,
    opponent_mix: Sequence[OpponentMixItem] | None,
    rng_game: np.random.Generator,
    rng_opponent: np.random.Generator,
) -> tuple[ScheduledTrainingGame, ...]:
    """Costruisce l'intera schedule prima che la policy inizi ad aggiornarsi.

    ``serial`` conserva il comportamento storico: un seed e un campionamento opponent
    per partita; ``seat_fair`` alterna soltanto la seat. ``paired`` campiona seed e
    opponent una volta per coppia e richiede update completi, così nessuna coppia viene
    divisa tra due versioni dei pesi o lasciata fuori dall'ultimo update.
    """
    if num_games <= 0:
        raise ValueError("num_games deve essere > 0")
    if update_every <= 0:
        raise ValueError("update_every deve essere > 0")
    if not default_opponent_name.strip():
        raise ValueError("default_opponent_name non può essere vuoto")
    if mode not in ("serial", "paired"):
        raise ValueError(f"Training schedule non supportata: {mode!r}")

    if mode == "paired" and num_games % 2 != 0:
        raise ValueError("La training schedule paired richiede --num-games pari")
    if mode == "paired" and update_every % 2 != 0:
        raise ValueError("La training schedule paired richiede --update-every pari")
    if mode == "paired" and num_games % update_every != 0:
        raise ValueError(
            "La training schedule paired richiede --num-games multiplo di --update-every, "
            "per non lasciare una coppia in un update parziale"
        )

    stream = TrainingGameScheduleStream(
        mode=mode,
        seat_fair=seat_fair,
        default_opponent_name=default_opponent_name,
        opponent_mix=opponent_mix,
        rng_game=rng_game,
        rng_opponent=rng_opponent,
    )
    return stream.take(num_games)


def training_schedule_sha256(schedule: Sequence[ScheduledTrainingGame]) -> str:
    """Hash-chain stabile, uguale al digest prodotto dallo stream."""
    digest = _initial_digest()
    for game in schedule:
        digest = _extend_digest(digest, game)
    return digest.hex()
