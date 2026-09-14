"""
Helper privacy per payload destinati al dataset.

Gli snapshot `ObservationDTO` servono sia alla UI sia al training. Per la UI i
nomi dei giocatori sono utili; per il dataset invece sono superflui e possono
contenere input libero dell'utente. Questo modulo produce copie JSON-safe in cui
i nomi vengono sostituiti con etichette deterministicamente derivate dall'indice
del giocatore (`player_0`, `player_1`, ...).
"""

from __future__ import annotations

from typing import Any


def sanitize_dataset_payload(value: Any) -> Any:
    """
    Ritorna una copia del payload senza nomi liberi dei giocatori.

    La funzione è conservativa:
    - conserva la forma dei DTO (`name` resta presente dove previsto);
    - non tocca `client_id`, perché è lo pseudonimo usato per split train/val;
    - non modifica carte, azioni, punteggi, reward o one-hot feature.
    """

    if isinstance(value, list):
        return [sanitize_dataset_payload(item) for item in value]

    if not isinstance(value, dict):
        return value

    cleaned = {str(k): sanitize_dataset_payload(v) for k, v in value.items()}

    index = cleaned.get("index")
    if isinstance(index, int) and isinstance(cleaned.get("name"), str):
        cleaned["name"] = f"player_{index}"

    winner_index = cleaned.get("winner_index")
    if isinstance(winner_index, int) and isinstance(cleaned.get("winner_name"), str):
        cleaned["winner_name"] = f"player_{winner_index}"

    player_names = cleaned.get("player_names")
    if isinstance(player_names, list):
        cleaned["player_names"] = [f"player_{idx}" for idx, _ in enumerate(player_names)]

    return cleaned
