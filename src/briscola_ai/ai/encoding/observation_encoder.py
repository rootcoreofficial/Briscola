"""
Encoder didattico: ObservationDTO -> feature vector + action mask (40 carte).

Contesto
--------
Il backend produce `ObservationDTO` (vedi `backend/dto.py`) che è già una vista
parziale e lecita (anti-cheat). Per un primo modello supervisionato (Behavior
Cloning) vogliamo trasformare questa observation in:
- un vettore di feature numeriche `x`
- una action mask `m` lunga 40 che limita le azioni alle carte realmente in mano

Nota:
Questo encoder è volutamente "semplice ma utile" e pensato per 2-player.
In 4-player funziona parzialmente, ma molte feature (es. avversario unico)
vanno ripensate per il team-play.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...domain.models import Card, Suit
from ...domain.observation import PlayerObservation
from .card_action_space import build_card_features, card_dto_to_action_id


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """
    Output dell'encoder.

    - `features`: lista di float (dimensione fissa)
    - `action_mask`: lista di bool (dimensione 40)
    """

    features: list[float]
    action_mask: list[bool]


_CARD_FEATURES = build_card_features()

# Costanti di “contratto” (didattiche).
#
# L'encoder v1 costruisce un vettore di feature a dimensione fissa:
# - 6 blocchi da 40 (mano onehot/points/strength + tavolo onehot/points/strength) = 240
# - briscola onehot sui 4 semi = 4  -> 244
# - 4 scalari = 4                 -> 248
FEATURE_DIM_2P_V1 = 248
FEATURE_DIM_2P_V2 = FEATURE_DIM_2P_V1 + 40
# v3 = v2 + 22 feature strategiche aggregate (vedi `_compute_v3_extra_features`).
FEATURE_DIM_2P_V3 = FEATURE_DIM_2P_V2 + 22
# v4 = v3 + 59 feature dalla storia ORDINATA/ATTRIBUITA delle prese (vedi `_compute_v4_extra_features`):
# 11 aggregate sul comportamento dell'avversario + 12 x 4 sulle ultime prese.
FEATURE_DIM_2P_V4 = FEATURE_DIM_2P_V3 + 59
ACTION_DIM = 40

EncoderVersion = Literal["v1", "v2", "v3", "v4"]

# Numero di feature aggiunte da v3 sopra v2 (contratto esplicito per i test).
_V3_EXTRA_DIM = 22

# Contratto v4 (esplicito per i test): quante prese recenti encodiamo e con quante feature l'una.
_V4_LAST_TRICKS = 4
_V4_TRICK_FEATURES = 12
_V4_BEHAVIOR_FEATURES = 11
_V4_EXTRA_DIM = _V4_BEHAVIOR_FEATURES + _V4_LAST_TRICKS * _V4_TRICK_FEATURES

# --- Indici PUBBLICI per diagnostica (usati da scripts/style_feature_probe.py) ---
# Esposti apposta: permettono di isolare/perturbare blocchi dell'encoder senza toccare i
# simboli privati. Gli offset seguono i layout documentati in `_compute_v3_extra_features`
# (blocco v3, base = FEATURE_DIM_2P_V2) e `_compute_v4_extra_features` (blocco v4, base = V3).
#
# Blocco extra v4 (opponent modeling + storia prese): [FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4).
V4_EXTRA_SLICE = slice(FEATURE_DIM_2P_V3, FEATURE_DIM_2P_V4)
# Contatori aggregati di STILE dentro il blocco v4:
V4_IDX_OPP_TRUMP_RESPONSE = FEATURE_DIM_2P_V3 + 5  # "tagli": risposte di briscola a lead non-briscola
V4_IDX_OPP_LEAD_LOAD = FEATURE_DIM_2P_V3 + 9  # aperture di carico (asso/tre) dell'avversario
# Feature di FASE dentro il blocco v3:
V3_IDX_DECK_SIZE = FEATURE_DIM_2P_V2 + 16
V3_IDX_HAND_SIZE = FEATURE_DIM_2P_V2 + 17
V3_IDX_IS_ENDGAME = FEATURE_DIM_2P_V2 + 18


def _one_hot_suit(suit: str | None) -> list[float]:
    """One-hot sui 4 semi (clubs, cups, coins, swords)."""
    out = [0.0, 0.0, 0.0, 0.0]
    if suit is None:
        return out
    idx = {"clubs": 0, "cups": 1, "coins": 2, "swords": 3}.get(suit)
    if idx is None:
        return out
    out[idx] = 1.0
    return out


def _card_to_action_id_fast(card: Card) -> int:
    """
    Converte una `Card` dominio in action id senza passare da DTO/dict.

    Questo helper è usato nel path caldo training/evaluation. Mantiene la stessa convenzione
    dello spazio azioni canonico: `suit_index * 10 + (number - 1)`.
    """
    if card.suit is Suit.CLUBS:
        suit_index = 0
    elif card.suit is Suit.CUPS:
        suit_index = 1
    elif card.suit is Suit.COINS:
        suit_index = 2
    elif card.suit is Suit.SWORDS:
        suit_index = 3
    else:
        raise ValueError(f"Seme non supportato: {card.suit!r}")
    return suit_index * 10 + (int(card.rank.number) - 1)


def _one_hot_suit_from_card(card: Card | None) -> list[float]:
    """One-hot sui 4 semi partendo direttamente da una carta dominio."""
    out = [0.0, 0.0, 0.0, 0.0]
    if card is None:
        return out
    if card.suit is Suit.CLUBS:
        out[0] = 1.0
    elif card.suit is Suit.CUPS:
        out[1] = 1.0
    elif card.suit is Suit.COINS:
        out[2] = 1.0
    elif card.suit is Suit.SWORDS:
        out[3] = 1.0
    return out


_SUIT_STR_TO_INDEX = {"clubs": 0, "cups": 1, "coins": 2, "swords": 3}


def _suit_str_to_index(suit: str | None) -> int | None:
    """Indice seme (clubs=0, cups=1, coins=2, swords=3) o None se assente/sconosciuto."""
    if not isinstance(suit, str):
        return None
    return _SUIT_STR_TO_INDEX.get(suit)


def _suit_to_index(suit: Suit | None) -> int | None:
    """Indice seme da `Suit` (stesso ordine dello spazio azioni)."""
    if suit is Suit.CLUBS:
        return 0
    if suit is Suit.CUPS:
        return 1
    if suit is Suit.COINS:
        return 2
    if suit is Suit.SWORDS:
        return 3
    return None


def _onehot_to_id_set(raw: object) -> set[int]:
    """
    Converte una one-hot (list/tuple di 0/1, lunghezza 40) in set di card id.

    Tollerante per backward compatibility: `None` o lista vuota -> set vuoto (dataset vecchi senza
    il campo). Una lunghezza diversa da 0/40 o valori non binari sono invece un errore esplicito.
    """
    if raw is None:
        return set()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("one-hot deve essere list/tuple")
    if len(raw) == 0:
        return set()
    if len(raw) != 40:
        raise ValueError(f"one-hot len={len(raw)} (atteso 40)")
    ids: set[int] = set()
    for i, value in enumerate(raw):
        if isinstance(value, bool) or isinstance(value, (int, float)) and value in (0, 1):
            iv = int(value)
        else:
            raise ValueError("one-hot deve contenere solo 0/1")
        if iv:
            ids.add(i)
    return ids


def _compute_v3_extra_features(
    *,
    my_hand_ids: set[int],
    seen_ids: set[int],
    out_of_play_ids: set[int],
    trump_suit_index: int | None,
    table_action_ids: list[int],
    deck_size: int,
    my_hand_size: int,
) -> list[float]:
    """
    Calcola il blocco v3 (22 feature) condiviso dai path dict e oggetto (parità garantita).

    Definizione "ignota" (anti-cheat): `unknown = not seen and not in_my_hand`. Poiché la briscola
    scoperta è SEMPRE in `seen` (anche quando è nel mazzo/in mano), questa formula esclude
    correttamente la briscola pubblica dalle carte ignote, senza bisogno di conoscerne l'id.
    Le feature `*_out_of_play` usano invece `out_of_play` (solo prese + tavolo).

    Layout (22): [unknown_trumps_norm, unk_high_ace, unk_high_three, unk_high_king]
    + per seme(×4) [ace_out_of_play, three_out_of_play, unknown_load_norm]
    + [deck_size_norm, my_hand_size_norm, is_endgame]
    + [current_trick_points_norm, current_trick_lead_strength_norm, current_trick_lead_is_trump].
    """
    points_by_id = _CARD_FEATURES.points_by_action_id
    strength_by_id = _CARD_FEATURES.strength_by_action_id
    unknown_ids = set(range(40)) - seen_ids - my_hand_ids

    # Blocco briscole (Asso=offset 0, Tre=offset 2, Re=offset 9 dentro il seme).
    if trump_suit_index is None:
        out: list[float] = [0.0, 0.0, 0.0, 0.0]
    else:
        base = trump_suit_index * 10
        unknown_trumps = sum(1 for cid in range(base, base + 10) if cid in unknown_ids)
        out = [
            float(unknown_trumps) / 10.0,
            1.0 if (base + 0) in unknown_ids else 0.0,
            1.0 if (base + 2) in unknown_ids else 0.0,
            1.0 if (base + 9) in unknown_ids else 0.0,
        ]

    # Per seme: assi/tre usciti (out_of_play) + carichi (Asso/Tre) ignoti.
    for suit_index in range(4):
        ace_id = suit_index * 10 + 0
        three_id = suit_index * 10 + 2
        load_unknown = (1 if ace_id in unknown_ids else 0) + (1 if three_id in unknown_ids else 0)
        out += [
            1.0 if ace_id in out_of_play_ids else 0.0,
            1.0 if three_id in out_of_play_ids else 0.0,
            float(load_unknown) / 2.0,
        ]

    # Fase partita.
    out += [float(deck_size) / 40.0, float(my_hand_size) / 3.0, 1.0 if deck_size == 0 else 0.0]

    # Presa corrente (lead = prima carta giocata sul tavolo; in 2-player al più 1 prima della mossa).
    if table_action_ids:
        lead_id = table_action_ids[0]
        trick_points = sum(float(points_by_id[a]) for a in table_action_ids)
        lead_strength = float(strength_by_id[lead_id])
        lead_is_trump = 1.0 if (trump_suit_index is not None and (lead_id // 10) == trump_suit_index) else 0.0
    else:
        trick_points = 0.0
        lead_strength = 0.0
        lead_is_trump = 0.0
    out += [trick_points / 11.0, lead_strength / 10.0, lead_is_trump]
    return out


def _compute_v4_extra_features(
    *,
    tricks: list[tuple[list[tuple[int, int]], int, int]],
    my_index: int,
    trump_suit_index: int | None,
) -> list[float]:
    """
    Feature v4: storia ordinata/attribuita delle prese (2-player).

    Input normalizzato (comune al path dict e al path `PlayerObservation`):
    `tricks` è la lista cronologica delle prese completate, ognuna
    `(plays, winner_index, points)` con `plays = [(action_id, player_index), ...]`
    nell'ordine di gioco (in 2p: esattamente lead e risposta).

    Perché queste feature (opponent modeling comportamentale):
    l'aggregato `seen`/`out_of_play` dice COSA è uscito, ma non CHI l'ha giocato né COME.
    Qui encodiamo il comportamento dell'avversario (semi giocati, tagli, risposte a seme,
    scarti, lead di briscola/carichi) e il dettaglio delle ultime `_V4_LAST_TRICKS` prese.
    Tutto deriva da informazione pubblica (carte giocate a tavolo scoperto): anti-cheat
    per costruzione.

    Blocco A — comportamento avversario aggregato (11):
      1-4  carte giocate dall'avversario per seme, /10
      5    prese in cui l'avversario era di mano (lead), /10
      6    risposte dell'avversario DI BRISCOLA a un lead non-briscola ("tagli"), /10
      7    risposte dell'avversario nello stesso seme del lead, /10
      8    risposte dell'avversario né a seme né briscola ("scarti"), /10
      9    lead di briscola dell'avversario, /10
      10   lead "carichi" (asso/tre) dell'avversario, /10
      11   prese completate, /20

    Blocco B — ultime `_V4_LAST_TRICKS` prese, dalla più recente (12 l'una, zero-padded):
      1    slot presente
      2    ero io di mano (lead)
      3-6  seme del lead (one-hot)
      7    forza del lead, /10
      8    punti del lead, /11
      9    la risposta ha seguito il seme del lead
      10   la risposta era briscola
      11   ho vinto io la presa
      12   punti della presa, /21
    """
    opp_index = 1 - int(my_index)

    opp_played_per_suit = [0, 0, 0, 0]
    opp_lead_count = 0
    opp_trump_response_count = 0
    opp_followed_suit_count = 0
    opp_discard_count = 0
    opp_lead_trump_count = 0
    opp_lead_load_count = 0

    for plays, _winner_index, _points in tricks:
        if len(plays) != 2:
            raise ValueError(f"Presa 2-player invalida nella storia: attese 2 giocate, trovate {len(plays)}")
        (lead_id, lead_player), (resp_id, _resp_player) = plays
        lead_suit = lead_id // 10
        resp_suit = resp_id // 10

        # Conteggio per seme delle carte giocate dall'avversario (lead o risposta).
        for action_id, player_index in plays:
            if player_index == opp_index:
                opp_played_per_suit[action_id // 10] += 1

        if lead_player == opp_index:
            opp_lead_count += 1
            if trump_suit_index is not None and lead_suit == trump_suit_index:
                opp_lead_trump_count += 1
            if _CARD_FEATURES.points_by_action_id[lead_id] >= 10:  # asso (11) o tre (10)
                opp_lead_load_count += 1
        else:
            # L'avversario ha risposto al mio lead.
            lead_is_trump = trump_suit_index is not None and lead_suit == trump_suit_index
            resp_is_trump = trump_suit_index is not None and resp_suit == trump_suit_index
            if resp_suit == lead_suit:
                opp_followed_suit_count += 1
            elif resp_is_trump and not lead_is_trump:
                opp_trump_response_count += 1
            else:
                opp_discard_count += 1

    behavior: list[float] = [
        opp_played_per_suit[0] / 10.0,
        opp_played_per_suit[1] / 10.0,
        opp_played_per_suit[2] / 10.0,
        opp_played_per_suit[3] / 10.0,
        opp_lead_count / 10.0,
        opp_trump_response_count / 10.0,
        opp_followed_suit_count / 10.0,
        opp_discard_count / 10.0,
        opp_lead_trump_count / 10.0,
        opp_lead_load_count / 10.0,
        len(tricks) / 20.0,
    ]

    recent: list[float] = []
    for slot in range(_V4_LAST_TRICKS):
        idx = len(tricks) - 1 - slot  # dalla più recente
        if idx < 0:
            recent.extend([0.0] * _V4_TRICK_FEATURES)
            continue
        plays, winner_index, points = tricks[idx]
        (lead_id, lead_player), (resp_id, _resp_player) = plays
        lead_suit = lead_id // 10
        resp_suit = resp_id // 10
        lead_suit_onehot = [0.0, 0.0, 0.0, 0.0]
        lead_suit_onehot[lead_suit] = 1.0
        recent.extend(
            [
                1.0,
                1.0 if lead_player == my_index else 0.0,
                *lead_suit_onehot,
                float(_CARD_FEATURES.strength_by_action_id[lead_id]) / 10.0,
                float(_CARD_FEATURES.points_by_action_id[lead_id]) / 11.0,
                1.0 if resp_suit == lead_suit else 0.0,
                1.0 if (trump_suit_index is not None and resp_suit == trump_suit_index) else 0.0,
                1.0 if winner_index == my_index else 0.0,
                float(points) / 21.0,
            ]
        )

    return behavior + recent


def _tricks_from_dto(raw: object) -> list[tuple[list[tuple[int, int]], int, int]]:
    """
    Normalizza `ObservationDTO.trick_history` (lista di dict) nel formato comune v4.

    Tollerante per backward compatibility: `None`/lista vuota -> storia vuota (dataset e
    payload precedenti a v4 non hanno il campo: le feature v4 degradano a zero, come
    documentato per `seen`/`out_of_play` in v2/v3).
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("trick_history deve essere una lista")
    tricks: list[tuple[list[tuple[int, int]], int, int]] = []
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("trick_history: record invalido (atteso dict)")
        plays: list[tuple[int, int]] = []
        for item in record.get("cards") or []:
            if not (isinstance(item, dict) and isinstance(item.get("card"), dict)):
                raise ValueError("trick_history: carta invalida nel record")
            plays.append((card_dto_to_action_id(item["card"]), int(item["player_index"])))
        tricks.append((plays, int(record["winner_index"]), int(record["points"])))
    return tricks


def encode_observation_2p(observation: dict) -> EncodedObservation:
    """
    Encoda una `ObservationDTO` (come dict JSON) in feature+mask.

    Feature (ordine, didattico):
    - my_hand_onehot[40]
    - my_hand_points[40]     (onehot * punti carta)
    - my_hand_strength[40]   (onehot * forza carta)
    - table_onehot[40]
    - table_points[40]
    - table_strength[40]
    - trump_suit_onehot[4]
    - scalari: deck_size/40, my_points/120, opp_points/120, is_second_in_trick
    """
    if not isinstance(observation, dict):
        raise TypeError("ObservationDTO attesa come dict")

    if observation.get("num_players") != 2:
        raise ValueError("Questo encoder è pensato per 2-player (num_players=2).")

    my_index = observation.get("my_index")
    if not isinstance(my_index, int):
        raise ValueError("ObservationDTO invalida: my_index mancante/non int")

    my_hand = observation.get("my_hand") or []
    if not isinstance(my_hand, list):
        raise ValueError("ObservationDTO invalida: my_hand non list")

    # Action mask e feature della mano (40 carte).
    mask = [False] * 40
    hand_onehot = [0.0] * 40
    for card in my_hand:
        action_id = card_dto_to_action_id(card)
        mask[action_id] = True
        hand_onehot[action_id] = 1.0

    hand_points = [hand_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    hand_strength = [hand_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    # Tavolo: carte pubbliche (al massimo 1 prima della nostra azione, in 2-player).
    table_cards = observation.get("table_cards") or []
    if not isinstance(table_cards, list):
        raise ValueError("ObservationDTO invalida: table_cards non list")

    table_onehot = [0.0] * 40
    for item in table_cards:
        if not isinstance(item, dict):
            continue
        card = item.get("card")
        if not isinstance(card, dict):
            continue
        action_id = card_dto_to_action_id(card)
        table_onehot[action_id] = 1.0

    table_points = [table_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    table_strength = [table_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    # Trump suit (pubblico).
    trump_suit = observation.get("trump_suit")
    trump_onehot = _one_hot_suit(trump_suit if isinstance(trump_suit, str) else None)

    # Scalari "stato partita".
    deck_size = observation.get("cards_remaining_in_deck")
    my_points = observation.get("my_points")
    if not isinstance(deck_size, int) or not isinstance(my_points, int):
        raise ValueError("ObservationDTO invalida: cards_remaining_in_deck/my_points mancanti")

    players = observation.get("players") or []
    opp_points = 0
    if isinstance(players, list):
        for p in players:
            if isinstance(p, dict) and p.get("index") != my_index:
                opp_points = int(p.get("points", 0))
                break

    is_second_in_trick = 1.0 if (observation.get("my_turn") is True and len(table_cards) == 1) else 0.0

    features: list[float] = (
        hand_onehot
        + hand_points
        + hand_strength
        + table_onehot
        + table_points
        + table_strength
        + trump_onehot
        + [
            float(deck_size) / 40.0,
            float(my_points) / 120.0,
            float(opp_points) / 120.0,
            is_second_in_trick,
        ]
    )

    return EncodedObservation(features=features, action_mask=mask)


def _seen_cards_onehot_to_floats(raw: object) -> list[float]:
    """
    Normalizza `seen_cards_onehot` in una lista di float lunga 40.

    Per compatibilità:
    - se il campo manca/None, ritorniamo 40 zeri (utile per dataset/modelli vecchi).
    - accettiamo `int`/`bool` e convertiamo in 0.0/1.0.
    """
    if raw is None:
        return [0.0] * 40

    if not isinstance(raw, list):
        raise ValueError("ObservationDTO invalida: seen_cards_onehot non list")

    if len(raw) != 40:
        raise ValueError(f"ObservationDTO invalida: seen_cards_onehot len={len(raw)} (atteso 40)")

    out: list[float] = []
    for v in raw:
        if isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        elif isinstance(v, int):
            if v not in (0, 1):
                raise ValueError("ObservationDTO invalida: seen_cards_onehot deve contenere solo 0/1")
            out.append(float(v))
        else:
            raise ValueError("ObservationDTO invalida: seen_cards_onehot deve contenere int/bool")
    return out


def encode_observation_2p_v2(observation: dict) -> EncodedObservation:
    """
    Encoder 2-player v2 = v1 + storia pubblica (card counting lecito).

    Aggiungiamo in coda al vettore v1:
    - `seen_cards_onehot[40]`: 1 se la carta è già stata vista/giocata nella partita.

    Perché è anti-cheat?
    - Sono **solo** carte pubbliche: briscola scoperta + carte sul tavolo + carte già uscite.
    - Non include l'ordine del mazzo né la mano avversaria.

    Compatibilità:
    - Se `seen_cards_onehot` manca, viene trattato come 40 zeri (utile per dataset vecchi).
      Per training v2 “serio” è comunque consigliato che il backend lo popoli sempre.
    """
    base = encode_observation_2p(observation)
    seen = _seen_cards_onehot_to_floats(observation.get("seen_cards_onehot"))
    features = list(base.features) + seen
    return EncodedObservation(features=features, action_mask=base.action_mask)


def encode_observation_2p_v3(observation: dict) -> EncodedObservation:
    """
    Encoder 2-player v3 = v2 + 22 feature strategiche aggregate (vedi `_compute_v3_extra_features`).

    Usa `seen_cards_onehot` (per le carte "ignote", che escludono la briscola pubblica) e
    `out_of_play_cards_onehot` (per assi/tre usciti). Su dataset vecchi senza `out_of_play` le
    feature relative degradano a 0: per un training v3 "serio" serve un re-export aggiornato.
    """
    base = encode_observation_2p_v2(observation)

    my_index = observation.get("my_index")
    my_hand = observation.get("my_hand") or []
    my_hand_ids = {card_dto_to_action_id(card) for card in my_hand if isinstance(card, dict)}

    table_cards = observation.get("table_cards") or []
    table_action_ids: list[int] = []
    for item in table_cards:
        if isinstance(item, dict) and isinstance(item.get("card"), dict):
            table_action_ids.append(card_dto_to_action_id(item["card"]))

    deck_size = observation.get("cards_remaining_in_deck")
    if not isinstance(deck_size, int):
        raise ValueError("ObservationDTO invalida: cards_remaining_in_deck mancante")

    trump_suit = observation.get("trump_suit")
    extra = _compute_v3_extra_features(
        my_hand_ids=my_hand_ids,
        seen_ids=_onehot_to_id_set(observation.get("seen_cards_onehot")),
        out_of_play_ids=_onehot_to_id_set(observation.get("out_of_play_cards_onehot")),
        trump_suit_index=_suit_str_to_index(trump_suit if isinstance(trump_suit, str) else None),
        table_action_ids=table_action_ids,
        deck_size=deck_size,
        my_hand_size=len(my_hand_ids) if isinstance(my_index, int) else len(my_hand),
    )
    return EncodedObservation(features=list(base.features) + extra, action_mask=base.action_mask)


def encode_observation_2p_v4(observation: dict) -> EncodedObservation:
    """
    Encoder 2-player v4 = v3 + storia ordinata/attribuita delle prese.

    Usa `trick_history` (vedi `TrickRecordDTO`): comportamento aggregato dell'avversario
    e dettaglio delle ultime prese. Su dataset/payload precedenti senza `trick_history`
    le feature v4 degradano a zero: per un training v4 "serio" serve un re-export.
    """
    base = encode_observation_2p_v3(observation)

    my_index = observation.get("my_index")
    if not isinstance(my_index, int):
        raise ValueError("ObservationDTO invalida: my_index mancante")
    trump_suit = observation.get("trump_suit")
    extra = _compute_v4_extra_features(
        tricks=_tricks_from_dto(observation.get("trick_history")),
        my_index=my_index,
        trump_suit_index=_suit_str_to_index(trump_suit if isinstance(trump_suit, str) else None),
    )
    return EncodedObservation(features=list(base.features) + extra, action_mask=base.action_mask)


def encode_observation_2p_with_version(observation: dict, *, version: EncoderVersion) -> EncodedObservation:
    """Selettore esplicito dell'encoder 2-player (v1/v2/v3/v4)."""
    if version == "v1":
        return encode_observation_2p(observation)
    if version == "v2":
        return encode_observation_2p_v2(observation)
    if version == "v3":
        return encode_observation_2p_v3(observation)
    if version == "v4":
        return encode_observation_2p_v4(observation)
    raise ValueError(f"Encoder version non supportata: {version!r}")


def feature_dim_for_encoder_version(version: EncoderVersion) -> int:
    """Ritorna la feature_dim attesa dall'encoder 2-player (v1/v2/v3/v4)."""
    if version == "v1":
        return int(FEATURE_DIM_2P_V1)
    if version == "v2":
        return int(FEATURE_DIM_2P_V2)
    if version == "v3":
        return int(FEATURE_DIM_2P_V3)
    if version == "v4":
        return int(FEATURE_DIM_2P_V4)
    raise ValueError(f"Encoder version non supportata: {version!r}")


def encode_player_observation_2p(
    observation: PlayerObservation, *, version: EncoderVersion = "v1"
) -> EncodedObservation:
    """
    Encoda una `PlayerObservation` (dominio) in feature+mask (2-player).

    Per coerenza con il training BC, convertiamo la `PlayerObservation` in un dict
    compatibile con `ObservationDTO` e poi riusiamo `encode_observation_2p`.

    Nota:
    - `PlayerObservation` non contiene informazione nascosta (anti-cheat).
    - Usiamo solo i campi necessari all'encoder (mano, tavolo, briscola, punti, deck_size).
    """
    if observation.num_players != 2:
        raise ValueError("Questo encoder è pensato per 2-player (num_players=2).")

    my_index = int(observation.player_index)
    if my_index < 0 or my_index >= observation.num_players:
        raise ValueError(f"player_index fuori range: {my_index} (num_players={observation.num_players})")

    # Action mask e feature mano. Questo duplica intenzionalmente la costruzione di
    # `encode_observation_2p`, evitando però la conversione PlayerObservation -> dict DTO
    # nel path caldo training/evaluation.
    mask = [False] * 40
    hand_onehot = [0.0] * 40
    for card in observation.hand:
        action_id = _card_to_action_id_fast(card)
        mask[action_id] = True
        hand_onehot[action_id] = 1.0

    hand_points = [hand_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    hand_strength = [hand_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    table_onehot = [0.0] * 40
    for card, _ in observation.table_cards:
        table_onehot[_card_to_action_id_fast(card)] = 1.0

    table_points = [table_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    table_strength = [table_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    if len(observation.players_points) != observation.num_players:
        raise ValueError("PlayerObservation invalida: players_points non coerente con num_players")

    opp_index = 1 - my_index
    is_second_in_trick = (
        1.0
        if (observation.current_turn == my_index and (not observation.game_over) and len(observation.table_cards) == 1)
        else 0.0
    )

    features: list[float] = (
        hand_onehot
        + hand_points
        + hand_strength
        + table_onehot
        + table_points
        + table_strength
        + _one_hot_suit_from_card(observation.trump_card)
        + [
            float(observation.deck_size) / 40.0,
            float(observation.players_points[my_index]) / 120.0,
            float(observation.players_points[opp_index]) / 120.0,
            is_second_in_trick,
        ]
    )

    if version == "v1":
        return EncodedObservation(features=features, action_mask=mask)

    seen = _seen_cards_onehot_to_floats(list(observation.seen_cards_onehot))
    if version == "v2":
        return EncodedObservation(features=features + seen, action_mask=mask)
    if version in ("v3", "v4"):
        extra = _compute_v3_extra_features(
            my_hand_ids={_card_to_action_id_fast(card) for card in observation.hand},
            seen_ids=_onehot_to_id_set(list(observation.seen_cards_onehot)),
            out_of_play_ids=_onehot_to_id_set(list(observation.out_of_play_cards_onehot)),
            trump_suit_index=_suit_to_index(observation.trump_card.suit if observation.trump_card else None),
            table_action_ids=[_card_to_action_id_fast(card) for card, _ in observation.table_cards],
            deck_size=int(observation.deck_size),
            my_hand_size=len(observation.hand),
        )
        if version == "v3":
            return EncodedObservation(features=features + seen + extra, action_mask=mask)

        # v4: normalizza la storia dal dominio (TrickRecord -> action id) e appende il blocco.
        tricks = [
            (
                [(_card_to_action_id_fast(card), int(player_idx)) for card, player_idx in record.cards],
                int(record.winner_index),
                int(record.points),
            )
            for record in observation.trick_history
        ]
        extra_v4 = _compute_v4_extra_features(
            tricks=tricks,
            my_index=my_index,
            trump_suit_index=_suit_to_index(observation.trump_card.suit if observation.trump_card else None),
        )
        return EncodedObservation(features=features + seen + extra + extra_v4, action_mask=mask)
    raise ValueError(f"Encoder version non supportata: {version!r}")
