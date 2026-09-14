"""
Test della diagnostica di equivarianza alla rinomina dei semi.

La proprieta' centrale e' semantica: una permutazione deve rinominare insieme tutte le
carte pubbliche e private dell'osservatore, le one-hot e la storia, senza cambiare
giocatori, punteggi o ordine delle giocate. I test sul modello distinguono inoltre una
policy davvero equivariant da una che assegna un significato assoluto ai nomi dei semi.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4, encode_player_observation_2p
from briscola_ai.ai.evaluation.suit_symmetry import (
    IDENTITY_SUIT_PERMUTATION,
    all_suit_permutations,
    evaluate_observation_suit_symmetry,
    inverse_suit_permutation,
    jensen_shannon_divergence_bits,
    masked_softmax,
    permute_action_vector,
    permute_card,
    permute_card_id,
    permute_player_observation,
    validate_suit_permutation,
)
from briscola_ai.ai.models.bc_model import BCModelAgent, MLPBCModel
from briscola_ai.domain.card_id import SUIT_ORDER, card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation
from briscola_ai.domain.state import TrickRecord


def _three_cycle() -> tuple[Suit, Suit, Suit, Suit]:
    """Ciclo clubs -> cups -> coins -> clubs, con swords fissa."""
    return (Suit.CUPS, Suit.COINS, Suit.CLUBS, Suit.SWORDS)


def _complete_observation() -> PlayerObservation:
    """Osservazione v4 non banale: esercita ogni campo che contiene carte."""
    hand = (
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.SWORDS, Rank.KING),
    )
    trump_card = Card(Suit.COINS, Rank.SEVEN)
    table_cards = ((Card(Suit.CUPS, Rank.TWO), 1),)
    trick_history = (
        TrickRecord(
            cards=((Card(Suit.SWORDS, Rank.TWO), 0), (Card(Suit.CLUBS, Rank.THREE), 1)),
            winner_index=0,
            points=10,
        ),
        TrickRecord(
            cards=((Card(Suit.COINS, Rank.FOUR), 1), (Card(Suit.CUPS, Rank.KING), 0)),
            winner_index=1,
            points=4,
        ),
    )

    seen = [0] * 40
    out_of_play = [0] * 40
    public_cards = [
        trump_card,
        table_cards[0][0],
        *(card for record in trick_history for card, _ in record.cards),
    ]
    for card in public_cards:
        seen[card_to_id(card)] = 1
    for card in public_cards[1:]:  # la briscola scoperta e' vista, ma resta in gioco
        out_of_play[card_to_id(card)] = 1

    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="P0",
        hand=hand,
        trump_card=trump_card,
        deck_size=14,
        table_cards=table_cards,
        current_turn=0,
        first_player=1,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(10, 4),
        players_hand_sizes=(3, 3),
        seen_cards_onehot=tuple(seen),
        out_of_play_cards_onehot=tuple(out_of_play),
        trick_history=trick_history,
    )


def _synthetic_agent(*, suit_bias: bool) -> BCModelAgent:
    """
    Costruisce una MLP sintetica la cui policy dipende soltanto dal rango o dal seme.

    Le feature entrano con pesi nulli: la mask derivata dalla mano seleziona le azioni
    legali, mentre `b2` rende esplicita la simmetria che vogliamo verificare.
    """
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden_dim = 2
    if suit_bias:
        # Il contributo dominante e' l'indice assoluto del seme: una rinomina deve quindi
        # produrre un segnale diagnostico. Il termine di rango evita pareggi accidentali.
        action_bias = np.asarray(
            [2.0 * (action_id // 10) + 0.01 * (action_id % 10) for action_id in range(40)],
            dtype=np.float32,
        )
    else:
        # Stesso score per lo stesso rango in tutti i semi: equivarianza esatta.
        action_bias = np.asarray([float(action_id % 10) for action_id in range(40)], dtype=np.float32)

    model = MLPBCModel(
        w1=np.zeros((feature_dim, hidden_dim), dtype=np.float32),
        b1=np.zeros((hidden_dim,), dtype=np.float32),
        w2=np.zeros((hidden_dim, 40), dtype=np.float32),
        b2=action_bias,
        metadata={"format": "mlp_bc_v1", "feature_dim": feature_dim, "encoder_version": "v4"},
    )
    return BCModelAgent(
        model=model,
        model_path=Path("synthetic_v4.npz"),
        encoder_version="v4",
        overkill_guard_enabled=False,
    )


def test_all_suit_permutations_are_the_24_bijections_with_identity_first() -> None:
    """Le permutazioni devono essere esaustive, uniche e deterministiche."""
    permutations = all_suit_permutations()

    assert len(permutations) == 24
    assert len(set(permutations)) == 24
    assert permutations[0] == IDENTITY_SUIT_PERMUTATION == SUIT_ORDER
    assert all(set(permutation) == set(SUIT_ORDER) for permutation in permutations)


def test_inverse_permutation_roundtrip_including_three_cycle() -> None:
    """L'inversa deve annullare anche cicli non riducibili a un semplice swap."""
    cycle = _three_cycle()
    assert inverse_suit_permutation(cycle) == (Suit.COINS, Suit.CLUBS, Suit.CUPS, Suit.SWORDS)

    for permutation in all_suit_permutations():
        inverse = inverse_suit_permutation(permutation)
        assert inverse_suit_permutation(inverse) == permutation
        for source_index, source_suit in enumerate(SUIT_ORDER):
            target_suit = permutation[source_index]
            assert inverse[SUIT_ORDER.index(target_suit)] == source_suit


def test_permutation_validation_rejects_non_bijections_and_invalid_values() -> None:
    """Errori di configurazione devono fallire prima di produrre confronti ambigui."""
    assert validate_suit_permutation(list(SUIT_ORDER)) == SUIT_ORDER

    with pytest.raises(ValueError, match="lunghezza"):
        validate_suit_permutation(SUIT_ORDER[:3])
    with pytest.raises(ValueError, match="biiezione"):
        validate_suit_permutation((Suit.CLUBS, Suit.CLUBS, Suit.COINS, Suit.SWORDS))
    with pytest.raises(ValueError, match="valori Suit"):
        validate_suit_permutation((Suit.CLUBS, Suit.CUPS, Suit.COINS, "swords"))  # type: ignore[arg-type]


def test_all_card_ids_preserve_rank_and_roundtrip_for_every_permutation() -> None:
    """La rinomina cambia soltanto il blocco-seme dei 40 action id."""
    for permutation in all_suit_permutations():
        inverse = inverse_suit_permutation(permutation)
        mapped_ids = [permute_card_id(card_id, permutation) for card_id in range(40)]
        assert len(set(mapped_ids)) == 40
        for card_id, mapped_id in enumerate(mapped_ids):
            assert mapped_id % 10 == card_id % 10
            assert permute_card_id(mapped_id, inverse) == card_id

    with pytest.raises(ValueError, match="fuori range"):
        permute_card_id(-1, IDENTITY_SUIT_PERMUTATION)
    with pytest.raises(ValueError, match="fuori range"):
        permute_card_id(40, IDENTITY_SUIT_PERMUTATION)


def test_action_vector_push_forward_has_unambiguous_direction_and_roundtrip() -> None:
    """`out[permute(old)] = original[old]`: questo test intercetta l'uso accidentale dell'inversa."""
    permutation = _three_cycle()
    inverse = inverse_suit_permutation(permutation)
    values = tuple(range(40))

    mapped = permute_action_vector(values, permutation)
    for old_id, value in enumerate(values):
        assert mapped[permute_card_id(old_id, permutation)] == value
    assert permute_action_vector(mapped, inverse) == values

    mask = tuple(card_id in {0, 12, 39} for card_id in range(40))
    mapped_mask = permute_action_vector(mask, permutation)
    assert {i for i, enabled in enumerate(mapped_mask) if enabled} == {
        permute_card_id(card_id, permutation) for card_id in (0, 12, 39)
    }

    with pytest.raises(ValueError, match="lunghezza"):
        permute_action_vector(values[:-1], permutation)


def test_complete_observation_roundtrip_preserves_non_card_fields() -> None:
    """Mano, tavolo, one-hot e storia cambiano seme; il resto dell'osservazione no."""
    observation = _complete_observation()
    invariant_fields = (
        "num_players",
        "is_team_game",
        "teams",
        "player_index",
        "player_name",
        "deck_size",
        "current_turn",
        "first_player",
        "game_over",
        "winner_index",
        "winning_team",
        "players_points",
        "players_hand_sizes",
    )

    for permutation in all_suit_permutations():
        transformed = permute_player_observation(observation, permutation)
        restored = permute_player_observation(transformed, inverse_suit_permutation(permutation))

        assert restored == observation
        for field_name in invariant_fields:
            assert getattr(transformed, field_name) == getattr(observation, field_name)
        assert transformed.hand == tuple(permute_card(card, permutation) for card in observation.hand)
        assert transformed.trump_card == permute_card(observation.trump_card, permutation)
        assert transformed.table_cards == tuple(
            (permute_card(card, permutation), player_index) for card, player_index in observation.table_cards
        )
        for old_record, new_record in zip(observation.trick_history, transformed.trick_history, strict=True):
            assert new_record.winner_index == old_record.winner_index
            assert new_record.points == old_record.points
            assert [player for _, player in new_record.cards] == [player for _, player in old_record.cards]


def test_observation_without_trump_or_history_roundtrips() -> None:
    """I campi opzionali/vuoti non devono richiedere casi speciali al chiamante."""
    observation = replace(
        _complete_observation(),
        trump_card=None,
        table_cards=(),
        seen_cards_onehot=(0,) * 40,
        out_of_play_cards_onehot=(0,) * 40,
        trick_history=(),
    )

    for permutation in all_suit_permutations():
        transformed = permute_player_observation(observation, permutation)
        assert transformed.trump_card is None
        assert transformed.trick_history == ()
        assert permute_player_observation(transformed, inverse_suit_permutation(permutation)) == observation


def test_encoder_v4_action_mask_is_covariant_under_every_permutation() -> None:
    """Le azioni legali devono seguire esattamente la rinomina della mano."""
    observation = _complete_observation()
    baseline = encode_player_observation_2p(observation, version="v4")

    for permutation in all_suit_permutations():
        transformed = permute_player_observation(observation, permutation)
        encoded = encode_player_observation_2p(transformed, version="v4")
        expected_mask = permute_action_vector(baseline.action_mask, permutation)

        assert tuple(encoded.action_mask) == expected_mask
        assert sum(encoded.action_mask) == len(observation.hand)
        assert len(encoded.features) == FEATURE_DIM_2P_V4


def test_masked_softmax_is_stable_and_ignores_illegal_logits() -> None:
    """Un logit illegale enorme, o non finito, non deve contaminare la distribuzione legale."""
    logits = np.zeros(40, dtype=np.float64)
    logits[0] = np.nan  # illegale: viene ignorato prima di ogni operazione numerica
    logits[7] = 10_000.0
    logits[9] = 9_999.0
    mask = np.zeros(40, dtype=bool)
    mask[[7, 9]] = True

    probabilities = masked_softmax(logits, mask)
    expected_best = 1.0 / (1.0 + np.exp(-1.0))
    assert np.all(np.isfinite(probabilities))
    assert float(np.sum(probabilities)) == pytest.approx(1.0)
    assert probabilities[7] == pytest.approx(expected_best)
    assert probabilities[9] == pytest.approx(1.0 - expected_best)
    assert probabilities[~mask].tolist() == [0.0] * 38

    logits[7] = np.nan
    with pytest.raises(ValueError, match="non finiti"):
        masked_softmax(logits, mask)
    with pytest.raises(ValueError, match="mask vuota"):
        masked_softmax(np.zeros(40), np.zeros(40, dtype=bool))


def test_jensen_shannon_is_zero_symmetric_and_one_for_disjoint_distributions() -> None:
    """La JS in bit deve rispettare i casi limite teorici senza produrre NaN sugli zeri."""
    p = np.zeros(40, dtype=np.float64)
    q = np.zeros(40, dtype=np.float64)
    p[[1, 3]] = (0.25, 0.75)
    q[[1, 3]] = (0.60, 0.40)

    assert jensen_shannon_divergence_bits(p, p) == pytest.approx(0.0, abs=1e-15)
    assert jensen_shannon_divergence_bits(p, q) == pytest.approx(
        jensen_shannon_divergence_bits(q, p),
        abs=1e-15,
    )

    left = np.zeros(40, dtype=np.float64)
    right = np.zeros(40, dtype=np.float64)
    left[0] = 1.0
    right[39] = 1.0
    assert jensen_shannon_divergence_bits(left, right) == pytest.approx(1.0, abs=1e-15)

    p[1] = np.nan
    with pytest.raises(ValueError, match="non finite"):
        jensen_shannon_divergence_bits(p, q)


def test_rank_only_bc_model_is_equivariant_on_all_24_renamings() -> None:
    """Un modello sintetico condiviso per rango deve avere agreement 100% e JS nulla."""
    result = evaluate_observation_suit_symmetry(_synthetic_agent(suit_bias=False), _complete_observation())

    assert result.baseline_action_id == card_to_id(Card(Suit.SWORDS, Rank.KING))
    assert result.baseline_top2_gap > 0.0
    assert len(result.comparisons) == 24
    assert len(result.remapped_probabilities) == 24
    assert all(sum(distribution) == pytest.approx(1.0) for distribution in result.remapped_probabilities)
    assert all(
        distribution == pytest.approx(result.remapped_probabilities[0], abs=1e-15)
        for distribution in result.remapped_probabilities
    )
    assert sum(comparison.is_identity for comparison in result.comparisons) == 1
    assert all(comparison.agreement for comparison in result.comparisons)
    assert all(not comparison.near_tie for comparison in result.comparisons)
    assert all(comparison.remapped_top2_gap > 0.0 for comparison in result.comparisons)
    assert all(comparison.js_divergence_bits == pytest.approx(0.0, abs=1e-15) for comparison in result.comparisons)
    assert all(
        comparison.max_abs_probability_delta == pytest.approx(0.0, abs=1e-15) for comparison in result.comparisons
    )


def test_suit_biased_bc_model_produces_detectable_asymmetry_signal() -> None:
    """Una policy che preferisce nomi di seme assoluti deve essere segnalata dalla sonda."""
    result = evaluate_observation_suit_symmetry(_synthetic_agent(suit_bias=True), _complete_observation())
    non_identity = [comparison for comparison in result.comparisons if not comparison.is_identity]

    assert non_identity
    assert any(not comparison.agreement for comparison in non_identity)
    assert any(comparison.js_divergence_bits > 1e-6 for comparison in non_identity)
    assert any(comparison.max_abs_probability_delta > 1e-6 for comparison in non_identity)


def test_near_tie_checks_each_remapped_distribution_not_only_baseline() -> None:
    """Una rinomina quasi-pari va filtrata anche quando la baseline originale è netta."""
    agent = _synthetic_agent(suit_bias=True)
    assert isinstance(agent.model, MLPBCModel)
    agent.model.b2[:] = np.repeat(np.asarray([3.0, 2.0, 2.0, 1.0], dtype=np.float32), 10)
    observation = replace(
        _complete_observation(),
        hand=(
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CUPS, Rank.ACE),
            Card(Suit.SWORDS, Rank.ACE),
        ),
    )

    result = evaluate_observation_suit_symmetry(agent, observation, near_tie_threshold=0.0)

    assert result.baseline_top2_gap > 0.0
    assert result.comparisons[0].near_tie is False
    assert any(
        comparison.remapped_top2_gap == pytest.approx(0.0, abs=1e-15) and comparison.near_tie
        for comparison in result.comparisons[1:]
    )


@pytest.mark.slow
def test_suit_symmetry_probe_cli_is_reproducible_and_json_safe(tmp_path: Path) -> None:
    """Lo smoke attraversa raccolta, quote, bootstrap e JSON due volte, senza valori NaN."""
    model_path = tmp_path / "synthetic_v4.npz"
    hidden_dim = 4
    np.savez(
        model_path,
        w1=np.zeros((int(FEATURE_DIM_2P_V4), hidden_dim), dtype=np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=np.zeros((hidden_dim, 40), dtype=np.float32),
        b2=np.asarray([float(action_id % 10) for action_id in range(40)], dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "mlp_bc_v1",
                "feature_dim": int(FEATURE_DIM_2P_V4),
                "encoder_version": "v4",
                "inference_overkill_guard": False,
            }
        ),
    )
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "probe_suit_symmetry.py"
    reports = (tmp_path / "first.json", tmp_path / "second.json")

    for output_path in reports:
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--model",
                str(model_path),
                "--seed-count",
                "1",
                "--samples-per-cell",
                "1",
                "--opponents",
                "random",
                "--bootstrap-reps",
                "10",
                "--worst-cases",
                "1",
                "--out-json",
                str(output_path),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    assert reports[0].read_bytes() == reports[1].read_bytes()
    payload = json.loads(reports[0].read_text(encoding="utf-8"), parse_constant=lambda value: pytest.fail(value))
    assert payload["schema"] == "briscola.suit_symmetry_probe.v1"
    assert payload["identity_control"]["passed"] is True
    assert payload["coverage"]["selected_observations"] == 4
    assert payload["overall"]["identity_vs_23"]["comparisons"] == 4 * 23
    assert payload["overall"]["all_276_pairs"]["comparisons"] == 4 * 276
