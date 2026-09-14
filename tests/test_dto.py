"""
Test per i DTO (Data Transfer Objects) dei messaggi WebSocket.

Verifica che la conversione dal dominio ai DTO funzioni correttamente
e che i DTO producano JSON con la struttura attesa.
"""

from briscola_ai.backend.dto import (
    AiCardRevealDTO,
    CardDTO,
    GameStateDTO,
    ObservationDTO,
    PlayerInfoDTO,
    PlayerStateDTO,
    TableCardDTO,
    TrickResultDTO,
)
from briscola_ai.domain.models import Card, Rank, Suit


class TestCardDTO:
    """Test per CardDTO."""

    def test_from_domain_converts_card_correctly(self) -> None:
        """Verifica che from_domain converta tutti i campi."""
        card = Card(Suit.CUPS, Rank.ACE)
        dto = CardDTO.from_domain(card)

        assert dto.suit == "cups"
        assert dto.rank == "ACE"
        assert dto.number == 1
        assert dto.points == 11

    def test_from_domain_handles_zero_points(self) -> None:
        """Verifica che le carte senza punti siano gestite correttamente."""
        card = Card(Suit.SWORDS, Rank.FOUR)
        dto = CardDTO.from_domain(card)

        assert dto.points == 0
        assert dto.number == 4


class TestTableCardDTO:
    """Test per TableCardDTO."""

    def test_from_domain_creates_nested_structure(self) -> None:
        """Verifica che TableCardDTO contenga CardDTO e player_index."""
        card = Card(Suit.COINS, Rank.KING)
        dto = TableCardDTO.from_domain(card, player_index=1)

        assert dto.player_index == 1
        assert dto.card.suit == "coins"
        assert dto.card.rank == "KING"


class TestObservationDTO:
    """Test per ObservationDTO."""

    def test_has_type_observation(self) -> None:
        """Verifica che il campo type sia sempre 'observation'."""
        dto = ObservationDTO(
            server_version=1,
            my_index=0,
            my_hand=[],
            my_points=0,
            my_turn=True,
            trump_card=None,
            trump_suit=None,
            table_cards=[],
            cards_remaining_in_deck=33,
            valid_actions=[0, 1, 2],
            game_over=False,
            num_players=2,
            is_team_game=False,
            players=[],
        )

        assert dto.type == "observation"

    def test_json_output_has_required_fields(self) -> None:
        """Verifica che il JSON contenga tutti i campi richiesti."""
        dto = ObservationDTO(
            server_version=5,
            my_index=0,
            my_hand=[CardDTO(suit="cups", rank="THREE", number=3, points=10)],
            my_points=21,
            my_turn=False,
            trump_card=CardDTO(suit="clubs", rank="ACE", number=1, points=11),
            trump_suit="clubs",
            table_cards=[],
            cards_remaining_in_deck=20,
            valid_actions=[],
            game_over=False,
            num_players=2,
            is_team_game=False,
            players=[
                PlayerInfoDTO(index=0, name="Player", points=21, hand_size=3),
                PlayerInfoDTO(index=1, name="AI", points=10, hand_size=3),
            ],
            seen_cards_onehot=[0] * 40,
        )

        data = dto.model_dump()

        assert data["type"] == "observation"
        assert data["server_version"] == 5
        assert len(data["my_hand"]) == 1
        assert len(data["players"]) == 2
        assert data["players"][0]["name"] == "Player"
        assert data["players"][1]["name"] == "AI"
        assert len(data["seen_cards_onehot"]) == 40
        # Step 3: il nuovo campo è sempre presente nel dump (default lista vuota).
        assert "out_of_play_cards_onehot" in data

    def test_accepts_old_payload_without_out_of_play_field(self) -> None:
        """Backward compatibility: payload vecchi (senza out_of_play) restano validi."""
        payload = {
            "server_version": 1,
            "my_index": 0,
            "my_hand": [],
            "my_points": 0,
            "my_turn": True,
            "trump_card": None,
            "trump_suit": "clubs",
            "table_cards": [],
            "cards_remaining_in_deck": 0,
            "valid_actions": [],
            "game_over": False,
            "num_players": 2,
            "is_team_game": False,
            "players": [],
            "seen_cards_onehot": [0] * 40,
            # nota: nessun `out_of_play_cards_onehot`
        }

        dto = ObservationDTO.model_validate(payload)

        # Il campo mancante prende il default (lista vuota), senza errori di validazione.
        assert dto.out_of_play_cards_onehot == []


class TestAiCardRevealDTO:
    """Test per AiCardRevealDTO."""

    def test_has_type_ai_card_reveal(self) -> None:
        """Verifica che il campo type sia 'ai_card_reveal'."""
        dto = AiCardRevealDTO(
            card_index=1,
            card=CardDTO(suit="swords", rank="SEVEN", number=7, points=0),
            decision_type="lookahead",
        )

        assert dto.type == "ai_card_reveal"
        assert dto.decision_type == "lookahead"


class TestGameStateDTO:
    """Test per GameStateDTO completo usato da debug/spectator."""

    def test_debug_state_can_include_next_deck_card(self) -> None:
        """Lo stato completo può esporre la prossima carta mazzo; ObservationDTO no."""
        dto = GameStateDTO(
            server_version=2,
            num_players=2,
            is_team_game=False,
            trump_card=None,
            trump_suit="cups",
            next_deck_card=CardDTO(suit="coins", rank="ACE", number=1, points=11),
            table_cards=[],
            current_turn=0,
            first_player=0,
            cards_remaining_in_deck=12,
            valid_actions=[0, 1, 2],
            game_over=False,
            trick_in_progress=False,
            trick_size=0,
            expected_trick_size=2,
            players=[
                PlayerStateDTO(index=0, name="Player", points=0, hand=[], hand_size=0, captured_cards=[]),
                PlayerStateDTO(index=1, name="AI", points=0, hand=[], hand_size=0, captured_cards=[]),
            ],
        )

        data = dto.model_dump()

        assert data["type"] == "game_state"
        assert data["next_deck_card"]["rank"] == "ACE"


class TestTrickResultDTO:
    """Test per TrickResultDTO."""

    def test_has_type_trick_result(self) -> None:
        """Verifica che il campo type sia 'trick_result'."""
        dto = TrickResultDTO(
            trick_cards=[],
            winner_index=0,
            winner_name="Player",
            points=14,
            server_version=3,
        )

        assert dto.type == "trick_result"

    def test_trick_cards_contains_table_card_dtos(self) -> None:
        """Verifica che trick_cards contenga TableCardDTO."""
        card1 = CardDTO(suit="cups", rank="ACE", number=1, points=11)
        card2 = CardDTO(suit="cups", rank="THREE", number=3, points=10)

        dto = TrickResultDTO(
            trick_cards=[
                TableCardDTO(card=card1, player_index=0),
                TableCardDTO(card=card2, player_index=1),
            ],
            winner_index=0,
            winner_name="Player",
            points=21,
            server_version=4,
        )

        assert len(dto.trick_cards) == 2
        assert dto.trick_cards[0].player_index == 0
        assert dto.trick_cards[1].player_index == 1
