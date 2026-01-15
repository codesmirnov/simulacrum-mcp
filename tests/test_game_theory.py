"""
Tests for game theory functionality.
"""

import pytest
from simulacrum.tools.game_theory import GameTheoryDynamics
from simulacrum.validation.errors import ValidationError


class TestGameTheoryDynamics:
    """Test suite for game theory dynamics."""

    @pytest.fixture
    def game_analyzer(self):
        """Create game theory dynamics analyzer."""
        return GameTheoryDynamics()

    @pytest.fixture
    def sample_game_definition(self):
        """Sample game definition for testing."""
        return {
            "name": "Prisoner's Dilemma",
            "players": ["player1", "player2"],
            "strategies": {
                "player1": ["cooperate", "defect"],
                "player2": ["cooperate", "defect"]
            },
            "payoffs": {
                "cooperate_cooperate": {"player1": 3, "player2": 3},
                "cooperate_defect": {"player1": 0, "player2": 5},
                "defect_cooperate": {"player1": 5, "player2": 0},
                "defect_defect": {"player1": 1, "player2": 1}
            }
        }

    def test_analyze_game_theory_success(self, game_analyzer, sample_game_definition):
        """Test successful game theory dynamics analysis."""
        result = game_analyzer.analyze_game_theory_dynamics(sample_game_definition)

        assert result["status"] == "success"

    def test_analyze_game_theory_invalid_data(self, game_analyzer):
        """Test error handling with invalid data."""
        with pytest.raises(ValidationError):
            game_analyzer.analyze_game_theory_dynamics({})

    def test_game_analyzer_initialization(self):
        """Test game theory analyzer initialization."""
        analyzer = GameTheoryDynamics()
        assert analyzer is not None
