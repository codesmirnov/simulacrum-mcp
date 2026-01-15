"""
Tests for scenario comparison functionality.
"""

import pytest
from simulacrum.tools.comparison import ScenarioComparator
from simulacrum.validation.errors import ValidationError


class TestScenarioComparator:
    """Test suite for scenario comparison."""

    @pytest.fixture
    def comparator(self):
        """Create scenario comparator."""
        return ScenarioComparator()

    @pytest.fixture
    def sample_scenarios(self):
        """Sample scenarios for testing."""
        return [
            {
                "name": "Base Case",
                "variables": [
                    {"name": "x", "initial_value": 1.0},
                    {"name": "y", "initial_value": 2.0}
                ],
                "equations": [
                    {"target_variable": "x", "expression": "x + 0.1"},
                    {"target_variable": "y", "expression": "y + 0.05"}
                ]
            },
            {
                "name": "Modified Case",
                "variables": [
                    {"name": "x", "initial_value": 1.0},
                    {"name": "y", "initial_value": 2.0}
                ],
                "equations": [
                    {"target_variable": "x", "expression": "x + 0.2"},
                    {"target_variable": "y", "expression": "y + 0.1"}
                ]
            }
        ]

    def test_compare_scenarios_success(self, comparator, sample_scenarios):
        """Test successful scenario comparison."""
        result = comparator.compare_scenarios(sample_scenarios)

        assert result["status"] == "success"
        assert "individual_results" in result
        assert "comparison" in result
        assert len(result["individual_results"]) == 2

    def test_compare_scenarios_insufficient_scenarios(self, comparator):
        """Test error with insufficient scenarios."""
        with pytest.raises(ValidationError):
            comparator.compare_scenarios([{"name": "Single", "variables": [], "equations": []}])

    def test_compare_scenarios_invalid_scenario(self, comparator):
        """Test error with invalid scenario."""
        invalid_scenarios = [
            {"name": "Valid", "variables": [{"name": "x", "initial_value": 1.0}], "equations": []},
            {"name": "Invalid", "variables": [], "equations": []}  # Invalid: no variables
        ]

        with pytest.raises(ValidationError):
            comparator.compare_scenarios(invalid_scenarios)

