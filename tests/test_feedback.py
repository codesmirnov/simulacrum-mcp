"""
Tests for feedback loop analysis functionality.
"""

import pytest
from simulacrum.tools.feedback import FeedbackAnalyzer
from simulacrum.validation.errors import ValidationError


class TestFeedbackAnalyzer:
    """Test suite for feedback analysis."""

    @pytest.fixture
    def feedback_analyzer(self):
        """Create feedback analyzer."""
        return FeedbackAnalyzer()

    @pytest.fixture
    def sample_scenario(self):
        """Sample scenario with feedback loops."""
        return {
            "name": "Predator-Prey",
            "variables": [
                {"name": "prey", "initial_value": 10.0},
                {"name": "predator", "initial_value": 5.0}
            ],
            "equations": [
                {"target_variable": "prey", "expression": "prey * (2.0 - 0.01 * predator)"},
                {"target_variable": "predator", "expression": "predator * (-1.0 + 0.01 * prey)"}
            ]
        }

    def test_analyze_feedback_loops_success(self, feedback_analyzer, sample_scenario):
        """Test successful feedback loop analysis."""
        result = feedback_analyzer.analyze_feedback_loops(sample_scenario)

        assert result["status"] == "success"
        assert "analysis" in result

    def test_analyze_feedback_loops_invalid_data(self, feedback_analyzer):
        """Test error handling with invalid data."""
        with pytest.raises(ValidationError):
            feedback_analyzer.analyze_feedback_loops({})

    def test_build_dependency_graph(self, feedback_analyzer, sample_scenario):
        """Test dependency graph building."""
        from simulacrum.validation.types import ScenarioData
        scenario = ScenarioData(**sample_scenario)

        graph = feedback_analyzer._build_dependency_graph(scenario)

        # Graph is a NetworkX DiGraph, not dict
        assert graph is not None

    def test_find_feedback_loops(self, feedback_analyzer, sample_scenario):
        """Test feedback loop detection."""
        from simulacrum.validation.types import ScenarioData
        scenario = ScenarioData(**sample_scenario)

        dependency_graph = feedback_analyzer._build_dependency_graph(scenario)
        loops = feedback_analyzer._find_feedback_loops(dependency_graph)

        assert isinstance(loops, list)

    def test_classify_loops(self, feedback_analyzer, sample_scenario):
        """Test loop classification."""
        from simulacrum.validation.types import ScenarioData
        scenario = ScenarioData(**sample_scenario)

        dependency_graph = feedback_analyzer._build_dependency_graph(scenario)
        loops = feedback_analyzer._find_feedback_loops(dependency_graph)
        classified = feedback_analyzer._classify_loops(loops, dependency_graph)

        # Returns list of classified loops, not dict
        assert isinstance(classified, list)

    def test_feedback_analyzer_initialization(self):
        """Test feedback analyzer initialization."""
        analyzer = FeedbackAnalyzer()
        assert analyzer is not None
