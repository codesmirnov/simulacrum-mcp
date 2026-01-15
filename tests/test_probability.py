"""
Tests for probability analysis functionality.
"""

import pytest
from simulacrum.tools.probability import ProbabilityAnalyzer
from simulacrum.validation.errors import ValidationError


class TestProbabilityAnalyzer:
    """Test suite for probability analysis."""

    @pytest.fixture
    def prob_analyzer(self):
        """Create probability analyzer."""
        return ProbabilityAnalyzer()

    @pytest.fixture
    def sample_analysis_request(self):
        """Sample analysis request for testing."""
        return {
            "analysis_type": "bayesian_update",
            "prior": {"A": 0.3, "B": 0.7},
            "likelihood": {"A": 0.8, "B": 0.2},
            "evidence": 0.5
        }


    def test_analyze_probability_invalid_data(self, prob_analyzer):
        """Test error handling with invalid data."""
        with pytest.raises(ValidationError):
            prob_analyzer.analyze_probability({})

    def test_probability_analyzer_initialization(self):
        """Test probability analyzer initialization."""
        analyzer = ProbabilityAnalyzer()
        assert analyzer is not None
