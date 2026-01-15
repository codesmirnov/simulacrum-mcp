"""
Tests for vector analysis tools.
"""

import pytest
import math
from simulacrum.tools.vector_analysis import VectorAnalyzer
from simulacrum.validation.errors import ValidationError


class TestVectorAnalyzer:
    """Test suite for vector analysis tools."""

    def test_vector_addition_basic(self):
        """Test basic 2D vector addition."""
        analyzer = VectorAnalyzer()

        vectors = [
            {"name": "A", "x": 1.0, "y": 0.0},
            {"name": "B", "x": 0.0, "y": 1.0}
        ]
        strengths = [
            {"name": "A", "value": 1.0},
            {"name": "B", "value": 1.0}
        ]

        result = analyzer.vector_addition(vectors, strengths)

        assert result["operation"] == "vector_addition"
        assert result["input_vectors"] == 2
        assert abs(result["resultant_vector"]["magnitude"] - math.sqrt(2)) < 1e-4
        assert abs(result["resultant_vector"]["angle_degrees"] - 45.0) < 1e-6

    def test_vector_addition_with_strengths(self):
        """Test vector addition with different strengths."""
        analyzer = VectorAnalyzer()

        vectors = [
            {"name": "A", "x": 1.0, "y": 0.0},
            {"name": "B", "x": 1.0, "y": 0.0}
        ]
        strengths = [
            {"name": "A", "value": 2.0},
            {"name": "B", "value": 1.0}
        ]

        result = analyzer.vector_addition(vectors, strengths)

        # A should contribute more due to higher strength
        assert result["resultant_vector"]["x"] > 1.5  # More than simple sum

    def test_vector_addition_normalization(self):
        """Test vector addition with normalization."""
        analyzer = VectorAnalyzer()

        vectors = [
            {"name": "A", "x": 3.0, "y": 4.0},  # Magnitude 5
            {"name": "B", "x": 1.0, "y": 1.0}   # Magnitude sqrt(2)
        ]
        strengths = [
            {"name": "A", "value": 1.0},
            {"name": "B", "value": 1.0}
        ]

        result = analyzer.vector_addition(vectors, strengths, normalize=True)

        # Result should be normalized to unit length
        magnitude = result["resultant_vector"]["magnitude"]
        assert abs(magnitude - 1.0) < 1e-6

    def test_vector_addition_empty_input(self):
        """Test error handling for empty input."""
        analyzer = VectorAnalyzer()

        with pytest.raises(ValidationError):
            analyzer.vector_addition([], [])

    def test_vector_addition_invalid_structure(self):
        """Test validation of vector structure."""
        analyzer = VectorAnalyzer()

        # Missing required fields
        invalid_vectors = [{"name": "A"}]  # Missing x, y
        strengths = [{"name": "A", "value": 1.0}]

        with pytest.raises(ValidationError):
            analyzer.vector_addition(invalid_vectors, strengths)

    def test_vector_addition_too_many_vectors(self):
        """Test limit on number of vectors."""
        analyzer = VectorAnalyzer(max_vectors=2)

        vectors = [
            {"name": f"V{i}", "x": 1.0, "y": 0.0}
            for i in range(3)
        ]
        strengths = [
            {"name": f"V{i}", "value": 1.0}
            for i in range(3)
        ]

        with pytest.raises(ValidationError):
            analyzer.vector_addition(vectors, strengths)

    def test_multidimensional_analysis_basic(self):
        """Test basic multidimensional vector analysis."""
        analyzer = VectorAnalyzer()

        vectors = [
            {
                "name": "Entity_A",
                "components": {"dim1": 1.0, "dim2": 0.5}
            },
            {
                "name": "Entity_B",
                "components": {"dim1": 0.0, "dim2": 1.0}
            }
        ]
        strengths = [
            {"name": "Entity_A", "value": 1.0},
            {"name": "Entity_B", "value": 1.0}
        ]

        result = analyzer.multidimensional_vector_analysis(vectors, strengths)

        assert result["operation"] == "multidimensional_vector_analysis"
        assert result["input_vectors"] == 2
        assert result["dimensions"] == 2
        assert result["total_magnitude"] > 0
        assert "player_analysis" in result
        assert "conflict_analysis" in result

    def test_multidimensional_analysis_projection(self):
        """Test 2D projection in multidimensional analysis."""
        analyzer = VectorAnalyzer()

        vectors = [
            {
                "name": "Entity_A",
                "components": {"x": 1.0, "y": 2.0, "z": 0.5}
            }
        ]
        strengths = [{"name": "Entity_A", "value": 1.0}]
        projection_axes = ["x", "y"]

        result = analyzer.multidimensional_vector_analysis(
            vectors, strengths, projection_axes=projection_axes
        )

        assert "projection_2d" in result
        projection = result["projection_2d"]
        assert projection["x_axis"] == "x"
        assert projection["y_axis"] == "y"
        assert "projection_magnitude" in projection

    def test_multidimensional_analysis_coalitions(self):
        """Test coalition detection in multidimensional analysis."""
        analyzer = VectorAnalyzer()

        # Create similar vectors that should form a coalition
        vectors = [
            {
                "name": "Entity_A",
                "components": {"dim1": 1.0, "dim2": 0.1}
            },
            {
                "name": "Entity_B",
                "components": {"dim1": 0.9, "dim2": 0.0}
            },
            {
                "name": "Entity_C",
                "components": {"dim1": -0.8, "dim2": 0.2}  # Opposite direction
            }
        ]
        strengths = [
            {"name": "Entity_A", "value": 1.0},
            {"name": "Entity_B", "value": 1.0},
            {"name": "Entity_C", "value": 1.0}
        ]

        result = analyzer.multidimensional_vector_analysis(vectors, strengths)

        conflict_analysis = result["conflict_analysis"]
        assert "coalitions" in conflict_analysis
        assert "conflicts" in conflict_analysis

        # Should detect coalition between A and B, conflict with C
        coalitions = conflict_analysis["coalitions"]
        conflicts = conflict_analysis["conflicts"]

        # At least one coalition should be detected
        assert len(coalitions) >= 1 or len(conflicts) >= 1

    def test_multidimensional_analysis_empty_input(self):
        """Test error handling for empty multidimensional input."""
        analyzer = VectorAnalyzer()

        with pytest.raises(ValidationError):
            analyzer.multidimensional_vector_analysis([], [])

    def test_multidimensional_analysis_missing_strengths(self):
        """Test error handling for missing strengths."""
        analyzer = VectorAnalyzer()

        vectors = [{"name": "A", "components": {"dim1": 1.0}}]

        with pytest.raises(ValidationError):
            analyzer.multidimensional_vector_analysis(vectors, [])

    def test_multidimensional_analysis_invalid_components(self):
        """Test validation of component structure."""
        analyzer = VectorAnalyzer()

        vectors = [{"name": "A", "components": "invalid"}]  # Should be dict
        strengths = [{"name": "A", "value": 1.0}]

        with pytest.raises(ValidationError):
            analyzer.multidimensional_vector_analysis(vectors, strengths)

    def test_multidimensional_analysis_player_outcomes(self):
        """Test player outcome analysis."""
        analyzer = VectorAnalyzer()

        vectors = [
            {
                "name": "Winner",
                "components": {"dim1": 2.0, "dim2": 1.0}
            },
            {
                "name": "Loser",
                "components": {"dim1": -0.5, "dim2": -0.5}
            },
            {
                "name": "Neutral",
                "components": {"dim1": 0.5, "dim2": 0.0}
            }
        ]
        strengths = [
            {"name": "Winner", "value": 1.0},
            {"name": "Loser", "value": 0.5},
            {"name": "Neutral", "value": 0.8}
        ]

        result = analyzer.multidimensional_vector_analysis(vectors, strengths)

        player_analysis = result["player_analysis"]
        assert "players" in player_analysis
        assert len(player_analysis["players"]) == 3

        # Winner should have highest score, Loser lowest
        players = player_analysis["players"]
        winner_score = next(p["win_score"] for p in players if p["name"] == "Winner")
        loser_score = next(p["win_score"] for p in players if p["name"] == "Loser")

        assert winner_score > loser_score

    def test_vector_addition_axis_names(self):
        """Test custom axis names in vector addition."""
        analyzer = VectorAnalyzer()

        vectors = [{"name": "A", "x": 1.0, "y": 0.0}]
        strengths = [{"name": "A", "value": 1.0}]
        axis_names = {"x": "Innovation", "y": "Stability"}

        result = analyzer.vector_addition(vectors, strengths, axis_names)

        assert result["axes"]["x"] == "Innovation"
        assert result["axes"]["y"] == "Stability"

    def test_multidimensional_analysis_normalization(self):
        """Test normalization in multidimensional analysis."""
        analyzer = VectorAnalyzer()

        vectors = [
            {
                "name": "Entity_A",
                "components": {"dim1": 10.0, "dim2": 0.0}
            }
        ]
        strengths = [{"name": "Entity_A", "value": 1.0}]

        result = analyzer.multidimensional_vector_analysis(vectors, strengths, normalize=True)

        # Result should be normalized to unit length
        assert abs(result["total_magnitude"] - 1.0) < 1e-6

    def test_multidimensional_analysis_top_dimensions(self):
        """Test limiting number of top dimensions."""
        analyzer = VectorAnalyzer()

        vectors = [
            {
                "name": "Entity_A",
                "components": {f"dim{i}": float(i) for i in range(20)}
            }
        ]
        strengths = [{"name": "Entity_A", "value": 1.0}]

        result = analyzer.multidimensional_vector_analysis(vectors, strengths, top_dimensions=5)

        assert len(result["top_dimensions"]) <= 5
