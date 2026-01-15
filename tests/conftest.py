"""
Pytest configuration and fixtures for Simulacrum MCP tests.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from simulacrum.validation.types import ScenarioData, SimulationConfig
from simulacrum.tools.vector_analysis import VectorAnalyzer


@pytest.fixture
def sample_scenario_data() -> Dict[str, Any]:
    """Sample scenario data for testing."""
    return {
        "name": "Test Scenario",
        "description": "A simple test scenario",
        "variables": [
            {"name": "x", "initial_value": 1.0, "min_value": 0.0, "max_value": 10.0},
            {"name": "y", "initial_value": 2.0, "min_value": 0.0}
        ],
        "equations": [
            {"target_variable": "x", "expression": "-0.1 * x + 0.05 * y"},
            {"target_variable": "y", "expression": "0.1 * x - 0.2 * y"}
        ],
        "config": {
            "time_config": {
                "start_time": 0.0,
                "end_time": 5.0,
                "time_step": 0.1
            },
            "convergence": {
                "max_iterations": 100,
                "tolerance": 1e-6
            }
        }
    }


@pytest.fixture
def sample_scenario(sample_scenario_data: Dict[str, Any]) -> ScenarioData:
    """Pydantic-validated scenario object."""
    return ScenarioData(**sample_scenario_data)


@pytest.fixture
def time_series_data() -> Dict[str, Any]:
    """Sample time series data for chaos analysis."""
    np.random.seed(42)  # For reproducible tests
    t = np.linspace(0, 10, 100)
    # Simple damped oscillator with some noise
    signal = np.exp(-0.1 * t) * np.sin(2 * t) + 0.1 * np.random.randn(100)

    return {
        "time_series": signal.tolist(),
        "time_points": t.tolist()
    }


@pytest.fixture
def bayesian_test_data() -> Dict[str, Any]:
    """Sample data for Bayesian analysis testing."""
    return {
        "analysis_type": "bayesian_update",
        "prior": {"hypothesis_A": 0.6, "hypothesis_B": 0.4},
        "likelihood": {
            "evidence_X": {"hypothesis_A": 0.8, "hypothesis_B": 0.3},
            "evidence_Y": {"hypothesis_A": 0.2, "hypothesis_B": 0.9}
        },
        "evidence": [
            {"hypothesis": "hypothesis_A", "observation": "evidence_X", "strength": 1.0},
            {"hypothesis": "hypothesis_B", "observation": "evidence_Y", "strength": 0.8}
        ]
    }


@pytest.fixture
def game_theory_data() -> Dict[str, Any]:
    """Sample game theory data (Prisoner's Dilemma)."""
    return {
        "players": ["player1", "player2"],
        "strategies": {
            "player1": ["Cooperate", "Defect"],
            "player2": ["Cooperate", "Defect"]
        },
        "payoffs": {
            "Cooperate,Cooperate": [3, 3],
            "Cooperate,Defect": [0, 5],
            "Defect,Cooperate": [5, 0],
            "Defect,Defect": [1, 1]
        }
    }


@pytest.fixture
def belief_dynamics_data() -> List[Dict[str, Any]]:
    """Sample belief dynamics data."""
    return [
        {
            "agent_id": "alice",
            "beliefs": {"policy_A": 0.8, "policy_B": 0.2},
            "evidence_weight": 1.0,
            "adaptability": 0.1
        },
        {
            "agent_id": "bob",
            "beliefs": {"policy_A": 0.3, "policy_B": 0.7},
            "evidence_weight": 0.8,
            "adaptability": 0.15
        }
    ]


@pytest.fixture
def mock_equation_parser():
    """Mock equation parser for testing."""
    class MockEquationParser:
        def parse(self, expression: str):
            return f"parsed_{expression}"

        def evaluate(self, parsed_expression, variables: Dict[str, float]) -> float:
            # Simple mock evaluation
            if "x" in str(parsed_expression) and "x" in variables:
                return -0.1 * variables["x"]
            return 0.0

        def get_dependencies(self, expression: str) -> List[str]:
            # Extract variable names (simplified)
            import re
            return re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)

    return MockEquationParser()


@pytest.fixture
def vector_analyzer():
    """Vector analyzer instance for testing."""
    return VectorAnalyzer()


@pytest.fixture
def sample_2d_vectors():
    """Sample 2D vectors for testing."""
    return [
        {"name": "Technology", "x": 0.8, "y": 0.3, "description": "Innovation focus"},
        {"name": "Environment", "x": -0.2, "y": 0.9, "description": "Sustainability focus"},
        {"name": "Finance", "x": 0.5, "y": -0.4, "description": "Profit focus"}
    ]


@pytest.fixture
def sample_vector_strengths():
    """Sample vector strengths for testing."""
    return [
        {"name": "Technology", "value": 0.9},
        {"name": "Environment", "value": 0.7},
        {"name": "Finance", "value": 0.8}
    ]


@pytest.fixture
def sample_multidimensional_vectors():
    """Sample multidimensional vectors for testing."""
    return [
        {
            "name": "Company_A",
            "components": {
                "profit": 0.8,
                "sustainability": 0.3,
                "innovation": 0.9,
                "risk": -0.2
            },
            "description": "Tech company"
        },
        {
            "name": "Company_B",
            "components": {
                "profit": 0.6,
                "sustainability": 0.8,
                "innovation": 0.2,
                "risk": 0.1
            },
            "description": "Green company"
        },
        {
            "name": "Company_C",
            "components": {
                "profit": 0.9,
                "sustainability": -0.3,
                "innovation": 0.4,
                "risk": 0.8
            },
            "description": "Finance company"
        }
    ]


@pytest.fixture
def sample_multidimensional_strengths():
    """Sample strengths for multidimensional vectors."""
    return [
        {"name": "Company_A", "value": 0.95},
        {"name": "Company_B", "value": 0.88},
        {"name": "Company_C", "value": 0.92}
    ]
