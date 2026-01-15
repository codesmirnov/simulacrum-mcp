"""
Tests for dynamics simulation functionality.
"""

import pytest
import numpy as np
from simulacrum.tools.dynamics import DynamicsSimulator
from simulacrum.validation.errors import ValidationError


class TestDynamicsSimulator:
    """Test suite for dynamics simulation."""

    def test_simulate_dynamics_valid_scenario(self, sample_scenario_data):
        """Test successful simulation with valid scenario."""
        simulator = DynamicsSimulator()

        result = simulator.simulate_dynamics(sample_scenario_data)

        assert result["status"] == "success"
        assert result["converged"] is not None
        assert result["execution_time"] > 0
        assert len(result["time_points"]) > 1
        assert "x" in result["variables"]
        assert "y" in result["variables"]
        assert len(result["variables"]["x"]) == len(result["time_points"])

    def test_simulate_dynamics_invalid_scenario(self):
        """Test error handling with invalid scenario."""
        simulator = DynamicsSimulator()

        invalid_scenario = {
            "name": "Invalid",
            "variables": [],  # Empty variables should fail
            "equations": []
        }

        with pytest.raises(ValidationError):
            simulator.simulate_dynamics(invalid_scenario)

    def test_validate_scenario_data_valid(self, sample_scenario_data):
        """Test scenario validation with valid data."""
        simulator = DynamicsSimulator()

        errors = simulator.validate_scenario_data(sample_scenario_data)

        assert len(errors) == 0

    def test_validate_scenario_data_invalid(self):
        """Test scenario validation with invalid data."""
        simulator = DynamicsSimulator()

        invalid_data = {
            "name": "Invalid",
            "variables": [
                {"name": "x", "initial_value": 1.0},
                {"name": "x", "initial_value": 2.0}  # Duplicate name
            ],
            "equations": [
                {"target_variable": "z", "expression": "x"}  # References undefined variable
            ]
        }

        errors = simulator.validate_scenario_data(invalid_data)

        assert len(errors) > 0
        assert any("unique" in error.lower() or "undefined" in error.lower() for error in errors)

    def test_simulation_results_structure(self, sample_scenario_data):
        """Test that simulation results have expected structure."""
        simulator = DynamicsSimulator()

        result = simulator.simulate_dynamics(sample_scenario_data)

        required_keys = ["status", "converged", "execution_time", "time_points", "variables", "summary"]
        for key in required_keys:
            assert key in result

        # Check summary structure
        summary = result["summary"]
        assert "variable_statistics" in summary
        assert "x" in summary["variable_statistics"]
        assert "y" in summary["variable_statistics"]

        # Check variable statistics
        x_stats = summary["variable_statistics"]["x"]
        required_stats = ["initial_value", "final_value", "min_value", "max_value", "mean_value"]
        for stat in required_stats:
            assert stat in x_stats

    def test_simulation_conservation_properties(self, sample_scenario_data):
        """Test that simulation maintains basic physical properties."""
        simulator = DynamicsSimulator()

        result = simulator.simulate_dynamics(sample_scenario_data)

        variables = result["variables"]

        # Check that values stay within bounds (if specified)
        for var_name, values in variables.items():
            var_def = next(v for v in sample_scenario_data["variables"] if v["name"] == var_name)

            if "min_value" in var_def:
                assert all(v >= var_def["min_value"] for v in values), f"{var_name} violates minimum bound"

            if "max_value" in var_def:
                assert all(v <= var_def["max_value"] for v in values), f"{var_name} violates maximum bound"

    def test_equation_syntax_info(self):
        """Test that equation syntax information is provided."""
        simulator = DynamicsSimulator()

        syntax = simulator.get_supported_equation_syntax()

        required_keys = ["operators", "functions", "constants", "variables", "examples"]
        for key in required_keys:
            assert key in syntax

        assert len(syntax["examples"]) > 0
        assert isinstance(syntax["operators"], list)
        assert isinstance(syntax["functions"], list)

    def test_empty_scenario_handling(self):
        """Test handling of edge case with minimal scenario."""
        simulator = DynamicsSimulator()

        minimal_scenario = {
            "name": "Minimal",
            "variables": [
                {"name": "x", "initial_value": 1.0}
            ],
            "equations": [
                {"target_variable": "x", "expression": "0"}  # Constant equation
            ]
        }

        result = simulator.simulate_dynamics(minimal_scenario)

        assert result["status"] == "success"
        assert len(result["variables"]["x"]) > 1
        # For constant equation, all values should be the same
        assert all(v == result["variables"]["x"][0] for v in result["variables"]["x"])


class TestDynamicsIntegration:
    """Integration tests for dynamics simulation."""

    def test_predator_prey_simulation(self):
        """Test classic Lotka-Volterra predator-prey model."""
        simulator = DynamicsSimulator()

        predator_prey = {
            "name": "Predator-Prey",
            "variables": [
                {"name": "prey", "initial_value": 10.0, "min_value": 0.1},
                {"name": "predator", "initial_value": 5.0, "min_value": 0.1}
            ],
            "equations": [
                {"target_variable": "prey", "expression": "prey * (2.0 - 0.01 * predator)"},
                {"target_variable": "predator", "expression": "predator * (-1.0 + 0.01 * prey)"}
            ],
            "config": {
                "time_config": {
                    "end_time": 20.0,
                    "time_step": 0.1
                }
            }
        }

        result = simulator.simulate_dynamics(predator_prey)

        assert result["status"] == "success"

        # Check oscillatory behavior (typical for predator-prey)
        prey_values = result["variables"]["prey"]
        predator_values = result["variables"]["predator"]

        # Should have multiple peaks/valleys
        prey_peaks = self._count_peaks_troughs(prey_values)
        assert prey_peaks > 2, "Predator-prey system should show oscillatory behavior"

    def test_logistic_growth_simulation(self):
        """Test logistic growth model."""
        simulator = DynamicsSimulator()

        logistic = {
            "name": "Logistic Growth",
            "variables": [
                {"name": "population", "initial_value": 1.0, "max_value": 100.0}
            ],
            "equations": [
                {"target_variable": "population", "expression": "population * (0.5 - 0.005 * population)"}
            ],
            "config": {
                "time_config": {
                    "end_time": 15.0,
                    "time_step": 0.1
                }
            }
        }

        result = simulator.simulate_dynamics(logistic)

        assert result["status"] == "success"

        population = result["variables"]["population"]

        # Should start slow, accelerate, then slow down (sigmoid shape)
        initial_growth = population[10] - population[0]
        middle_growth = population[50] - population[40]
        final_growth = population[-1] - population[-11]

        assert initial_growth < middle_growth, "Growth should accelerate initially"
        assert final_growth < middle_growth, "Growth should decelerate at end"

        # Should approach carrying capacity
        assert population[-1] < 100.0, "Should not exceed carrying capacity"

    def _count_peaks_troughs(self, values: list, threshold: float = 0.1) -> int:
        """Count significant peaks and troughs in a time series."""
        if len(values) < 3:
            return 0

        peaks_troughs = 0
        direction_changes = 0

        for i in range(1, len(values) - 1):
            prev_diff = values[i] - values[i-1]
            next_diff = values[i+1] - values[i]

            # Check for direction change (peak or trough)
            if prev_diff * next_diff < 0:
                # Check if change is significant
                change_magnitude = abs(prev_diff) + abs(next_diff)
                if change_magnitude > threshold * np.mean(values):
                    direction_changes += 1

        return direction_changes
