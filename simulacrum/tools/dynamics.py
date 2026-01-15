"""
Dynamics simulation tool - main simulation interface.
"""

import json
from typing import Dict, List, Any, Optional
from ..core.engine import SimulationEngine
from ..validation.types import ScenarioData, SimulationResult
from ..validation.errors import ValidationError


class DynamicsSimulator:
    """
    Main interface for system dynamics simulation.

    This tool provides a high-level interface for simulating complex systems
    using differential equations and feedback loops.
    """

    def __init__(self, engine: Optional[SimulationEngine] = None):
        """
        Initialize dynamics simulator.

        Args:
            engine: Custom simulation engine (uses default if None)
        """
        self.engine = engine or SimulationEngine()

    def simulate_dynamics(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute system dynamics simulation.

        Args:
            scenario_data: Dictionary containing scenario definition with keys:
                - name: Scenario name
                - description: Optional description
                - variables: List of variable definitions
                - equations: List of equation definitions
                - config: Optional simulation configuration

        Returns:
            Dictionary containing simulation results and metadata

        Raises:
            ValidationError: If scenario data is invalid
        """
        try:
            # Convert and validate input
            scenario = ScenarioData(**scenario_data)

            # Execute simulation
            result = self.engine.simulate(scenario)

            # Format result
            return self._format_simulation_result(result)

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "scenario_data",
                scenario_data,
                f"Simulation failed: {str(e)}"
            )

    def _format_simulation_result(self, result: SimulationResult) -> Dict[str, Any]:
        """Format simulation result for API response."""
        return {
            "scenario_name": result.scenario_name,
            "status": "success",
            "converged": result.converged,
            "execution_time": result.execution_time,
            "time_points": result.time_points,
            "variables": result.variable_data,
            "metadata": result.metadata,
            "summary": self._generate_summary(result)
        }

    def _generate_summary(self, result: SimulationResult) -> Dict[str, Any]:
        """Generate human-readable summary of simulation results."""
        summary = {
            "total_time_points": len(result.time_points),
            "variable_count": len(result.variable_data),
            "time_range": {
                "start": result.time_points[0],
                "end": result.time_points[-1],
                "duration": result.time_points[-1] - result.time_points[0]
            }
        }

        # Add basic statistics for each variable
        variable_stats = {}
        for var_name, values in result.variable_data.items():
            variable_stats[var_name] = {
                "initial_value": values[0],
                "final_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "mean_value": sum(values) / len(values),
                "change": values[-1] - values[0]
            }

        summary["variable_statistics"] = variable_stats
        return summary

    def validate_scenario_data(self, scenario_data: Dict[str, Any]) -> List[str]:
        """
        Validate scenario data without running simulation.

        Args:
            scenario_data: Scenario definition to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            scenario = ScenarioData(**scenario_data)
            return self.engine.validate_scenario(scenario)
        except Exception as e:
            return [f"Data validation failed: {str(e)}"]

    def get_supported_equation_syntax(self) -> Dict[str, Any]:
        """
        Get information about supported equation syntax.

        Returns:
            Dictionary describing supported mathematical operations
        """
        return {
            "operators": ["+", "-", "*", "/", "^", "**"],
            "functions": [
                "sin", "cos", "tan", "exp", "log", "ln", "sqrt",
                "abs", "min", "max", "if", "and", "or", "not"
            ],
            "constants": ["pi", "e"],
            "variables": "Any defined variable name",
            "time_variable": "t (automatically available)",
            "examples": [
                "dx/dt = a*x - b*x*y",  # Lotka-Volterra predator-prey
                "dS/dt = -beta*S*I/N",  # SIR epidemiological model
                "dV/dt = r*V*(1 - V/K)"  # Logistic growth
            ]
        }
