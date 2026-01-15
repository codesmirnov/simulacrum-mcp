"""
Main simulation engine coordinating all simulation components.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from .interfaces import ISimulator, INumericalSolver, IEquationParser
from .numerical import RungeKuttaSolver
from .equation_parser import SimpleEquationParser
from ..validation.types import ScenarioData, SimulationResult, SimulationConfig
from ..validation.errors import SimulationError, ConvergenceError

logger = logging.getLogger(__name__)


class SimulationEngine(ISimulator):
    """Main simulation engine implementing the facade pattern."""

    def __init__(self,
                 numerical_solver: Optional[INumericalSolver] = None,
                 equation_parser: Optional[IEquationParser] = None):
        """
        Initialize simulation engine.

        Args:
            numerical_solver: Custom numerical solver (defaults to RungeKutta)
            equation_parser: Custom equation parser
        """
        self.numerical_solver = numerical_solver or RungeKuttaSolver()
        self.equation_parser = equation_parser or SimpleEquationParser()
        self._initialized = False

    def simulate(self, scenario: ScenarioData) -> SimulationResult:
        """
        Execute simulation for given scenario.

        Args:
            scenario: Complete scenario definition

        Returns:
            SimulationResult: Complete simulation results

        Raises:
            SimulationError: If simulation fails
        """
        start_time = time.time()

        try:
            logger.info(f"Starting simulation: {scenario.name}")

            # Validate scenario
            validation_errors = self.validate_scenario(scenario)
            if validation_errors:
                raise SimulationError(
                    "validation",
                    f"Scenario validation failed: {', '.join(validation_errors)}",
                    {"errors": validation_errors}
                )

            # Prepare simulation parameters
            time_points = self._generate_time_points(scenario.config.time_config)
            initial_conditions = {
                var.name: var.initial_value for var in scenario.variables
            }

            # Parse equations
            parsed_equations = []
            for eq in scenario.equations:
                try:
                    parsed = self.equation_parser.parse(eq.expression)
                    parsed_equations.append((eq.target_variable, parsed))
                except Exception as e:
                    raise SimulationError(
                        "equation_parsing",
                        f"Failed to parse equation for {eq.target_variable}: {str(e)}"
                    )

            # Execute numerical simulation
            variable_data = self.numerical_solver.solve(
                parsed_equations,
                initial_conditions,
                scenario.config.time_config
            )

            # Check convergence
            converged = self._check_convergence(
                variable_data,
                scenario.config.convergence
            )

            execution_time = time.time() - start_time

            result = SimulationResult(
                scenario_name=scenario.name,
                time_points=time_points,
                variable_data=variable_data,
                metadata={
                    "solver": self.numerical_solver.get_method_name(),
                    "config": scenario.config.dict(),
                    "numerical_error": self.numerical_solver.estimate_error()
                },
                execution_time=execution_time,
                converged=converged
            )

            logger.info(f"Simulation completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Simulation failed after {execution_time:.3f}s: {str(e)}")
            if isinstance(e, SimulationError):
                raise
            raise SimulationError("execution", str(e)) from e

    def validate_scenario(self, scenario: ScenarioData) -> List[str]:
        """
        Validate scenario before simulation.

        Args:
            scenario: Scenario to validate

        Returns:
            List of validation error messages
        """
        errors = []

        # Check variable references in equations
        variable_names = {var.name for var in scenario.variables}

        for eq in scenario.equations:
            try:
                dependencies = self.equation_parser.get_dependencies(eq.expression)
                missing_vars = set(dependencies) - variable_names
                if missing_vars:
                    errors.append(
                        f"Equation for '{eq.target_variable}' references "
                        f"undefined variables: {', '.join(missing_vars)}"
                    )
            except Exception as e:
                errors.append(
                    f"Failed to analyze equation for '{eq.target_variable}': {str(e)}"
                )

        # Check for circular dependencies (allow cycles in differential equations)
        # Note: Cycles are normal in systems of differential equations
        dependency_graph = {}
        for eq in scenario.equations:
            dependencies = self.equation_parser.get_dependencies(eq.expression)
            dependency_graph[eq.target_variable] = dependencies

        # Only report cycles as warnings for now, since they're normal in ODEs
        # In the future, we could add more sophisticated cycle analysis
        if self._has_cycles(dependency_graph):
            # For now, allow cycles but log them
            logger.info(f"Circular dependencies detected in scenario '{scenario.name}' - this is normal for systems of differential equations")

        return errors

    def _generate_time_points(self, time_config) -> List[float]:
        """Generate time points for simulation."""
        start = time_config.start_time
        end = time_config.end_time
        step = time_config.time_step

        points = []
        current = start
        while current <= end:
            points.append(current)
            current += step

        return points

    def _check_convergence(self, variable_data: Dict[str, List[float]],
                          convergence: Any) -> bool:
        """
        Check if simulation converged based on criteria.

        For now, simple check based on final stability.
        More sophisticated convergence detection can be added.
        """
        if not variable_data:
            return False

        # Simple convergence check: look at last few points
        window_size = min(10, len(next(iter(variable_data.values()))))

        for var_name, values in variable_data.items():
            if len(values) < window_size:
                return False

            # Check if values are stable in the last window
            last_values = values[-window_size:]
            mean_val = sum(last_values) / len(last_values)
            variance = sum((x - mean_val) ** 2 for x in last_values) / len(last_values)

            if variance > convergence.tolerance:
                return False

        return True

    def _has_cycles(self, graph: Dict[str, List[str]]) -> bool:
        """Check for cycles in dependency graph using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True

        return False
