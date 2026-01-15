"""
Numerical solvers for differential equations.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from .interfaces import INumericalSolver
from ..validation.errors import NumericError, ConvergenceError


class RungeKuttaSolver(INumericalSolver):
    """Fourth-order Runge-Kutta numerical integration."""

    def __init__(self, adaptive_step: bool = False, max_step_size: float = 1.0):
        """
        Initialize RK4 solver.

        Args:
            adaptive_step: Whether to use adaptive step sizing
            max_step_size: Maximum allowed step size
        """
        self.adaptive_step = adaptive_step
        self.max_step_size = max_step_size
        self._last_error_estimate = 0.0

    def solve(self, equations: List[Tuple[str, Any]], initial_conditions: Dict[str, float],
             time_config: Any) -> Dict[str, List[float]]:
        """
        Solve system of equations using RK4 method.

        Args:
            equations: List of (variable_name, parsed_equation) tuples
            initial_conditions: Initial values for all variables
            time_config: Time configuration

        Returns:
            Dictionary mapping variable names to time series
        """
        try:
            # Generate time points
            time_points = self._generate_time_points(time_config)

            # Initialize result storage
            results = {var: [initial_conditions[var]] for var in initial_conditions}
            results['time'] = time_points

            # Current state
            current_state = initial_conditions.copy()
            current_time = time_config.start_time
            dt = time_config.time_step

            for i in range(1, len(time_points)):
                # RK4 integration step
                k1 = self._compute_derivatives(current_state, equations)
                k2_state = {var: current_state[var] + 0.5 * dt * k1[var] for var in current_state}
                k2 = self._compute_derivatives(k2_state, equations)
                k3_state = {var: current_state[var] + 0.5 * dt * k2[var] for var in current_state}
                k3 = self._compute_derivatives(k3_state, equations)
                k4_state = {var: current_state[var] + dt * k3[var] for var in current_state}
                k4 = self._compute_derivatives(k4_state, equations)

                # Update state
                for var in current_state:
                    derivative = (k1[var] + 2*k2[var] + 2*k3[var] + k4[var]) / 6
                    new_value = current_state[var] + dt * derivative
                    current_state[var] = new_value
                    results[var].append(new_value)

                current_time += dt

                # Check for numerical instabilities
                if self._check_numerical_instability(current_state):
                    raise NumericError(
                        "integration",
                        "Numerical instability detected",
                        {"time": current_time, "state": current_state}
                    )

            return {k: v for k, v in results.items() if k != 'time'}

        except Exception as e:
            if isinstance(e, NumericError):
                raise
            raise NumericError("solve", f"Numerical integration failed: {str(e)}") from e

    def _compute_derivatives(self, state: Dict[str, float],
                           equations: List[Tuple[str, Any]]) -> Dict[str, float]:
        """Compute derivatives for current state."""
        derivatives = {}
        from .equation_parser import SimpleEquationParser

        parser = SimpleEquationParser()

        for var_name, parsed_eq in equations:
            try:
                # Evaluate the derivative expression
                derivative = parser.evaluate(parsed_eq, state)
                derivatives[var_name] = derivative
            except Exception as e:
                raise NumericError(
                    "derivative_computation",
                    f"Failed to compute derivative for {var_name}: {str(e)}"
                )

        return derivatives

    def _generate_time_points(self, time_config: Any) -> List[float]:
        """Generate time points for integration."""
        start = time_config.start_time
        end = time_config.end_time
        step = time_config.time_step

        points = []
        current = start
        while current <= end + step/2:  # Add small epsilon for floating point
            points.append(current)
            current += step

        return points

    def _check_numerical_instability(self, state: Dict[str, float]) -> bool:
        """Check for numerical instabilities (NaN, inf values)."""
        for var, value in state.items():
            if not np.isfinite(value):
                return True
        return False

    def get_method_name(self) -> str:
        """Return solver method name."""
        return "Runge-Kutta 4th Order"

    def estimate_error(self) -> float:
        """Estimate numerical error (placeholder for now)."""
        # In a full implementation, this would track truncation error
        return self._last_error_estimate
