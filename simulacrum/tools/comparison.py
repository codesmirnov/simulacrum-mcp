"""
Scenario comparison tool for A/B testing of reality.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ..core.engine import SimulationEngine
from ..validation.types import ScenarioData, SimulationResult
from ..validation.errors import ValidationError


class ScenarioComparator:
    """
    Tool for comparing multiple simulation scenarios.

    Enables A/B testing of different assumptions, parameters, or model structures
    to understand how changes affect system behavior.
    """

    def __init__(self, engine: Optional[SimulationEngine] = None):
        """
        Initialize scenario comparator.

        Args:
            engine: Custom simulation engine (uses default if None)
        """
        self.engine = engine or SimulationEngine()

    def compare_scenarios(self, scenarios_data: List[Dict[str, Any]],
                         comparison_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple simulation scenarios.

        Args:
            scenarios_data: List of scenario definitions
            comparison_metrics: List of metrics to compute (optional)

        Returns:
            Dictionary containing comparison results and analysis

        Raises:
            ValidationError: If scenarios are invalid or incomparable
        """
        if len(scenarios_data) < 2:
            raise ValidationError(
                "scenarios_data",
                scenarios_data,
                "At least 2 scenarios required for comparison"
            )

        try:
            # Validate and convert scenarios
            scenarios = [ScenarioData(**data) for data in scenarios_data]

            # Run simulations
            results = []
            for scenario in scenarios:
                result = self.engine.simulate(scenario)
                results.append(result)

            # Perform comparison analysis
            comparison = self._analyze_comparison(results, comparison_metrics or [])

            return {
                "status": "success",
                "scenario_count": len(scenarios),
                "scenario_names": [r.scenario_name for r in results],
                "comparison": comparison,
                "individual_results": [self._format_simulation_result(r) for r in results]
            }

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "scenarios_data",
                scenarios_data,
                f"Comparison failed: {str(e)}"
            )

    def _analyze_comparison(self, results: List[SimulationResult],
                          metrics: List[str]) -> Dict[str, Any]:
        """Analyze differences between simulation results."""
        analysis = {
            "similarity_metrics": self._compute_similarity_metrics(results),
            "divergence_points": self._find_divergence_points(results),
            "stability_comparison": self._compare_stability(results),
            "execution_comparison": self._compare_execution_times(results)
        }

        # Add requested custom metrics
        for metric in metrics:
            if metric == "trajectory_distance":
                analysis["trajectory_distance"] = self._compute_trajectory_distances(results)
            elif metric == "phase_space_analysis":
                analysis["phase_space_analysis"] = self._analyze_phase_space(results)
            elif metric == "sensitivity_analysis":
                analysis["sensitivity_analysis"] = self._perform_sensitivity_analysis(results)

        return analysis

    def _compute_similarity_metrics(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Compute various similarity metrics between trajectories."""
        if len(results) < 2:
            return {}

        base_result = results[0]
        similarities = {}

        for i, result in enumerate(results[1:], 1):
            scenario_similarities = {}

            # Compare each variable
            for var_name in base_result.variable_data.keys():
                if var_name in result.variable_data:
                    base_values = np.array(base_result.variable_data[var_name])
                    comp_values = np.array(result.variable_data[var_name])

                    # Normalize lengths if different
                    min_len = min(len(base_values), len(comp_values))
                    base_norm = base_values[:min_len]
                    comp_norm = comp_values[:min_len]

                    # Compute metrics
                    mse = np.mean((base_norm - comp_norm) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(base_norm - comp_norm))

                    # Correlation coefficient
                    correlation = np.corrcoef(base_norm, comp_norm)[0, 1] if len(base_norm) > 1 else 0

                    scenario_similarities[var_name] = {
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "correlation": float(correlation),
                        "max_difference": float(np.max(np.abs(base_norm - comp_norm)))
                    }

            similarities[f"scenario_{i}_vs_base"] = scenario_similarities

        return similarities

    def _find_divergence_points(self, results: List[SimulationResult]) -> List[Dict[str, Any]]:
        """Find points where trajectories start to diverge significantly."""
        if len(results) < 2:
            return []

        divergence_points = []
        base_result = results[0]
        threshold = 0.1  # 10% relative difference threshold

        for var_name in base_result.variable_data.keys():
            base_values = np.array(base_result.variable_data[var_name])

            for i, result in enumerate(results[1:], 1):
                if var_name in result.variable_data:
                    comp_values = np.array(result.variable_data[var_name])
                    min_len = min(len(base_values), len(comp_values))

                    for t_idx in range(min_len):
                        if t_idx == 0:
                            continue

                        base_val = base_values[t_idx]
                        comp_val = comp_values[t_idx]

                        if abs(base_val) > 1e-10:  # Avoid division by zero
                            relative_diff = abs(base_val - comp_val) / abs(base_val)
                            if relative_diff > threshold:
                                divergence_points.append({
                                    "variable": var_name,
                                    "time_index": t_idx,
                                    "time_value": base_result.time_points[t_idx],
                                    "scenario_a": base_result.scenario_name,
                                    "scenario_b": result.scenario_name,
                                    "value_a": float(base_val),
                                    "value_b": float(comp_val),
                                    "relative_difference": float(relative_diff)
                                })
                                break  # Only record first divergence per variable pair

        return divergence_points

    def _compare_stability(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Compare convergence and stability across scenarios."""
        return {
            "convergence_status": {
                result.scenario_name: result.converged for result in results
            },
            "execution_times": {
                result.scenario_name: result.execution_time for result in results
            },
            "overall_stability": all(result.converged for result in results)
        }

    def _compare_execution_times(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Compare execution performance."""
        times = [r.execution_time for r in results]
        return {
            "fastest_scenario": results[np.argmin(times)].scenario_name,
            "slowest_scenario": results[np.argmax(times)].scenario_name,
            "average_time": float(np.mean(times)),
            "time_variance": float(np.var(times))
        }

    def _compute_trajectory_distances(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Compute trajectory distances using dynamic time warping or similar."""
        # Placeholder for trajectory distance computation
        return {"note": "Trajectory distance analysis not yet implemented"}

    def _analyze_phase_space(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze behavior in phase space."""
        # Placeholder for phase space analysis
        return {"note": "Phase space analysis not yet implemented"}

    def _perform_sensitivity_analysis(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Perform sensitivity analysis on parameters."""
        # Placeholder for sensitivity analysis
        return {"note": "Sensitivity analysis not yet implemented"}

    def _format_simulation_result(self, result: SimulationResult) -> Dict[str, Any]:
        """Format individual simulation result for inclusion in comparison."""
        return {
            "scenario_name": result.scenario_name,
            "converged": result.converged,
            "execution_time": result.execution_time,
            "time_points_count": len(result.time_points),
            "variables_count": len(result.variable_data)
        }
