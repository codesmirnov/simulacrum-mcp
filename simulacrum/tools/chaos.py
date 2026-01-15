"""
Chaos analysis tool for detecting black swan events and system instabilities.
"""

import math
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union
from ..validation.errors import ValidationError


class ChaosAnalyzer:
    """
    Tool for analyzing chaotic behavior and detecting potential black swan events.

    Identifies system instabilities, tipping points, and early warning signals
    for catastrophic changes.
    """

    def __init__(self, lyapunov_tolerance: float = 1e-6, embedding_dimension: int = 3):
        """
        Initialize chaos analyzer.

        Args:
            lyapunov_tolerance: Tolerance for Lyapunov exponent calculation
            embedding_dimension: Default embedding dimension for phase space reconstruction
        """
        self.lyapunov_tolerance = lyapunov_tolerance
        self.embedding_dimension = embedding_dimension

    def analyze_chaos(self, time_series_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze time series for chaotic behavior and instability indicators.

        Args:
            time_series_data: Dictionary containing:
                - time_series: List of time series values
                - time_points: Optional time points
                - variables: Optional multiple variable time series

        Returns:
            Dictionary containing chaos analysis results

        Raises:
            ValidationError: If data is invalid
        """
        try:
            # Extract time series
            if 'time_series' in time_series_data:
                time_series = [np.array(time_series_data['time_series'])]
                variable_names = ['main_series']
            elif 'variables' in time_series_data:
                variables = time_series_data['variables']
                time_series = [np.array(var_data) for var_data in variables.values()]
                variable_names = list(variables.keys())
            else:
                raise ValidationError(
                    "time_series_data",
                    time_series_data,
                    "Must provide either 'time_series' or 'variables'"
                )

            # Validate data
            for i, series in enumerate(time_series):
                if len(series) < 10:
                    raise ValidationError(
                        "time_series",
                        series,
                        f"Time series {variable_names[i]} too short (minimum 10 points)"
                    )

            results = {}

            # Analyze each time series
            for i, series in enumerate(time_series):
                var_name = variable_names[i]
                results[var_name] = self._analyze_single_series(series, var_name)

            # Cross-variable analysis if multiple series
            if len(time_series) > 1:
                results['system_analysis'] = self._analyze_system_chaos(time_series, variable_names)

            # Overall assessment
            results['overall_assessment'] = self._generate_overall_assessment(results)

            return {
                "status": "success",
                "analysis_type": "chaos_analysis",
                "variables_analyzed": variable_names,
                "results": results,
                "early_warning_signals": self._detect_early_warnings(results)
            }

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "time_series_data",
                time_series_data,
                f"Chaos analysis failed: {str(e)}"
            )

    def _analyze_single_series(self, series: np.ndarray, name: str) -> Dict[str, Any]:
        """Analyze a single time series for chaotic indicators."""
        analysis = {}

        # Basic statistical properties
        analysis['statistics'] = self._compute_basic_statistics(series)

        # Lyapunov exponent estimation
        analysis['lyapunov_exponent'] = self._estimate_lyapunov_exponent(series)

        # Correlation dimension
        analysis['correlation_dimension'] = self._estimate_correlation_dimension(series)

        # Entropy measures
        analysis['entropy'] = self._compute_entropy_measures(series)

        # Phase space analysis
        analysis['phase_space'] = self._analyze_phase_space(series)

        # Bifurcation detection
        analysis['bifurcations'] = self._detect_bifurcations(series)

        # Stability assessment
        analysis['stability'] = self._assess_stability(analysis)

        return analysis

    def _compute_basic_statistics(self, series: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistical properties."""
        return {
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            "variance": float(np.var(series)),
            "skewness": float(stats.skew(series)),
            "kurtosis": float(stats.kurtosis(series)),
            "autocorrelation": self._compute_autocorrelation(series),
            "hurst_exponent": self._estimate_hurst_exponent(series)
        }

    def _compute_autocorrelation(self, series: np.ndarray, max_lag: int = 20) -> List[float]:
        """Compute autocorrelation function."""
        n = len(series)
        mean = np.mean(series)
        var = np.var(series)

        autocorr = []
        for lag in range(min(max_lag, n-1)):
            cov = np.sum((series[:-lag-1] - mean) * (series[lag+1:] - mean)) / (n - lag)
            autocorr.append(float(cov / var))

        return autocorr

    def _estimate_hurst_exponent(self, series: np.ndarray) -> float:
        """Estimate Hurst exponent for long-range dependence."""
        # Simplified Hurst exponent estimation
        n = len(series)
        if n < 10:
            return 0.5  # Random walk default

        # R/S analysis
        rs_values = []
        for subset_size in range(10, min(n//2, 100), 10):
            subsets = [series[i:i+subset_size] for i in range(0, n-subset_size+1, subset_size//2)]
            rs_subset = []

            for subset in subsets:
                mean_val = np.mean(subset)
                cumsum = np.cumsum(subset - mean_val)
                r = max(cumsum) - min(cumsum)
                s = np.std(subset)
                if s > 0:
                    rs_subset.append(r / s)

            if rs_subset:
                rs_values.append(np.mean(rs_subset))

        if len(rs_values) > 1:
            # Fit log-log regression
            x = np.log(range(10, 10 + 10*len(rs_values), 10))
            y = np.log(rs_values)
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)

        return 0.5

    def _estimate_lyapunov_exponent(self, series: np.ndarray) -> Dict[str, Any]:
        """Estimate largest Lyapunov exponent."""
        # Simplified Lyapunov exponent estimation using Rosenstein method
        m = min(self.embedding_dimension, len(series)//10)
        tau = 1  # Time delay

        if len(series) < 2*m + 1:
            return {"exponent": 0.0, "confidence": "insufficient_data"}

        # Phase space reconstruction
        embedded = self._embed_series(series, m, tau)

        # Find nearest neighbors and track divergence
        divergences = []

        for i in range(len(embedded)):
            distances = []
            for j in range(len(embedded)):
                if i != j:
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append((dist, j))

            if distances:
                # Find closest neighbor
                distances.sort()
                nearest_idx = distances[0][1]
                initial_distance = distances[0][0]

                if initial_distance > 0:
                    # Track divergence
                    divergence = [initial_distance]
                    for t in range(1, min(20, len(embedded) - max(i, nearest_idx))):
                        dist_t = np.linalg.norm(embedded[i+t] - embedded[nearest_idx+t])
                        divergence.append(dist_t)

                    if len(divergence) > 5:
                        # Fit exponential
                        t_vals = np.arange(len(divergence))
                        log_div = np.log(np.array(divergence) + 1e-10)  # Avoid log(0)

                        # Linear fit for slope (Lyapunov exponent)
                        if len(log_div) > 1:
                            slope = np.polyfit(t_vals, log_div, 1)[0]
                            divergences.append(slope)

        if divergences:
            lyapunov = float(np.mean(divergences))
            std_lyapunov = float(np.std(divergences))

            return {
                "exponent": lyapunov,
                "std": std_lyapunov,
                "is_chaotic": lyapunov > 0.1,  # Positive Lyapunov indicates chaos
                "confidence": "estimated" if std_lyapunov < abs(lyapunov) else "uncertain"
            }

        return {"exponent": 0.0, "confidence": "insufficient_data"}

    def _embed_series(self, series: np.ndarray, m: int, tau: int) -> np.ndarray:
        """Create phase space embedding."""
        n = len(series) - (m-1)*tau
        embedded = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                embedded[i, j] = series[i + j*tau]

        return embedded

    def _estimate_correlation_dimension(self, series: np.ndarray) -> Dict[str, Any]:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
        # Simplified correlation sum calculation
        m = min(self.embedding_dimension, len(series)//10)

        if len(series) < 2*m:
            return {"dimension": 0.0, "confidence": "insufficient_data"}

        embedded = self._embed_series(series, m, 1)

        # Calculate correlation sum for different radii
        radii = np.logspace(-3, 0, 20)
        correlation_sums = []

        for r in radii:
            count = 0
            total_pairs = 0

            for i in range(len(embedded)):
                for j in range(i+1, len(embedded)):
                    distance = np.linalg.norm(embedded[i] - embedded[j])
                    if distance <= r:
                        count += 1
                    total_pairs += 1

            correlation_sum = count / total_pairs if total_pairs > 0 else 0
            correlation_sums.append(correlation_sum)

        # Estimate dimension from scaling region
        log_r = np.log(radii)
        log_c = np.log(np.array(correlation_sums) + 1e-10)

        # Find scaling region (middle portion)
        start_idx = len(log_r) // 3
        end_idx = 2 * len(log_r) // 3

        if end_idx > start_idx:
            slope = np.polyfit(log_r[start_idx:end_idx], log_c[start_idx:end_idx], 1)[0]
            dimension = float(slope)

            return {
                "dimension": dimension,
                "is_fractal": 0 < dimension < m,  # Non-integer dimension suggests chaos
                "confidence": "estimated"
            }

        return {"dimension": 0.0, "confidence": "insufficient_data"}

    def _compute_entropy_measures(self, series: np.ndarray) -> Dict[str, Any]:
        """Compute various entropy measures."""
        # Approximate entropy
        approx_entropy = self._approximate_entropy(series, m=2, r=0.2)

        # Sample entropy
        sample_entropy = self._sample_entropy(series, m=2, r=0.2)

        # Permutation entropy
        perm_entropy = self._permutation_entropy(series, order=3)

        return {
            "approximate_entropy": float(approx_entropy),
            "sample_entropy": float(sample_entropy),
            "permutation_entropy": float(perm_entropy),
            "regularity": "high" if sample_entropy < 1.0 else "low"
        }

    def _approximate_entropy(self, series: np.ndarray, m: int, r: float) -> float:
        """Calculate approximate entropy."""
        def phi(m_val):
            patterns = []
            for i in range(len(series) - m_val + 1):
                pattern = series[i:i+m_val]
                patterns.append(pattern)

            distances = []
            for i in range(len(patterns)):
                count = 0
                for j in range(len(patterns)):
                    if i != j:
                        dist = np.max(np.abs(patterns[i] - patterns[j]))
                        if dist <= r:
                            count += 1
                distances.append(count / (len(patterns) - 1))

            return np.mean(np.log(np.array(distances) + 1e-10))

        return phi(m) - phi(m + 1)

    def _sample_entropy(self, series: np.ndarray, m: int, r: float) -> float:
        """Calculate sample entropy."""
        def count_matches(patterns, m_val):
            count = 0
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):  # Compare each pair only once
                    dist = np.max(np.abs(patterns[i] - patterns[j]))
                    if dist <= r:
                        count += 1
            return count

        # Create all patterns of length m
        patterns_m = []
        for i in range(len(series) - m + 1):
            patterns_m.append(series[i:i+m])

        # Create all patterns of length m+1
        patterns_m1 = []
        for i in range(len(series) - m):
            patterns_m1.append(series[i:i+m+1])

        matches_m = count_matches(patterns_m, m)
        matches_m1 = count_matches(patterns_m1, m+1)

        if matches_m == 0:
            return float('inf')

        return -np.log(matches_m1 / matches_m)

    def _permutation_entropy(self, series: np.ndarray, order: int) -> float:
        """Calculate permutation entropy."""
        n = len(series)
        permutations = {}

        for i in range(n - order):
            window = series[i:i+order]
            # Get permutation pattern
            sorted_indices = np.argsort(window)
            pattern = tuple(np.argsort(sorted_indices))

            permutations[pattern] = permutations.get(pattern, 0) + 1

        # Calculate entropy
        total = sum(permutations.values())
        entropy = 0

        for count in permutations.values():
            p = count / total
            entropy -= p * np.log2(p)

        return entropy / np.log2(math.factorial(order))

    def _analyze_phase_space(self, series: np.ndarray) -> Dict[str, Any]:
        """Analyze phase space properties."""
        m = min(self.embedding_dimension, len(series)//10)

        if len(series) < 2*m:
            return {"attractor_type": "unknown", "confidence": "insufficient_data"}

        embedded = self._embed_series(series, m, 1)

        # Simple attractor classification
        centroid = np.mean(embedded, axis=0)
        distances = [np.linalg.norm(point - centroid) for point in embedded]

        # Check for strange attractor properties
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Rough classification
        if std_distance / mean_distance < 0.1:
            attractor_type = "fixed_point"
        elif std_distance / mean_distance < 0.5:
            attractor_type = "limit_cycle"
        else:
            attractor_type = "strange_attractor"

        return {
            "attractor_type": attractor_type,
            "embedding_dimension": m,
            "phase_space_volume": float(np.var(distances)),
            "centroid": centroid.tolist()
        }

    def _detect_bifurcations(self, series: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential bifurcation points."""
        # Simple bifurcation detection based on variance changes
        window_size = max(10, len(series)//20)
        bifurcations = []

        for i in range(window_size, len(series) - window_size, window_size//2):
            before = series[i-window_size:i]
            after = series[i:i+window_size]

            var_before = np.var(before)
            var_after = np.var(after)

            if var_after > 2 * var_before and var_before > 0:
                bifurcations.append({
                    "position": i,
                    "variance_ratio": float(var_after / var_before),
                    "type": "variance_increase",
                    "confidence": "medium"
                })

        return bifurcations

    def _assess_stability(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system stability."""
        lyapunov = analysis.get('lyapunov_exponent', {}).get('exponent', 0)
        correlation_dim = analysis.get('correlation_dimension', {}).get('dimension', 0)
        entropy = analysis.get('entropy', {}).get('sample_entropy', 1)

        # Stability indicators
        stability_score = 0

        # Lyapunov exponent (negative = stable, positive = chaotic)
        if lyapunov < -0.1:
            stability_score += 2
        elif lyapunov > 0.1:
            stability_score -= 2

        # Correlation dimension (low = more ordered)
        if correlation_dim < 2:
            stability_score += 1
        elif correlation_dim > 3:
            stability_score -= 1

        # Entropy (low = more regular/ordered)
        if entropy < 0.5:
            stability_score += 1
        elif entropy > 1.5:
            stability_score -= 1

        stability_level = "unknown"
        if stability_score >= 2:
            stability_level = "stable"
        elif stability_score <= -2:
            stability_level = "chaotic"
        else:
            stability_level = "transitional"

        return {
            "stability_level": stability_level,
            "stability_score": stability_score,
            "risk_level": "high" if stability_level == "chaotic" else "medium" if stability_level == "transitional" else "low",
            "indicators": {
                "lyapunov_contribution": -1 if lyapunov > 0.1 else 1 if lyapunov < -0.1 else 0,
                "dimension_contribution": 1 if correlation_dim < 2 else -1 if correlation_dim > 3 else 0,
                "entropy_contribution": 1 if entropy < 0.5 else -1 if entropy > 1.5 else 0
            }
        }

    def _analyze_system_chaos(self, time_series: List[np.ndarray],
                            variable_names: List[str]) -> Dict[str, Any]:
        """Analyze chaos in multi-variable system."""
        # Cross-correlation analysis
        correlations = {}
        for i in range(len(time_series)):
            for j in range(i+1, len(time_series)):
                corr = np.corrcoef(time_series[i], time_series[j])[0, 1]
                correlations[f"{variable_names[i]}_{variable_names[j]}"] = float(corr)

        # Synchrony analysis
        synchrony = self._analyze_synchrony(time_series)

        return {
            "cross_correlations": correlations,
            "system_synchrony": synchrony,
            "coupling_strength": float(np.mean(np.abs(list(correlations.values()))))
        }

    def _analyze_synchrony(self, time_series: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze synchronization between variables."""
        if len(time_series) < 2:
            return {"synchrony_level": "single_variable"}

        # Phase synchrony (simplified)
        phases = []
        for series in time_series:
            # Hilbert transform for phase extraction (simplified)
            analytic = series + 1j * self._hilbert_transform(series)
            phase = np.angle(analytic)
            phases.append(phase)

        # Mean phase coherence
        coherence_values = []
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                phase_diff = phases[i] - phases[j]
                coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
                coherence_values.append(coherence)

        mean_coherence = float(np.mean(coherence_values)) if coherence_values else 0.0

        synchrony_level = "high" if mean_coherence > 0.8 else "medium" if mean_coherence > 0.5 else "low"

        return {
            "synchrony_level": synchrony_level,
            "mean_phase_coherence": mean_coherence,
            "coherence_values": [float(c) for c in coherence_values]
        }

    def _hilbert_transform(self, series: np.ndarray) -> np.ndarray:
        """Simple Hilbert transform approximation."""
        # Very basic approximation - in practice use scipy.signal.hilbert
        n = len(series)
        hilbert = np.zeros(n)

        for i in range(n):
            sum_val = 0
            for j in range(n):
                if i != j:
                    sum_val += series[j] / (i - j + 1e-10)  # Avoid division by zero
            hilbert[i] = sum_val / np.pi

        return hilbert

    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall chaos assessment."""
        stability_levels = []
        lyapunov_values = []
        dimension_values = []

        for var_name, var_results in results.items():
            if isinstance(var_results, dict) and 'stability' in var_results:
                stability_levels.append(var_results['stability']['stability_level'])

                if 'lyapunov_exponent' in var_results:
                    lyapunov = var_results['lyapunov_exponent'].get('exponent', 0)
                    lyapunov_values.append(lyapunov)

                if 'correlation_dimension' in var_results:
                    dim = var_results['correlation_dimension'].get('dimension', 0)
                    dimension_values.append(dim)

        # Overall assessment
        chaotic_vars = sum(1 for level in stability_levels if level == "chaotic")
        stable_vars = sum(1 for level in stability_levels if level == "stable")

        positive_lyapunov = sum(1 for l in lyapunov_values if l > 0.1)

        if chaotic_vars > stable_vars or positive_lyapunov > len(lyapunov_values) / 2:
            system_state = "chaotic"
            risk_level = "high"
        elif stable_vars > chaotic_vars:
            system_state = "stable"
            risk_level = "low"
        else:
            system_state = "transitional"
            risk_level = "medium"

        return {
            "system_state": system_state,
            "risk_level": risk_level,
            "chaotic_variables": chaotic_vars,
            "stable_variables": stable_vars,
            "positive_lyapunov_count": positive_lyapunov,
            "recommendations": self._generate_recommendations(system_state, risk_level)
        }

    def _generate_recommendations(self, system_state: str, risk_level: str) -> List[str]:
        """Generate recommendations based on system state."""
        recommendations = []

        if risk_level == "high":
            recommendations.extend([
                "Implement robust monitoring systems",
                "Prepare contingency plans for black swan events",
                "Consider system redesign to reduce coupling",
                "Increase feedback loop monitoring"
            ])

        if system_state == "chaotic":
            recommendations.extend([
                "Look for early warning signals",
                "Monitor for bifurcation points",
                "Consider stabilizing interventions",
                "Increase system resilience measures"
            ])

        if system_state == "transitional":
            recommendations.extend([
                "Monitor for tipping points",
                "Track key indicators regularly",
                "Prepare for multiple scenarios"
            ])

        if not recommendations:
            recommendations.append("System appears stable - continue normal monitoring")

        return recommendations

    def _detect_early_warnings(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect early warning signals for critical transitions."""
        warnings = []

        for var_name, var_results in results.items():
            if not isinstance(var_results, dict):
                continue

            # Check for critical slowing down indicators
            if 'statistics' in var_results:
                stats = var_results['statistics']

                # Increasing autocorrelation (critical slowing down)
                autocorr = stats.get('autocorrelation', [])
                if len(autocorr) > 5:
                    recent_autocorr = np.mean(autocorr[-3:])  # Last 3 lags
                    earlier_autocorr = np.mean(autocorr[:3])  # First 3 lags

                    if recent_autocorr > earlier_autocorr * 1.5:
                        warnings.append({
                            "type": "critical_slowing_down",
                            "variable": var_name,
                            "indicator": "autocorrelation_increase",
                            "severity": "medium",
                            "description": f"Increasing autocorrelation in {var_name} suggests approaching tipping point"
                        })

                # Increasing variance (another CSD indicator)
                # This would require time-windowed variance analysis

            # Check for flickering (oscillations near bifurcation)
            if 'bifurcations' in var_results and var_results['bifurcations']:
                warnings.append({
                    "type": "bifurcation_warning",
                    "variable": var_name,
                    "indicator": "bifurcation_detected",
                    "severity": "high",
                    "description": f"Bifurcation detected in {var_name} - system may be near critical transition"
                })

        return warnings
