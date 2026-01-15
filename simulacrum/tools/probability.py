"""
Probability analysis tool for Bayesian reasoning and uncertainty modeling.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
from ..validation.errors import ValidationError


class ProbabilityAnalyzer:
    """
    Tool for probabilistic analysis and Bayesian reasoning.

    Enables quantitative assessment of uncertainty, hypothesis testing,
    and evidence-based decision making.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize probability analyzer.

        Args:
            confidence_level: Default confidence level for intervals (0-1)
        """
        self.confidence_level = confidence_level

    def analyze_probability(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform probabilistic analysis based on request type.

        Args:
            analysis_request: Dictionary containing analysis parameters with:
                - analysis_type: "bayesian_update", "hypothesis_test", "uncertainty_analysis", etc.
                - data: Input data for analysis
                - parameters: Analysis-specific parameters

        Returns:
            Dictionary containing analysis results

        Raises:
            ValidationError: If request is invalid
        """
        try:
            analysis_type = analysis_request.get('analysis_type')

            if analysis_type == 'bayesian_update':
                return self._bayesian_update(analysis_request)
            elif analysis_type == 'hypothesis_test':
                return self._hypothesis_test(analysis_request)
            elif analysis_type == 'uncertainty_analysis':
                return self._uncertainty_analysis(analysis_request)
            elif analysis_type == 'decision_analysis':
                return self._decision_analysis(analysis_request)
            else:
                raise ValidationError(
                    "analysis_type",
                    analysis_type,
                    f"Unsupported analysis type: {analysis_type}"
                )

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "analysis_request",
                analysis_request,
                f"Probability analysis failed: {str(e)}"
            )

    def _bayesian_update(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Bayesian belief updating."""
        prior = request.get('prior', {})
        likelihood = request.get('likelihood', {})
        evidence = request.get('evidence', [])

        if not isinstance(prior, dict) or not prior:
            raise ValidationError("prior", prior, "Prior must be non-empty dictionary")

        # Normalize prior
        total_prior = sum(prior.values())
        normalized_prior = {k: v/total_prior for k, v in prior.items()}

        posterior = normalized_prior.copy()

        # Apply evidence sequentially
        update_history = []

        for evidence_item in evidence:
            hypothesis = evidence_item.get('hypothesis')
            observation = evidence_item.get('observation')
            strength = evidence_item.get('strength', 1.0)

            if hypothesis not in posterior:
                continue

            # Simple likelihood update (placeholder for more sophisticated model)
            likelihood_ratio = likelihood.get(observation, {}).get(hypothesis, 1.0)
            likelihood_ratio **= strength

            # Update posterior
            for hyp in posterior:
                if hyp == hypothesis:
                    posterior[hyp] *= likelihood_ratio
                else:
                    posterior[hyp] *= (1 - likelihood_ratio * 0.1)  # Small adjustment for alternatives

            # Normalize
            total = sum(posterior.values())
            posterior = {k: v/total for k, v in posterior.items()}

            update_history.append({
                "evidence": evidence_item,
                "posterior": posterior.copy()
            })

        # Compute confidence intervals
        sorted_posterior = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
        confidence_intervals = self._compute_probability_intervals(posterior)

        return {
            "status": "success",
            "analysis_type": "bayesian_update",
            "initial_prior": normalized_prior,
            "final_posterior": posterior,
            "most_likely_hypothesis": sorted_posterior[0][0],
            "confidence_intervals": confidence_intervals,
            "update_history": update_history,
            "evidence_strength": len(evidence),
            "summary": self._generate_bayesian_summary(posterior, sorted_posterior)
        }

    def _hypothesis_test(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical hypothesis testing."""
        data = request.get('data', [])
        null_hypothesis = request.get('null_hypothesis', {})
        alternative_hypothesis = request.get('alternative_hypothesis', {})
        test_type = request.get('test_type', 't_test')

        if not data:
            raise ValidationError("data", data, "Data cannot be empty for hypothesis testing")

        try:
            if test_type == 't_test':
                result = self._perform_t_test(data, null_hypothesis)
            elif test_type == 'proportion_test':
                result = self._perform_proportion_test(data, null_hypothesis)
            elif test_type == 'chi_square':
                result = self._perform_chi_square_test(data, null_hypothesis)
            else:
                result = {"error": f"Unsupported test type: {test_type}"}

            result.update({
                "status": "success",
                "analysis_type": "hypothesis_test",
                "test_type": test_type,
                "sample_size": len(data),
                "null_hypothesis": null_hypothesis,
                "alternative_hypothesis": alternative_hypothesis
            })

            return result

        except Exception as e:
            raise ValidationError("hypothesis_test", request, f"Hypothesis test failed: {str(e)}")

    def _perform_t_test(self, data: List[float], null_hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one-sample t-test."""
        mu_0 = null_hypothesis.get('mean', 0)

        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        n = len(data)

        t_statistic = (sample_mean - mu_0) / (sample_std / np.sqrt(n))

        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))

        # Confidence interval
        se = sample_std / np.sqrt(n)
        t_critical = stats.t.ppf(1 - (1 - self.confidence_level)/2, df=n-1)
        ci_lower = sample_mean - t_critical * se
        ci_upper = sample_mean + t_critical * se

        return {
            "test_statistic": float(t_statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": n - 1,
            "sample_mean": float(sample_mean),
            "sample_std": float(sample_std),
            "null_mean": mu_0,
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "reject_null": p_value < (1 - self.confidence_level),
            "effect_size": float((sample_mean - mu_0) / sample_std)
        }

    def _perform_proportion_test(self, data: List[int], null_hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform proportion test."""
        p_0 = null_hypothesis.get('proportion', 0.5)

        successes = sum(data)
        n = len(data)
        sample_proportion = successes / n

        # Z-test for proportion
        se = np.sqrt(p_0 * (1 - p_0) / n)
        z_statistic = (sample_proportion - p_0) / se

        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

        # Confidence interval
        se_sample = np.sqrt(sample_proportion * (1 - sample_proportion) / n)
        z_critical = stats.norm.ppf(1 - (1 - self.confidence_level)/2)
        ci_lower = sample_proportion - z_critical * se_sample
        ci_upper = sample_proportion + z_critical * se_sample

        return {
            "test_statistic": float(z_statistic),
            "p_value": float(p_value),
            "sample_proportion": float(sample_proportion),
            "null_proportion": p_0,
            "successes": successes,
            "sample_size": n,
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "reject_null": p_value < (1 - self.confidence_level)
        }

    def _perform_chi_square_test(self, data: List[List[int]], null_hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform chi-square test for independence."""
        if not data or not all(isinstance(row, list) for row in data):
            raise ValidationError("data", data, "Chi-square test requires contingency table")

        observed = np.array(data)

        # Calculate expected frequencies
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        total = observed.sum()

        expected = np.outer(row_totals, col_totals) / total

        # Chi-square statistic
        chi_square = np.sum((observed - expected) ** 2 / expected)

        # Degrees of freedom
        df = (len(row_totals) - 1) * (len(col_totals) - 1)

        # P-value
        p_value = 1 - stats.chi2.cdf(chi_square, df)

        return {
            "test_statistic": float(chi_square),
            "p_value": float(p_value),
            "degrees_of_freedom": df,
            "reject_null": p_value < (1 - self.confidence_level)
        }

    def _uncertainty_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze uncertainty in parameters or predictions."""
        parameters = request.get('parameters', {})
        distribution_types = request.get('distributions', {})

        uncertainty_results = {}

        for param_name, param_info in parameters.items():
            dist_type = distribution_types.get(param_name, 'normal')
            uncertainty_results[param_name] = self._analyze_parameter_uncertainty(
                param_name, param_info, dist_type
            )

        # Monte Carlo simulation if requested
        monte_carlo = request.get('monte_carlo', {})
        if monte_carlo.get('enabled', False):
            mc_results = self._run_monte_carlo(uncertainty_results, monte_carlo)
            uncertainty_results['monte_carlo'] = mc_results

        return {
            "status": "success",
            "analysis_type": "uncertainty_analysis",
            "parameter_uncertainty": uncertainty_results,
            "overall_uncertainty": self._compute_overall_uncertainty(uncertainty_results)
        }

    def _analyze_parameter_uncertainty(self, name: str, param_info: Dict[str, Any],
                                     dist_type: str) -> Dict[str, Any]:
        """Analyze uncertainty for a single parameter."""
        if dist_type == 'normal':
            mean = param_info.get('mean', 0)
            std = param_info.get('std', 1)
            dist = stats.norm(mean, std)
        elif dist_type == 'beta':
            a = param_info.get('alpha', 1)
            b = param_info.get('beta', 1)
            dist = stats.beta(a, b)
        else:
            # Default to uniform
            low = param_info.get('low', 0)
            high = param_info.get('high', 1)
            dist = stats.uniform(low, high - low)

        # Compute confidence intervals
        ci_lower, ci_upper = dist.ppf([(1 - self.confidence_level)/2, 1 - (1 - self.confidence_level)/2])

        return {
            "distribution": dist_type,
            "parameters": param_info,
            "mean": float(dist.mean()),
            "variance": float(dist.var()),
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "credibility_interval": [float(ci_lower), float(ci_upper)]
        }

    def _run_monte_carlo(self, uncertainty_results: Dict[str, Any],
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for uncertainty propagation."""
        n_samples = config.get('n_samples', 1000)

        # Generate samples for each parameter
        samples = {}
        for param_name, param_uncertainty in uncertainty_results.items():
            if param_name == 'monte_carlo':
                continue

            dist_type = param_uncertainty['distribution']
            params = param_uncertainty['parameters']

            if dist_type == 'normal':
                samples[param_name] = np.random.normal(
                    params.get('mean', 0),
                    params.get('std', 1),
                    n_samples
                )
            elif dist_type == 'beta':
                samples[param_name] = np.random.beta(
                    params.get('alpha', 1),
                    params.get('beta', 1),
                    n_samples
                )
            else:
                samples[param_name] = np.random.uniform(
                    params.get('low', 0),
                    params.get('high', 1),
                    n_samples
                )

        # Compute output statistics
        # This would be customized based on the actual model
        output_samples = np.mean(list(samples.values()), axis=0)  # Simple average for demo

        return {
            "n_samples": n_samples,
            "output_mean": float(np.mean(output_samples)),
            "output_std": float(np.std(output_samples)),
            "output_percentiles": {
                "5th": float(np.percentile(output_samples, 5)),
                "25th": float(np.percentile(output_samples, 25)),
                "50th": float(np.percentile(output_samples, 50)),
                "75th": float(np.percentile(output_samples, 75)),
                "95th": float(np.percentile(output_samples, 95))
            }
        }

    def _compute_overall_uncertainty(self, uncertainty_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall system uncertainty."""
        # Simplified uncertainty aggregation
        variances = []
        for param_name, param_uncertainty in uncertainty_results.items():
            if param_name not in ['monte_carlo']:
                variances.append(param_uncertainty.get('variance', 0))

        if variances:
            total_variance = sum(variances)
            return {
                "total_variance": float(total_variance),
                "system_uncertainty": float(np.sqrt(total_variance)),
                "dominant_parameter": max(
                    uncertainty_results.keys(),
                    key=lambda x: uncertainty_results[x].get('variance', 0) if x != 'monte_carlo' else 0
                )
            }

        return {"total_variance": 0.0, "system_uncertainty": 0.0}

    def _decision_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform decision analysis under uncertainty."""
        options = request.get('options', [])
        criteria = request.get('criteria', [])
        weights = request.get('weights', [])

        if not options or not criteria:
            raise ValidationError("decision_analysis", request,
                                "Options and criteria cannot be empty")

        # Simple weighted sum model (placeholder)
        scores = {}
        for option in options:
            total_score = 0
            for i, criterion in enumerate(criteria):
                weight = weights[i] if i < len(weights) else 1.0 / len(criteria)
                value = option.get(criterion, 0)
                total_score += weight * value
            scores[option.get('name', str(option))] = total_score

        # Rank options
        ranked_options = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "status": "success",
            "analysis_type": "decision_analysis",
            "ranking": [{"option": opt, "score": score} for opt, score in ranked_options],
            "best_option": ranked_options[0][0] if ranked_options else None,
            "scores": scores
        }

    def _compute_probability_intervals(self, distribution: Dict[str, float]) -> Dict[str, Any]:
        """Compute confidence/credibility intervals."""
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        # Highest density interval approximation
        total_prob = sum(distribution.values())
        cumulative = 0
        hdi_items = []

        for item, prob in sorted_items:
            hdi_items.append(item)
            cumulative += prob
            if cumulative >= self.confidence_level * total_prob:
                break

        return {
            "confidence_level": self.confidence_level,
            "highest_density_interval": hdi_items,
            "interval_coverage": cumulative / total_prob
        }

    def _generate_bayesian_summary(self, posterior: Dict[str, float],
                                 sorted_posterior: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Generate summary of Bayesian analysis."""
        most_likely = sorted_posterior[0][1]
        least_likely = sorted_posterior[-1][1]

        return {
            "most_likely_probability": float(most_likely),
            "belief_strength_ratio": float(most_likely / least_likely) if least_likely > 0 else float('inf'),
            "uncertainty_level": float(1 - most_likely),  # 1 - max probability
            "key_insight": f"Strongest belief in {sorted_posterior[0][0]} with {most_likely:.1%} confidence"
        }
