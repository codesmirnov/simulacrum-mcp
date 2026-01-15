"""
Tests for chaos analysis functionality.
"""

import pytest
import numpy as np
from simulacrum.tools.chaos import ChaosAnalyzer
from simulacrum.validation.errors import ValidationError


class TestChaosAnalyzer:
    """Test suite for chaos analysis."""

    @pytest.fixture
    def chaos_analyzer(self):
        """Create chaos analyzer."""
        return ChaosAnalyzer(lyapunov_tolerance=1e-6, embedding_dimension=3)

    @pytest.fixture
    def sample_time_series(self):
        """Generate sample time series for testing."""
        # Create a simple logistic map time series (chaotic)
        x = 0.5
        series = []
        for _ in range(100):
            x = 4 * x * (1 - x)
            series.append(x)
        return series

    @pytest.fixture
    def stable_time_series(self):
        """Generate stable time series."""
        # Simple damped oscillation
        t = np.linspace(0, 10, 100)
        return list(np.exp(-0.1 * t) * np.cos(2 * t))

    def test_analyze_chaos_single_series(self, chaos_analyzer, sample_time_series):
        """Test chaos analysis with single time series."""
        data = {"time_series": sample_time_series}

        result = chaos_analyzer.analyze_chaos(data)

        assert result["status"] == "success"
        assert result["analysis_type"] == "chaos_analysis"
        assert "main_series" in result["variables_analyzed"]
        assert "results" in result
        assert "main_series" in result["results"]
        assert "overall_assessment" in result["results"]
        assert "early_warning_signals" in result

    def test_analyze_chaos_multiple_variables(self, chaos_analyzer, sample_time_series, stable_time_series):
        """Test chaos analysis with multiple variables."""
        data = {
            "variables": {
                "chaotic": sample_time_series,
                "stable": stable_time_series
            }
        }

        result = chaos_analyzer.analyze_chaos(data)

        assert result["status"] == "success"
        assert len(result["variables_analyzed"]) == 2
        assert "chaotic" in result["results"]
        assert "stable" in result["results"]
        assert "system_analysis" in result["results"]

    def test_analyze_chaos_invalid_data(self, chaos_analyzer):
        """Test error handling with invalid data."""
        # No time_series or variables
        with pytest.raises(ValidationError):
            chaos_analyzer.analyze_chaos({})

        # Too short time series
        with pytest.raises(ValidationError):
            chaos_analyzer.analyze_chaos({"time_series": [1, 2, 3]})

    def test_analyze_single_series(self, chaos_analyzer, sample_time_series):
        """Test analysis of single time series."""
        series = np.array(sample_time_series)
        result = chaos_analyzer._analyze_single_series(series, "test_series")

        required_keys = [
            "statistics", "lyapunov_exponent", "correlation_dimension",
            "entropy", "phase_space", "bifurcations"
        ]

        for key in required_keys:
            assert key in result

    def test_compute_basic_statistics(self, chaos_analyzer, sample_time_series):
        """Test basic statistics computation."""
        series = np.array(sample_time_series)
        stats = chaos_analyzer._compute_basic_statistics(series)

        required_stats = [
            "mean", "std", "variance", "skewness", "kurtosis",
            "autocorrelation", "hurst_exponent"
        ]

        for stat in required_stats:
            assert stat in stats

        assert stats["mean"] == pytest.approx(np.mean(series), rel=1e-10)
        assert stats["std"] == pytest.approx(np.std(series), rel=1e-10)

    def test_compute_autocorrelation(self, chaos_analyzer, sample_time_series):
        """Test autocorrelation computation."""
        series = np.array(sample_time_series)
        autocorr = chaos_analyzer._compute_autocorrelation(series, max_lag=10)

        assert isinstance(autocorr, list)
        assert len(autocorr) == min(10, len(sample_time_series)-1)  # lag 0 to min(max_lag-1, n-2)

    def test_estimate_hurst_exponent(self, chaos_analyzer, sample_time_series):
        """Test Hurst exponent estimation."""
        series = np.array(sample_time_series)
        hurst = chaos_analyzer._estimate_hurst_exponent(series)

        assert isinstance(hurst, float)
        # Hurst exponent should be between 0 and 1 for meaningful series
        assert 0 <= hurst <= 1

    def test_estimate_lyapunov_exponent(self, chaos_analyzer, sample_time_series):
        """Test Lyapunov exponent estimation."""
        series = np.array(sample_time_series)
        lyapunov = chaos_analyzer._estimate_lyapunov_exponent(series)

        assert isinstance(lyapunov, dict)
        assert "exponent" in lyapunov

    def test_embed_series(self, chaos_analyzer, sample_time_series):
        """Test phase space embedding."""
        series = np.array(sample_time_series)
        embedded = chaos_analyzer._embed_series(series, m=2, tau=1)

        assert isinstance(embedded, np.ndarray)
        assert embedded.shape[1] == 2  # embedding dimension

    def test_chaos_analyzer_initialization(self):
        """Test chaos analyzer initialization."""
        analyzer = ChaosAnalyzer(lyapunov_tolerance=1e-5, embedding_dimension=5)

        assert analyzer.lyapunov_tolerance == 1e-5
        assert analyzer.embedding_dimension == 5
