"""
Simulacrum MCP - Reality Engine for AI Systems

This package provides a comprehensive toolkit for simulating complex dynamics,
enabling AI systems to understand and predict real-world phenomena through
mathematical modeling and system dynamics simulation.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"
__author__ = "codesmirnov"
__description__ = "Reality Engine MCP - Advanced simulation toolkit for AI systems"

from .core.engine import SimulationEngine
from .tools.dynamics import DynamicsSimulator
from .tools.feedback import FeedbackAnalyzer
from .tools.belief import BeliefDynamics
from .tools.game_theory import GameTheoryDynamics
from .tools.probability import ProbabilityAnalyzer
from .tools.chaos import ChaosAnalyzer
from .tools.comparison import ScenarioComparator
from .tools.vector_analysis import VectorAnalyzer

__all__ = [
    "SimulationEngine",
    "DynamicsSimulator",
    "FeedbackAnalyzer",
    "BeliefDynamics",
    "GameTheoryDynamics",
    "ProbabilityAnalyzer",
    "ChaosAnalyzer",
    "ScenarioComparator",
    "VectorAnalyzer",
]
