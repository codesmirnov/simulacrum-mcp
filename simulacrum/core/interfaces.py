"""
Abstract interfaces for the simulation system following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Protocol
from ..validation.types import ScenarioData, SimulationResult


class ISimulator(Protocol):
    """Interface for simulation execution."""

    def simulate(self, scenario: ScenarioData) -> SimulationResult:
        """Execute simulation for given scenario."""
        ...

    def validate_scenario(self, scenario: ScenarioData) -> List[str]:
        """Validate scenario before simulation."""
        ...


class IValidator(ABC):
    """Abstract base class for validation logic."""

    @abstractmethod
    def validate(self, data: Any) -> List[str]:
        """Validate input data and return list of error messages."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return validation schema."""
        pass


class IResultFormatter(ABC):
    """Abstract base class for result formatting."""

    @abstractmethod
    def format_result(self, result: SimulationResult, format_type: str = "json") -> str:
        """Format simulation result into specified format."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported output formats."""
        pass


class INumericalSolver(ABC):
    """Abstract base class for numerical integration methods."""

    @abstractmethod
    def solve(self, equations: List[Any], initial_conditions: Dict[str, float],
             time_config: Any) -> Dict[str, List[float]]:
        """Solve system of equations numerically."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return name of the numerical method."""
        pass

    @abstractmethod
    def estimate_error(self) -> float:
        """Estimate numerical error of the solution."""
        pass


class IEquationParser(ABC):
    """Abstract base class for equation parsing and evaluation."""

    @abstractmethod
    def parse(self, expression: str) -> Any:
        """Parse mathematical expression into executable form."""
        pass

    @abstractmethod
    def evaluate(self, parsed_expression: Any, variables: Dict[str, float]) -> float:
        """Evaluate parsed expression with given variable values."""
        pass

    @abstractmethod
    def get_dependencies(self, expression: str) -> List[str]:
        """Extract variable dependencies from expression."""
        pass
