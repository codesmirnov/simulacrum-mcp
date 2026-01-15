"""
Custom exception classes for Simulacrum MCP.
"""

from typing import Any, Dict, Optional


class SimulacrumError(Exception):
    """Base exception for all Simulacrum errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERIC_ERROR"
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(SimulacrumError):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: Any, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Validation failed for field '{field}': {reason}"
        super().__init__(message, "VALIDATION_ERROR", details or {})
        self.field = field
        self.value = value
        self.reason = reason


class SimulationError(SimulacrumError):
    """Raised when simulation execution fails."""

    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Simulation operation '{operation}' failed: {reason}"
        super().__init__(message, "SIMULATION_ERROR", details or {})
        self.operation = operation
        self.reason = reason


class ConvergenceError(SimulationError):
    """Raised when simulation fails to converge."""

    def __init__(self, max_iterations: int, tolerance: float, final_error: float):
        message = f"Simulation failed to converge after {max_iterations} iterations"
        details = {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "final_error": final_error,
        }
        super().__init__("convergence", message, details)


class NumericError(SimulationError):
    """Raised when numerical computation fails."""

    def __init__(self, operation: str, reason: str, values: Optional[Dict[str, Any]] = None):
        message = f"Numerical error in '{operation}': {reason}"
        details = values or {}
        super().__init__(operation, message, details)
