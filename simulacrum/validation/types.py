"""
Pydantic models for data validation and type safety.
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, model_validator
import numpy as np


class SimulationMode(str, Enum):
    """Enumeration of supported simulation modes."""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    MONTE_CARLO = "monte_carlo"


class ConvergenceCriteria(BaseModel):
    """Configuration for simulation convergence criteria."""

    max_iterations: int = Field(default=1000, ge=1, le=100000)
    tolerance: float = Field(default=1e-6, gt=0, le=1e-3)
    adaptive_tolerance: bool = Field(default=True)
    patience: int = Field(default=50, ge=1, le=1000)

    @validator('tolerance')
    def validate_tolerance(cls, v):
        if v <= 0:
            raise ValueError("Tolerance must be positive")
        if v > 1e-3:
            raise ValueError("Tolerance too loose, may cause inaccurate results")
        return v


class TimeConfig(BaseModel):
    """Configuration for simulation time parameters."""

    start_time: float = Field(default=0.0)
    end_time: float = Field(default=10.0, gt=0)
    time_step: float = Field(default=0.01, gt=0)
    adaptive_step: bool = Field(default=False)

    @model_validator(mode='before')
    @classmethod
    def validate_time_range(cls, values):
        if isinstance(values, dict):
            start_time = values.get('start_time', 0.0)
            end_time = values.get('end_time', 10.0)
            if end_time <= start_time:
                raise ValueError("End time must be greater than start time")
        return values


class SimulationConfig(BaseModel):
    """Main configuration for simulation runs."""

    mode: SimulationMode = Field(default=SimulationMode.DETERMINISTIC)
    time_config: TimeConfig = Field(default_factory=TimeConfig)
    convergence: ConvergenceCriteria = Field(default_factory=ConvergenceCriteria)
    random_seed: Optional[int] = Field(default=None, ge=0)

    class Config:
        validate_assignment = True


class VariableDefinition(BaseModel):
    """Definition of a simulation variable."""

    name: str = Field(..., min_length=1, max_length=50)
    initial_value: float = Field(...)
    min_value: Optional[float] = Field(default=None)
    max_value: Optional[float] = Field(default=None)
    description: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode='before')
    @classmethod
    def validate_bounds(cls, values):
        if isinstance(values, dict):
            min_val = values.get('min_value')
            max_val = values.get('max_value')
            initial = values.get('initial_value')

            if min_val is not None and max_val is not None and min_val >= max_val:
                raise ValueError("Minimum value must be less than maximum value")

            if min_val is not None and initial < min_val:
                raise ValueError("Initial value cannot be less than minimum value")

            if max_val is not None and initial > max_val:
                raise ValueError("Initial value cannot be greater than maximum value")
        return values


class EquationDefinition(BaseModel):
    """Definition of a differential equation or relationship."""

    target_variable: str = Field(..., min_length=1, max_length=50)
    expression: str = Field(..., min_length=1, max_length=500)
    parameters: Dict[str, float] = Field(default_factory=dict)

    @validator('expression')
    def validate_expression(cls, v):
        # Basic validation - check for balanced parentheses
        if v.count('(') != v.count(')'):
            raise ValueError("Unbalanced parentheses in expression")
        return v


class ScenarioData(BaseModel):
    """Complete scenario definition for simulation."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    variables: List[VariableDefinition] = Field(..., min_items=1)
    equations: List[EquationDefinition] = Field(..., min_items=1)
    config: SimulationConfig = Field(default_factory=SimulationConfig)

    @validator('variables')
    def validate_unique_variable_names(cls, v):
        names = [var.name for var in v]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique")
        return v

    @validator('equations')
    def validate_equation_targets(cls, v):
        targets = [eq.target_variable for eq in v]
        if len(targets) != len(set(targets)):
            raise ValueError("Equation targets must be unique")
        return v


class FeedbackLoop(BaseModel):
    """Representation of a feedback loop in a system."""

    name: str = Field(..., min_length=1, max_length=50)
    variables: List[str] = Field(..., min_items=2)
    loop_type: str = Field(..., pattern=r'^(reinforcing|balancing)$')
    strength: float = Field(default=1.0, ge=0, le=10)
    description: Optional[str] = Field(default=None, max_length=200)


class BeliefState(BaseModel):
    """Representation of an agent's belief state."""

    agent_id: str = Field(..., min_length=1, max_length=50)
    beliefs: Dict[str, float] = Field(..., min_items=1)  # belief_name -> confidence (0-1)
    evidence_weight: float = Field(default=1.0, ge=0, le=1)
    adaptability: float = Field(default=0.1, ge=0, le=1)

    @validator('beliefs')
    def validate_belief_confidence(cls, v):
        for belief, confidence in v.items():
            if not (0 <= confidence <= 1):
                raise ValueError(f"Belief confidence for '{belief}' must be between 0 and 1")
        return v


class SimulationResult(BaseModel):
    """Container for simulation results."""

    scenario_name: str
    time_points: List[float]
    variable_data: Dict[str, List[float]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    converged: bool

    @validator('time_points')
    def validate_time_points(cls, v):
        if len(v) < 2:
            raise ValueError("Must have at least 2 time points")
        if not all(v[i] <= v[i+1] for i in range(len(v)-1)):
            raise ValueError("Time points must be monotonically increasing")
        return v

    @validator('variable_data')
    def validate_variable_data(cls, v):
        if not v:
            raise ValueError("Variable data cannot be empty")
        lengths = [len(data) for data in v.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All variable data arrays must have the same length")
        return v
