"""
MCP Server for Simulacrum - Reality Engine for AI Systems.

This server provides Model Context Protocol (MCP) integration for Cursor AI,
enabling access to advanced simulation and analysis tools.
"""

import logging
from typing import Any, Dict
from mcp.server import FastMCP

from .tools.dynamics import DynamicsSimulator
from .tools.feedback import FeedbackAnalyzer
from .tools.belief import BeliefDynamics
from .tools.game_theory import GameTheoryDynamics
from .tools.probability import ProbabilityAnalyzer
from .tools.chaos import ChaosAnalyzer
from .tools.comparison import ScenarioComparator
from .tools.vector_analysis import VectorAnalyzer
from .validation.errors import ValidationError

logger = logging.getLogger(__name__)

# Initialize FastMCP app
app = FastMCP(
    name="simulacrum-mcp",
    instructions="Reality Engine for AI Systems - Advanced simulation and analysis tools"
)

# Initialize tools
dynamics_simulator = DynamicsSimulator()
feedback_analyzer = FeedbackAnalyzer()
belief_dynamics = BeliefDynamics()
game_theory_dynamics = GameTheoryDynamics()
probability_analyzer = ProbabilityAnalyzer()
chaos_analyzer = ChaosAnalyzer()
scenario_comparator = ScenarioComparator()
vector_analyzer = VectorAnalyzer()


def _format_dict_result(result: Dict[str, Any], indent: int = 0) -> str:
    """Format dictionary result for display."""
    lines = []
    prefix = "  " * indent

    for key, value in result.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_dict_result(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value[:5]:  # Limit list display
                if isinstance(item, dict):
                    lines.append(_format_dict_result(item, indent + 1))
                else:
                    lines.append(f"{prefix}  - {item}")
            if len(value) > 5:
                lines.append(f"{prefix}  ... and {len(value) - 5} more items")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


# Tool definitions
@app.tool()
async def simulate_dynamics(scenario_data: Dict[str, Any]) -> str:
    """
    Execute system dynamics simulation.

    This is the main simulation tool that allows modeling complex systems
    using differential equations and feedback loops.

    Args:
        scenario_data: Dictionary containing scenario definition with:
            - name: Scenario name (string)
            - description: Optional description (string)
            - variables: List of variable definitions (list)
            - equations: List of equation definitions (list)
            - config: Optional simulation configuration (dict)

    Returns:
        Simulation results including time series data and analysis
    """
    try:
        result = dynamics_simulator.simulate_dynamics(scenario_data)
        return f"Simulation completed successfully.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return f"Simulation Error: {str(e)}"


@app.tool()
async def compare_scenarios(scenarios_data: list) -> str:
    """
    Compare multiple simulation scenarios (A/B testing for reality).

    Enables comparative analysis of different assumptions, parameters, or model structures
    to understand how changes affect outcomes.

    Args:
        scenarios_data: List of scenario definitions to compare

    Returns:
        Comparison results including similarity metrics and key differences
    """
    try:
        result = scenario_comparator.compare_scenarios(scenarios_data)
        return f"Scenario comparison completed.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        return f"Comparison Error: {str(e)}"


@app.tool()
async def analyze_feedback_loops(scenario_data: Dict[str, Any]) -> str:
    """
    Analyze feedback loops in dynamic systems.

    Identifies reinforcing and balancing loops, analyzes their impact
    on system behavior, and provides insights for system understanding.

    Args:
        scenario_data: Scenario definition for feedback loop analysis

    Returns:
        Feedback loop analysis including loop classification and system insights
    """
    try:
        result = feedback_analyzer.analyze_feedback_loops(scenario_data)
        return f"Feedback loop analysis completed.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Feedback analysis failed: {str(e)}")
        return f"Analysis Error: {str(e)}"


@app.tool()
async def simulate_belief_dynamics(agents_data: list, interactions: list, time_steps: int = 100) -> str:
    """
    Simulate belief evolution among agents (Theory of Mind).

    Models how agents update their beliefs through social interactions,
    enabling analysis of complex social dynamics and decision-making.

    Args:
        agents_data: List of agent belief state definitions
        interactions: List of interaction patterns between agents
        time_steps: Number of time steps to simulate (default: 100)

    Returns:
        Belief evolution trajectories and social dynamics analysis
    """
    try:
        result = belief_dynamics.simulate_belief_dynamics(agents_data, interactions, time_steps)
        return f"Belief dynamics simulation completed.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Belief simulation failed: {str(e)}")
        return f"Simulation Error: {str(e)}"


@app.tool()
async def analyze_game_theory_dynamics(game_definition: Dict[str, Any]) -> str:
    """
    Analyze strategic interactions using game theory.

    Supports equilibrium analysis, learning dynamics, and strategic stability
    assessment for understanding rational behavior in competitive situations.

    Args:
        game_definition: Game definition with players, strategies, and payoffs

    Returns:
        Game theory analysis including equilibria and strategic insights
    """
    try:
        result = game_theory_dynamics.analyze_game_theory_dynamics(game_definition)
        return f"Game theory analysis completed.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Game theory analysis failed: {str(e)}")
        return f"Analysis Error: {str(e)}"


@app.tool()
async def analyze_probability(analysis_request: Dict[str, Any]) -> str:
    """
    Perform probabilistic analysis and Bayesian reasoning.

    Enables quantitative assessment of uncertainty, hypothesis testing,
    and evidence-based decision making.

    Args:
        analysis_request: Analysis specification with type and parameters

    Returns:
        Probabilistic analysis results
    """
    try:
        result = probability_analyzer.analyze_probability(analysis_request)
        return f"Probability analysis completed.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Probability analysis failed: {str(e)}")
        return f"Analysis Error: {str(e)}"


@app.tool()
async def analyze_chaos(time_series_data: Dict[str, Any]) -> str:
    """
    Analyze time series for chaotic behavior and black swan detection.

    Identifies system instabilities, early warning signals, and potential
    catastrophic changes using chaos theory and nonlinear dynamics.

    Args:
        time_series_data: Time series data for chaos analysis

    Returns:
        Chaos analysis including stability assessment and risk evaluation
    """
    try:
        result = chaos_analyzer.analyze_chaos(time_series_data)
        return f"Chaos analysis completed.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Chaos analysis failed: {str(e)}")
        return f"Analysis Error: {str(e)}"


@app.tool()
async def vector_addition(vectors: list, strengths: list, axis_names: Dict[str, str] = None, normalize: bool = False) -> str:
    """
    Perform 2D vector addition of concepts with strength weighting.

    Combines multiple 2D vectors representing concepts, each with associated
    strength values. Vectors are normalized to unit length before strength
    application, ensuring fair combination regardless of original magnitude.

    Args:
        vectors: List of 2D vectors with x, y coordinates and names
        strengths: List of strength values for each vector
        axis_names: Optional names for X and Y axes
        normalize: Whether to normalize the resultant vector to unit length

    Returns:
        Complete vector addition analysis including resultant vector and contributions
    """
    try:
        result = vector_analyzer.vector_addition(vectors, strengths, axis_names, normalize)
        return f"Vector addition completed successfully.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Vector addition failed: {str(e)}")
        return f"Vector Addition Error: {str(e)}"


@app.tool()
async def multidimensional_vector_analysis(vectors: list, strengths: list, normalize: bool = False, top_dimensions: int = 10, projection_axes: list = None) -> str:
    """
    Perform multidimensional vector analysis for complex concept relationships.

    Analyzes relationships in high-dimensional spaces, identifying conflicts,
    coalitions, winners, and losers based on vector alignments and strengths.

    Args:
        vectors: List of multidimensional vectors with components
        strengths: List of strength values for each vector
        normalize: Whether to normalize the resultant vector
        top_dimensions: Number of top dimensions to display
        projection_axes: Optional 2-axis projection for visualization

    Returns:
        Comprehensive analysis including dimension contributions, conflict/coalition analysis, and player outcomes
    """
    try:
        result = vector_analyzer.multidimensional_vector_analysis(vectors, strengths, normalize, top_dimensions, projection_axes)
        return f"Multidimensional vector analysis completed.\n\n{_format_dict_result(result)}"
    except ValidationError as e:
        return f"Validation Error: {e.message}\nDetails: {e.details}"
    except Exception as e:
        logger.error(f"Multidimensional analysis failed: {str(e)}")
        return f"Multidimensional Analysis Error: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server
    app.run()