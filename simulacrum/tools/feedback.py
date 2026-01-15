"""
Feedback loop analysis tool for system dynamics.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
from ..validation.types import ScenarioData, FeedbackLoop
from ..validation.errors import ValidationError


class FeedbackAnalyzer:
    """
    Tool for analyzing feedback loops in dynamic systems.

    Identifies reinforcing and balancing loops, analyzes their impact,
    and provides insights for system behavior understanding.
    """

    def __init__(self):
        """Initialize feedback analyzer."""
        pass

    def analyze_feedback_loops(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feedback loops in a system dynamics scenario.

        Args:
            scenario_data: Dictionary containing scenario definition

        Returns:
            Dictionary containing feedback loop analysis results

        Raises:
            ValidationError: If scenario data is invalid
        """
        try:
            scenario = ScenarioData(**scenario_data)

            # Build dependency graph
            dependency_graph = self._build_dependency_graph(scenario)

            # Find all feedback loops
            loops = self._find_feedback_loops(dependency_graph)

            # Classify loops
            classified_loops = self._classify_loops(loops, dependency_graph)

            # Analyze loop impact
            loop_analysis = self._analyze_loop_impact(classified_loops, scenario)

            return {
                "status": "success",
                "scenario_name": scenario.name,
                "dependency_graph": {
                    "nodes": list(dependency_graph.nodes()),
                    "edges": list(dependency_graph.edges())
                },
                "feedback_loops": classified_loops,
                "analysis": loop_analysis,
                "summary": self._generate_summary(classified_loops, loop_analysis)
            }

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "scenario_data",
                scenario_data,
                f"Feedback loop analysis failed: {str(e)}"
            )

    def _build_dependency_graph(self, scenario: ScenarioData) -> nx.DiGraph:
        """Build directed graph of variable dependencies."""
        graph = nx.DiGraph()

        # Add all variables as nodes
        for var in scenario.variables:
            graph.add_node(var.name)

        # Add edges based on equation dependencies
        for eq in scenario.equations:
            # This would use the equation parser to extract dependencies
            # For now, simple placeholder logic
            dependencies = self._extract_dependencies_simple(eq.expression)

            for dep in dependencies:
                if dep in [var.name for var in scenario.variables]:
                    graph.add_edge(dep, eq.target_variable)

        return graph

    def _extract_dependencies_simple(self, expression: str) -> List[str]:
        """Simple dependency extraction (placeholder for full parser)."""
        # This is a placeholder - real implementation would use equation parser
        # For now, assume variables are single letters or common names
        import re
        # Find potential variable names (simplified)
        potential_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
        # Filter out functions and keywords
        functions = {'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs', 'if'}
        return [var for var in potential_vars if var not in functions and len(var) > 1]

    def _find_feedback_loops(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find all feedback loops in the dependency graph."""
        loops = []

        # Find all simple cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            loops.extend(cycles)
        except:
            # If cycle detection fails, try alternative approach
            pass

        # Also check for longer loops using DFS
        visited = set()
        for node in graph.nodes():
            if node not in visited:
                self._dfs_find_loops(graph, node, visited, [], set(), loops)

        # Remove duplicates and sort
        unique_loops = []
        seen = set()
        for loop in loops:
            loop_tuple = tuple(sorted(loop))
            if loop_tuple not in seen:
                seen.add(loop_tuple)
                unique_loops.append(loop)

        return unique_loops

    def _dfs_find_loops(self, graph: nx.DiGraph, current: str, visited: Set[str],
                        path: List[str], path_set: Set[str], loops: List[List[str]]):
        """DFS-based loop finding."""
        visited.add(current)
        path.append(current)
        path_set.add(current)

        for neighbor in graph.successors(current):
            if neighbor not in path_set:
                self._dfs_find_loops(graph, neighbor, visited, path, path_set, loops)
            elif neighbor == path[0] and len(path) > 2:  # Found a loop
                loop = path[path.index(neighbor):] + [neighbor]
                if len(loop) > 2:  # Only loops with 3+ nodes
                    loops.append(loop[:-1])  # Remove duplicate end

        path.pop()
        path_set.remove(current)

    def _classify_loops(self, loops: List[List[str]],
                        graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Classify feedback loops as reinforcing or balancing."""
        classified = []

        for loop in loops:
            loop_type = self._determine_loop_type(loop, graph)

            classified.append({
                "variables": loop,
                "type": loop_type,
                "length": len(loop),
                "strength": self._estimate_loop_strength(loop, graph),
                "description": self._generate_loop_description(loop, loop_type)
            })

        return classified

    def _determine_loop_type(self, loop: List[str], graph: nx.DiGraph) -> str:
        """Determine if loop is reinforcing or balancing."""
        # Count negative relationships in the loop
        negative_count = 0

        for i in range(len(loop)):
            current = loop[i]
            next_var = loop[(i + 1) % len(loop)]

            # Check edge weight/type (placeholder - would use actual relationship type)
            edge_data = graph.get_edge_data(current, next_var)
            if edge_data and edge_data.get('relationship') == 'negative':
                negative_count += 1

        # Even number of negative relationships = reinforcing
        # Odd number = balancing
        return "reinforcing" if negative_count % 2 == 0 else "balancing"

    def _estimate_loop_strength(self, loop: List[str], graph: nx.DiGraph) -> float:
        """Estimate the strength of a feedback loop."""
        # Placeholder - would analyze loop gain, delays, etc.
        return 1.0  # Default strength

    def _generate_loop_description(self, loop: List[str], loop_type: str) -> str:
        """Generate human-readable description of feedback loop."""
        loop_str = " → ".join(loop)
        type_desc = "reinforcing (amplifying)" if loop_type == "reinforcing" else "balancing (stabilizing)"
        return f"{type_desc} feedback loop: {loop_str}"

    def _analyze_loop_impact(self, classified_loops: List[Dict[str, Any]],
                           scenario: ScenarioData) -> Dict[str, Any]:
        """Analyze the impact of feedback loops on system behavior."""
        reinforcing_count = sum(1 for loop in classified_loops if loop['type'] == 'reinforcing')
        balancing_count = sum(1 for loop in classified_loops if loop['type'] == 'balancing')

        # Analyze dominant loop types
        total_loops = len(classified_loops)
        dominant_type = "balanced"
        if reinforcing_count > balancing_count:
            dominant_type = "growth_oriented"
        elif balancing_count > reinforcing_count:
            dominant_type = "stability_oriented"

        # Identify potential system behaviors
        behaviors = []
        if reinforcing_count > 0:
            behaviors.append("exponential_growth_or_decline")
        if balancing_count > 0:
            behaviors.append("goal_seeking_behavior")
        if total_loops > 5:
            behaviors.append("complex_system_behavior")
        if self._has_conflicting_loops(classified_loops):
            behaviors.append("oscillatory_behavior")

        return {
            "reinforcing_loops": reinforcing_count,
            "balancing_loops": balancing_count,
            "dominant_characteristic": dominant_type,
            "predicted_behaviors": behaviors,
            "system_complexity": self._assess_complexity(classified_loops)
        }

    def _has_conflicting_loops(self, loops: List[Dict[str, Any]]) -> bool:
        """Check if system has conflicting feedback loops."""
        # Simple check: both reinforcing and balancing loops present
        types = {loop['type'] for loop in loops}
        return len(types) > 1

    def _assess_complexity(self, loops: List[Dict[str, Any]]) -> str:
        """Assess overall system complexity based on loops."""
        total_loops = len(loops)
        if total_loops <= 2:
            return "simple"
        elif total_loops <= 5:
            return "moderate"
        else:
            return "complex"

    def _generate_summary(self, loops: List[Dict[str, Any]],
                         analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of feedback loop analysis."""
        return {
            "total_loops_found": len(loops),
            "system_characteristics": analysis["dominant_characteristic"],
            "behavior_predictions": analysis["predicted_behaviors"],
            "complexity_level": analysis["system_complexity"],
            "key_insights": self._generate_insights(loops, analysis)
        }

    def _generate_insights(self, loops: List[Dict[str, Any]],
                          analysis: Dict[str, Any]) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []

        if analysis["reinforcing_loops"] > analysis["balancing_loops"]:
            insights.append("System dominated by growth loops - watch for exponential behavior")
        elif analysis["balancing_loops"] > analysis["reinforcing_loops"]:
            insights.append("System dominated by balancing loops - tends toward stability")

        if len(loops) > 3:
            insights.append("Complex system with multiple feedback loops - behavior may be unpredictable")

        if analysis["system_complexity"] == "complex":
            insights.append("High system complexity suggests need for careful parameter tuning")

        return insights
