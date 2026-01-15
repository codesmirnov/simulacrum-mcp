"""
Vector analysis tools for Simulacrum MCP.

Provides 2D vector addition and multidimensional vector analysis capabilities
for understanding complex relationships and conflicts in multi-dimensional spaces.
"""

import math
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from ..validation.errors import ValidationError


class VectorAnalyzer:
    """
    Tool for vector-based analysis of concepts and relationships.

    Supports both 2D vector addition for simple concept combination and
    multidimensional analysis for complex relationship modeling.
    """

    def __init__(self, max_vectors: int = 50, max_dimensions: int = 100):
        """
        Initialize vector analyzer.

        Args:
            max_vectors: Maximum number of vectors to process
            max_dimensions: Maximum number of dimensions to analyze
        """
        self.max_vectors = max_vectors
        self.max_dimensions = max_dimensions

    def vector_addition(self, vectors: List[Dict[str, Any]],
                       strengths: List[Dict[str, Any]],
                       axis_names: Optional[Dict[str, str]] = None,
                       normalize: bool = False) -> Dict[str, Any]:
        """
        Perform 2D vector addition of concepts.

        Args:
            vectors: List of 2D vectors with x, y coordinates and names
            strengths: List of strength values for each vector
            axis_names: Optional names for X and Y axes
            normalize: Whether to normalize the result vector

        Returns:
            Dictionary containing addition results and analysis

        Raises:
            ValidationError: If input data is invalid
        """
        try:
            self._validate_vector_addition_input(vectors, strengths)

            axis_names = axis_names or {"x": "X-axis", "y": "Y-axis"}

            # Create strength lookup dictionary
            strength_dict = {s["name"]: s["value"] for s in strengths}

            # Add default strengths for vectors without specified strength
            for vector in vectors:
                if vector["name"] not in strength_dict:
                    strength_dict[vector["name"]] = 1.0

            # Calculate resultant vector
            total_x, total_y = self._calculate_resultant_vector(vectors, strength_dict)

            # Normalize if requested
            if normalize and (total_x != 0 or total_y != 0):
                magnitude = math.sqrt(total_x**2 + total_y**2)
                total_x /= magnitude
                total_y /= magnitude

            # Calculate vector properties
            magnitude = math.sqrt(total_x**2 + total_y**2)
            angle_rad = math.atan2(total_y, total_x)
            angle_deg = math.degrees(angle_rad)

            # Determine direction
            direction = self._calculate_direction(angle_deg)

            result = {
                "operation": "vector_addition",
                "input_vectors": len(vectors),
                "resultant_vector": {
                    "x": round(total_x, 4),
                    "y": round(total_y, 4),
                    "magnitude": round(magnitude, 4),
                    "angle_degrees": round(angle_deg, 1),
                    "direction": direction
                },
                "axes": axis_names,
                "normalization_applied": normalize,
                "vector_details": self._analyze_vector_contributions(vectors, strength_dict, total_x, total_y),
                "interpretation": self._interpret_vector_result(total_x, total_y, magnitude, axis_names)
            }

            return result

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "vector_addition",
                {"vectors": vectors, "strengths": strengths},
                f"Vector addition failed: {str(e)}"
            )

    def multidimensional_vector_analysis(self, vectors: List[Dict[str, Any]],
                                       strengths: List[Dict[str, Any]],
                                       normalize: bool = False,
                                       top_dimensions: int = 10,
                                       projection_axes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform multidimensional vector analysis for complex relationships.

        Args:
            vectors: List of multidimensional vectors with components
            strengths: List of strength values for each vector
            normalize: Whether to normalize the result vector
            top_dimensions: Number of top dimensions to show
            projection_axes: Optional axes for 2D projection

        Returns:
            Dictionary containing comprehensive analysis results

        Raises:
            ValidationError: If input data is invalid
        """
        try:
            self._validate_multidimensional_input(vectors, strengths)

            # Create strength lookup
            strength_dict = {s["name"]: s["value"] for s in strengths}

            # Collect all dimensions
            all_dimensions = self._collect_dimensions(vectors)

            # Calculate resultant vector
            result_vector = self._calculate_multidimensional_resultant(
                vectors, strength_dict, all_dimensions
            )

            # Normalize if requested
            if normalize:
                magnitude = math.sqrt(sum(val**2 for val in result_vector.values()))
                if magnitude > 0:
                    result_vector = {dim: val/magnitude for dim, val in result_vector.items()}

            # Calculate total magnitude
            total_magnitude = math.sqrt(sum(val**2 for val in result_vector.values()))

            # Analyze dimension contributions
            dimension_analysis = self._analyze_dimension_contributions(
                result_vector, all_dimensions, total_magnitude
            )

            result = {
                "operation": "multidimensional_vector_analysis",
                "input_vectors": len(vectors),
                "dimensions": len(all_dimensions),
                "total_magnitude": round(total_magnitude, 4),
                "normalization_applied": normalize,
                "top_dimensions": dimension_analysis[:top_dimensions],
                "vector_strengths": strength_dict,
                "vector_contributions": self._analyze_vector_contributions_multidimensional(
                    vectors, strength_dict, result_vector, all_dimensions, total_magnitude
                )
            }

            # Add 2D projection if requested
            if projection_axes and len(projection_axes) == 2:
                result["projection_2d"] = self._calculate_2d_projection(
                    result_vector, projection_axes
                )

            # Add conflict and coalition analysis
            result["conflict_analysis"] = self._analyze_conflicts_and_coalitions(
                vectors, all_dimensions
            )

            # Add player win/loss analysis
            result["player_analysis"] = self._analyze_player_outcomes(
                vectors, strength_dict, result_vector, all_dimensions, total_magnitude
            )

            result["interpretation"] = self._interpret_multidimensional_result(
                result_vector, dimension_analysis, total_magnitude
            )

            return result

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "multidimensional_analysis",
                {"vectors": vectors, "strengths": strengths},
                f"Multidimensional analysis failed: {str(e)}"
            )

    def _validate_vector_addition_input(self, vectors: List[Dict[str, Any]],
                                       strengths: List[Dict[str, Any]]) -> None:
        """Validate input for 2D vector addition."""
        if not vectors:
            raise ValidationError("vectors", vectors, "At least one vector required")

        if len(vectors) > self.max_vectors:
            raise ValidationError(
                "vectors", vectors,
                f"Too many vectors: maximum {self.max_vectors} allowed"
            )

        # Check vector structure
        required_keys = {"name", "x", "y"}
        for i, vector in enumerate(vectors):
            if not isinstance(vector, dict):
                raise ValidationError(
                    f"vectors[{i}]", vector, "Each vector must be a dictionary"
                )

            missing_keys = required_keys - set(vector.keys())
            if missing_keys:
                raise ValidationError(
                    f"vectors[{i}]", vector,
                    f"Missing required keys: {', '.join(missing_keys)}"
                )

            # Validate coordinates are numeric
            try:
                float(vector["x"])
                float(vector["y"])
            except (ValueError, TypeError):
                raise ValidationError(
                    f"vectors[{i}]", vector,
                    "Vector coordinates x and y must be numeric"
                )

    def _validate_multidimensional_input(self, vectors: List[Dict[str, Any]],
                                       strengths: List[Dict[str, Any]]) -> None:
        """Validate input for multidimensional analysis."""
        if not vectors:
            raise ValidationError("vectors", vectors, "At least one vector required")

        if not strengths:
            raise ValidationError("strengths", strengths, "Strengths must be provided")

        if len(vectors) > self.max_vectors:
            raise ValidationError(
                "vectors", vectors,
                f"Too many vectors: maximum {self.max_vectors} allowed"
            )

        # Check vector structure
        for i, vector in enumerate(vectors):
            if not isinstance(vector, dict):
                raise ValidationError(
                    f"vectors[{i}]", vector, "Each vector must be a dictionary"
                )

            if "name" not in vector:
                raise ValidationError(
                    f"vectors[{i}]", vector, "Each vector must have a 'name' field"
                )

            if "components" not in vector:
                raise ValidationError(
                    f"vectors[{i}]", vector, "Each vector must have 'components' field"
                )

            components = vector["components"]
            if not isinstance(components, dict):
                raise ValidationError(
                    f"vectors[{i}].components", components, "Components must be a dictionary"
                )

            if len(components) > self.max_dimensions:
                raise ValidationError(
                    f"vectors[{i}]", vector,
                    f"Too many dimensions: maximum {self.max_dimensions} allowed"
                )

    def _collect_dimensions(self, vectors: List[Dict[str, Any]]) -> List[str]:
        """Collect all unique dimensions from vectors."""
        all_dimensions = set()
        for vector in vectors:
            all_dimensions.update(vector.get("components", {}).keys())
        return sorted(all_dimensions)

    def _calculate_resultant_vector(self, vectors: List[Dict[str, Any]],
                                  strength_dict: Dict[str, float]) -> Tuple[float, float]:
        """Calculate the resultant 2D vector."""
        total_x = 0.0
        total_y = 0.0

        for vector in vectors:
            strength = strength_dict[vector["name"]]
            x, y = vector["x"], vector["y"]

            # Normalize to unit vector before applying strength
            magnitude = math.sqrt(x**2 + y**2)
            if magnitude > 0:
                normalized_x = x / magnitude
                normalized_y = y / magnitude
                total_x += normalized_x * strength
                total_y += normalized_y * strength

        return total_x, total_y

    def _calculate_multidimensional_resultant(self, vectors: List[Dict[str, Any]],
                                            strength_dict: Dict[str, float],
                                            all_dimensions: List[str]) -> Dict[str, float]:
        """Calculate resultant multidimensional vector."""
        result = defaultdict(float)

        for vector in vectors:
            components = vector.get("components", {})
            strength = strength_dict[vector["name"]]

            # Calculate vector magnitude for normalization
            magnitude = math.sqrt(sum(val**2 for val in components.values()))
            if magnitude == 0:
                continue  # Skip zero vectors

            # Normalize and apply strength
            normalized_components = {
                dim: components.get(dim, 0.0) / magnitude
                for dim in all_dimensions
            }

            for dim in all_dimensions:
                result[dim] += normalized_components[dim] * strength

        return dict(result)

    def _calculate_direction(self, angle_deg: float) -> str:
        """Calculate cardinal direction from angle."""
        directions = [
            "East", "Northeast", "North", "Northwest",
            "West", "Southwest", "South", "Southeast"
        ]
        direction_index = round(angle_deg / 45) % 8
        return directions[direction_index]

    def _analyze_vector_contributions(self, vectors: List[Dict[str, Any]],
                                    strength_dict: Dict[str, float],
                                    total_x: float, total_y: float) -> List[Dict[str, Any]]:
        """Analyze individual vector contributions to result."""
        contributions = []

        for vector in vectors:
            strength = strength_dict[vector["name"]]
            x, y = vector["x"], vector["y"]

            # Calculate contribution of normalized vector
            magnitude = math.sqrt(x**2 + y**2)
            if magnitude > 0:
                normalized_x = x / magnitude
                normalized_y = y / magnitude
                weighted_x = normalized_x * strength
                weighted_y = normalized_y * strength

                total_magnitude = math.sqrt(total_x**2 + total_y**2)
                if total_magnitude > 0:
                    percent_contribution = (math.sqrt(weighted_x**2 + weighted_y**2) /
                                          total_magnitude * 100)
                else:
                    percent_contribution = 0

                contributions.append({
                    "name": vector["name"],
                    "strength": round(strength, 3),
                    "contribution_percent": round(percent_contribution, 1),
                    "description": vector.get("description", "")
                })

        return contributions

    def _analyze_dimension_contributions(self, result_vector: Dict[str, float],
                                       all_dimensions: List[str],
                                       total_magnitude: float) -> List[Dict[str, Any]]:
        """Analyze contribution of each dimension."""
        contributions = []

        for dim in all_dimensions:
            value = result_vector[dim]
            if total_magnitude > 0:
                contribution = abs(value) / total_magnitude
            else:
                contribution = 0

            contributions.append({
                "dimension": dim,
                "value": round(value, 4),
                "contribution": round(contribution, 4),
                "direction": "positive" if value > 0 else "negative" if value < 0 else "neutral"
            })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
        return contributions

    def _analyze_vector_contributions_multidimensional(self, vectors: List[Dict[str, Any]],
                                                     strength_dict: Dict[str, float],
                                                     result_vector: Dict[str, float],
                                                     all_dimensions: List[str],
                                                     total_magnitude: float) -> List[Dict[str, Any]]:
        """Analyze vector contributions in multidimensional space."""
        contributions = []

        for vector in vectors:
            components = vector.get("components", {})
            strength = strength_dict[vector["name"]]

            # Calculate vector magnitude
            magnitude = math.sqrt(sum(val**2 for val in components.values()))
            if magnitude == 0:
                contribution = 0
            else:
                # Normalize vector and calculate alignment with result
                normalized_vector = {
                    dim: components.get(dim, 0.0) / magnitude
                    for dim in all_dimensions
                }

                # Calculate cosine similarity with result vector
                if total_magnitude > 0:
                    normalized_result = {
                        dim: result_vector[dim] / total_magnitude
                        for dim in all_dimensions
                    }

                    cosine_similarity = sum(
                        normalized_vector[dim] * normalized_result[dim]
                        for dim in all_dimensions
                    )

                    contribution = cosine_similarity * strength
                else:
                    contribution = 0

            contributions.append({
                "name": vector["name"],
                "contribution": round(contribution, 3),
                "strength": round(strength, 3),
                "description": vector.get("description", "")
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions

    def _calculate_2d_projection(self, result_vector: Dict[str, float],
                               projection_axes: List[str]) -> Dict[str, Any]:
        """Calculate 2D projection of multidimensional vector."""
        x_axis, y_axis = projection_axes

        if x_axis not in result_vector or y_axis not in result_vector:
            raise ValidationError(
                "projection_axes", projection_axes,
                f"Projection axes {x_axis} and {y_axis} not found in vector dimensions"
            )

        x_val = result_vector[x_axis]
        y_val = result_vector[y_axis]

        projection_magnitude = math.sqrt(x_val**2 + y_val**2)
        angle_rad = math.atan2(y_val, x_val)

        return {
            "x_axis": x_axis,
            "y_axis": y_axis,
            "x_value": round(x_val, 4),
            "y_value": round(y_val, 4),
            "projection_magnitude": round(projection_magnitude, 4),
            "angle_degrees": round(math.degrees(angle_rad), 1)
        }

    def _analyze_conflicts_and_coalitions(self, vectors: List[Dict[str, Any]],
                                        all_dimensions: List[str]) -> Dict[str, Any]:
        """Analyze potential conflicts and coalitions between vectors."""
        conflicts = []
        coalitions = []

        coalition_threshold = 0.7  # Similarity threshold for coalitions
        conflict_threshold = -0.7  # Dissimilarity threshold for conflicts

        # Compare each pair of vectors
        for i, vector1 in enumerate(vectors):
            for vector2 in enumerate(vectors[i+1:], i+1):
                j, vector2 = vector2

                components1 = vector1.get("components", {})
                components2 = vector2.get("components", {})

                # Calculate vector magnitudes
                mag1 = math.sqrt(sum(val**2 for val in components1.values()))
                mag2 = math.sqrt(sum(val**2 for val in components2.values()))

                if mag1 == 0 or mag2 == 0:
                    continue

                # Calculate cosine similarity
                similarity = 0
                for dim in all_dimensions:
                    val1 = components1.get(dim, 0.0) / mag1
                    val2 = components2.get(dim, 0.0) / mag2
                    similarity += val1 * val2

                if similarity > coalition_threshold:
                    coalitions.append({
                        "vectors": [vector1["name"], vector2["name"]],
                        "similarity": round(similarity, 3),
                        "average_strength": None  # Will be filled later
                    })
                elif similarity < conflict_threshold:
                    angle_deg = math.degrees(math.acos(max(-1, min(1, abs(similarity)))))
                    conflicts.append({
                        "vectors": [vector1["name"], vector2["name"]],
                        "similarity": round(similarity, 3),
                        "angle_degrees": round(angle_deg, 1)
                    })

        return {
            "coalitions": coalitions,
            "conflicts": conflicts,
            "coalition_threshold": coalition_threshold,
            "conflict_threshold": conflict_threshold
        }

    def _analyze_player_outcomes(self, vectors: List[Dict[str, Any]],
                               strength_dict: Dict[str, float],
                               result_vector: Dict[str, float],
                               all_dimensions: List[str],
                               total_magnitude: float) -> Dict[str, Any]:
        """Analyze win/loss outcomes for each vector/player."""
        players = []

        if total_magnitude == 0:
            return {"players": [], "summary": "No clear winners - zero resultant vector"}

        # Normalize result vector
        normalized_result = {
            dim: result_vector[dim] / total_magnitude
            for dim in all_dimensions
        }

        for vector in vectors:
            components = vector.get("components", {})
            strength = strength_dict[vector["name"]]

            # Calculate vector magnitude
            magnitude = math.sqrt(sum(val**2 for val in components.values()))
            if magnitude == 0:
                win_score = 0
                cosine_similarity = 0
            else:
                # Normalize player vector
                normalized_player = {
                    dim: components.get(dim, 0.0) / magnitude
                    for dim in all_dimensions
                }

                # Calculate cosine similarity (alignment with result)
                cosine_similarity = sum(
                    normalized_player[dim] * normalized_result[dim]
                    for dim in all_dimensions
                )

                win_score = cosine_similarity * strength

            angle_deg = math.degrees(math.acos(max(-1, min(1, cosine_similarity))))

            players.append({
                "name": vector["name"],
                "win_score": round(win_score, 3),
                "cosine_similarity": round(cosine_similarity, 3),
                "angle_degrees": round(angle_deg, 1),
                "strength": round(strength, 3),
                "description": vector.get("description", "")
            })

        # Sort by win score
        players.sort(key=lambda x: x["win_score"], reverse=True)

        # Calculate summary statistics
        winners = sum(1 for p in players if p["win_score"] > 0.1)
        losers = sum(1 for p in players if p["win_score"] < -0.1)
        neutral = len(players) - winners - losers

        return {
            "players": players,
            "summary": {
                "winners": winners,
                "neutral": neutral,
                "losers": losers,
                "total_players": len(players)
            }
        }

    def _interpret_vector_result(self, total_x: float, total_y: float,
                               magnitude: float, axis_names: Dict[str, str]) -> str:
        """Generate interpretation of 2D vector result."""
        if magnitude < 0.1:
            return "Vectors largely cancel each other out - no dominant direction."

        interpretations = []

        # Analyze axis dominance
        abs_x, abs_y = abs(total_x), abs(total_y)
        if abs_x > abs_y * 1.5:
            axis_name = axis_names.get("x", "X-axis")
            direction = "positive" if total_x > 0 else "negative"
            interpretations.append(f"Dominant direction along {axis_name} ({direction})")
        elif abs_y > abs_x * 1.5:
            axis_name = axis_names.get("y", "Y-axis")
            direction = "positive" if total_y > 0 else "negative"
            interpretations.append(f"Dominant direction along {axis_name} ({direction})")
        else:
            interpretations.append("Balanced influence across both axes")

        return " ".join(interpretations)

    def _interpret_multidimensional_result(self, result_vector: Dict[str, float],
                                         dimension_analysis: List[Dict[str, Any]],
                                         total_magnitude: float) -> Dict[str, Any]:
        """Generate interpretation of multidimensional analysis."""
        interpretation = {}

        # Analyze balance
        if not dimension_analysis:
            return {"overall": "No dimensions to analyze"}

        max_contrib = max(abs(d["value"]) for d in dimension_analysis)
        min_contrib = min(abs(d["value"]) for d in dimension_analysis)

        if max_contrib == 0:
            balance = "neutral"
        else:
            balance_ratio = min_contrib / max_contrib
            if balance_ratio > 0.8:
                balance = "well_balanced"
            elif balance_ratio > 0.5:
                balance = "moderately_balanced"
            else:
                balance = "highly_unbalanced"

        interpretation["balance"] = balance

        # Find dominant dimensions
        top_dimensions = [d for d in dimension_analysis[:3] if abs(d["value"]) > 0.1]
        if top_dimensions:
            interpretation["dominant_dimensions"] = [d["dimension"] for d in top_dimensions]

        # Analyze direction distribution
        positive = sum(1 for d in result_vector.values() if d > 0)
        negative = sum(1 for d in result_vector.values() if d < 0)
        neutral = len(result_vector) - positive - negative

        interpretation["direction_distribution"] = {
            "positive": positive,
            "negative": negative,
            "neutral": neutral
        }

        # Generate summary
        if balance == "well_balanced":
            interpretation["summary"] = "System is well-balanced across dimensions"
        elif balance == "highly_unbalanced":
            interpretation["summary"] = "System is highly unbalanced - few dimensions dominate"
        else:
            interpretation["summary"] = "System shows moderate balance with some dominant factors"

        return interpretation
