"""
Belief dynamics tool for modeling agent belief evolution.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ..validation.types import BeliefState
from ..validation.errors import ValidationError


class BeliefDynamics:
    """
    Tool for modeling how agents update and evolve their beliefs.

    Implements Theory of Mind simulation for understanding complex social
    dynamics and decision-making processes.
    """

    def __init__(self, time_step: float = 0.1, max_iterations: int = 1000):
        """
        Initialize belief dynamics simulator.

        Args:
            time_step: Time step for belief evolution
            max_iterations: Maximum iterations for convergence
        """
        self.time_step = time_step
        self.max_iterations = max_iterations

    def simulate_belief_dynamics(self, agents_data: List[Dict[str, Any]],
                               interactions: List[Dict[str, Any]],
                               time_steps: int = 100) -> Dict[str, Any]:
        """
        Simulate belief evolution among multiple agents.

        Args:
            agents_data: List of agent belief state definitions
            interactions: List of interaction patterns between agents
            time_steps: Number of time steps to simulate

        Returns:
            Dictionary containing belief evolution results

        Raises:
            ValidationError: If input data is invalid
        """
        try:
            # Convert and validate agents
            agents = [BeliefState(**agent_data) for agent_data in agents_data]

            # Initialize belief trajectories
            trajectories = self._initialize_trajectories(agents, time_steps)

            # Simulate belief evolution
            for t in range(1, time_steps):
                current_beliefs = {agent.agent_id: trajectories[agent.agent_id][t-1]
                                 for agent in agents}

                # Update beliefs based on interactions
                for interaction in interactions:
                    self._apply_interaction(current_beliefs, interaction, trajectories, t)

                # Apply individual belief adaptation
                for agent in agents:
                    self._apply_adaptation(agent, current_beliefs, trajectories, t)

            # Analyze results
            analysis = self._analyze_belief_evolution(trajectories, agents)

            return {
                "status": "success",
                "agent_count": len(agents),
                "time_steps": time_steps,
                "trajectories": trajectories,
                "analysis": analysis,
                "final_state": self._get_final_state(trajectories, agents)
            }

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "agents_data",
                agents_data,
                f"Belief dynamics simulation failed: {str(e)}"
            )

    def _initialize_trajectories(self, agents: List[BeliefState],
                               time_steps: int) -> Dict[str, List[Dict[str, float]]]:
        """Initialize belief trajectories for all agents."""
        trajectories = {}

        for agent in agents:
            trajectory = []
            for _ in range(time_steps):
                trajectory.append(agent.beliefs.copy())
            trajectories[agent.agent_id] = trajectory

        return trajectories

    def _apply_interaction(self, current_beliefs: Dict[str, Dict[str, float]],
                          interaction: Dict[str, Any],
                          trajectories: Dict[str, List[Dict[str, float]]],
                          time_step: int):
        """Apply social interaction effects on beliefs."""
        agent_a = interaction.get('agent_a')
        agent_b = interaction.get('agent_b')
        influence_strength = interaction.get('influence_strength', 0.1)
        topics = interaction.get('topics', [])

        if agent_a not in current_beliefs or agent_b not in current_beliefs:
            return

        beliefs_a = current_beliefs[agent_a]
        beliefs_b = current_beliefs[agent_b]

        # Social learning: agents adjust beliefs toward each other
        for topic in topics:
            if topic in beliefs_a and topic in beliefs_b:
                diff = beliefs_b[topic] - beliefs_a[topic]
                adjustment = influence_strength * diff * self.time_step

                # Update trajectory
                trajectories[agent_a][time_step][topic] += adjustment
                trajectories[agent_b][time_step][topic] -= adjustment * 0.5  # Less adjustment for influencer

                # Clamp to [0, 1]
                trajectories[agent_a][time_step][topic] = np.clip(
                    trajectories[agent_a][time_step][topic], 0.0, 1.0)
                trajectories[agent_b][time_step][topic] = np.clip(
                    trajectories[agent_b][time_step][topic], 0.0, 1.0)

    def _apply_adaptation(self, agent: BeliefState,
                         current_beliefs: Dict[str, Dict[str, float]],
                         trajectories: Dict[str, List[Dict[str, float]]],
                         time_step: int):
        """Apply individual belief adaptation based on agent characteristics."""
        current_trajectory = trajectories[agent.agent_id][time_step]

        # Simple adaptation: beliefs tend toward moderate values (0.5)
        # More adaptable agents change faster
        for belief, confidence in current_trajectory.items():
            target = 0.5  # Moderate belief
            diff = target - confidence
            adaptation_rate = agent.adaptability * self.time_step
            adjustment = adaptation_rate * diff

            current_trajectory[belief] += adjustment

            # Add some noise for realism
            noise = np.random.normal(0, 0.01)
            current_trajectory[belief] += noise

            # Clamp to [0, 1]
            current_trajectory[belief] = np.clip(current_trajectory[belief], 0.0, 1.0)

    def _analyze_belief_evolution(self, trajectories: Dict[str, List[Dict[str, float]]],
                                agents: List[BeliefState]) -> Dict[str, Any]:
        """Analyze the evolution of beliefs over time."""
        analysis = {
            "convergence": {},
            "polarization": {},
            "consensus": {},
            "belief_volatility": {}
        }

        # Analyze convergence for each belief topic
        all_topics = set()
        for agent in agents:
            all_topics.update(agent.beliefs.keys())

        for topic in all_topics:
            topic_values = []

            for agent_id, trajectory in trajectories.items():
                topic_trajectory = [step.get(topic, 0.5) for step in trajectory]
                topic_values.append(topic_trajectory)

            if topic_values:
                # Compute convergence metrics
                final_values = [series[-1] for series in topic_values]
                initial_values = [series[0] for series in topic_values]

                convergence = {
                    "initial_std": float(np.std(initial_values)),
                    "final_std": float(np.std(final_values)),
                    "convergence_ratio": float(np.std(final_values) / max(np.std(initial_values), 1e-10)),
                    "consensus_reached": np.std(final_values) < 0.1  # Low variance indicates consensus
                }

                # Polarization analysis
                polarization = self._analyze_polarization(final_values)

                # Volatility analysis
                volatility = self._analyze_volatility(topic_values)

                analysis["convergence"][topic] = convergence
                analysis["polarization"][topic] = polarization
                analysis["belief_volatility"][topic] = volatility

        # Overall consensus analysis
        analysis["consensus"] = self._analyze_overall_consensus(analysis["convergence"])

        return analysis

    def _analyze_polarization(self, values: List[float]) -> Dict[str, Any]:
        """Analyze belief polarization."""
        if len(values) < 2:
            return {"polarized": False, "extremes_count": 0}

        # Count extreme beliefs (close to 0 or 1)
        extreme_threshold = 0.1
        extremes = sum(1 for v in values if v < extreme_threshold or v > (1 - extreme_threshold))

        # Check for bimodal distribution
        sorted_values = sorted(values)
        median = np.median(sorted_values)

        # Simple polarization detection
        polarized = extremes >= len(values) * 0.6  # 60% extreme beliefs

        return {
            "polarized": polarized,
            "extremes_count": extremes,
            "extremes_ratio": extremes / len(values),
            "belief_spread": float(max(values) - min(values))
        }

    def _analyze_volatility(self, trajectories: List[List[float]]) -> Dict[str, Any]:
        """Analyze belief volatility over time."""
        if not trajectories:
            return {"average_volatility": 0.0}

        volatilities = []
        for trajectory in trajectories:
            if len(trajectory) > 1:
                changes = [abs(trajectory[i+1] - trajectory[i])
                          for i in range(len(trajectory)-1)]
                volatility = np.mean(changes) if changes else 0.0
                volatilities.append(volatility)

        return {
            "average_volatility": float(np.mean(volatilities)) if volatilities else 0.0,
            "max_volatility": float(max(volatilities)) if volatilities else 0.0,
            "volatility_std": float(np.std(volatilities)) if volatilities else 0.0
        }

    def _analyze_overall_consensus(self, convergence_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall consensus across all topics."""
        if not convergence_data:
            return {"overall_consensus": False}

        consensus_topics = sum(1 for topic_data in convergence_data.values()
                             if topic_data.get("consensus_reached", False))

        total_topics = len(convergence_data)

        return {
            "consensus_topics": consensus_topics,
            "total_topics": total_topics,
            "consensus_ratio": consensus_topics / total_topics if total_topics > 0 else 0,
            "overall_consensus": consensus_topics == total_topics
        }

    def _get_final_state(self, trajectories: Dict[str, List[Dict[str, float]]],
                        agents: List[BeliefState]) -> Dict[str, Any]:
        """Get final belief state for all agents."""
        final_state = {}

        for agent in agents:
            trajectory = trajectories[agent.agent_id]
            final_beliefs = trajectory[-1] if trajectory else agent.beliefs

            final_state[agent.agent_id] = {
                "beliefs": final_beliefs,
                "agent_info": {
                    "evidence_weight": agent.evidence_weight,
                    "adaptability": agent.adaptability
                }
            }

        return final_state
