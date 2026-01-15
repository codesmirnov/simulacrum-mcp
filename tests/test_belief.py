"""
Tests for belief dynamics functionality.
"""

import pytest
import numpy as np
from simulacrum.tools.belief import BeliefDynamics
from simulacrum.validation.errors import ValidationError


class TestBeliefDynamics:
    """Test suite for belief dynamics simulation."""

    @pytest.fixture
    def belief_simulator(self):
        """Create belief dynamics simulator."""
        return BeliefDynamics(time_step=0.1, max_iterations=1000)

    @pytest.fixture
    def sample_agents_data(self):
        """Sample agent data for testing."""
        return [
            {
                "agent_id": "agent1",
                "beliefs": {"topic_a": 0.8, "topic_b": 0.2},
                "evidence_weight": 0.8,
                "adaptability": 0.3
            },
            {
                "agent_id": "agent2",
                "beliefs": {"topic_a": 0.3, "topic_b": 0.9},
                "evidence_weight": 0.6,
                "adaptability": 0.5
            }
        ]

    @pytest.fixture
    def sample_interactions(self):
        """Sample interaction data for testing."""
        return [
            {
                "agent_a": "agent1",
                "agent_b": "agent2",
                "influence_strength": 0.2,
                "topics": ["topic_a"]
            }
        ]

    def test_simulate_belief_dynamics_success(self, belief_simulator, sample_agents_data, sample_interactions):
        """Test successful belief dynamics simulation."""
        result = belief_simulator.simulate_belief_dynamics(
            sample_agents_data, sample_interactions, time_steps=10
        )

        assert result["status"] == "success"
        assert result["agent_count"] == 2
        assert result["time_steps"] == 10
        assert "trajectories" in result
        assert "analysis" in result
        assert "final_state" in result

        # Check trajectories
        assert "agent1" in result["trajectories"]
        assert "agent2" in result["trajectories"]
        assert len(result["trajectories"]["agent1"]) == 10
        assert len(result["trajectories"]["agent2"]) == 10

        # Check that beliefs are within [0, 1]
        for agent_id, trajectory in result["trajectories"].items():
            for step in trajectory:
                for belief_value in step.values():
                    assert 0.0 <= belief_value <= 1.0

    def test_simulate_belief_dynamics_no_interactions(self, belief_simulator, sample_agents_data):
        """Test belief dynamics with no interactions."""
        result = belief_simulator.simulate_belief_dynamics(
            sample_agents_data, [], time_steps=5
        )

        assert result["status"] == "success"
        # Beliefs should still evolve due to adaptation
        initial_belief_a1 = result["trajectories"]["agent1"][0]["topic_a"]
        final_belief_a1 = result["trajectories"]["agent1"][-1]["topic_a"]
        assert abs(initial_belief_a1 - final_belief_a1) > 0  # Should change due to adaptation

    def test_simulate_belief_dynamics_invalid_agent_data(self, belief_simulator):
        """Test error handling with invalid agent data."""
        invalid_agents = [
            {
                "agent_id": "agent1",
                "beliefs": {"topic_a": 1.5},  # Invalid confidence > 1
                "evidence_weight": 0.8,
                "adaptability": 0.3
            }
        ]

        with pytest.raises(ValidationError):
            belief_simulator.simulate_belief_dynamics(invalid_agents, [], time_steps=10)

    def test_simulate_belief_dynamics_empty_agents(self, belief_simulator):
        """Test handling of empty agents list."""
        result = belief_simulator.simulate_belief_dynamics([], [], time_steps=10)

        assert result["status"] == "success"
        assert result["agent_count"] == 0
        assert result["trajectories"] == {}
        assert result["final_state"] == {}

    def test_initialize_trajectories(self, belief_simulator, sample_agents_data):
        """Test trajectory initialization."""
        from simulacrum.validation.types import BeliefState
        agents = [BeliefState(**agent) for agent in sample_agents_data]

        trajectories = belief_simulator._initialize_trajectories(agents, 5)

        assert len(trajectories) == 2
        assert "agent1" in trajectories
        assert "agent2" in trajectories
        assert len(trajectories["agent1"]) == 5
        assert len(trajectories["agent2"]) == 5

        # Check that initial beliefs are copied correctly
        assert trajectories["agent1"][0]["topic_a"] == 0.8
        assert trajectories["agent2"][0]["topic_b"] == 0.9

    def test_apply_interaction_valid(self, belief_simulator, sample_agents_data, sample_interactions):
        """Test applying valid social interactions."""
        from simulacrum.validation.types import BeliefState
        agents = [BeliefState(**agent) for agent in sample_agents_data]

        trajectories = belief_simulator._initialize_trajectories(agents, 3)
        current_beliefs = {
            "agent1": {"topic_a": 0.8, "topic_b": 0.2},
            "agent2": {"topic_a": 0.3, "topic_b": 0.9}
        }

        belief_simulator._apply_interaction(current_beliefs, sample_interactions[0], trajectories, 1)

        # Check that beliefs changed
        assert trajectories["agent1"][1]["topic_a"] != trajectories["agent1"][0]["topic_a"]
        assert trajectories["agent2"][1]["topic_a"] != trajectories["agent2"][0]["topic_a"]

        # Check bounds
        assert 0.0 <= trajectories["agent1"][1]["topic_a"] <= 1.0
        assert 0.0 <= trajectories["agent2"][1]["topic_a"] <= 1.0

    def test_apply_interaction_missing_agent(self, belief_simulator, sample_agents_data):
        """Test interaction with non-existent agent."""
        from simulacrum.validation.types import BeliefState
        agents = [BeliefState(**agent) for agent in sample_agents_data]

        trajectories = belief_simulator._initialize_trajectories(agents, 3)
        current_beliefs = {
            "agent1": {"topic_a": 0.8, "topic_b": 0.2},
            "agent2": {"topic_a": 0.3, "topic_b": 0.9}
        }

        invalid_interaction = {
            "agent_a": "agent1",
            "agent_b": "nonexistent",
            "influence_strength": 0.2,
            "topics": ["topic_a"]
        }

        # Should not crash, just return early
        belief_simulator._apply_interaction(current_beliefs, invalid_interaction, trajectories, 1)

        # Beliefs should remain unchanged
        assert trajectories["agent1"][1]["topic_a"] == trajectories["agent1"][0]["topic_a"]

    def test_apply_interaction_missing_topic(self, belief_simulator, sample_agents_data):
        """Test interaction with topic not present in agent beliefs."""
        from simulacrum.validation.types import BeliefState
        agents = [BeliefState(**agent) for agent in sample_agents_data]

        trajectories = belief_simulator._initialize_trajectories(agents, 3)
        current_beliefs = {
            "agent1": {"topic_a": 0.8, "topic_b": 0.2},
            "agent2": {"topic_a": 0.3, "topic_b": 0.9}
        }

        interaction = {
            "agent_a": "agent1",
            "agent_b": "agent2",
            "influence_strength": 0.2,
            "topics": ["nonexistent_topic"]
        }

        belief_simulator._apply_interaction(current_beliefs, interaction, trajectories, 1)

        # Beliefs should remain unchanged for that topic
        assert trajectories["agent1"][1]["topic_a"] == trajectories["agent1"][0]["topic_a"]

    def test_apply_adaptation(self, belief_simulator, sample_agents_data):
        """Test individual belief adaptation."""
        from simulacrum.validation.types import BeliefState
        agents = [BeliefState(**agent) for agent in sample_agents_data]

        trajectories = belief_simulator._initialize_trajectories(agents, 3)
        current_beliefs = {
            "agent1": {"topic_a": 0.8, "topic_b": 0.2},
            "agent2": {"topic_a": 0.3, "topic_b": 0.9}
        }

        belief_simulator._apply_adaptation(agents[0], current_beliefs, trajectories, 1)

        # Beliefs should change due to adaptation toward 0.5
        assert trajectories["agent1"][1]["topic_a"] != trajectories["agent1"][0]["topic_a"]
        assert 0.0 <= trajectories["agent1"][1]["topic_a"] <= 1.0

    def test_analyze_belief_evolution(self, belief_simulator, sample_agents_data):
        """Test belief evolution analysis."""
        from simulacrum.validation.types import BeliefState
        agents = [BeliefState(**agent) for agent in sample_agents_data]

        # Create mock trajectories
        trajectories = {
            "agent1": [
                {"topic_a": 0.8, "topic_b": 0.2},
                {"topic_a": 0.7, "topic_b": 0.3},
                {"topic_a": 0.6, "topic_b": 0.4}
            ],
            "agent2": [
                {"topic_a": 0.3, "topic_b": 0.9},
                {"topic_a": 0.4, "topic_b": 0.8},
                {"topic_a": 0.5, "topic_b": 0.7}
            ]
        }

        analysis = belief_simulator._analyze_belief_evolution(trajectories, agents)

        assert "convergence" in analysis
        assert "polarization" in analysis
        assert "consensus" in analysis
        assert "belief_volatility" in analysis

        assert "topic_a" in analysis["convergence"]
        assert "topic_b" in analysis["convergence"]

    def test_analyze_polarization(self, belief_simulator):
        """Test polarization analysis."""
        # Test with polarized values
        polarized_values = [0.05, 0.95, 0.08, 0.92]  # Mostly extreme
        result = belief_simulator._analyze_polarization(polarized_values)

        assert result["polarized"] is True
        assert result["extremes_count"] == 4
        assert result["extremes_ratio"] == 1.0

        # Test with moderate values
        moderate_values = [0.4, 0.5, 0.6, 0.55]
        result = belief_simulator._analyze_polarization(moderate_values)

        assert result["polarized"] is False
        assert result["extremes_count"] == 0

        # Test with single value
        single_value = [0.5]
        result = belief_simulator._analyze_polarization(single_value)

        assert result["polarized"] is False
        assert result["extremes_count"] == 0

    def test_analyze_volatility(self, belief_simulator):
        """Test volatility analysis."""
        # Test with changing trajectories
        trajectories = [
            [0.5, 0.6, 0.4, 0.7, 0.5],
            [0.3, 0.2, 0.4, 0.3, 0.5]
        ]
        result = belief_simulator._analyze_volatility(trajectories)

        assert "average_volatility" in result
        assert "max_volatility" in result
        assert "volatility_std" in result
        assert result["average_volatility"] > 0

        # Test with empty trajectories
        result = belief_simulator._analyze_volatility([])
        assert result["average_volatility"] == 0.0

        # Test with single point trajectories
        single_point = [[0.5]]
        result = belief_simulator._analyze_volatility(single_point)
        assert result["average_volatility"] == 0.0

    def test_analyze_overall_consensus(self, belief_simulator):
        """Test overall consensus analysis."""
        # Mock convergence data
        convergence_data = {
            "topic_a": {"consensus_reached": True},
            "topic_b": {"consensus_reached": False},
            "topic_c": {"consensus_reached": True}
        }

        result = belief_simulator._analyze_overall_consensus(convergence_data)

        assert result["consensus_topics"] == 2
        assert result["total_topics"] == 3
        assert result["consensus_ratio"] == 2/3
        assert result["overall_consensus"] is False

        # Test with all consensus
        all_consensus = {
            "topic_a": {"consensus_reached": True},
            "topic_b": {"consensus_reached": True}
        }

        result = belief_simulator._analyze_overall_consensus(all_consensus)
        assert result["overall_consensus"] is True

        # Test with empty data
        result = belief_simulator._analyze_overall_consensus({})
        assert result["overall_consensus"] is False

    def test_get_final_state(self, belief_simulator, sample_agents_data):
        """Test getting final belief state."""
        from simulacrum.validation.types import BeliefState
        agents = [BeliefState(**agent) for agent in sample_agents_data]

        trajectories = {
            "agent1": [
                {"topic_a": 0.8, "topic_b": 0.2},
                {"topic_a": 0.6, "topic_b": 0.4}
            ],
            "agent2": [
                {"topic_a": 0.3, "topic_b": 0.9},
                {"topic_a": 0.5, "topic_b": 0.7}
            ]
        }

        final_state = belief_simulator._get_final_state(trajectories, agents)

        assert "agent1" in final_state
        assert "agent2" in final_state
        assert final_state["agent1"]["beliefs"]["topic_a"] == 0.6
        assert final_state["agent1"]["agent_info"]["adaptability"] == 0.3

    def test_simulate_belief_dynamics_convergence_analysis(self, belief_simulator):
        """Test that convergence analysis works correctly."""
        agents_data = [
            {
                "agent_id": "agent1",
                "beliefs": {"topic": 0.9},
                "evidence_weight": 0.8,
                "adaptability": 0.9  # High adaptability
            },
            {
                "agent_id": "agent2",
                "beliefs": {"topic": 0.1},
                "evidence_weight": 0.8,
                "adaptability": 0.9
            }
        ]

        interactions = [
            {
                "agent_a": "agent1",
                "agent_b": "agent2",
                "influence_strength": 0.5,
                "topics": ["topic"]
            }
        ]

        result = belief_simulator.simulate_belief_dynamics(agents_data, interactions, time_steps=20)

        analysis = result["analysis"]

        # With high influence and adaptability, should reach consensus
        assert "topic" in analysis["convergence"]
        convergence = analysis["convergence"]["topic"]

        # Final std should be less than initial
        assert convergence["final_std"] < convergence["initial_std"]

    def test_simulate_belief_dynamics_polarization_analysis(self, belief_simulator):
        """Test polarization analysis in simulation results."""
        # Create agents with extreme initial beliefs
        agents_data = [
            {
                "agent_id": "agent1",
                "beliefs": {"topic": 0.95},
                "evidence_weight": 0.8,
                "adaptability": 0.1  # Low adaptability
            },
            {
                "agent_id": "agent2",
                "beliefs": {"topic": 0.05},
                "evidence_weight": 0.8,
                "adaptability": 0.1
            }
        ]

        result = belief_simulator.simulate_belief_dynamics(agents_data, [], time_steps=10)

        analysis = result["analysis"]
        assert "topic" in analysis["polarization"]

        polarization = analysis["polarization"]["topic"]
        assert "polarized" in polarization
        assert "extremes_count" in polarization
