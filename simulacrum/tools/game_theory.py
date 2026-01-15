"""
Game theory dynamics tool for strategic analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from itertools import product
from ..validation.errors import ValidationError


class StrategyProfile(NamedTuple):
    """Represents a strategy profile for all players."""
    strategies: Tuple[str, ...]
    payoffs: Tuple[float, ...]


class GameTheoryDynamics:
    """
    Tool for analyzing strategic interactions using game theory.

    Supports various equilibrium concepts and dynamic analysis of
    how strategies evolve in repeated games.
    """

    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 1000):
        """
        Initialize game theory dynamics simulator.

        Args:
            learning_rate: Learning rate for strategy adaptation
            max_iterations: Maximum iterations for convergence
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def analyze_game_theory_dynamics(self, game_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze strategic dynamics in a game.

        Args:
            game_definition: Dictionary containing game definition with:
                - players: List of player names
                - strategies: Dict mapping player to list of strategies
                - payoffs: Dict mapping strategy profiles to payoff vectors
                - game_type: "normal_form", "extensive_form", etc.

        Returns:
            Dictionary containing game analysis results

        Raises:
            ValidationError: If game definition is invalid
        """
        try:
            game_type = game_definition.get('game_type', 'normal_form')

            if game_type == 'normal_form':
                return self._analyze_normal_form_game(game_definition)
            elif game_type == 'extensive_form':
                return self._analyze_extensive_form_game(game_definition)
            else:
                raise ValidationError(
                    "game_type",
                    game_type,
                    f"Unsupported game type: {game_type}"
                )

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                "game_definition",
                game_definition,
                f"Game theory analysis failed: {str(e)}"
            )

    def _analyze_normal_form_game(self, game_def: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze normal form game."""
        players = game_def['players']
        strategies = game_def['strategies']
        payoffs = game_def['payoffs']

        # Find Nash equilibria
        nash_equilibria = self._find_nash_equilibria(players, strategies, payoffs)

        # Compute best responses
        best_responses = self._compute_best_responses(players, strategies, payoffs)

        # Analyze strategic stability
        stability_analysis = self._analyze_strategic_stability(nash_equilibria, payoffs)

        # Simulate learning dynamics
        learning_dynamics = self._simulate_learning_dynamics(players, strategies, payoffs)

        return {
            "status": "success",
            "game_type": "normal_form",
            "players": players,
            "strategy_spaces": strategies,
            "nash_equilibria": nash_equilibria,
            "best_responses": best_responses,
            "stability_analysis": stability_analysis,
            "learning_dynamics": learning_dynamics,
            "summary": self._generate_game_summary(nash_equilibria, stability_analysis)
        }

    def _find_nash_equilibria(self, players: List[str],
                             strategies: Dict[str, List[str]],
                             payoffs: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Find Nash equilibria using brute force enumeration."""
        equilibria = []

        # Generate all possible strategy profiles
        strategy_lists = [strategies[player] for player in players]
        all_profiles = list(product(*strategy_lists))

        for profile in all_profiles:
            profile_dict = dict(zip(players, profile))
            profile_key = ','.join(profile)

            if profile_key in payoffs:
                player_payoffs = payoffs[profile_key]

                # Check if this is a Nash equilibrium
                is_nash = True
                for i, player in enumerate(players):
                    current_payoff = player_payoffs[i]

                    # Check if any unilateral deviation is profitable
                    for strategy in strategies[player]:
                        if strategy != profile[i]:
                            deviation_profile = list(profile)
                            deviation_profile[i] = strategy
                            deviation_key = ','.join(deviation_profile)

                            if deviation_key in payoffs:
                                deviation_payoff = payoffs[deviation_key][i]
                                if deviation_payoff > current_payoff:
                                    is_nash = False
                                    break

                    if not is_nash:
                        break

                if is_nash:
                    equilibria.append({
                        "strategy_profile": profile_dict,
                        "payoffs": dict(zip(players, player_payoffs)),
                        "profile_key": profile_key
                    })

        return equilibria

    def _compute_best_responses(self, players: List[str],
                              strategies: Dict[str, List[str]],
                              payoffs: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compute best response correspondences."""
        best_responses = {}

        for player in players:
            player_index = players.index(player)
            player_best_responses = {}

            # For each possible strategy profile of other players
            other_players = [p for p in players if p != player]
            other_strategies = [strategies[p] for p in other_players]

            if other_strategies:
                for other_profile in product(*other_strategies):
                    other_profile_dict = dict(zip(other_players, other_profile))

                    # Find best response for this player
                    best_payoff = float('-inf')
                    best_strategies = []

                    for strategy in strategies[player]:
                        test_profile = list(other_profile)
                        test_profile.insert(player_index, strategy)
                        profile_key = ','.join(test_profile)

                        if profile_key in payoffs:
                            payoff = payoffs[profile_key][player_index]
                            if payoff > best_payoff:
                                best_payoff = payoff
                                best_strategies = [strategy]
                            elif payoff == best_payoff:
                                best_strategies.append(strategy)

                    player_best_responses[str(other_profile_dict)] = {
                        "best_strategies": best_strategies,
                        "payoff": best_payoff
                    }

            best_responses[player] = player_best_responses

        return best_responses

    def _analyze_strategic_stability(self, equilibria: List[Dict[str, Any]],
                                   payoffs: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze stability properties of equilibria."""
        stability = {
            "pareto_optimal": [],
            "risk_dominant": None,
            "payoff_dominant": None,
            "stability_classification": []
        }

        if not equilibria:
            return stability

        # Check Pareto optimality
        for eq in equilibria:
            profile_key = eq["profile_key"]
            eq_payoffs = eq["payoffs"]

            is_pareto_optimal = True
            for other_key, other_payoffs in payoffs.items():
                if other_key != profile_key:
                    # Check if other profile Pareto dominates this one
                    dominates = True
                    strictly_better = False

                    for player, eq_payoff in eq_payoffs.items():
                        player_idx = list(eq_payoffs.keys()).index(player)
                        other_payoff = other_payoffs[player_idx]

                        if other_payoff < eq_payoff:
                            dominates = False
                            break
                        elif other_payoff > eq_payoff:
                            strictly_better = True

                    if dominates and strictly_better:
                        is_pareto_optimal = False
                        break

            if is_pareto_optimal:
                stability["pareto_optimal"].append(eq)

        # Simple risk dominance check (for 2x2 games)
        if len(equilibria) == 2:
            stability["risk_dominant"] = self._check_risk_dominance(equilibria[0], equilibria[1], payoffs)

        # Payoff dominance
        if equilibria:
            max_total_payoff = max(sum(eq["payoffs"].values()) for eq in equilibria)
            stability["payoff_dominant"] = next(
                (eq for eq in equilibria if sum(eq["payoffs"].values()) == max_total_payoff),
                None
            )

        return stability

    def _check_risk_dominance(self, eq1: Dict[str, Any], eq2: Dict[str, Any],
                            payoffs: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
        """Check risk dominance between two equilibria."""
        # Simplified risk dominance check
        # In a full implementation, this would use proper risk dominance criteria
        return None  # Placeholder

    def _simulate_learning_dynamics(self, players: List[str],
                                  strategies: Dict[str, List[str]],
                                  payoffs: Dict[str, List[float]]) -> Dict[str, Any]:
        """Simulate how strategies evolve through learning."""
        # Simple fictitious play simulation
        strategy_counts = {player: {strat: 1.0 for strat in strategies[player]}
                          for player in players}

        mixed_strategies = {player: {strat: 1.0/len(strategies[player])
                                   for strat in strategies[player]}
                          for player in players}

        # Simulate learning over time
        history = []
        for iteration in range(min(100, self.max_iterations)):
            # Record current state
            history.append({
                "iteration": iteration,
                "mixed_strategies": mixed_strategies.copy()
            })

            # Each player best responds to current beliefs about others
            for player in players:
                # Compute expected payoffs for each strategy
                expected_payoffs = {}
                for strategy in strategies[player]:
                    total_payoff = 0.0
                    total_prob = 0.0

                    # Average over other players' mixed strategies
                    other_players = [p for p in players if p != player]
                    other_strategy_profiles = list(product(*[strategies[p] for p in other_players]))

                    for other_profile in other_strategy_profiles:
                        prob = np.prod([mixed_strategies[p][s] for p, s in zip(other_players, other_profile)])

                        full_profile = list(other_profile)
                        full_profile.insert(players.index(player), strategy)
                        profile_key = ','.join(full_profile)

                        if profile_key in payoffs:
                            payoff = payoffs[profile_key][players.index(player)]
                            total_payoff += prob * payoff
                            total_prob += prob

                    expected_payoffs[strategy] = total_payoff / total_prob if total_prob > 0 else 0

                # Update mixed strategy (simple reinforcement learning)
                max_payoff = max(expected_payoffs.values())
                best_strategies = [s for s, p in expected_payoffs.items() if p == max_payoff]

                for strategy in strategies[player]:
                    if strategy in best_strategies:
                        mixed_strategies[player][strategy] += self.learning_rate * (1 - mixed_strategies[player][strategy])
                    else:
                        mixed_strategies[player][strategy] *= (1 - self.learning_rate)

                # Normalize
                total = sum(mixed_strategies[player].values())
                for strategy in strategies[player]:
                    mixed_strategies[player][strategy] /= total

        return {
            "final_mixed_strategies": mixed_strategies,
            "convergence_history": history[-10:],  # Last 10 iterations
            "learning_method": "fictitious_play"
        }

    def _generate_game_summary(self, equilibria: List[Dict[str, Any]],
                             stability: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of game analysis."""
        return {
            "equilibria_count": len(equilibria),
            "has_equilibria": len(equilibria) > 0,
            "pareto_optimal_count": len(stability.get("pareto_optimal", [])),
            "key_insights": self._generate_game_insights(equilibria, stability)
        }

    def _generate_game_insights(self, equilibria: List[Dict[str, Any]],
                               stability: Dict[str, Any]) -> List[str]:
        """Generate key insights from game analysis."""
        insights = []

        if not equilibria:
            insights.append("No Nash equilibria found - game may have no stable outcomes")
        elif len(equilibria) == 1:
            insights.append("Single Nash equilibrium - clear strategic prediction")
        else:
            insights.append(f"Multiple equilibria ({len(equilibria)}) - coordination problem likely")

        if stability.get("pareto_optimal"):
            insights.append(f"{len(stability['pareto_optimal'])} Pareto-optimal equilibria found")

        return insights
