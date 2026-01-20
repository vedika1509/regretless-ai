"""Monte Carlo simulation engine."""
import numpy as np
from typing import Dict, List, Callable, Optional
from models.decision import StructuredDecision, VariableDistribution
from models.scenario import OutcomeMetrics, ScenarioResult
from services.causal_graph import CausalGraph


class MonteCarloSimulator:
    """Monte Carlo simulator for decision outcomes."""
    
    def __init__(self, n_iterations: int = 3000):
        """
        Initialize simulator.
        
        Args:
            n_iterations: Number of simulation runs
        """
        self.n_iterations = n_iterations
        self.causal_graph = CausalGraph()
    
    def sample_distribution(self, distribution: VariableDistribution) -> float:
        """
        Sample a value from a probability distribution.
        
        Args:
            distribution: VariableDistribution object
        
        Returns:
            Sampled value
        """
        dist_type = distribution.distribution.lower()
        params = distribution.params
        
        if dist_type == "normal":
            return np.random.normal(
                loc=params.get("mean", distribution.value),
                scale=params.get("std", 0.1)
            )
        elif dist_type == "beta":
            alpha = params.get("alpha", 2.0)
            beta = params.get("beta", 2.0)
            return np.random.beta(alpha, beta)
        elif dist_type == "uniform":
            min_val = params.get("min", -0.5)
            max_val = params.get("max", 0.5)
            return np.random.uniform(min_val, max_val)
        elif dist_type == "bernoulli":
            prob = params.get("probability", 0.5)
            return float(np.random.binomial(1, prob))
        else:
            # Default: return base value with small noise
            return distribution.value + np.random.normal(0, 0.1)
    
    def run_simulation(
        self, 
        decision: StructuredDecision,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, float]]:
        """
        Run Monte Carlo simulation.
        
        Args:
            decision: Structured decision
            progress_callback: Optional callback function(current, total) for progress updates
        
        Returns:
            List of simulation results, each containing outcome scores
        """
        results = []
        
        for i in range(self.n_iterations):
            # Sample values for all variables
            sampled_values = {}
            for var_name, distribution in decision.variables.items():
                sampled_values[var_name] = self.sample_distribution(distribution)
            
            # Compute outcomes using causal graph
            outcomes = self.causal_graph.compute_outcomes(decision, sampled_values)
            
            # Calculate composite scores
            financial_score = outcomes.get("financial_satisfaction", 0.5)
            satisfaction_score = outcomes.get("satisfaction_score", 0.5)
            risk_score = outcomes.get("risk_score", 0.5)
            
            # Overall score (weighted average)
            overall_score = (
                0.4 * financial_score +
                0.4 * satisfaction_score +
                0.2 * (1 - risk_score)  # Lower risk is better
            )
            
            result = {
                "satisfaction_score": satisfaction_score,
                "financial_score": financial_score,
                "risk_score": risk_score,
                "overall_score": overall_score,
                "outcomes": outcomes,
                "sampled_values": sampled_values
            }
            
            results.append(result)
            
            # Update progress
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, self.n_iterations)
        
        # Final progress update
        if progress_callback:
            progress_callback(self.n_iterations, self.n_iterations)
        
        return results
    
    def extract_scenarios(self, results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Extract best case, worst case, and most likely scenarios.
        
        Args:
            results: List of simulation results
        
        Returns:
            Dictionary with best_case, worst_case, most_likely scenarios
        """
        if not results:
            return {}
        
        # Sort by overall score
        sorted_results = sorted(results, key=lambda x: x["overall_score"])
        
        # Best case: top 10%
        best_start = int(0.9 * len(sorted_results))
        best_case_data = sorted_results[best_start:]
        best_case = {
            "overall_score": np.mean([r["overall_score"] for r in best_case_data]),
            "satisfaction_score": np.mean([r["satisfaction_score"] for r in best_case_data]),
            "financial_score": np.mean([r["financial_score"] for r in best_case_data]),
            "risk_score": np.mean([r["risk_score"] for r in best_case_data]),
            "probability": 0.1,
        }
        
        # Worst case: bottom 10%
        worst_case_data = sorted_results[:int(0.1 * len(sorted_results))]
        worst_case = {
            "overall_score": np.mean([r["overall_score"] for r in worst_case_data]),
            "satisfaction_score": np.mean([r["satisfaction_score"] for r in worst_case_data]),
            "financial_score": np.mean([r["financial_score"] for r in worst_case_data]),
            "risk_score": np.mean([r["risk_score"] for r in worst_case_data]),
            "probability": 0.1,
        }
        
        # Most likely: median cluster (45th to 55th percentile)
        p45_idx = int(0.45 * len(sorted_results))
        p55_idx = int(0.55 * len(sorted_results))
        most_likely_data = sorted_results[p45_idx:p55_idx]
        most_likely = {
            "overall_score": np.mean([r["overall_score"] for r in most_likely_data]),
            "satisfaction_score": np.mean([r["satisfaction_score"] for r in most_likely_data]),
            "financial_score": np.mean([r["financial_score"] for r in most_likely_data]),
            "risk_score": np.mean([r["risk_score"] for r in most_likely_data]),
            "probability": 0.1,
        }
        
        return {
            "best_case": best_case,
            "worst_case": worst_case,
            "most_likely": most_likely,
        }
    
    def calculate_confidence(self, results: List[Dict[str, float]]) -> float:
        """
        Calculate confidence score based on outcome consistency.
        
        Args:
            results: List of simulation results
        
        Returns:
            Confidence score between 0 and 1
        """
        if not results or len(results) < 2:
            return 0.5
        
        # Calculate coefficient of variation for overall scores
        overall_scores = [r["overall_score"] for r in results]
        mean_score = np.mean(overall_scores)
        std_score = np.std(overall_scores)
        
        # Handle edge cases
        if mean_score == 0 or abs(mean_score) < 1e-10:
            return 0.5
        
        # Handle zero or very small standard deviation
        if std_score < 1e-10:
            # All scores are nearly identical - high confidence
            return 0.95
        
        # Lower coefficient of variation = higher confidence
        cv = std_score / abs(mean_score)
        
        # Convert to confidence: lower CV = higher confidence
        # Use exponential decay: confidence = exp(-2*cv)
        confidence = np.exp(-2 * cv)
        
        # Normalize to [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
