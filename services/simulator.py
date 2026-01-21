"""Monte Carlo simulation engine."""
import numpy as np
from typing import Dict, List, Callable, Optional
from models.decision import StructuredDecision, VariableDistribution
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
            x = np.random.beta(alpha, beta)  # in [0, 1]

            # Optional scaling to a custom range.
            # - If params include {min, max}, map into [min, max]
            # - If params include {range: [min, max]}, map into that range
            if "min" in params and "max" in params:
                min_val = float(params["min"])
                max_val = float(params["max"])
                return min_val + x * (max_val - min_val)
            if "range" in params and isinstance(params["range"], (list, tuple)) and len(params["range"]) == 2:
                min_val = float(params["range"][0])
                max_val = float(params["range"][1])
                return min_val + x * (max_val - min_val)

            # If a variable's base value is outside [0, 1], treat this beta sample as a signed factor.
            # This prevents "negative" concepts (e.g., toxicity) from always sampling as positive.
            if distribution.value < 0.0 or distribution.value > 1.0:
                signed = (x * 2.0) - 1.0  # in [-1, 1]
                # Optional: dampen/scale the signed factor
                scale = float(params.get("scale", 1.0))
                return signed * scale

            return x
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
            # The causal model often routes many human factors into overall_satisfaction.
            # Blend it into satisfaction_score so those drivers actually affect displayed results.
            satisfaction_score = (
                0.6 * outcomes.get("satisfaction_score", 0.5)
                + 0.4 * outcomes.get("overall_satisfaction", 0.5)
            )

            # Blend risk with (lack of) job security so stability drivers influence risk outcomes.
            base_risk = outcomes.get("risk_score", 0.5)
            job_insecurity = 1.0 - outcomes.get("job_security", 0.5)
            risk_score = (0.7 * base_risk) + (0.3 * job_insecurity)
            
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

        n = len(sorted_results)
        # Product choice: use 25/50/25 buckets for interpretability.
        # - Worst Case: bottom 25%
        # - Most Likely: middle 50%
        # - Best Case: top 25%
        tail = max(1, int(0.25 * n))

        # Buckets: worst 25%, best 25%, most_likely = middle 50%
        worst_case_data = sorted_results[:tail]
        best_case_data = sorted_results[n - tail :]
        most_likely_data = sorted_results[tail : n - tail] if n - 2 * tail >= 1 else sorted_results

        # Compute overall means for driver deltas
        all_vars = set()
        for r in results:
            all_vars.update((r.get("sampled_values") or {}).keys())
        all_vars = sorted(all_vars)

        overall_means = {}
        for var in all_vars:
            vals = [r["sampled_values"][var] for r in results if "sampled_values" in r and var in r["sampled_values"]]
            if vals:
                overall_means[var] = float(np.mean(vals))

        def _drivers_for(bucket: List[Dict[str, float]], top_k: int = 3):
            if not bucket or not overall_means:
                return []
            driver_rows = []
            for var, overall_mean in overall_means.items():
                vals = [r["sampled_values"][var] for r in bucket if "sampled_values" in r and var in r["sampled_values"]]
                if not vals:
                    continue
                bucket_mean = float(np.mean(vals))
                delta = bucket_mean - overall_mean
                driver_rows.append({"variable": var, "mean": bucket_mean, "delta": float(delta)})
            driver_rows.sort(key=lambda x: abs(x["delta"]), reverse=True)
            return driver_rows[:top_k]

        def _scenario_stats(bucket: List[Dict[str, float]]):
            if not bucket:
                return {
                    "overall_score": 0.5,
                    "satisfaction_score": 0.5,
                    "financial_score": 0.5,
                    "risk_score": 0.5,
                    # Extra real-world dimensions (0–1) for human-facing metrics
                    "job_security": 0.5,
                    "financial_satisfaction": 0.5,
                    "overall_satisfaction": 0.5,
                    "stress": 0.5,
                }

            def _mean_outcome(key: str, default: float = 0.5) -> float:
                vals = []
                for r in bucket:
                    out = r.get("outcomes") or {}
                    v = out.get(key, None)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                return float(np.mean(vals)) if vals else float(default)

            return {
                "overall_score": float(np.mean([r["overall_score"] for r in bucket])),
                "satisfaction_score": float(np.mean([r["satisfaction_score"] for r in bucket])),
                "financial_score": float(np.mean([r["financial_score"] for r in bucket])),
                "risk_score": float(np.mean([r["risk_score"] for r in bucket])),
                "job_security": _mean_outcome("job_security", 0.5),
                "financial_satisfaction": _mean_outcome("financial_satisfaction", 0.5),
                "overall_satisfaction": _mean_outcome("overall_satisfaction", 0.5),
                # Stress is modeled as "higher = worse" but still returned 0–1 for readability
                "stress": _mean_outcome("stress", 0.5),
            }

        best_case = _scenario_stats(best_case_data)
        best_case["probability"] = len(best_case_data) / n
        best_case["drivers"] = _drivers_for(best_case_data)

        worst_case = _scenario_stats(worst_case_data)
        worst_case["probability"] = len(worst_case_data) / n
        worst_case["drivers"] = _drivers_for(worst_case_data)

        most_likely = _scenario_stats(most_likely_data)
        most_likely["probability"] = len(most_likely_data) / n
        most_likely["drivers"] = _drivers_for(most_likely_data)

        return {"best_case": best_case, "worst_case": worst_case, "most_likely": most_likely}
    
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

        # Guardrail: if the input space is too low-dimensional, don't claim very high confidence.
        # (Common when the parser falls back or extracts very few variables.)
        try:
            sample0 = results[0].get("sampled_values") or {}
            var_count = len(sample0)
            if var_count <= 1:
                confidence = min(confidence, 0.45)
            elif var_count == 2:
                confidence = min(confidence, 0.65)
        except Exception:
            pass

        return confidence
