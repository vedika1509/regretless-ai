"""Counterfactual Explorer - What-if scenarios for changing outcomes."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from models.scenario import AnalysisResult


class CounterfactualExplorer:
    """Analyze what variable changes would shift outcomes."""
    
    def __init__(self):
        """Initialize counterfactual explorer."""
        pass
    
    def compute_centroid(self, results: List[Dict[str, float]], metric: str = "overall_score") -> float:
        """
        Compute centroid of a cluster of results.
        
        Args:
            results: List of simulation results
            metric: Metric to compute centroid for
            
        Returns:
            Centroid value
        """
        if not results:
            return 0.5
        
        values = [r[metric] for r in results]
        return np.mean(values)
    
    def compute_variable_centroid(self, results: List[Dict[str, float]], var_name: str) -> float:
        """
        Compute centroid for a specific variable.
        
        Args:
            results: List of simulation results
            var_name: Name of variable
            
        Returns:
            Variable centroid value
        """
        if not results or not results[0].get("sampled_values"):
            return 0.0
        
        values = [
            r["sampled_values"].get(var_name, 0.0) 
            for r in results 
            if r.get("sampled_values")
        ]
        
        if not values:
            return 0.0
        
        return np.mean(values)
    
    def compute_gradients(
        self, 
        results: List[Dict[str, float]], 
        variable_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute gradients (marginal impact) for each variable.
        
        Args:
            results: List of simulation results
            variable_names: List of variable names
            
        Returns:
            Dictionary mapping variable names to gradient values
        """
        gradients = {}
        
        if not results or not variable_names:
            return gradients
        
        # For each variable, compute correlation with overall_score
        overall_scores = [r["overall_score"] for r in results]
        
        for var_name in variable_names:
            if not results[0].get("sampled_values") or var_name not in results[0]["sampled_values"]:
                continue
            
            var_values = [
                r["sampled_values"].get(var_name, 0.0) 
                for r in results 
                if r.get("sampled_values")
            ]
            
            if len(var_values) < 10:
                continue
            
            # Compute correlation as proxy for gradient
            correlation = np.corrcoef(var_values, overall_scores)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0
            
            # Gradient is the correlation scaled by standard deviation
            var_std = np.std(var_values)
            overall_std = np.std(overall_scores)
            
            if var_std > 1e-10 and overall_std > 1e-10:
                gradient = correlation * (overall_std / var_std)
            else:
                gradient = 0.0
            
            gradients[var_name] = gradient
        
        return gradients
    
    def find_minimum_changes(
        self,
        result: AnalysisResult,
        simulation_results: List[Dict[str, float]],
        variable_names: List[str],
        target_shift: str = "worst_to_best"
    ) -> List[Dict[str, any]]:
        """
        Find minimum variable changes needed to shift scenarios.
        
        Args:
            result: AnalysisResult with scenarios
            simulation_results: List of simulation results
            variable_names: List of variable names
            target_shift: Target shift (worst_to_best, worst_to_likely, likely_to_best)
            
        Returns:
            List of counterfactual changes
        """
        counterfactuals = []
        
        if not result.scenarios or not simulation_results:
            return counterfactuals
        
        # Compute gradients
        gradients = self.compute_gradients(simulation_results, variable_names)
        
        if not gradients:
            return counterfactuals
        
        # Rank variables by absolute gradient (impact)
        ranked_vars = sorted(
            gradients.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Calculate target score difference
        worst_score = result.scenarios.get("worst_case", {}).outcomes.overall_score if result.scenarios.get("worst_case") else 0.0
        most_likely_score = result.scenarios.get("most_likely", {}).outcomes.overall_score if result.scenarios.get("most_likely") else 0.5
        best_score = result.scenarios.get("best_case", {}).outcomes.overall_score if result.scenarios.get("best_case") else 1.0
        
        # Calculate score gaps
        worst_to_likely_gap = most_likely_score - worst_score
        worst_to_best_gap = best_score - worst_score
        likely_to_best_gap = best_score - most_likely_score
        
        # Determine target gap
        if target_shift == "worst_to_best":
            target_gap = worst_to_best_gap
            from_scenario = "worst case"
            to_scenario = "best case"
        elif target_shift == "worst_to_likely":
            target_gap = worst_to_likely_gap
            from_scenario = "worst case"
            to_scenario = "most likely"
        else:  # likely_to_best
            target_gap = likely_to_best_gap
            from_scenario = "most likely"
            to_scenario = "best case"
        
        # Get variable centroids for reference
        worst_results = [r for r in simulation_results if r["overall_score"] <= worst_score + 0.1]
        best_results = [r for r in simulation_results if r["overall_score"] >= best_score - 0.1]
        
        # Find top 5 variables with highest impact
        for var_name, gradient in ranked_vars[:5]:
            if abs(gradient) < 1e-10:
                continue
            
            # Compute current variable value
            current_value = self.compute_variable_centroid(
                worst_results if "worst" in from_scenario else simulation_results,
                var_name
            )
            
            # Calculate required change
            # gradient = Δ outcome / Δ variable
            # So: Δ variable = Δ outcome / gradient
            required_change = target_gap / gradient if abs(gradient) > 1e-10 else 0.0
            
            # Format change as percentage or absolute
            if abs(current_value) > 1e-10:
                change_percent = (required_change / abs(current_value)) * 100
            else:
                change_percent = 0.0
            
            # Create counterfactual description
            if required_change > 0:
                direction = "increase"
            else:
                direction = "decrease"
                required_change = abs(required_change)
                change_percent = abs(change_percent)
            
            counterfactuals.append({
                "variable": var_name,
                "direction": direction,
                "required_change": required_change,
                "change_percent": change_percent,
                "impact": abs(gradient),
                "explanation": f"If {var_name.replace('_', ' ')} {direction}s by {abs(change_percent):.1f}%, outcome would shift from {from_scenario} to {to_scenario}."
            })
        
        # Sort by impact
        counterfactuals.sort(key=lambda x: x["impact"], reverse=True)
        
        return counterfactuals


def generate_counterfactual_summary(
    counterfactuals: List[Dict[str, any]],
    result: AnalysisResult
) -> str:
    """
    Generate natural language summary of counterfactuals.
    
    Args:
        counterfactuals: List of counterfactual changes
        result: AnalysisResult
        
    Returns:
        Summary string
    """
    if not counterfactuals:
        return "Unable to identify specific variable changes that would significantly impact outcomes."
    
    top_change = counterfactuals[0]
    var_name = top_change["variable"].replace('_', ' ')
    direction = top_change["direction"]
    change = top_change["change_percent"]
    
    summary = f"To shift outcomes from worst to best case, the most impactful change would be to {direction} {var_name} by {change:.1f}%. "
    
    if len(counterfactuals) > 1:
        alt_change = counterfactuals[1]
        alt_var = alt_change["variable"].replace('_', ' ')
        alt_dir = alt_change["direction"]
        alt_change_val = alt_change["change_percent"]
        summary += f"Alternatively, {alt_dir}ing {alt_var} by {alt_change_val:.1f}% would also have significant impact."
    
    return summary
