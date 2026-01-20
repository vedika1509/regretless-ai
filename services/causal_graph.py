"""Service to build causal relationships between decision variables and outcomes."""
from typing import Dict, List, Tuple
from models.decision import StructuredDecision


class CausalGraph:
    """Simple causal graph builder for decision variables."""
    
    def __init__(self):
        """Initialize causal relationships."""
        # Define default causal relationships
        # Format: (variable_pattern, outcome_dimension, impact_type, strength)
        self.causal_rules = [
            # Financial impacts
            ("salary", "financial_satisfaction", "positive", 0.8),
            ("income", "financial_satisfaction", "positive", 0.8),
            ("cost", "financial_satisfaction", "negative", 0.7),
            ("rent", "financial_satisfaction", "negative", 0.7),
            ("price", "financial_satisfaction", "negative", 0.7),
            
            # Stability and security
            ("stability", "job_security", "positive", 0.9),
            ("company_stability", "job_security", "positive", 0.9),
            ("market_risk", "job_security", "negative", 0.6),
            ("uncertainty", "risk_score", "positive", 0.7),
            
            # Time and stress
            ("stress", "overall_satisfaction", "negative", 0.7),
            ("work_life", "overall_satisfaction", "positive", 0.6),
            ("work_life_balance", "overall_satisfaction", "positive", 0.6),
            ("time", "overall_satisfaction", "positive", 0.5),
            ("hours", "overall_satisfaction", "negative", 0.5),
            
            # Career growth
            ("growth", "satisfaction_score", "positive", 0.7),
            ("career", "satisfaction_score", "positive", 0.7),
            ("opportunity", "satisfaction_score", "positive", 0.6),
            
            # Default catch-all for unknown variables
            ("default", "overall_satisfaction", "positive", 0.3),
        ]
        
        # Special handling for job switches
        self.job_switch_rules = [
            ("job_switch", "stress", "positive", 0.5, "short_term"),
            ("job_switch", "satisfaction_score", "positive", 0.4, "long_term"),
        ]
    
    def build_graph(self, decision: StructuredDecision) -> Dict[str, List[Tuple[str, str, float, str]]]:
        """
        Build causal graph for a decision.
        
        Args:
            decision: Structured decision with variables
        
        Returns:
            Dictionary mapping outcome dimensions to list of (variable, impact_type, strength, duration)
        """
        graph = {
            "financial_satisfaction": [],
            "job_security": [],
            "overall_satisfaction": [],
            "satisfaction_score": [],
            "risk_score": [],
            "stress": [],
        }
        
        decision_type = decision.decision_type.lower()
        
        # Add special rules for job switches
        if "job" in decision_type:
            for var_name, outcome, impact, strength, duration in self.job_switch_rules:
                if outcome in graph:
                    graph[outcome].append((var_name, impact, strength, duration))
        
        # Match variables to causal rules
        for var_name in decision.variables.keys():
            var_lower = var_name.lower()
            matched = False
            
            for pattern, outcome, impact_type, strength in self.causal_rules:
                if pattern in var_lower or var_lower in pattern:
                    if outcome in graph:
                        graph[outcome].append((var_name, impact_type, strength, "long_term"))
                        matched = True
                        break
            
            # Default mapping if no match
            if not matched and "default" in [r[0] for r in self.causal_rules]:
                graph["overall_satisfaction"].append((var_name, "positive", 0.3, "long_term"))
        
        return graph
    
    def compute_outcomes(self, decision: StructuredDecision, sampled_values: Dict[str, float]) -> Dict[str, float]:
        """
        Compute outcome dimensions from sampled variable values.
        
        Args:
            decision: Structured decision
            sampled_values: Dictionary of variable names to sampled values
        
        Returns:
            Dictionary of outcome dimensions to computed scores
        """
        graph = self.build_graph(decision)
        outcomes = {
            "financial_satisfaction": 0.5,  # Baseline
            "job_security": 0.5,
            "overall_satisfaction": 0.5,
            "satisfaction_score": 0.5,
            "risk_score": 0.5,
            "stress": 0.0,  # Baseline stress
        }
        
        # Apply causal relationships
        for outcome_dim, relationships in graph.items():
            contributions = []
            for var_name, impact_type, strength, duration in relationships:
                if var_name in sampled_values:
                    value = sampled_values[var_name]
                    
                    # Normalize value to [-1, 1] range if needed
                    if abs(value) > 1:
                        value = max(-1, min(1, value))
                    
                    # Apply impact based on type
                    if impact_type == "positive":
                        contribution = value * strength
                    else:  # negative
                        contribution = -value * strength
                    
                    contributions.append(contribution)
            
            # Aggregate contributions
            if contributions:
                base_value = outcomes[outcome_dim]
                net_contribution = sum(contributions) / len(contributions)
                outcomes[outcome_dim] = max(0.0, min(1.0, base_value + net_contribution))
        
        return outcomes
