"""Calculate Regret Score - proprietary metric for decision regret risk."""
from typing import Dict
from models.scenario import AnalysisResult


def calculate_regret(result: AnalysisResult) -> float:
    """
    Calculate Regret Score using the formula:
    Regret = Σ(Pi × Lossi × Emotional_Costi)
    
    Where:
    - Pi = Probability of scenario i
    - Lossi = Best outcome - Scenario outcome (downside magnitude)
    - Emotional_Costi = 1 + (risk_score × 0.5) (risk amplifies regret)
    
    Args:
        result: AnalysisResult with scenarios
        
    Returns:
        Regret Score between 0.0 and 1.0
    """
    if not result.scenarios or len(result.scenarios) == 0:
        return 0.0
    
    # Get best case score (reference point)
    best_score = result.scenarios.get("best_case")
    if not best_score:
        return 0.0
    
    best_outcome = best_score.outcomes.overall_score
    regret = 0.0
    
    # Calculate regret for each scenario
    for scenario_type, scenario in result.scenarios.items():
        # Skip best case (no regret)
        if scenario_type == "best_case":
            continue
        
        # Calculate loss (downside magnitude)
        loss = max(0.0, best_outcome - scenario.outcomes.overall_score)
        
        # Probability weight
        probability = scenario.probability
        
        # Emotional cost factor (risk amplifies regret)
        # Higher risk scenarios create more emotional regret
        emotional_cost = 1.0 + (scenario.outcomes.risk_score * 0.5)
        
        # Add to total regret
        regret += probability * loss * emotional_cost
    
    # Normalize to [0, 1] range
    # Maximum regret would be if worst case had 100% probability
    # and best case had 0% probability
    max_regret = 1.0 * 1.0 * 1.5  # Worst case: loss=1.0, prob=1.0, emotional=1.5
    normalized_regret = min(1.0, regret / max_regret if max_regret > 0 else 0.0)
    
    return normalized_regret


def get_regret_level(regret_score: float) -> str:
    """
    Get regret level interpretation.
    
    Args:
        regret_score: Regret score between 0.0 and 1.0
        
    Returns:
        Level string: "Low", "Medium", or "High"
    """
    if regret_score < 0.3:
        return "Low"
    elif regret_score < 0.6:
        return "Medium"
    else:
        return "High"


def get_regret_color(regret_score: float) -> str:
    """
    Get color for regret score display.
    
    Args:
        regret_score: Regret score between 0.0 and 1.0
        
    Returns:
        Color name: "green", "orange", or "red"
    """
    level = get_regret_level(regret_score)
    if level == "Low":
        return "green"
    elif level == "Medium":
        return "orange"
    else:
        return "red"


def get_regret_explanation(regret_score: float) -> str:
    """
    Get human-readable explanation of regret score.
    
    Args:
        regret_score: Regret score between 0.0 and 1.0
        
    Returns:
        Explanation string
    """
    level = get_regret_level(regret_score)
    
    explanations = {
        "Low": "Low risk of regretting this decision. Expected emotional cost is minimal even in worst-case scenarios.",
        "Medium": "Moderate risk of regret. Consider additional factors or mitigation strategies.",
        "High": "High risk of regretting this decision. Strongly consider alternatives or additional information before proceeding."
    }
    
    return explanations.get(level, "Unable to interpret regret score.")
