"""Service to generate human-readable explanations using LLM."""
import os
import google.generativeai as genai
from typing import Dict, Any
from dotenv import load_dotenv

from models.scenario import ScenarioResult, RiskFlag

load_dotenv()


def initialize_gemini() -> genai.GenerativeModel:
    """Initialize Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')


def generate_scenario_explanation(
    scenario_type: str,
    metrics: Dict[str, float],
    decision_text: str
) -> str:
    """
    Generate human-readable explanation for a scenario.
    
    Args:
        scenario_type: Type of scenario (best_case, worst_case, most_likely)
        metrics: Outcome metrics for the scenario
        decision_text: Original decision text
    
    Returns:
        Human-readable explanation
    """
    model = initialize_gemini()
    
    scenario_names = {
        "best_case": "Best Case",
        "worst_case": "Worst Case",
        "most_likely": "Most Likely"
    }
    
    scenario_name = scenario_names.get(scenario_type, scenario_type)
    
    prompt = f"""You are a decision analysis expert providing empathetic, clear explanations of decision outcomes.

Original Decision: {decision_text}

Scenario Type: {scenario_name}

Outcome Metrics:
- Overall Score: {metrics.get('overall_score', 0.5):.2f} (0 = worst, 1 = best)
- Financial Score: {metrics.get('financial_score', 0.5):.2f}
- Satisfaction Score: {metrics.get('satisfaction_score', 0.5):.2f}
- Risk Score: {metrics.get('risk_score', 0.5):.2f}

Generate a clear, empathetic, 2-3 sentence explanation of what this scenario means for the decision-maker. 

Guidelines:
- Be specific about what would happen in this scenario
- Use natural, conversational language
- Focus on practical implications
- For best case: emphasize opportunities and positive outcomes
- For worst case: acknowledge challenges but remain constructive
- For most likely: provide a balanced, realistic view

Return ONLY the explanation text, no labels or formatting."""

    try:
        response = model.generate_content(prompt)
        explanation = response.text.strip()
        
        # Clean up response
        if explanation.startswith('"') and explanation.endswith('"'):
            explanation = explanation[1:-1]
        
        return explanation
    
    except Exception as e:
        # Fallback explanation
        overall = metrics.get('overall_score', 0.5)
        financial = metrics.get('financial_score', 0.5)
        satisfaction = metrics.get('satisfaction_score', 0.5)
        
        if scenario_type == "best_case":
            return f"In the best-case scenario, you experience strong outcomes with a financial score of {financial:.1%} and satisfaction of {satisfaction:.1%}. This represents the most favorable combination of factors."
        elif scenario_type == "worst_case":
            return f"In the worst-case scenario, outcomes are challenging with a financial score of {financial:.1%} and satisfaction of {satisfaction:.1%}. This highlights important risks to consider."
        else:
            return f"In the most likely scenario, you can expect moderate outcomes with a financial score of {financial:.1%} and satisfaction of {satisfaction:.1%}. This represents the most probable result based on the analysis."


def generate_risk_explanation(risk: RiskFlag, decision_text: str) -> str:
    """
    Generate enhanced explanation for a risk flag.
    
    Args:
        risk: RiskFlag object
        decision_text: Original decision text
    
    Returns:
        Enhanced risk description
    """
    # For now, return the risk description as-is
    # Could be enhanced with LLM if needed
    return risk.description
