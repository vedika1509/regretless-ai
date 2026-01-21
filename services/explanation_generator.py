"""Service to generate human-readable explanations using LLM."""

from typing import Dict

from dotenv import load_dotenv
from models.scenario import RiskFlag

from services.llm.groq_client import safe_chat
from services.llm.guardrails import redact_invented_numbers

load_dotenv()


def detect_emotional_factors(decision_text: str) -> Dict[str, bool]:
    """
    Detect emotional/psychological keywords in decision text.

    Args:
        decision_text: Natural language description of the decision

    Returns:
        Dictionary indicating presence of different emotional factors
    """
    text_lower = decision_text.lower()

    toxicity_keywords = ["toxic", "hostile", "unhealthy", "abusive", "bullying", "harassment"]
    mental_health_keywords = [
        "mentally",
        "depressed",
        "anxious",
        "stress",
        "stressed",
        "burnout",
        "exhausted",
        "overwhelmed",
        "drained",
        "unhappy",
    ]
    negative_keywords = ["bad", "terrible", "awful", "hate", "horrible", "miserable"]

    return {
        "has_toxicity": any(keyword in text_lower for keyword in toxicity_keywords),
        "has_mental_health": any(keyword in text_lower for keyword in mental_health_keywords),
        "has_negative_env": any(keyword in text_lower for keyword in negative_keywords),
    }


def generate_scenario_explanation(
    scenario_type: str, metrics: Dict[str, float], decision_text: str
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
    # Detect emotional factors
    emotional_factors = detect_emotional_factors(decision_text)

    scenario_names = {
        "best_case": "Best Case",
        "worst_case": "Worst Case",
        "most_likely": "Most Likely",
    }

    scenario_name = scenario_names.get(scenario_type, scenario_type)

    # Build emotional context emphasis
    emotional_emphasis = ""
    if emotional_factors["has_toxicity"] or emotional_factors["has_mental_health"]:
        emotional_emphasis = (
            "\nIMPORTANT CONTEXT: The decision text mentions emotional/psychological concerns:\n"
        )
        if emotional_factors["has_toxicity"]:
            emotional_emphasis += "- Toxicity/toxic environment mentioned - this significantly impacts well-being and satisfaction\n"
        if emotional_factors["has_mental_health"]:
            emotional_emphasis += (
                "- Mental health concerns mentioned (stress, burnout, mental well-being)\n"
            )

        emotional_emphasis += """
You MUST emphasize these factors in your explanation:
- For best case: If leaving a toxic environment, emphasize improved mental health, reduced stress, and better well-being
- For worst case: If staying in toxic environment, explicitly mention continued mental health impact, potential burnout, and long-term consequences
- For most likely: Discuss how emotional/psychological factors affect outcomes more than financial metrics
- Reference the user's specific concerns (toxicity, mental health) directly
- Be empathetic and acknowledge that these factors are critical to long-term satisfaction
"""

    # Allowed numeric tokens (the LLM must not introduce any new numbers).
    allowed_numbers = [
        f"{metrics.get('overall_score', 0.5):.2f}",
        f"{metrics.get('financial_score', 0.5):.2f}",
        f"{metrics.get('satisfaction_score', 0.5):.2f}",
        f"{metrics.get('risk_score', 0.5):.2f}",
    ]

    prompt = f"""You are Regretless AI.

You interpret simulation outputs. You do NOT predict the future.
Strict rule: Never invent numbers, dates, or probabilities. Do not add any numbers beyond the scores provided.

Original Decision: {decision_text}

{emotional_emphasis}

Scenario Type: {scenario_name}

Outcome Metrics (0 = worst, 1 = best):
- Overall Score: {metrics.get("overall_score", 0.5):.2f}
- Financial Score: {metrics.get("financial_score", 0.5):.2f}
- Satisfaction Score: {metrics.get("satisfaction_score", 0.5):.2f}
- Risk Score: {metrics.get("risk_score", 0.5):.2f}

Write a clear, empathetic 2–3 sentence explanation of what this scenario means.

Guidelines:
- Be specific about consequences (money, stability, wellbeing) WITHOUT adding timelines or new numbers.
- For best case: emphasize opportunities and positive outcomes
- For worst case: acknowledge challenges but remain constructive
- For most likely: balanced and realistic
- If emotional/psychological factors are present, make them central

Return ONLY the explanation text."""

    try:
        response = safe_chat(
            [{"role": "system", "content": "Return only plain text."}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=220,
        )
        explanation = response.text.strip()

        # Clean up response
        if explanation.startswith('"') and explanation.endswith('"'):
            explanation = explanation[1:-1]

        cleaned, _extras = redact_invented_numbers(explanation, allowed_numbers=allowed_numbers)
        return cleaned.strip()

    except Exception:
        # Fallback explanation
        financial = metrics.get("financial_score", 0.5)
        satisfaction = metrics.get("satisfaction_score", 0.5)

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
