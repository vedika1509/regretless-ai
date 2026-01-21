"""Service to parse natural language decisions into structured variables using LLM."""
import os
import json
import google.generativeai as genai
from typing import Dict, Any
from dotenv import load_dotenv

from models.decision import StructuredDecision, VariableDistribution

load_dotenv()


def initialize_gemini() -> genai.GenerativeModel:
    """Initialize Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')


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
    mental_health_keywords = ["mentally", "depressed", "anxious", "stress", "stressed", 
                             "burnout", "exhausted", "overwhelmed", "drained", "unhappy"]
    negative_keywords = ["bad", "terrible", "awful", "hate", "horrible", "miserable"]
    
    return {
        "has_toxicity": any(keyword in text_lower for keyword in toxicity_keywords),
        "has_mental_health": any(keyword in text_lower for keyword in mental_health_keywords),
        "has_negative_env": any(keyword in text_lower for keyword in negative_keywords)
    }


def parse_decision(decision_text: str, decision_type: str = "custom") -> StructuredDecision:
    """
    Parse a natural language decision into structured variables.
    
    Args:
        decision_text: Natural language description of the decision
        decision_type: Type of decision (job, rent, hiring, custom)
    
    Returns:
        StructuredDecision object with extracted variables
    """
    model = initialize_gemini()
    
    # Detect emotional factors
    emotional_factors = detect_emotional_factors(decision_text)
    
    # Build enhanced prompt with emotional context
    emotional_emphasis = ""
    if emotional_factors["has_toxicity"] or emotional_factors["has_mental_health"] or emotional_factors["has_negative_env"]:
        emotional_emphasis = """
IMPORTANT: The decision text contains emotional/psychological concerns:
"""
        if emotional_factors["has_toxicity"]:
            emotional_emphasis += "- Toxicity indicators detected (toxic, hostile, abusive environment)\n"
        if emotional_factors["has_mental_health"]:
            emotional_emphasis += "- Mental health concerns detected (stress, burnout, mental well-being)\n"
        if emotional_factors["has_negative_env"]:
            emotional_emphasis += "- Negative environment indicators detected\n"
        
        emotional_emphasis += """
You MUST create variables that capture these emotional/psychological factors:
1. toxicity_level (if toxicity mentioned) - Should have high impact on satisfaction and risk scores
   - Base value: -0.6 to -0.8 (negative impact on satisfaction)
   - Should significantly increase risk_score
   
2. mental_health_impact (if mental health mentioned) - Should heavily weight satisfaction calculations
   - Base value: -0.5 to -0.7 (negative impact on satisfaction)
   - Should increase risk_score substantially

These factors should be weighted MORE HEAVILY than financial factors in determining satisfaction and risk outcomes.
"""
    
    prompt = f"""You are a decision analysis expert. Extract structured variables from the following decision.

Decision Type: {decision_type}
Decision Text: {decision_text}

{emotional_emphasis}

Your task is to identify key variables that affect the outcome of this decision and assign them probability distributions.

For each variable, provide:
1. A base value (numeric, typically between -1 and 1 or 0 and 1)
2. A probability distribution type (normal, beta, uniform, or bernoulli)
3. Distribution parameters (mean, std for normal; alpha, beta for beta; min, max for uniform; probability for bernoulli)

Focus on variables like:
- Financial changes (salary, cost, savings)
- Time/effort changes
- Risk factors (stability, uncertainty)
- Quality of life factors (work-life balance, stress, satisfaction)
- Emotional/psychological factors (toxicity, mental health, well-being) - if mentioned, these are CRITICAL

Return a JSON object with this structure:
{{
    "decision_type": "{decision_type}",
    "variables": {{
        "variable_name": {{
            "value": <base_value>,
            "distribution": "<distribution_type>",
            "params": {{<distribution_params>}}
        }}
    }}
}}

Example for a job switch decision:
{{
    "decision_type": "job_switch",
    "variables": {{
        "salary_change": {{
            "value": 0.3,
            "distribution": "normal",
            "params": {{"mean": 0.3, "std": 0.1}}
        }},
        "company_stability": {{
            "value": 0.5,
            "distribution": "beta",
            "params": {{"alpha": 2, "beta": 2}}
        }},
        "work_life_balance": {{
            "value": 0.0,
            "distribution": "uniform",
            "params": {{"min": -0.2, "max": 0.2}}
        }}
    }}
}}

Return ONLY the JSON object, no other text."""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        parsed_data = json.loads(response_text)
        
        # Convert to StructuredDecision
        variables = {}
        for var_name, var_data in parsed_data.get("variables", {}).items():
            variables[var_name] = VariableDistribution(
                value=var_data["value"],
                distribution=var_data["distribution"],
                params=var_data["params"]
            )
        
        # If emotional factors detected but not in parsed data, add them
        if emotional_factors["has_toxicity"] and "toxicity_level" not in variables:
            variables["toxicity_level"] = VariableDistribution(
                value=-0.7,  # High negative impact
                # Use a signed distribution to reflect negative impact explicitly.
                # Uniform here avoids the common beta-in-[0,1] pitfall for "negative" concepts.
                distribution="uniform",
                params={"min": -1.0, "max": -0.4},
            )
        
        if emotional_factors["has_mental_health"] and "mental_health_impact" not in variables:
            variables["mental_health_impact"] = VariableDistribution(
                value=-0.6,  # Significant negative impact
                distribution="uniform",
                params={"min": -1.0, "max": -0.3},
            )
        
        return StructuredDecision(
            decision_type=parsed_data.get("decision_type", decision_type),
            variables=variables,
            raw_text=decision_text
        )
    
    except Exception as e:
        # Fallback to a simple default structure if parsing fails
        return StructuredDecision(
            decision_type=decision_type,
            variables={
                "outcome_uncertainty": VariableDistribution(
                    value=0.5,
                    distribution="beta",
                    params={"alpha": 2, "beta": 2}
                )
            },
            raw_text=decision_text
        )
