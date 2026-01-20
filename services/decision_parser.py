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
    
    prompt = f"""You are a decision analysis expert. Extract structured variables from the following decision.

Decision Type: {decision_type}
Decision Text: {decision_text}

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
