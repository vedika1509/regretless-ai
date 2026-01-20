"""Generate actionable recommendations."""
import os
import google.generativeai as genai
from typing import Dict, List
from dotenv import load_dotenv

from models.scenario import AnalysisResult

load_dotenv()


def initialize_gemini() -> genai.GenerativeModel:
    """Initialize Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')


def generate_recommendations(result: AnalysisResult, decision_text: str) -> Dict:
    """Generate actionable recommendations based on analysis."""
    model = initialize_gemini()
    
    # Format scenario summaries
    scenario_summaries = []
    for scenario_type, scenario in result.scenarios.items():
        scenario_name = scenario_type.replace('_', ' ').title()
        scenario_summaries.append(
            f"{scenario_name}: {scenario.explanation}"
        )
    
    # Format risk summaries
    risk_summaries = []
    for risk in result.risks:
        risk_summaries.append(f"{risk.risk_type.replace('_', ' ').title()} ({risk.severity}): {risk.description}")
    
    prompt = f"""Based on this decision analysis, provide 3-5 specific, actionable recommendations.

Decision: {decision_text}

Analysis Summary:
- Confidence: {result.confidence:.1%}
- Scenarios analyzed: {len(result.scenarios)}
- Risks detected: {len(result.risks)}

Scenarios:
{chr(10).join(scenario_summaries)}

Risks:
{chr(10).join(risk_summaries) if risk_summaries else 'No significant risks detected.'}

Provide recommendations in this format:
1. [Specific action] - [Reason why]
2. [Specific action] - [Reason why]
3. [Specific action] - [Reason why]

Focus on practical, actionable steps the decision-maker can take. Be specific about what they should do and why.
Return ONLY the numbered list, no other text."""

    try:
        response = model.generate_content(prompt)
        recommendations_text = response.text.strip()
        
        # Clean up response
        if recommendations_text.startswith('"') and recommendations_text.endswith('"'):
            recommendations_text = recommendations_text[1:-1]
        
        # Split into individual recommendations
        recommendations_list = [
            rec.strip() for rec in recommendations_text.split('\n') 
            if rec.strip() and (rec.strip()[0].isdigit() or rec.strip().startswith('-'))
        ]
        
        # Calculate priority (based on risk level and confidence)
        if result.risks:
            high_risk_count = sum(1 for r in result.risks if r.severity == "high")
            priority = "high" if high_risk_count > 0 else "medium"
        else:
            priority = "low"
        
        return {
            "recommendations": recommendations_list,
            "timeline": "short_term",  # Could be enhanced
            "priority": priority
        }
    
    except Exception as e:
        # Fallback recommendations
        recommendations_list = [
            "Review the analysis results carefully before making your decision",
            "Consider the identified risks and plan mitigation strategies",
            "Discuss the decision with trusted advisors or stakeholders"
        ]
        
        return {
            "recommendations": recommendations_list,
            "timeline": "short_term",
            "priority": "medium"
        }
