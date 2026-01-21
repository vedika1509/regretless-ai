"""Generate actionable recommendations."""
import json
from typing import Dict
from dotenv import load_dotenv

from models.scenario import AnalysisResult
from services.llm.groq_client import safe_chat

load_dotenv()


def generate_recommendations(result: AnalysisResult, decision_text: str) -> Dict:
    """Generate actionable recommendations based on analysis."""
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

    prompt = f"""You are Regretless AI.
You interpret simulation outputs and propose risk-reducing next steps.

Rules:
- Never invent numbers, probabilities, or timelines with numeric durations.
- Avoid absolute directives; prefer preparation and checkpoints.
- Recommendations must be actionable and directly reduce risk or regret.

Return ONLY JSON with keys: recommendations (list), priority (high|medium|low), timeline (immediate|1-3_months|3-6_months).

Decision: {decision_text}

Analysis Summary:
- Confidence: {result.confidence:.1%}
- Scenarios analyzed: {len(result.scenarios)}
- Risks detected: {len(result.risks)}

Scenarios:
{chr(10).join(scenario_summaries)}

Risks:
{chr(10).join(risk_summaries) if risk_summaries else 'No significant risks detected.'}
""".strip()

    try:
        resp = safe_chat(
            [{"role": "system", "content": "Return only JSON."}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        data = json.loads(resp.text.strip().strip("`"))
        recommendations_list = [r.strip() for r in data.get("recommendations", []) if isinstance(r, str) and r.strip()]
        
        # Calculate priority (based on risk level and confidence)
        if result.risks:
            high_risk_count = sum(1 for r in result.risks if r.severity == "high")
            priority = "high" if high_risk_count > 0 else "medium"
        else:
            priority = "low"
        
        return {
            "recommendations": recommendations_list,
            "timeline": data.get("timeline", "short_term"),
            "priority": data.get("priority", priority),
        }
    
    except Exception:
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
