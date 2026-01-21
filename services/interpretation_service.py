"""Post-simulation LLM interpretation layer.

Principle:
- Simulation produces all numeric outputs.
- LLM converts them into meaning, trade-offs, stories, regret framing, and next steps.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from models.scenario import AnalysisResult
from services.llm.groq_client import safe_chat


_FENCE_RE = re.compile(r"^```(?:json)?\s*|```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(s: str) -> str:
    return _FENCE_RE.sub("", (s or "").strip()).strip()


class ScenarioStory(BaseModel):
    scenario_type: str
    title: str
    timeline: str
    probability_pct: int = Field(..., ge=0, le=100)
    emotional_outcome: str
    bullets: List[str] = Field(default_factory=list, max_length=6)


class HiddenRisk(BaseModel):
    title: str
    why_it_matters: str
    mitigation: List[str] = Field(default_factory=list, max_length=6)


class InterpretationResult(BaseModel):
    readiness: str
    readiness_rationale: str
    readiness_conditions: Optional[str] = None

    likely_outcome_label: str
    likely_outcome_explanation: str

    tradeoffs: List[str] = Field(default_factory=list, max_length=6)
    biggest_hidden_risk: HiddenRisk
    regret_paths: List[str] = Field(default_factory=list, max_length=6)
    recommendations: List[str] = Field(default_factory=list, max_length=8)
    followup_questions: List[str] = Field(default_factory=list, max_length=5)

    scenario_stories: List[ScenarioStory] = Field(default_factory=list, max_length=3)


def build_simulation_payload(
    result: AnalysisResult,
    *,
    decision_text: str,
    context: Optional[Dict[str, Any]] = None,
    counterfactuals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    context = context or {}
    scenario_payload = {}
    for k, s in (result.scenarios or {}).items():
        scenario_payload[k] = {
            "probability": float(s.probability),
            "outcomes": {
                "overall_score": float(s.outcomes.overall_score),
                "financial_score": float(s.outcomes.financial_score),
                "satisfaction_score": float(s.outcomes.satisfaction_score),
                "risk_score": float(s.outcomes.risk_score),
            },
            "drivers": [
                {"variable": d.variable, "delta": float(d.delta)} for d in (getattr(s, "drivers", None) or [])
            ][:3],
        }

    return {
        "decision_text": decision_text,
        "context": context,
        "simulation": {
            "simulation_count": int(result.simulation_count),
            "confidence": float(result.confidence),
            "regret_score": float(result.regret_score),
            "risks": [
                {"risk_type": r.risk_type, "severity": r.severity, "description": r.description}
                for r in (result.risks or [])
            ],
            "scenarios": scenario_payload,
            "counterfactuals": (counterfactuals or [])[:5],
        },
    }


def generate_interpretation(
    result: AnalysisResult,
    *,
    decision_text: str,
    context: Optional[Dict[str, Any]] = None,
    counterfactuals: Optional[List[Dict[str, Any]]] = None,
) -> Optional[InterpretationResult]:
    """
    Return a structured interpretation. If LLM fails, return None (UI should fall back).
    """
    payload = build_simulation_payload(
        result, decision_text=decision_text, context=context, counterfactuals=counterfactuals
    )

    system = (
        "You are Regretless AI — a decision intelligence assistant.\n"
        "Role: interpret Monte Carlo simulation results into human meaning.\n\n"
        "Strict rules:\n"
        "- Never invent numbers or probabilities.\n"
        "- Never output any numeric value unless it is present in the provided JSON.\n"
        "- Do not override the simulation.\n"
        "- Explain uncertainty and trade-offs.\n"
        "- Challenge high-risk or irreversible decisions with preparation-first framing.\n"
        "Return ONLY JSON.\n"
    )

    user = (
        "You are given simulation outputs as JSON below.\n"
        "Produce an interpretation JSON that matches this schema (no extra keys):\n"
        "{\n"
        '  "readiness": "NOT READY YET|PROCEED CAREFULLY|SAFE TO PROCEED (WITH CONDITIONS)",\n'
        '  "readiness_rationale": "string",\n'
        '  "readiness_conditions": "string|null",\n'
        '  "likely_outcome_label": "STRONGLY POSITIVE|MODERATELY POSITIVE|MIXED / UNCERTAIN|MODERATELY NEGATIVE|STRONGLY NEGATIVE",\n'
        '  "likely_outcome_explanation": "string",\n'
        '  "tradeoffs": ["string", "..."],\n'
        '  "biggest_hidden_risk": {"title":"string","why_it_matters":"string","mitigation":["string","..."]},\n'
        '  "regret_paths": ["string","..."],\n'
        '  "recommendations": ["string","..."],\n'
        '  "followup_questions": ["string","..."],\n'
        '  "scenario_stories": [\n'
        '    {"scenario_type":"best_case|most_likely|worst_case","title":"string","timeline":"string","probability_pct":25,"emotional_outcome":"string","bullets":["string","..."]}\n'
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- Make stories concrete and human. Use the provided scenario probabilities.\n"
        "- Use counterfactuals (if present) to explain regret paths.\n"
        "- Recommendations must reduce risk or regret and be actionable.\n"
        "- If confidence is low or context missing, include 2–4 followup_questions.\n\n"
        "Simulation JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    try:
        resp = safe_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=1200,
        )
        raw = _strip_fences(resp.text)
        data = json.loads(raw)
        return InterpretationResult.model_validate(data)
    except (ValidationError, json.JSONDecodeError):
        return None
    except Exception:
        return None

