"""Decision parsing (LLM-assisted but simulation-owned math).

Key rule enforced here:
- The LLM may propose *qualitative* variables + context,
  but it must NOT output probability distributions, probabilities, or fabricated numbers.
- All distributions/params are assigned deterministically in `services.structurer`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

from models.decision import StructuredDecision
from services.llm.groq_client import safe_chat
from services.structurer import build_variables

load_dotenv()


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|```\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)%?")


def _extract_numbers(text: str) -> List[str]:
    return [m.group(0) for m in _NUM_RE.finditer(text or "")]


def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    s = _JSON_FENCE_RE.sub("", s).strip()
    return s

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
    parser_meta: Dict[str, Any] = {"used_llm": True, "fallback": False, "warnings": []}
    # Detect emotional factors (used to seed deterministic candidates if LLM is unavailable)
    emotional_factors = detect_emotional_factors(decision_text)
    
    # Hard rule: LLM must not output probabilities/distributions/numbers.
    # It may only propose qualitative variables and what to ask next.
    system = (
        "You are Regretless AI.\n"
        "You do NOT predict the future.\n"
        "You do NOT generate probabilities, distributions, or numeric parameters.\n"
        "You ONLY extract qualitative factors and missing context.\n"
        "Never invent numbers. If you reference numbers, they MUST be exact copies from the user's text.\n"
    )

    user = f"""
Decision type (user-selected): {decision_type}
Decision text:
{decision_text}

Extract a JSON object with this schema (and nothing else):
{{
  "decision_type_hint": "job|rent|hiring|custom",
  "context_hints": {{
    "runway_months": "unknown|0-1|1-3|3-6|6+",
    "plan_clarity": "unknown|no_plan|somewhat|clear",
    "driver": "unknown|stress|mixed|strategy"
  }},
  "variable_candidates": [
    {{
      "name": "snake_case_variable_name",
      "kind": "level|change|binary",
      "polarity": "positive|negative",
      "importance": "low|medium|high",
      "uncertainty": "low|medium|high",
      "evidence": "short quote or paraphrase from the user's text"
    }}
  ],
  "missing_questions": ["1-5 short clarifying questions"]
}}

Rules:
- DO NOT include probability distributions or any parameters.
- DO NOT include any numbers unless they appear verbatim in the decision text.
- Keep variable_candidates to 3–8 items max.
""".strip()

    try:
        resp = safe_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=900,
        )
        response_text = _strip_fences(resp.text)
        parsed = json.loads(response_text)

        # Validate number rule: any numbers in LLM output must appear in user text.
        in_nums = set(_extract_numbers(decision_text))
        out_nums = set(_extract_numbers(response_text))
        extra_nums = sorted([n for n in out_nums if n not in in_nums])
        if extra_nums:
            parser_meta["warnings"].append(
                "LLM output contained numbers not present in your text; ignoring numeric details to avoid hallucinations."
            )

        # Convert context_hints into simple deterministic context.
        context_hints = parsed.get("context_hints") or {}
        runway_map = {"0-1": 1, "1-3": 2, "3-6": 5, "6+": 6}
        runway_raw = (context_hints.get("runway_months") or "unknown").strip().lower()
        runway_months = runway_map.get(runway_raw)

        plan_raw = (context_hints.get("plan_clarity") or "unknown").strip().lower()
        plan = {"no_plan": "No plan", "somewhat": "Somewhat", "clear": "Clear"}.get(plan_raw)

        driver_raw = (context_hints.get("driver") or "unknown").strip().lower()
        driver = {"stress": "Stress", "mixed": "Mixed", "strategy": "Strategy"}.get(driver_raw)

        context = {"runway_months": runway_months, "plan_clarity": plan, "driver": driver}

        candidates = parsed.get("variable_candidates") or []

        # Seed deterministic candidates for emotional factors if LLM missed them.
        if emotional_factors.get("has_toxicity") and not any(
            "toxic" in str(c.get("name", "")).lower() or "toxicity" in str(c.get("name", "")).lower()
            for c in candidates
        ):
            candidates.append(
                {
                    "name": "toxicity_level",
                    "kind": "level",
                    "polarity": "negative",
                    "importance": "high",
                    "uncertainty": "medium",
                    "evidence": "toxicity mentioned",
                }
            )
        if emotional_factors.get("has_mental_health") and not any(
            "mental" in str(c.get("name", "")).lower() or "burnout" in str(c.get("name", "")).lower()
            for c in candidates
        ):
            candidates.append(
                {
                    "name": "stress_level",
                    "kind": "level",
                    "polarity": "negative",
                    "importance": "high",
                    "uncertainty": "medium",
                    "evidence": "stress/burnout mentioned",
                }
            )

        variables = build_variables(
            decision_text=decision_text, decision_type=decision_type, candidates=candidates, context=context
        )

        # Heuristic warning if too few variables.
        if len(variables) < 3:
            parser_meta["warnings"].append(
                "Very few factors were identified; results may be generic. Add more context for better realism."
            )

        parser_meta["llm_model"] = resp.model
        parser_meta["missing_questions"] = (parsed.get("missing_questions") or [])[:5]
        parser_meta["context_hints"] = context_hints

        return StructuredDecision(
            decision_type=parsed.get("decision_type_hint", decision_type) or decision_type,
            variables=variables,
            raw_text=decision_text,
            meta=parser_meta,
        )

    except Exception as e:
        # Fallback: deterministic minimal structure (no LLM).
        parser_meta["fallback"] = True
        parser_meta["used_llm"] = False
        parser_meta["warnings"].append(f"Parser fallback: {type(e).__name__}: {str(e)}")

        # Basic deterministic candidates using keyword heuristics.
        candidates: List[Dict[str, Any]] = []
        t = (decision_text or "").lower()
        if any(k in t for k in ("salary", "income", "pay", "money")):
            candidates.append(
                {"name": "salary_change", "kind": "change", "polarity": "positive", "importance": "high", "uncertainty": "high"}
            )
        if any(k in t for k in ("stress", "burnout", "overwhelmed")):
            candidates.append(
                {"name": "stress_level", "kind": "level", "polarity": "negative", "importance": "high", "uncertainty": "medium"}
            )
        if any(k in t for k in ("stability", "secure", "uncertain", "risk")):
            candidates.append(
                {"name": "company_stability", "kind": "level", "polarity": "positive", "importance": "medium", "uncertainty": "high"}
            )

        if emotional_factors.get("has_toxicity"):
            candidates.append(
                {"name": "toxicity_level", "kind": "level", "polarity": "negative", "importance": "high", "uncertainty": "medium"}
            )

        variables = build_variables(
            decision_text=decision_text, decision_type=decision_type, candidates=candidates, context={}
        )

        return StructuredDecision(
            decision_type=decision_type,
            variables=variables,
            raw_text=decision_text,
            meta=parser_meta,
        )
