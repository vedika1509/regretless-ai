"""Lightweight smoke tests (no network calls).

Run:
  python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Ensure imports work when running as a script.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from services.structurer import build_variables
    from services.llm.guardrails import redact_invented_numbers
    from services.safety import assess_safety
    from services.interpretation_service import InterpretationResult

    # Structurer should work deterministically.
    vars_ = build_variables(
        decision_text="Should I quit my job? I have 30% raise offer but I'm stressed.",
        decision_type="job",
        candidates=[
            {"name": "salary_change", "kind": "change", "polarity": "positive", "importance": "high", "uncertainty": "medium"},
            {"name": "stress_level", "kind": "level", "polarity": "negative", "importance": "high", "uncertainty": "medium"},
        ],
        context={"runway_months": 6, "plan_clarity": "Somewhat", "driver": "Mixed"},
    )
    assert "salary_change" in vars_
    assert "stress_level" in vars_

    # Guardrail should redact unknown numbers.
    allowed = {"0.50", "0.30"}
    text = "Your risk is 0.50 and the timeline is 12 months."
    out, extras = redact_invented_numbers(text, allowed_numbers=allowed)
    assert extras
    assert "12" not in out

    # Safety should detect high-risk domains.
    s = assess_safety("Can I drop out of college?")
    assert s.is_high_risk_domain

    # Interpretation schema validation works.
    InterpretationResult.model_validate(
        {
            "readiness": "PROCEED CAREFULLY",
            "readiness_rationale": "Because downside risk exists.",
            "readiness_conditions": None,
            "likely_outcome_label": "MIXED / UNCERTAIN",
            "likely_outcome_explanation": "Outcomes are balanced with some risk.",
            "tradeoffs": ["Stability vs wellbeing"],
            "biggest_hidden_risk": {"title": "Directionless exit", "why_it_matters": "It increases regret.", "mitigation": ["Define Plan A/Plan B"]},
            "regret_paths": ["Quit regret comes from instability."],
            "recommendations": ["Create a bridge plan."],
            "followup_questions": ["What would success look like?"],
            "scenario_stories": [
                {"scenario_type": "most_likely", "title": "Uncertain transition", "timeline": "12–18 months", "probability_pct": 50, "emotional_outcome": "Mixed", "bullets": ["Some uncertainty"]}
            ],
        }
    )

    print("smoke_test: ok")


if __name__ == "__main__":
    main()

