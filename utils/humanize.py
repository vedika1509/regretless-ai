"""Human-friendly interpretations for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.5


def to_10(x01: float) -> int:
    """Convert 0–1 score into a 0–10 (integer) scale."""
    x = clamp01(x01)
    return int(round(x * 10))


def band_label(score01: float) -> Tuple[str, str]:
    """
    Convert a 0–1 score into a qualitative outcome label + severity color.
    Intended for "Likely Outcome".
    """
    s = clamp01(score01)
    if s >= 0.75:
        return "STRONGLY POSITIVE", "green"
    if s >= 0.60:
        return "MODERATELY POSITIVE", "green"
    if s >= 0.45:
        return "MIXED / UNCERTAIN", "#D4A017"  # amber
    if s >= 0.30:
        return "MODERATELY NEGATIVE", "orange"
    return "STRONGLY NEGATIVE", "red"


def regret_band(regret01: float) -> str:
    r = clamp01(regret01)
    if r < 0.20:
        return "Low"
    if r < 0.35:
        return "Low–Moderate"
    if r < 0.55:
        return "Moderate"
    if r < 0.70:
        return "Moderate–High"
    return "High"


def emotional_outcome_label(mental_wellbeing01: float) -> str:
    m = clamp01(mental_wellbeing01)
    if m >= 0.75:
        return "High satisfaction"
    if m >= 0.55:
        return "Mostly okay"
    if m >= 0.40:
        return "Mixed"
    return "High stress"


def timeline_for_scenario(
    scenario_type: str, horizon: Optional[str] = None
) -> str:
    """
    Produce a human timeline string. If horizon is provided, we anchor around it.
    """
    # Horizon is a human hint like "6–12 months", "12–18 months", etc.
    if horizon:
        if scenario_type == "best_case":
            return horizon.replace("12–18", "6–12").replace("18–36", "12–18")
        if scenario_type == "worst_case":
            return horizon.replace("6–12", "12–18").replace("12–18", "18–36")
        return horizon

    # Reasonable defaults
    if scenario_type == "best_case":
        return "6–12 months"
    if scenario_type == "worst_case":
        return "18–36 months"
    return "12–18 months"


@dataclass
class ReadinessVerdict:
    label: str
    color: str
    rationale: str
    conditions: Optional[str] = None


def compute_readiness(
    regret01: float,
    confidence01: float,
    risks: list,
    context: Optional[Dict[str, Any]] = None,
) -> ReadinessVerdict:
    """
    Judge-friendly readiness verdict.
    Rules are intentionally simple + explainable.
    """
    context = context or {}
    runway = context.get("runway_months")  # int|None
    plan = context.get("plan_clarity")  # str|None
    driver = context.get("driver")  # str|None

    high_risk = any(getattr(r, "severity", "").lower() == "high" for r in (risks or []))
    r = clamp01(regret01)
    c = clamp01(confidence01)

    not_ready_reasons = []
    if high_risk or r >= 0.55:
        not_ready_reasons.append("high downside / regret risk")
    if c < 0.45:
        not_ready_reasons.append("low confidence (inputs too uncertain)")
    if plan in (None, "", "No plan"):
        not_ready_reasons.append("no clear Plan A / Plan B")
    if runway is not None and runway < 3:
        not_ready_reasons.append("thin financial runway")
    if driver in ("Stress", "Mostly stress"):
        not_ready_reasons.append("decision seems stress-driven")

    if not_ready_reasons:
        return ReadinessVerdict(
            label="NOT READY YET",
            color="#D4A017",
            rationale="This is a high-stakes decision. Right now, key inputs aren’t stabilized: "
            + ", ".join(not_ready_reasons[:3])
            + ".",
            conditions="Prepare first: define a Plan A/Plan B, secure runway (or a bridge), and reduce uncertainty before committing.",
        )

    if r < 0.35 and c >= 0.55:
        return ReadinessVerdict(
            label="SAFE TO PROCEED (WITH CONDITIONS)",
            color="green",
            rationale="Downside risk looks contained and the outcomes are reasonably consistent.",
            conditions="Proceed only if you keep a fallback plan and set a checkpoint (e.g., reassess in 8–12 weeks).",
        )

    return ReadinessVerdict(
        label="PROCEED CAREFULLY",
        color="orange",
        rationale="There’s meaningful upside, but you still have non-trivial downside risk.",
        conditions="Reduce regret risk by adding structure: runway, timeline, and clear next steps.",
    )


def scenario_story_title(scenario_type: str) -> str:
    if scenario_type == "best_case":
        return "Strategic Move"
    if scenario_type == "worst_case":
        return "Regret Loop"
    return "Uncertain Transition"


def scenario_bullets(metrics: Dict[str, Any]) -> list[str]:
    """
    Generate 3 human bullets from metrics (0–1).
    """
    job_sec = clamp01(metrics.get("job_security", 0.5))
    fin = clamp01(metrics.get("financial_satisfaction", metrics.get("financial_score", 0.5)))
    stress = clamp01(metrics.get("stress", 0.5))

    bullets = []

    # Career / stability
    if job_sec >= 0.7:
        bullets.append("Stability improves; your path feels more secure.")
    elif job_sec <= 0.4:
        bullets.append("Stability drops; you may face uncertainty or gaps.")
    else:
        bullets.append("Stability is mixed; expect some uncertainty and adjustment.")

    # Money
    if fin >= 0.7:
        bullets.append("Financially safe: income/savings likely cover the transition.")
    elif fin <= 0.4:
        bullets.append("Financial pressure increases; runway may feel tight.")
    else:
        bullets.append("Finances are workable but require budgeting and a plan.")

    # Emotion
    if stress >= 0.7:
        bullets.append("Stress stays high; self-doubt / pressure may build.")
    elif stress <= 0.4:
        bullets.append("Relief is likely; mental load reduces noticeably.")
    else:
        bullets.append("Emotions are mixed: initial relief + periodic anxiety.")

    return bullets[:3]

