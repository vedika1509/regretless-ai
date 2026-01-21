"""Ethical/safety safeguards for decision guidance.

This module is deterministic and does not call any LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SafetyAssessment:
    is_high_risk_domain: bool
    is_self_harm_risk: bool
    domains: List[str]
    guidance_prefix: str


def assess_safety(decision_text: str) -> SafetyAssessment:
    t = (decision_text or "").lower()

    # Self-harm / crisis indicators (keep conservative)
    self_harm_keywords = [
        "suicide",
        "kill myself",
        "end my life",
        "self harm",
        "self-harm",
        "hurt myself",
    ]
    is_self_harm = any(k in t for k in self_harm_keywords)

    domains: List[str] = []
    if any(k in t for k in ["study", "college", "degree", "school", "drop out", "dropout"]):
        domains.append("education")
    if any(k in t for k in ["loan", "debt", "rent", "mortgage", "bankrupt", "savings", "income"]):
        domains.append("finances")
    if any(k in t for k in ["health", "diagnosis", "medication", "therapy", "hospital"]):
        domains.append("health")
    if any(k in t for k in ["visa", "immigration", "legal", "lawsuit", "court"]):
        domains.append("legal")
    if any(k in t for k in ["quit my job", "resign", "fired", "offer", "layoff"]):
        domains.append("career")

    is_high_risk_domain = bool(domains)

    if is_self_harm:
        prefix = (
            "I’m really sorry you’re feeling this way. I can’t help with anything that encourages self-harm.\n"
            "If you’re in immediate danger, contact your local emergency number now.\n"
            "If you can, reach out to someone you trust or a local crisis hotline. "
            "If you tell me your country, I can help you find the right support resources.\n\n"
            "If you still want, we can focus on safer, immediate steps to reduce harm and get support."
        )
    elif is_high_risk_domain:
        prefix = (
            "This is a high-stakes decision. I won’t give absolute directives — I’ll help you reduce risk and regret.\n"
            "We’ll focus on preparation, optionality, and checkpoints rather than impulsive moves."
        )
    else:
        prefix = ""

    return SafetyAssessment(
        is_high_risk_domain=is_high_risk_domain,
        is_self_harm_risk=is_self_harm,
        domains=domains,
        guidance_prefix=prefix,
    )

