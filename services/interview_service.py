"""LLM-powered decision interviewing (clarifying questions only).

LLM role:
- Ask for missing context to make simulation inputs realistic.
- Never generate probabilities, distributions, or numeric outcomes.
"""

from __future__ import annotations

import json
import re
from typing import List

from services.llm.groq_client import safe_chat


_FENCE_RE = re.compile(r"^```(?:json)?\s*|```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(s: str) -> str:
    return _FENCE_RE.sub("", (s or "").strip()).strip()


def generate_clarifying_questions(
    decision_text: str, decision_type: str = "custom", *, max_questions: int = 5
) -> List[str]:
    """
    Generate 3–5 short clarifying questions.
    """
    system = (
        "You are Regretless AI, a decision intelligence assistant.\n"
        "Your job is to interview the user to collect missing context BEFORE simulation.\n"
        "Rules:\n"
        "- Never generate probabilities or numeric outcomes.\n"
        "- Do not invent numbers.\n"
        "- Ask short, practical questions.\n"
        "- Prefer questions about runway, alternatives, constraints, timeline, and whether the decision is stress-driven.\n"
        "Return ONLY JSON.\n"
    )

    user = f"""
Decision type: {decision_type}
Decision text:
{decision_text}

Return a JSON array of 3–{max_questions} questions (strings). No extra keys.
""".strip()

    resp = safe_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=300,
    )

    raw = _strip_fences(resp.text)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Invalid interviewer output (expected JSON list).")

    questions = []
    for q in data:
        if isinstance(q, str):
            q = q.strip()
            if q:
                questions.append(q)

    # Keep it tight
    return questions[:max_questions]

