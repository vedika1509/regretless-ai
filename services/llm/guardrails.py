"""LLM output guardrails.

These guardrails are intentionally lightweight and explainable.
They aim to prevent the LLM from fabricating numbers or over-claiming certainty.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set, Tuple


_NUM_TOKEN_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)%?")


def extract_number_tokens(text: str) -> List[str]:
    """Extract number-ish tokens like 0.5, 12, 30% as they appear."""
    return [m.group(0) for m in _NUM_TOKEN_RE.finditer(text or "")]


def redact_invented_numbers(text: str, allowed_numbers: Iterable[str]) -> Tuple[str, List[str]]:
    """
    If the LLM output includes numbers not present in allowed_numbers, redact them.

    Strategy: replace unknown numbers with qualitative placeholders.
    """
    allowed: Set[str] = set(str(x) for x in allowed_numbers if str(x).strip())
    found = extract_number_tokens(text)
    extras = sorted([x for x in set(found) if x not in allowed])

    if not extras:
        return text, []

    def _replace(m: re.Match) -> str:
        token = m.group(0)
        if token in allowed:
            return token
        # Use a qualitative term instead of making up a new number.
        return "a while" if "month" in (text or "").lower() else "some"

    redacted = _NUM_TOKEN_RE.sub(_replace, text or "")
    redacted = redacted.strip()

    # Add a short disclaimer if we had to redact.
    redacted += "\n\n(Notes: I avoided adding any new numbers that weren’t in your simulation output.)"
    return redacted, extras

