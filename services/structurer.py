"""Deterministic decision structurer.

This module intentionally contains **no LLM calls**.
It converts qualitative variable candidates + optional explicit facts into
`VariableDistribution`s for the Monte Carlo simulator.

Key principle: the LLM must not invent probabilities or distributions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models.decision import VariableDistribution


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def extract_explicit_percents(text: str) -> List[float]:
    """Extract explicit % values from text as 0–1 scaled."""
    percents = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", text):
        try:
            percents.append(float(m.group(1)) / 100.0)
        except Exception:
            continue
    return percents


def _importance_scale(importance: str) -> float:
    imp = (importance or "").strip().lower()
    if imp == "high":
        return 0.20
    if imp == "medium":
        return 0.12
    if imp == "low":
        return 0.06
    return 0.10


def _uncertainty_std(uncertainty: str) -> float:
    u = (uncertainty or "").strip().lower()
    if u == "low":
        return 0.06
    if u == "medium":
        return 0.12
    if u == "high":
        return 0.20
    return 0.12


def _polarity_sign(polarity: str) -> float:
    p = (polarity or "").strip().lower()
    return -1.0 if p in ("negative", "downside", "risk") else 1.0


def _as_snake(name: str) -> str:
    n = re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").strip())
    n = re.sub(r"_+", "_", n).strip("_").lower()
    return n or "unknown_factor"


def candidate_to_distribution(
    candidate: Dict[str, Any],
    *,
    defaults: Optional[Dict[str, Any]] = None,
) -> Tuple[str, VariableDistribution]:
    """
    Convert a qualitative candidate into a VariableDistribution.

    Supported candidate fields (all qualitative; no probabilities):
    - name: str
    - kind: "level" | "change" | "binary"
    - polarity: "positive" | "negative"
    - importance: "low" | "medium" | "high"
    - uncertainty: "low" | "medium" | "high"
    - explicit: dict (optional) like {"percent_change": 0.3} (only if copied from user text)
    """
    defaults = defaults or {}
    name = _as_snake(candidate.get("name") or defaults.get("name") or "unknown_factor")
    kind = (candidate.get("kind") or defaults.get("kind") or "level").strip().lower()
    polarity = (candidate.get("polarity") or defaults.get("polarity") or "positive").strip().lower()
    importance = (candidate.get("importance") or defaults.get("importance") or "medium").strip().lower()
    uncertainty = (candidate.get("uncertainty") or defaults.get("uncertainty") or "medium").strip().lower()
    explicit = candidate.get("explicit") or {}

    sign = _polarity_sign(polarity)
    strength = _importance_scale(importance)
    std = _uncertainty_std(uncertainty)

    # Level factors (0–1). Higher always means "more of the thing".
    if kind == "level":
        base = 0.5 + (sign * strength)
        base = _clamp01(base)
        # Map base/uncertainty into a Beta distribution by choosing alpha/beta.
        # Keep it simple + stable: higher uncertainty => smaller concentration.
        concentration = 14.0 if uncertainty == "low" else 8.0 if uncertainty == "medium" else 5.0
        alpha = max(1.1, base * concentration)
        beta = max(1.1, (1.0 - base) * concentration)
        return name, VariableDistribution(value=float(base), distribution="beta", params={"alpha": float(alpha), "beta": float(beta)})

    # Change factors (-1..1), e.g. salary change scaled. Prefer Normal.
    if kind == "change":
        mean = explicit.get("percent_change")
        if isinstance(mean, (int, float)):
            mean = float(mean)
        else:
            mean = sign * (strength * 1.5)  # heuristic: important changes have bigger effect
        mean = max(-1.0, min(1.0, float(mean)))
        return name, VariableDistribution(value=float(mean), distribution="normal", params={"mean": float(mean), "std": float(std)})

    # Binary factors (0/1). Without explicit truth, assume 50/50.
    if kind == "binary":
        p = explicit.get("probability")
        if isinstance(p, (int, float)):
            p = float(p)
        elif explicit.get("is_true") is True:
            p = 1.0
        elif explicit.get("is_true") is False:
            p = 0.0
        else:
            p = 0.5
        p = _clamp01(p)
        return name, VariableDistribution(value=float(p), distribution="bernoulli", params={"probability": float(p)})

    # Fallback
    base = 0.5 + (sign * strength)
    base = _clamp01(base)
    return name, VariableDistribution(value=float(base), distribution="beta", params={"alpha": 2.0, "beta": 2.0})


def build_variables(
    *,
    decision_text: str,
    decision_type: str,
    candidates: Iterable[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, VariableDistribution]:
    """
    Build a variables dict for the simulator using deterministic heuristics.
    """
    context = context or {}
    variables: Dict[str, VariableDistribution] = {}

    # Always include an uncertainty driver so risk can reflect ambiguity.
    variables["outcome_uncertainty"] = VariableDistribution(
        value=0.5, distribution="beta", params={"alpha": 2.0, "beta": 2.0}
    )

    # Translate some known context fields into variables (still deterministic).
    runway = context.get("runway_months")
    if isinstance(runway, int):
        # More runway => higher financial_satisfaction driver.
        # 0–1 mapping: 0 months ~ 0.2, 6+ months ~ 0.8
        fin_buffer = _clamp01(0.2 + (min(12, runway) / 12.0) * 0.6)
        variables["savings_buffer"] = VariableDistribution(
            value=float(fin_buffer),
            distribution="beta",
            params={"alpha": float(max(1.1, fin_buffer * 10)), "beta": float(max(1.1, (1.0 - fin_buffer) * 10))},
        )

    # Parse explicit % facts (e.g., “30% raise”) and attach to plausible variables if present.
    percents = extract_explicit_percents(decision_text)
    explicit_percent = percents[0] if percents else None

    for c in candidates or []:
        c = dict(c)
        name = _as_snake(c.get("name", ""))
        if not name:
            continue

        # If the candidate is salary/income related and we have an explicit percent, attach it.
        if explicit_percent is not None and any(k in name for k in ("salary", "income", "pay")):
            c.setdefault("explicit", {})
            c["explicit"].setdefault("percent_change", float(max(-1.0, min(1.0, explicit_percent))))
            c.setdefault("kind", "change")

        key, dist = candidate_to_distribution(c)
        variables[key] = dist

    # Decision-type defaults (non-LLM)
    dt = (decision_type or "").lower()
    if "job" in dt and "job_security" not in variables:
        variables["company_stability"] = VariableDistribution(
            value=0.55, distribution="beta", params={"alpha": 3.0, "beta": 2.5}
        )
    if "rent" in dt and "cost" not in variables:
        variables["rent_cost_pressure"] = VariableDistribution(
            value=0.55, distribution="beta", params={"alpha": 3.0, "beta": 2.5}
        )

    return variables

