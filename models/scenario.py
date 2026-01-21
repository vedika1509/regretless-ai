"""Scenario result models."""
from typing import Dict, List
from pydantic import BaseModel, Field


class OutcomeMetrics(BaseModel):
    """Outcome metrics for a scenario."""
    satisfaction_score: float = Field(..., ge=0.0, le=1.0)
    financial_score: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)


class ScenarioDriver(BaseModel):
    """A key driver that differentiates a scenario (based on sampled inputs)."""

    variable: str = Field(..., description="Variable name (snake_case)")
    mean: float = Field(..., description="Mean sampled value for this scenario bucket")
    delta: float = Field(..., description="Scenario mean minus overall mean")


class ScenarioResult(BaseModel):
    """Result for a single scenario."""
    scenario_type: str = Field(..., description="best_case, worst_case, or most_likely")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of this scenario")
    outcomes: OutcomeMetrics
    explanation: str = Field(..., description="Human-readable explanation of the scenario")
    drivers: List[ScenarioDriver] = Field(
        default_factory=list,
        description="Top drivers that explain why this scenario looks this way",
    )
    metrics: Dict[str, float] = Field(default_factory=dict, description="Additional metrics")


class RiskFlag(BaseModel):
    """Risk flag detected in analysis."""
    risk_type: str = Field(..., description="Type of risk (high_variance, long_tail, etc.)")
    severity: str = Field(..., description="Severity level (high, medium, low)")
    description: str = Field(..., description="Human-readable description of the risk")


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    scenarios: Dict[str, ScenarioResult] = Field(..., description="Best, worst, and most likely scenarios")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    regret_score: float = Field(..., ge=0.0, le=1.0, description="Regret Score - proprietary metric for decision regret risk")
    risks: List[RiskFlag] = Field(default_factory=list, description="List of detected risks")
    simulation_count: int = Field(..., description="Number of simulations run")
