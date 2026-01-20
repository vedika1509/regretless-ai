"""Decision data models."""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class VariableDistribution(BaseModel):
    """Probability distribution for a decision variable."""
    value: float = Field(..., description="Base value of the variable")
    distribution: str = Field(..., description="Type of distribution (normal, beta, uniform, etc.)")
    params: Dict[str, Any] = Field(..., description="Parameters for the distribution")


class DecisionInput(BaseModel):
    """Input model for a decision."""
    decision_type: str = Field(..., description="Type of decision (job_switch, rent, hiring, etc.)")
    decision_text: str = Field(..., description="Natural language description of the decision")
    variables: Optional[Dict[str, VariableDistribution]] = Field(
        None, 
        description="Structured variables extracted from the decision"
    )


class StructuredDecision(BaseModel):
    """Structured decision after parsing."""
    decision_type: str
    variables: Dict[str, VariableDistribution]
    raw_text: str
