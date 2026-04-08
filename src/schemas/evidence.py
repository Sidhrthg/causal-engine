"""Evidence packet schema for claims with value/unit/confidence/source."""

from pydantic import BaseModel, Field
from typing import Optional


class EvidencePacket(BaseModel):
    """Evidence packet with claim, value, unit, confidence, and source."""
    
    claim: str = Field(..., description="The claim being made")
    value: float = Field(..., description="Numeric value")
    unit: str = Field(..., description="Unit of measurement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    source: Optional[str] = Field(default=None, description="Source of the evidence")

