"""Schema validation for scenario configuration files."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import yaml


class TimeConfig(BaseModel):
    """Time configuration."""
    dt: float = Field(..., gt=0, description="Time step")
    start_year: int = Field(..., description="Start year")
    end_year: int = Field(..., description="End year")
    
    @model_validator(mode="after")
    def validate_end_year(self):
        """End year must be after start year."""
        if self.end_year <= self.start_year:
            raise ValueError("end_year must be after start_year")
        return self


class BaselineConfig(BaseModel):
    """Baseline state configuration."""
    P_ref: float = Field(..., gt=0, description="Reference price")
    P0: float = Field(..., gt=0, description="Initial price")
    K0: float = Field(..., gt=0, description="Initial capacity")
    I0: float = Field(..., ge=0, description="Initial inventory")
    D0: float = Field(..., gt=0, description="Initial demand")


class DemandGrowthConfig(BaseModel):
    """Demand growth configuration."""
    type: Literal["constant"] = Field(default="constant", description="Growth type")
    g: float = Field(..., gt=0, description="Growth rate per year")


class ParametersConfig(BaseModel):
    """Model parameters."""
    eps: float = Field(default=1e-9, gt=0, description="Small positive constant")
    
    # Utilization
    u0: float = Field(..., description="Base utilization rate")
    beta_u: float = Field(..., description="Utilization price elasticity")
    u_min: float = Field(..., ge=0, le=1, description="Minimum utilization")
    u_max: float = Field(..., ge=0, le=1, description="Maximum utilization")
    
    # Capacity dynamics
    tau_K: float = Field(..., gt=0, description="Capacity adjustment time (years)")
    eta_K: float = Field(..., description="Capacity price elasticity")
    retire_rate: float = Field(..., ge=0, description="Capacity retirement rate per year")
    
    # Demand
    eta_D: float = Field(..., description="Demand price elasticity")
    demand_growth: DemandGrowthConfig = Field(..., description="Demand growth configuration")
    
    # Price dynamics
    alpha_P: float = Field(..., gt=0, description="Price adjustment speed")
    cover_star: float = Field(..., gt=0, description="Target cover ratio")
    lambda_cover: float = Field(..., ge=0, description="Cover ratio feedback")
    sigma_P: float = Field(default=0.0, ge=0, description="Price noise std (0 for deterministic)")


class PolicyConfig(BaseModel):
    """Policy configuration."""
    substitution: float = Field(default=0.0, ge=0, le=1, description="Substitution rate")
    efficiency: float = Field(default=0.0, ge=0, le=1, description="Efficiency gain rate")
    subsidy: float = Field(default=0.0, ge=0, description="Capacity subsidy")
    stockpile_release: float = Field(default=0.0, ge=0, description="Stockpile release per step (tons/year, deprecated: use stockpile_release shock instead)")


class ShockConfig(BaseModel):
    """Configuration for a supply/demand shock."""

    type: Literal[
        "export_restriction",
        "demand_surge",
        "capex_shock",
        "stockpile_release",
        "capacity_reduction",
        "policy_shock",
        "macro_demand_shock",
    ] = Field(..., description="Shock type")
    start_year: int = Field(..., description="Start year")
    end_year: int = Field(..., description="End year")
    magnitude: float = Field(
        ...,
        description="For policy_shock: 0.3 = 30% quota reduction; for macro_demand_shock: -0.4 = 40% demand drop",
    )
    quota_reduction: Optional[float] = Field(default=None, description="Specific quota % (policy_shock)")
    demand_destruction: Optional[float] = Field(default=None, description="Demand drop % (macro_demand_shock)")


class OutputsConfig(BaseModel):
    """Output configuration."""
    out_dir: str = Field(default="runs", description="Output directory")
    save_csv: bool = Field(default=True, description="Save CSV timeseries")
    metrics: List[str] = Field(..., description="List of metrics to compute")


class ScenarioConfig(BaseModel):
    """Schema for scenario configuration files."""
    
    name: str = Field(..., description="Scenario name")
    commodity: str = Field(..., description="Commodity type")
    seed: int = Field(..., description="Random seed for reproducibility")
    description: Optional[str] = Field(default=None, description="Scenario description")
    
    time: TimeConfig = Field(..., description="Time configuration")
    baseline: BaselineConfig = Field(..., description="Baseline state")
    parameters: ParametersConfig = Field(..., description="Model parameters")
    policy: PolicyConfig = Field(default_factory=PolicyConfig, description="Policy configuration")
    shocks: List[ShockConfig] = Field(default_factory=list, description="Shock list")
    outputs: OutputsConfig = Field(..., description="Output configuration")
    
    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate commodity type."""
        if v.lower() != "graphite":
            raise ValueError(f"Unsupported commodity: {v}. Only 'graphite' is supported.")
        return v.lower()
    
    @property
    def years(self) -> List[int]:
        """Get list of years to simulate."""
        return list(range(self.time.start_year, self.time.end_year + 1))


def load_scenario(path: str) -> ScenarioConfig:
    """Load and validate scenario from YAML file."""
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    
    with open(scenario_path, "r") as f:
        data = yaml.safe_load(f)
    
    # Convert shocks list of dicts to list of ShockConfig
    if "shocks" in data and isinstance(data["shocks"], list):
        data["shocks"] = [ShockConfig(**s) if isinstance(s, dict) else s for s in data["shocks"]]
    
    # Convert outputs dict to OutputsConfig
    if "outputs" in data and isinstance(data["outputs"], dict):
        data["outputs"] = OutputsConfig(**data["outputs"])
    
    return ScenarioConfig(**data)
