"""TimeSeries schema for canonical, validated time series data."""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from pydantic import ConfigDict


class TimeSeriesSchema(BaseModel):
    """Canonical schema for annual minerals time series."""

    year: int = Field(..., description="Year (calendar)")

    # Optional canonical value columns
    P: Optional[float] = Field(default=None, description="Price level")
    D: Optional[float] = Field(default=None, description="Demand")
    Q: Optional[float] = Field(default=None, description="Production/Supply")
    I: Optional[float] = Field(default=None, description="Inventory/stock levels")

    # Optional metadata
    source: Optional[str] = Field(default=None, description="Data provenance")
    units: Optional[str] = Field(default=None, description="Units for numeric columns")
    notes: Optional[str] = Field(default=None, description="Freeform notes or caveats")

    model_config = ConfigDict(extra="forbid")


def validate_timeseries_df(df: pd.DataFrame) -> bool:
    """
    Validate a DataFrame conforms to TimeSeriesSchema.

    Args:
        df: pandas DataFrame

    Returns:
        True if valid, raises ValidationError/ValueError otherwise
    """
    if "year" not in df.columns:
        raise ValueError("Missing required column: year")

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        for key, value in list(row_dict.items()):
            if pd.isna(value):
                row_dict[key] = None
        if row_dict.get("year") is None:
            raise ValueError(f"Row {idx} missing required 'year'")
        try:
            TimeSeriesSchema.model_validate(row_dict)
        except ValidationError as exc:
            raise ValidationError(f"Row {idx} invalid: {exc}") from exc

    return True

