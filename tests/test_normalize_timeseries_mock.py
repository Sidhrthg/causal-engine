"""Tests for timeseries normalization with MockLLM."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import pytest

from src.ingest.mapping import infer_column_mapping, apply_mapping
from src.llm.providers import MockLLM
from src.schemas.timeseries import TimeSeriesSchema, validate_timeseries_df
from scripts.normalize_timeseries import normalize_timeseries


def test_mock_llm_deterministic_mapping():
    """Test that MockLLM produces deterministic mapping."""
    # Create messy dataframe
    df = pd.DataFrame({
        "yr": [2024, 2025, 2026],
        "price": [1.0, 1.1, 1.2],
        "demand": [100.0, 105.0, 110.0],
        "supply": [95.0, 100.0, 105.0],
        "stock": [20.0, 25.0, 30.0],
    })
    
    # Test with same seed - should be deterministic
    llm1 = MockLLM(seed=42)
    result1 = infer_column_mapping(list(df.columns), TimeSeriesSchema, llm1, df)
    
    llm2 = MockLLM(seed=42)
    result2 = infer_column_mapping(list(df.columns), TimeSeriesSchema, llm2, df)
    
    assert result1 == result2, "MockLLM should produce deterministic results with same seed"
    
    # Verify mapping makes sense
    assert "year" in result1["mapping"], "Should map 'year'"
    assert "P" in result1["mapping"], "Should map 'P'"
    assert result1["mapping"]["year"] == "yr", "Should map 'yr' to 'year'"
    assert result1["mapping"]["P"] == "price", "Should map 'price' to 'P'"


def test_apply_mapping():
    """Test applying mapping to DataFrame."""
    df = pd.DataFrame({
        "yr": [2024, 2025],
        "price": [1.0, 1.1],
        "demand": [100.0, 105.0],
    })
    
    mapping = {
        "year": "yr",
        "P": "price",
        "D": "demand"
    }
    
    mapped_df = apply_mapping(df, mapping, target_schema=TimeSeriesSchema)
    
    assert "year" in mapped_df.columns
    assert "P" in mapped_df.columns
    assert "D" in mapped_df.columns
    assert mapped_df["year"].iloc[0] == 2024
    assert mapped_df["P"].iloc[0] == 1.0


def test_normalize_timeseries_output_conforms():
    """Test that normalize_timeseries produces output conforming to TimeSeriesSchema."""
    # Create messy input CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "messy.csv"
        output_path = Path(tmpdir) / "normalized.csv"
        
        # Create messy dataframe
        df = pd.DataFrame({
            "yr": [2024, 2025, 2026],
            "price": [1.0, 1.1, 1.2],
            "demand": [100.0, 105.0, 110.0],
        })
        df.to_csv(input_path, index=False)
        
        # Normalize
        normalize_timeseries(input_path, output_path, seed=42)
        
        # Load and validate output
        output_df = pd.read_csv(output_path)
        
        # Must have required columns
        assert "year" in output_df.columns, "Output must have 'year' column"
        assert "P" in output_df.columns, "Output must have 'P' column"
        
        # Validate schema
        validate_timeseries_df(output_df)
        
        # Check mapping file exists
        mapping_path = output_path.with_suffix(".mapping.json")
        assert mapping_path.exists(), "Mapping file should exist"
        
        # Check schema file exists
        schema_path = output_path.with_suffix(".schema.json")
        assert schema_path.exists(), "Schema file should exist"


def test_normalize_timeseries_deterministic():
    """Test that normalization is deterministic with same seed."""
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "messy.csv"
        output1_path = Path(tmpdir) / "normalized1.csv"
        output2_path = Path(tmpdir) / "normalized2.csv"
        
        df = pd.DataFrame({
            "yr": [2024, 2025],
            "price": [1.0, 1.1],
        })
        df.to_csv(input_path, index=False)
        
        # Normalize twice with same seed
        normalize_timeseries(input_path, output1_path, seed=42)
        normalize_timeseries(input_path, output2_path, seed=42)
        
        # Outputs should be identical
        df1 = pd.read_csv(output1_path)
        df2 = pd.read_csv(output2_path)
        
        pd.testing.assert_frame_equal(df1, df2)
        
        # Mapping files should be identical
        mapping1 = json.loads((output1_path.with_suffix(".mapping.json")).read_text())
        mapping2 = json.loads((output2_path.with_suffix(".mapping.json")).read_text())
        
        assert mapping1 == mapping2, "Mapping should be deterministic"

