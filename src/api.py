"""
FastAPI application for causal modeling engine.

This module provides HTTP endpoints for causal effect estimation and
intervention simulation. It wraps the core estimation and simulation
functions with a REST API interface.
"""

from typing import Tuple, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import Config
from src.utils.logging_utils import get_logger, setup_logger
from src.ingest import load_dataset
from src.estimate import estimate_from_dag_path
from src.simulate import simulate_from_dag_path, Intervention

# Set up logging
setup_logger(__name__, Config.LOG_LEVEL)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Causal Modeling Engine API",
    description="API for causal effect estimation and policy simulation",
    version="0.1.0"
)


# Request/Response models
class EstimateRequest(BaseModel):
    """Request model for effect estimation."""
    dataset_path: str = Field(..., description="Path to the dataset file (CSV or Parquet)")
    treatment: str = Field(..., description="Name of the treatment variable")
    outcome: str = Field(..., description="Name of the outcome variable")
    controls: list[str] = Field(..., description="List of control/covariate variables")
    dag_path: str = Field(..., description="Path to the DAG file (DOT format)")


class EstimateResponse(BaseModel):
    """Response model for effect estimation."""
    ate: float = Field(..., description="Average Treatment Effect estimate")
    ate_ci: Tuple[float, float] = Field(..., description="Confidence interval as (lower, upper) tuple")
    method: str = Field(..., description="Method used for estimation")


class SimulationRequest(BaseModel):
    """Request model for intervention simulation."""
    dataset_path: str = Field(..., description="Path to the dataset file (CSV or Parquet)")
    treatment: str = Field(..., description="Name of the treatment variable")
    outcome: str = Field(..., description="Name of the outcome variable")
    controls: list[str] = Field(..., description="List of control/covariate variables")
    dag_path: str = Field(..., description="Path to the DAG file (DOT format)")
    node: str = Field(..., description="Name of the node to intervene on")
    value: Union[float, int, str] = Field(..., description="Value to set for the intervention")


class SimulationResponse(BaseModel):
    """Response model for intervention simulation."""
    outcomes: dict[str, float] = Field(..., description="Dictionary of outcome metrics")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Causal Modeling Engine API",
        "version": "0.1.0",
        "endpoints": ["/estimate", "/simulate"]
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/estimate", response_model=EstimateResponse)
async def estimate_endpoint(req: EstimateRequest) -> EstimateResponse:
    """
    Estimate Average Treatment Effect (ATE) from a dataset.
    
    This endpoint:
    1) Loads the dataset from req.dataset_path (supports CSV and Parquet)
    2) Calls estimate_from_dag_path from estimate.py
    3) Returns the ATE and confidence interval
    
    Args:
        req: Estimation request with dataset path, treatment, outcome, controls, and DAG path
    
    Returns:
        Estimation results including ATE and confidence interval
    
    Raises:
        HTTPException: If dataset or DAG file cannot be loaded
    """
    logger.info(f"Received estimation request: {req.treatment} -> {req.outcome}")
    logger.info(f"Dataset: {req.dataset_path}, DAG: {req.dag_path}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {req.dataset_path}")
        df = load_dataset(req.dataset_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {req.dataset_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {req.dataset_path}. Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load dataset from {req.dataset_path}: {str(e)}"
        )
    
    try:
        # Estimate ATE
        logger.info("Calling estimate_from_dag_path")
        result = estimate_from_dag_path(
            df=df,
            treatment=req.treatment,
            outcome=req.outcome,
            controls=req.controls,
            dag_path=req.dag_path
        )
        
        logger.info(f"Estimation completed: ATE = {result.ate:.6f}")
        
        return EstimateResponse(
            ate=result.ate,
            ate_ci=result.ate_ci,
            method=result.method
        )
    
    except FileNotFoundError as e:
        logger.error(f"DAG file not found: {req.dag_path}")
        raise HTTPException(
            status_code=404,
            detail=f"DAG file not found: {req.dag_path}. Error: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Estimation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Estimation failed: {str(e)}"
        )


@app.post("/simulate", response_model=SimulationResponse)
async def simulate_endpoint(req: SimulationRequest) -> SimulationResponse:
    """
    Simulate the effects of an intervention on a causal model.
    
    This endpoint:
    1) Loads the dataset from req.dataset_path
    2) Builds an Intervention from req.node and req.value
    3) Calls simulate_from_dag_path
    4) Returns the resulting outcomes
    
    Args:
        req: Simulation request with dataset path, treatment, outcome, controls, DAG path, and intervention details
    
    Returns:
        Simulation results with outcome metrics (baseline, intervened, differences, etc.)
    
    Raises:
        HTTPException: If dataset or DAG file cannot be loaded, or simulation fails
    """
    logger.info(f"Received simulation request: intervention on {req.node} = {req.value}")
    logger.info(f"Dataset: {req.dataset_path}, DAG: {req.dag_path}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {req.dataset_path}")
        df = load_dataset(req.dataset_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {req.dataset_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {req.dataset_path}. Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load dataset from {req.dataset_path}: {str(e)}"
        )
    
    try:
        # Build Intervention object
        intervention = Intervention(node=req.node, value=req.value)
        logger.info(f"Created intervention: {intervention.node} = {intervention.value}")
        
        # Run simulation
        logger.info("Calling simulate_from_dag_path")
        result = simulate_from_dag_path(
            df=df,
            treatment=req.treatment,
            outcome=req.outcome,
            controls=req.controls,
            dag_path=req.dag_path,
            intervention=intervention,
            num_samples=1000
        )
        
        logger.info(f"Simulation completed: baseline_mean = {result.outcomes.get('baseline_mean', 'N/A')}")
        
        return SimulationResponse(
            outcomes=result.outcomes
        )
    
    except FileNotFoundError as e:
        logger.error(f"DAG file not found: {req.dag_path}")
        raise HTTPException(
            status_code=404,
            detail=f"DAG file not found: {req.dag_path}. Error: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Causal Modeling Engine API on {Config.API_HOST}:{Config.API_PORT}")
    uvicorn.run(
        "src.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_DEBUG
    )
