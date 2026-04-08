"""
Configuration management for the causal modeling engine.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Application configuration settings."""
    
    # Base paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DAG_REGISTRY_DIR: Path = PROJECT_ROOT / "dag_registry"
    SKILLS_DIR: Path = PROJECT_ROOT / "skills"
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Model settings
    DEFAULT_RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DAG_REGISTRY_DIR.mkdir(exist_ok=True)
        cls.SKILLS_DIR.mkdir(exist_ok=True)
        cls.NOTEBOOKS_DIR.mkdir(exist_ok=True)


# Initialize directories on import
Config.ensure_directories()

