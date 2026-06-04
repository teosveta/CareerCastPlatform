"""Pydantic models for API requests."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class ExtractionMethod(str, Enum):
    """Skill extraction methods."""
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"

class ModelType(str, Enum):
    """Model types for predictions."""
    PROPHET = "prophet"
    ARIMA = "arima"
    ENSEMBLE = "ensemble"
    BEST = "best"

class ExperienceLevel(str, Enum):
    """Experience levels for salary prediction."""
    ENTRY = "Entry Level"
    MID = "Mid Level"
    SENIOR = "Senior Level"
    PRINCIPAL = "Principal Level"
    MANAGEMENT = "Management"

class EmploymentType(str, Enum):
    """Employment types."""
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    CONTRACT = "Contract"
    FREELANCE = "Freelance"

class SkillQuery(BaseModel):
    """Request model for skill extraction."""
    text: str = Field(..., min_length=10, max_length=10000, description="Job description text")
    method: ExtractionMethod = Field(ExtractionMethod.HYBRID, description="Extraction method to use")
    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence for skill extraction")

class ForecastRequest(BaseModel):
    """Request model for demand forecasting."""
    skills: List[str] = Field(..., min_items=1, max_items=10, description="Skills to forecast")
    forecast_horizon: int = Field(12, ge=1, le=24, description="Forecast horizon in months")
    model_type: ModelType = Field(ModelType.ENSEMBLE, description="Model type to use")
    include_confidence_intervals: bool = Field(True, description="Include confidence intervals")

class SalaryPredictionRequest(BaseModel):
    """Request model for salary prediction."""
    skills: List[str] = Field(..., min_items=1, max_items=20, description="Required skills")
    location: str = Field(..., description="Job location")
    experience_level: ExperienceLevel = Field(..., description="Experience level")
    employment_type: EmploymentType = Field(EmploymentType.FULL_TIME, description="Employment type")
    company_size: Optional[str] = Field(None, description="Company size category")
    remote_work: bool = Field(False, description="Remote work allowed")
    model_type: ModelType = Field(ModelType.BEST, description="Model type to use")

class CareerPathRequest(BaseModel):
    """Request model for career path analysis."""
    current_skills: List[str] = Field(..., min_items=1, description="Current skills")
    target_role: Optional[str] = Field(None, description="Target job role")
    experience_level: ExperienceLevel = Field(..., description="Current experience level")
    preferred_industries: Optional[List[str]] = Field(None, description="Preferred industries")
    location_preferences: Optional[List[str]] = Field(None, description="Location preferences")