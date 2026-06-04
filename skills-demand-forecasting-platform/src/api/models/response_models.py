"""Pydantic models for API responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class SkillResponse(BaseModel):
    """Response model for skill information."""
    skill_name: str
    category: str
    skill_type: str
    frequency: int
    frequency_percentage: float
    avg_confidence: float
    growth_trend: Optional[float] = None
    avg_salary_impact: Optional[float] = None

class SkillTrendResponse(BaseModel):
    """Response model for skill trend data."""
    skill_name: str
    period: str
    historical_data: Dict[str, List[Union[str, float]]]
    forecast_data: Dict[str, List[Union[str, float]]]
    trend_analysis: Dict[str, Any]
    growth_rate: float

class SkillForecast(BaseModel):
    """Individual skill forecast data."""
    skill_name: str
    forecast_dates: List[str]
    forecast_values: List[float]
    confidence_intervals: Dict[str, List[float]]
    trend_direction: str
    growth_rate: float
    model_performance: Dict[str, float]

class ForecastResponse(BaseModel):
    """Response model for demand forecasting."""
    forecasts: List[SkillForecast]
    forecast_horizon: int
    model_type: str
    generated_at: str

class SalaryPredictionResponse(BaseModel):
    """Response model for salary prediction."""
    predicted_salary: float
    salary_range: Dict[str, float]
    confidence_score: float
    factors_analysis: Dict[str, Any]
    model_used: str
    similar_jobs_count: int

class CareerPathResponse(BaseModel):
    """Response model for career path recommendations."""
    recommended_skills: List[Dict[str, Any]]
    skill_gaps: List[str]
    learning_path: List[Dict[str, Any]]
    salary_progression: Dict[str, float]
    job_market_outlook: Dict[str, Any]
