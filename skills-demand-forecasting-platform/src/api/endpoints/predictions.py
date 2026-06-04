"""Prediction-related API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.file_utils import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances
file_handler = FileHandler()


@router.post("/demand-forecast")
async def predict_skill_demand(request: dict):
    """Predict future demand for specific skills."""
    try:
        skills = request.get("skills", [])
        forecast_horizon = request.get("forecast_horizon", 12)
        model_type = request.get("model_type", "ensemble")

        if not skills:
            raise HTTPException(status_code=400, detail="Skills list is required")

        skill_forecasts = []

        for skill_name in skills:
            # Generate sample forecast data
            end_date = datetime.now()
            forecast_dates = [(end_date + timedelta(days=30 * i)).strftime("%Y-%m-%d") for i in
                              range(1, forecast_horizon + 1)]

            # Sample forecast values with growth trend
            base_value = 100
            forecast_values = [base_value + i * 5 + (i % 3) * 2 for i in range(forecast_horizon)]

            # Sample confidence intervals
            lower_bound = [max(0, val * 0.8) for val in forecast_values]
            upper_bound = [val * 1.2 for val in forecast_values]

            skill_forecast = {
                "skill_name": skill_name,
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values,
                "confidence_intervals": {
                    "lower": lower_bound,
                    "upper": upper_bound
                },
                "trend_direction": "increasing",
                "growth_rate": 15.2,
                "model_performance": {
                    "mae": 8.5,
                    "mse": 125.3,
                    "rmse": 11.2,
                    "mape": 12.8
                }
            }

            skill_forecasts.append(skill_forecast)

        response = {
            "forecasts": skill_forecasts,
            "forecast_horizon": forecast_horizon,
            "model_type": model_type,
            "generated_at": datetime.now().isoformat()
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating demand forecast: {e}")
        raise HTTPException(status_code=500, detail="Error generating demand forecast")


@router.post("/salary-prediction")
async def predict_salary(request: dict):
    """Predict salary based on job characteristics."""
    try:
        skills = request.get("skills", [])
        location = request.get("location", "")
        experience_level = request.get("experience_level", "")
        employment_type = request.get("employment_type", "Full-time")
        company_size = request.get("company_size", "")
        remote_work = request.get("remote_work", False)
        model_type = request.get("model_type", "best")

        if not skills:
            raise HTTPException(status_code=400, detail="Skills list is required")

        # Simple salary prediction logic
        base_salary = {
            "Entry Level": 70000,
            "Mid Level": 95000,
            "Senior Level": 130000,
            "Principal Level": 170000,
            "Management": 200000
        }.get(experience_level, 95000)

        # Location multiplier
        location_multiplier = 1.0
        if "San Francisco" in location or "SF" in location:
            location_multiplier = 1.4
        elif "New York" in location or "NYC" in location:
            location_multiplier = 1.3
        elif "Seattle" in location:
            location_multiplier = 1.25
        elif "Austin" in location:
            location_multiplier = 1.1

        # Skills bonus
        high_value_skills = {"Python", "AWS", "Kubernetes", "React", "Machine Learning", "Docker"}
        skill_bonus = len([skill for skill in skills if skill in high_value_skills]) * 5000

        predicted_salary = base_salary * location_multiplier + skill_bonus

        # Calculate salary range
        uncertainty = predicted_salary * 0.15
        salary_range = {
            "min": max(20000, predicted_salary - uncertainty),
            "max": predicted_salary + uncertainty
        }

        response = {
            "predicted_salary": predicted_salary,
            "salary_range": salary_range,
            "confidence_score": 0.85,
            "factors_analysis": {
                "high_impact_skills": [skill for skill in skills if skill in high_value_skills],
                "location_impact": "positive" if location_multiplier > 1 else "neutral",
                "experience_impact": "high" if experience_level in ["Senior Level", "Principal Level"] else "moderate"
            },
            "model_used": model_type,
            "similar_jobs_count": 150
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting salary: {e}")
        raise HTTPException(status_code=500, detail="Error predicting salary")


@router.get("/market-trends")
async def get_market_trends(
        period: str = Query("12m", pattern="^(3m|6m|12m|24m)$", description="Time period for trend analysis"),
        top_skills: int = Query(20, ge=5, le=50, description="Number of top skills to return")
):
    """Get overall market trends for top skills."""
    try:
        # Sample market trends data
        sample_skills = [
            ("Python", 25.4), ("JavaScript", 18.7), ("React", 15.2), ("AWS", 28.9),
            ("Docker", 22.1), ("Kubernetes", 35.6), ("PostgreSQL", 8.3), ("MongoDB", 12.4),
            ("Machine Learning", 31.2), ("TensorFlow", 19.8), ("Node.js", 14.6), ("Angular", 7.9),
            ("Vue.js", 16.3), ("Java", 5.2), ("C++", 3.1), ("Go", 29.7),
            ("Rust", 42.3), ("TypeScript", 21.5), ("GraphQL", 33.8), ("Redis", 18.4)
        ]

        # Sort by growth rate and take top skills
        top_growing = sorted(sample_skills, key=lambda x: x[1], reverse=True)[:top_skills]
        top_declining = [("jQuery", -8.2), ("AngularJS", -15.3), ("PHP", -5.1), ("Perl", -12.7), ("Flash", -25.6)]

        return {
            "period": period,
            "top_growing_skills": [{"skill": skill, "growth_rate": rate} for skill, rate in top_growing],
            "top_declining_skills": [{"skill": skill, "growth_rate": rate} for skill, rate in top_declining],
            "market_summary": {
                "total_skills_tracked": len(sample_skills),
                "avg_growth_rate": sum(rate for _, rate in sample_skills) / len(sample_skills),
                "high_demand_categories": ["Cloud Platforms", "Data Science & ML", "DevOps & Infrastructure"]
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching market trends: {e}")
        raise HTTPException(status_code=500, detail="Error fetching market trends")