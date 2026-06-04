"""Skills-related API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional, Any
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.file_utils import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances
file_handler = FileHandler()

# Simple request/response models (since Pydantic models might have import issues)
class SkillQuery:
    def __init__(self, text: str, method: str = "hybrid", confidence_threshold: float = 0.7):
        self.text = text
        self.method = method
        self.confidence_threshold = confidence_threshold

@router.get("/")
async def get_all_skills(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    skill_type: Optional[str] = Query(None)
):
    """Get all skills with optional filtering."""
    try:
        # Try to load skills data, create sample if not available
        try:
            skills_data = file_handler.load_json("data/processed/skills_taxonomy.json")
        except:
            # Create sample skills data
            skills_data = [
                {
                    "skill_name": "Python",
                    "category": "Programming Languages", 
                    "skill_type": "technical",
                    "frequency": 1250,
                    "frequency_percentage": 25.5,
                    "avg_confidence": 0.89,
                    "growth_trend": 15.2
                },
                {
                    "skill_name": "JavaScript",
                    "category": "Programming Languages",
                    "skill_type": "technical", 
                    "frequency": 1100,
                    "frequency_percentage": 22.8,
                    "avg_confidence": 0.87,
                    "growth_trend": 8.5
                },
                {
                    "skill_name": "React",
                    "category": "Web Frameworks",
                    "skill_type": "technical",
                    "frequency": 950, 
                    "frequency_percentage": 19.2,
                    "avg_confidence": 0.85,
                    "growth_trend": 22.1
                },
                {
                    "skill_name": "AWS",
                    "category": "Cloud Platforms",
                    "skill_type": "technical",
                    "frequency": 800,
                    "frequency_percentage": 16.4, 
                    "avg_confidence": 0.92,
                    "growth_trend": 28.3
                }
            ]
        
        # Apply filters
        filtered_skills = skills_data
        if category:
            filtered_skills = [s for s in filtered_skills if s.get("category") == category]
        if skill_type:
            filtered_skills = [s for s in filtered_skills if s.get("skill_type") == skill_type]
        
        # Apply pagination
        paginated_skills = filtered_skills[offset:offset + limit]
        
        return {
            "skills": paginated_skills,
            "total": len(filtered_skills),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error fetching skills: {e}")
        raise HTTPException(status_code=500, detail="Error fetching skills data")

@router.get("/{skill_name}")
async def get_skill_details(skill_name: str):
    """Get detailed information about a specific skill."""
    try:
        # Sample skill detail
        skill_data = {
            "skill_name": skill_name,
            "category": "Programming Languages",
            "skill_type": "technical",
            "frequency": 1250,
            "frequency_percentage": 25.5,
            "avg_confidence": 0.89,
            "growth_trend": 15.2,
            "avg_salary_impact": 12000,
            "related_skills": ["Django", "Flask", "FastAPI", "NumPy", "Pandas"],
            "top_companies": ["Google", "Microsoft", "Amazon", "Netflix", "Spotify"]
        }
        
        return skill_data
        
    except Exception as e:
        logger.error(f"Error fetching skill {skill_name}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching skill data")

@router.get("/{skill_name}/trends")
async def get_skill_trends(
    skill_name: str,
    period: str = Query("12m", pattern="^(3m|6m|12m|24m)$", description="Time period for trend analysis")
):
    """Get trend data for a specific skill."""
    try:
        # Sample trend data
        import datetime
        from datetime import timedelta
        
        # Generate sample dates
        end_date = datetime.datetime.now()
        period_months = {"3m": 3, "6m": 6, "12m": 12, "24m": 24}
        months = period_months[period]
        
        dates = []
        values = []
        base_value = 100
        
        for i in range(months):
            date = end_date - timedelta(days=30 * i)
            dates.append(date.strftime("%Y-%m-%d"))
            # Add some trend and noise
            trend_value = base_value + (i * 2) + (i % 3 - 1) * 10
            values.append(max(0, trend_value))
        
        dates.reverse()
        values.reverse()
        
        trend_response = {
            "skill_name": skill_name,
            "period": period,
            "historical_data": {
                "dates": dates,
                "values": values
            },
            "forecast_data": {
                "dates": [(end_date + timedelta(days=30 * i)).strftime("%Y-%m-%d") for i in range(1, 7)],
                "values": [values[-1] + i * 5 for i in range(1, 7)]
            },
            "trend_analysis": {
                "overall_direction": "increasing",
                "growth_rate": 15.2,
                "volatility": "low",
                "trend_strength": "strong"
            }
        }
        
        return trend_response
        
    except Exception as e:
        logger.error(f"Error fetching trends for {skill_name}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching trend data")

@router.post("/extract")
async def extract_skills_from_text(request: dict):
    """Extract skills from job description text."""
    try:
        text = request.get("text", "")
        method = request.get("method", "hybrid")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Simple skill extraction (replace with actual NLP when available)
        skills_keywords = [
            "python", "javascript", "react", "aws", "docker", "kubernetes", 
            "postgresql", "mongodb", "machine learning", "tensorflow",
            "java", "c++", "node.js", "angular", "vue.js", "git"
        ]
        
        text_lower = text.lower()
        found_skills = []
        confidence_scores = {}
        
        for skill in skills_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
                confidence_scores[skill.title()] = 0.85  # Default confidence
        
        return {
            "extracted_skills": found_skills,
            "confidence_scores": confidence_scores,
            "method_used": method,
            "total_skills_found": len(found_skills)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail="Error extracting skills from text")

@router.get("/categories/list")
async def get_skill_categories():
    """Get all available skill categories."""
    try:
        categories = [
            "Programming Languages",
            "Web Frameworks", 
            "Databases",
            "Cloud Platforms",
            "DevOps & Infrastructure",
            "Data Science & ML",
            "Mobile Development",
            "Soft Skills"
        ]
        
        return {
            "categories": categories,
            "total_categories": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail="Error fetching categories")
