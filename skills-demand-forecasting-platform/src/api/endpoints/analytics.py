"""Analytics-related API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/skill-correlation")
async def get_skill_correlations(
    skill: str = Query(..., description="Base skill to find correlations for"),
    top_k: int = Query(10, ge=1, le=20, description="Number of correlated skills to return")
):
    """Get skills that frequently appear together with the specified skill."""
    try:
        # Sample correlation data
        correlations = {
            "Python": [
                ("Django", 0.75), ("Flask", 0.68), ("Pandas", 0.82), ("NumPy", 0.79),
                ("scikit-learn", 0.71), ("TensorFlow", 0.66), ("AWS", 0.58), ("Docker", 0.54)
            ],
            "JavaScript": [
                ("React", 0.78), ("Node.js", 0.72), ("Express", 0.65), ("MongoDB", 0.59),
                ("TypeScript", 0.71), ("Vue.js", 0.48), ("Angular", 0.52), ("GraphQL", 0.44)
            ],
            "AWS": [
                ("Docker", 0.69), ("Kubernetes", 0.73), ("Python", 0.58), ("Terraform", 0.65),
                ("Linux", 0.61), ("Jenkins", 0.52), ("PostgreSQL", 0.48), ("Redis", 0.45)
            ]
        }

        skill_correlations = correlations.get(skill, [])[:top_k]

        return {
            "base_skill": skill,
            "correlations": [
                {"skill": corr_skill, "correlation_score": score}
                for corr_skill, score in skill_correlations
            ],
            "total_found": len(skill_correlations)
        }

    except Exception as e:
        logger.error(f"Error fetching skill correlations: {e}")
        raise HTTPException(status_code=500, detail="Error fetching skill correlations")

@router.get("/geographic-analysis")
async def get_geographic_analysis(
    skill: Optional[str] = Query(None, description="Skill to analyze geographically"),
    metric: str = Query("demand", pattern="^(demand|salary|growth)$", description="Metric to analyze")
):
    """Get geographic analysis of skills demand, salary, or growth."""
    try:
        # Sample geographic data
        locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
            "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
            "Atlanta, GA", "Remote"
        ]

        if metric == "demand":
            location_data = [
                {"location": loc, "value": 100 + (i * 25) + (i % 3) * 10, "rank": i + 1}
                for i, loc in enumerate(locations)
            ]
            metric_name = "Job Postings"
        elif metric == "salary":
            location_data = [
                {"location": loc, "value": 80000 + (i * 15000) + (i % 2) * 5000, "rank": i + 1}
                for i, loc in enumerate(locations)
            ]
            metric_name = "Average Salary"
        else:  # growth
            location_data = [
                {"location": loc, "value": 5.2 + (i * 2.8) + (i % 4) * 1.5, "rank": i + 1}
                for i, loc in enumerate(locations)
            ]
            metric_name = "Growth Rate (%)"

        return {
            "skill": skill or "All Skills",
            "metric": metric,
            "metric_name": metric_name,
            "geographic_data": sorted(location_data, key=lambda x: x["value"], reverse=True),
            "top_location": location_data[0]["location"] if location_data else None,
            "analysis_date": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching geographic analysis: {e}")
        raise HTTPException(status_code=500, detail="Error fetching geographic analysis")

@router.get("/emerging-technologies")
async def get_emerging_technologies(
    timeframe: str = Query("3m", pattern="^(1m|3m|6m|12m)$", description="Timeframe for emergence analysis"),
    threshold: float = Query(20.0, ge=0.0, le=100.0, description="Minimum growth rate to consider emerging")
):
    """Identify emerging technologies based on growth rate."""
    try:
        # Sample emerging technologies data
        emerging_techs = [
            {"skill": "Rust", "growth_rate": 45.3, "current_demand": 156, "trend": "exponential"},
            {"skill": "Kubernetes", "growth_rate": 38.7, "current_demand": 892, "trend": "strong_linear"},
            {"skill": "GraphQL", "growth_rate": 35.2, "current_demand": 423, "trend": "accelerating"},
            {"skill": "TypeScript", "growth_rate": 29.8, "current_demand": 1247, "trend": "steady"},
            {"skill": "Svelte", "growth_rate": 28.4, "current_demand": 89, "trend": "emerging"},
            {"skill": "Deno", "growth_rate": 26.1, "current_demand": 67, "trend": "early_adoption"},
            {"skill": "WebAssembly", "growth_rate": 24.7, "current_demand": 134, "trend": "growing"},
            {"skill": "Terraform", "growth_rate": 22.9, "current_demand": 678, "trend": "strong_linear"},
            {"skill": "Snowflake", "growth_rate": 21.5, "current_demand": 234, "trend": "accelerating"}
        ]

        # Filter by threshold
        filtered_techs = [tech for tech in emerging_techs if tech["growth_rate"] >= threshold]

        return {
            "timeframe": timeframe,
            "growth_threshold": threshold,
            "emerging_technologies": filtered_techs,
            "total_found": len(filtered_techs),
            "fastest_growing": filtered_techs[0] if filtered_techs else None,
            "analysis_summary": {
                "avg_growth_rate": sum(tech["growth_rate"] for tech in filtered_techs) / len(filtered_techs) if filtered_techs else 0,
                "total_demand": sum(tech["current_demand"] for tech in filtered_techs),
                "categories_represented": ["Programming Languages", "DevOps", "Web Technologies", "Databases"]
            }
        }

    except Exception as e:
        logger.error(f"Error fetching emerging technologies: {e}")
        raise HTTPException(status_code=500, detail="Error fetching emerging technologies")

@router.get("/industry-insights")
async def get_industry_insights(
    industry: Optional[str] = Query(None, description="Specific industry to analyze"),
    top_skills: int = Query(10, ge=5, le=25, description="Number of top skills to return per industry")
):
    """Get industry-specific skill insights."""
    try:
        # Sample industry data
        industries_data = {
            "Technology": {
                "top_skills": [
                    {"skill": "Python", "demand_score": 95, "avg_salary": 125000},
                    {"skill": "JavaScript", "demand_score": 92, "avg_salary": 115000},
                    {"skill": "React", "demand_score": 88, "avg_salary": 120000},
                    {"skill": "AWS", "demand_score": 85, "avg_salary": 135000},
                    {"skill": "Docker", "demand_score": 78, "avg_salary": 128000}
                ],
                "growth_trends": {"overall": 18.5, "ai_ml": 35.2, "cloud": 28.7},
                "avg_salary": 118000
            },
            "Finance": {
                "top_skills": [
                    {"skill": "Python", "demand_score": 89, "avg_salary": 140000},
                    {"skill": "SQL", "demand_score": 85, "avg_salary": 110000},
                    {"skill": "R", "demand_score": 72, "avg_salary": 125000},
                    {"skill": "Tableau", "demand_score": 68, "avg_salary": 105000},
                    {"skill": "Excel", "demand_score": 65, "avg_salary": 85000}
                ],
                "growth_trends": {"overall": 12.3, "fintech": 25.8, "data_analytics": 22.1},
                "avg_salary": 133000
            },
            "Healthcare": {
                "top_skills": [
                    {"skill": "Python", "demand_score": 78, "avg_salary": 115000},
                    {"skill": "R", "demand_score": 75, "avg_salary": 108000},
                    {"skill": "SQL", "demand_score": 72, "avg_salary": 95000},
                    {"skill": "Tableau", "demand_score": 65, "avg_salary": 98000},
                    {"skill": "HIPAA", "demand_score": 58, "avg_salary": 105000}
                ],
                "growth_trends": {"overall": 15.7, "health_tech": 28.3, "data_science": 24.6},
                "avg_salary": 104000
            }
        }

        if industry and industry in industries_data:
            data = industries_data[industry]
            return {
                "industry": industry,
                "top_skills": data["top_skills"][:top_skills],
                "growth_trends": data["growth_trends"],
                "industry_avg_salary": data["avg_salary"],
                "analysis_date": datetime.now().isoformat()
            }
        else:
            # Return overview of all industries
            return {
                "overview": True,
                "industries": [
                    {
                        "name": name,
                        "avg_salary": data["avg_salary"],
                        "growth_rate": data["growth_trends"]["overall"],
                        "top_skill": data["top_skills"][0]["skill"]
                    }
                    for name, data in industries_data.items()
                ],
                "available_industries": list(industries_data.keys())
            }

    except Exception as e:
        logger.error(f"Error fetching industry insights: {e}")
        raise HTTPException(status_code=500, detail="Error fetching industry insights")