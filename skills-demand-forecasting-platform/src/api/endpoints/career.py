"""Career guidance-related API endpoints."""

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

@router.post("/skill-gap-analysis")
async def analyze_skill_gap(request: dict):
    """Analyze skill gaps for career advancement."""
    try:
        current_skills = request.get("current_skills", [])
        target_role = request.get("target_role", "")
        experience_level = request.get("experience_level", "Mid Level")

        if not current_skills or not target_role:
            raise HTTPException(status_code=400, detail="Current skills and target role are required")

        # Sample role requirements
        role_requirements = {
            "Data Scientist": {
                "required_skills": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas", "NumPy"],
                "preferred_skills": ["R", "TensorFlow", "PyTorch", "Tableau", "AWS", "Jupyter"],
                "avg_salary": 125000
            },
            "Full Stack Developer": {
                "required_skills": ["JavaScript", "React", "Node.js", "SQL", "Git", "HTML/CSS"],
                "preferred_skills": ["TypeScript", "MongoDB", "AWS", "Docker", "GraphQL", "Next.js"],
                "avg_salary": 110000
            },
            "DevOps Engineer": {
                "required_skills": ["Linux", "Docker", "Kubernetes", "AWS", "Git", "Python"],
                "preferred_skills": ["Terraform", "Jenkins", "Prometheus", "Grafana", "Ansible", "Go"],
                "avg_salary": 130000
            },
            "Product Manager": {
                "required_skills": ["Product Strategy", "User Research", "Analytics", "SQL", "Project Management"],
                "preferred_skills": ["A/B Testing", "Tableau", "Figma", "JIRA", "Agile", "Python"],
                "avg_salary": 140000
            },
            "ML Engineer": {
                "required_skills": ["Python", "Machine Learning", "TensorFlow", "Docker", "Git", "SQL"],
                "preferred_skills": ["Kubernetes", "AWS", "MLflow", "PyTorch", "Spark", "Airflow"],
                "avg_salary": 145000
            }
        }

        role_data = role_requirements.get(target_role)
        if not role_data:
            return {"error": f"Role '{target_role}' not found in our database"}

        required_skills = role_data["required_skills"]
        preferred_skills = role_data["preferred_skills"]

        # Analyze gaps
        missing_required = [skill for skill in required_skills if skill not in current_skills]
        missing_preferred = [skill for skill in preferred_skills if skill not in current_skills]

        # Calculate readiness score
        required_coverage = (len(required_skills) - len(missing_required)) / len(required_skills) * 100
        preferred_coverage = (len(preferred_skills) - len(missing_preferred)) / len(preferred_skills) * 100
        overall_readiness = (required_coverage * 0.7) + (preferred_coverage * 0.3)

        return {
            "target_role": target_role,
            "current_skills": current_skills,
            "skill_gap_analysis": {
                "missing_required_skills": missing_required,
                "missing_preferred_skills": missing_preferred,
                "skills_you_have": [skill for skill in current_skills if skill in required_skills + preferred_skills],
                "readiness_score": round(overall_readiness, 1),
                "required_skills_coverage": round(required_coverage, 1),
                "preferred_skills_coverage": round(preferred_coverage, 1)
            },
            "recommendations": {
                "priority_skills": missing_required[:3],  # Top 3 missing required skills
                "bonus_skills": missing_preferred[:2],    # Top 2 missing preferred skills
                "estimated_learning_time": len(missing_required) * 6 + len(missing_preferred) * 3  # weeks
            },
            "role_info": {
                "average_salary": role_data["avg_salary"],
                "required_skills_total": len(required_skills),
                "preferred_skills_total": len(preferred_skills)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing skill gap: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing skill gap")

@router.post("/learning-path")
async def generate_learning_path(request: dict):
    """Generate a personalized learning path."""
    try:
        current_skills = request.get("current_skills", [])
        target_skills = request.get("target_skills", [])
        learning_preference = request.get("learning_preference", "balanced")  # fast, balanced, thorough
        time_commitment = request.get("time_commitment", "10")  # hours per week

        if not target_skills:
            raise HTTPException(status_code=400, detail="Target skills are required")

        # Sample learning resources and time estimates
        skill_learning_data = {
            "Python": {
                "difficulty": "medium",
                "time_weeks": 8,
                "prerequisites": [],
                "resources": ["Python.org Tutorial", "Automate the Boring Stuff", "Codecademy Python"]
            },
            "JavaScript": {
                "difficulty": "medium",
                "time_weeks": 6,
                "prerequisites": ["HTML/CSS"],
                "resources": ["MDN Web Docs", "JavaScript.info", "freeCodeCamp"]
            },
            "React": {
                "difficulty": "medium",
                "time_weeks": 4,
                "prerequisites": ["JavaScript", "HTML/CSS"],
                "resources": ["React Documentation", "React Tutorial", "Full Stack Open"]
            },
            "AWS": {
                "difficulty": "hard",
                "time_weeks": 12,
                "prerequisites": ["Linux", "Networking Basics"],
                "resources": ["AWS Training", "A Cloud Guru", "AWS Solutions Architect Course"]
            },
            "Docker": {
                "difficulty": "medium",
                "time_weeks": 3,
                "prerequisites": ["Linux"],
                "resources": ["Docker Documentation", "Docker Mastery Course", "Play with Docker"]
            },
            "Machine Learning": {
                "difficulty": "hard",
                "time_weeks": 16,
                "prerequisites": ["Python", "Statistics", "Linear Algebra"],
                "resources": ["Andrew Ng Course", "Hands-On ML Book", "Fast.ai"]
            }
        }

        # Calculate learning path
        skills_to_learn = [skill for skill in target_skills if skill not in current_skills]
        learning_path = []

        for skill in skills_to_learn:
            skill_data = skill_learning_data.get(skill, {
                "difficulty": "medium",
                "time_weeks": 6,
                "prerequisites": [],
                "resources": ["Online courses", "Documentation", "Practice projects"]
            })

            # Check if prerequisites are met
            unmet_prerequisites = [prereq for prereq in skill_data["prerequisites"]
                                 if prereq not in current_skills]

            learning_path.append({
                "skill": skill,
                "difficulty": skill_data["difficulty"],
                "estimated_weeks": skill_data["time_weeks"],
                "prerequisites": skill_data["prerequisites"],
                "unmet_prerequisites": unmet_prerequisites,
                "recommended_resources": skill_data["resources"],
                "priority": "high" if not unmet_prerequisites else "medium"
            })

        # Sort by priority and prerequisites
        learning_path.sort(key=lambda x: (len(x["unmet_prerequisites"]), x["estimated_weeks"]))

        # Calculate timeline
        total_weeks = sum(item["estimated_weeks"] for item in learning_path)
        weekly_hours = int(time_commitment)

        return {
            "learning_path": learning_path,
            "timeline": {
                "total_skills": len(skills_to_learn),
                "estimated_total_weeks": total_weeks,
                "weekly_time_commitment": f"{weekly_hours} hours",
                "estimated_completion": f"{total_weeks} weeks"
            },
            "recommendations": {
                "start_with": learning_path[0]["skill"] if learning_path else None,
                "parallel_learning": [item["skill"] for item in learning_path if not item["unmet_prerequisites"]][:2],
                "most_challenging": max(learning_path, key=lambda x: x["estimated_weeks"])["skill"] if learning_path else None
            },
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating learning path: {e}")
        raise HTTPException(status_code=500, detail="Error generating learning path")

@router.get("/career-paths")
async def get_career_paths(
    current_role: Optional[str] = Query(None, description="Current job role"),
    experience_level: str = Query("Mid Level", description="Current experience level"),
    interests: Optional[str] = Query(None, description="Comma-separated interests")
):
    """Get potential career advancement paths."""
    try:
        # Sample career progression paths
        career_paths = {
            "Software Engineer": [
                {
                    "target_role": "Senior Software Engineer",
                    "time_to_advance": "2-3 years",
                    "key_skills_needed": ["Advanced Programming", "System Design", "Mentoring"],
                    "salary_increase": 25000,
                    "probability": 85
                },
                {
                    "target_role": "Tech Lead",
                    "time_to_advance": "3-5 years",
                    "key_skills_needed": ["Leadership", "Architecture", "Project Management"],
                    "salary_increase": 35000,
                    "probability": 65
                },
                {
                    "target_role": "Engineering Manager",
                    "time_to_advance": "4-6 years",
                    "key_skills_needed": ["People Management", "Strategic Planning", "Communication"],
                    "salary_increase": 45000,
                    "probability": 45
                }
            ],
            "Data Scientist": [
                {
                    "target_role": "Senior Data Scientist",
                    "time_to_advance": "2-4 years",
                    "key_skills_needed": ["Advanced ML", "Deep Learning", "MLOps"],
                    "salary_increase": 30000,
                    "probability": 80
                },
                {
                    "target_role": "ML Engineer",
                    "time_to_advance": "1-3 years",
                    "key_skills_needed": ["Docker", "Kubernetes", "Model Deployment"],
                    "salary_increase": 25000,
                    "probability": 70
                },
                {
                    "target_role": "Data Science Manager",
                    "time_to_advance": "4-6 years",
                    "key_skills_needed": ["Team Leadership", "Business Strategy", "Stakeholder Management"],
                    "salary_increase": 50000,
                    "probability": 40
                }
            ]
        }

        if current_role and current_role in career_paths:
            paths = career_paths[current_role]
        else:
            # Return general career progression examples
            paths = [
                {
                    "target_role": "Senior Individual Contributor",
                    "time_to_advance": "2-4 years",
                    "key_skills_needed": ["Deep Technical Expertise", "Mentoring", "Complex Problem Solving"],
                    "salary_increase": 25000,
                    "probability": 75
                },
                {
                    "target_role": "Team Lead",
                    "time_to_advance": "3-5 years",
                    "key_skills_needed": ["Leadership", "Project Management", "Communication"],
                    "salary_increase": 35000,
                    "probability": 60
                },
                {
                    "target_role": "Manager",
                    "time_to_advance": "4-7 years",
                    "key_skills_needed": ["People Management", "Strategic Thinking", "Business Acumen"],
                    "salary_increase": 45000,
                    "probability": 45
                }
            ]

        return {
            "current_role": current_role or "General",
            "experience_level": experience_level,
            "career_paths": paths,
            "general_advice": {
                "focus_areas": ["Technical Excellence", "Communication Skills", "Domain Knowledge"],
                "networking_importance": "High",
                "recommended_certifications": ["Cloud Certifications", "Industry-specific Credentials"],
                "skill_development_priority": "Continuously update technical skills while developing soft skills"
            },
            "market_outlook": {
                "growth_rate": "12% (above average)",
                "job_availability": "High",
                "remote_opportunities": "Abundant"
            }
        }

    except Exception as e:
        logger.error(f"Error fetching career paths: {e}")
        raise HTTPException(status_code=500, detail="Error fetching career paths")

@router.post("/salary-negotiation")
async def get_salary_negotiation_insights(request: dict):
    """Get insights for salary negotiation."""
    try:
        current_salary = request.get("current_salary", 0)
        target_salary = request.get("target_salary", 0)
        location = request.get("location", "")
        skills = request.get("skills", [])
        experience_years = request.get("experience_years", 0)

        if not current_salary or not target_salary:
            raise HTTPException(status_code=400, detail="Current and target salary are required")

        # Calculate market data
        increase_percentage = ((target_salary - current_salary) / current_salary) * 100

        # Sample market ranges
        market_data = {
            "market_median": current_salary * 1.05,
            "market_75th_percentile": current_salary * 1.15,
            "market_90th_percentile": current_salary * 1.25
        }

        # Negotiation strategy
        strategy_tips = []

        if increase_percentage <= 10:
            strategy_tips.append("This is a reasonable increase within market standards.")
            negotiation_likelihood = "High"
        elif increase_percentage <= 20:
            strategy_tips.append("Prepare strong justification with concrete achievements.")
            negotiation_likelihood = "Medium-High"
        else:
            strategy_tips.append("Consider a phased approach or additional non-salary benefits.")
            negotiation_likelihood = "Medium"

        # Skills-based insights
        high_value_skills = ["AWS", "Kubernetes", "Machine Learning", "React", "Python"]
        valuable_skills_count = len([skill for skill in skills if skill in high_value_skills])

        if valuable_skills_count >= 3:
            strategy_tips.append("Highlight your high-demand technical skills.")

        return {
            "current_salary": current_salary,
            "target_salary": target_salary,
            "increase_percentage": round(increase_percentage, 1),
            "market_analysis": {
                "your_position": "Above median" if current_salary > market_data["market_median"] else "Below median",
                "market_ranges": market_data,
                "target_vs_market": "Within range" if target_salary <= market_data["market_90th_percentile"] else "Above market"
            },
            "negotiation_insights": {
                "likelihood_of_success": negotiation_likelihood,
                "strategy_tips": strategy_tips,
                "best_timing": "End of quarter/year or after major achievement",
                "preparation_checklist": [
                    "Document recent achievements and impact",
                    "Research industry salary data",
                    "Prepare alternative compensation options",
                    "Practice negotiation conversation"
                ]
            },
            "leverage_factors": {
                "high_demand_skills": valuable_skills_count,
                "experience_level": "Strong" if experience_years >= 5 else "Developing",
                "market_conditions": "Favorable for tech professionals"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating salary negotiation insights: {e}")
        raise HTTPException(status_code=500, detail="Error generating salary negotiation insights")