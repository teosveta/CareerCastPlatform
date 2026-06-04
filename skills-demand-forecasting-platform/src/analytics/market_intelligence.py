"""
Advanced market intelligence and analytics for comprehensive insights.
Provides business intelligence, trend analysis, and strategic recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class MarketIntelligence:
    """
    Advanced market intelligence providing comprehensive analytics,
    trend analysis, and strategic insights for decision making.
    """

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.industry_mappings = self._load_industry_mappings()
        self.skill_categories = self._load_skill_categories()
        self.location_tiers = self._load_location_tiers()

    def _load_industry_mappings(self) -> Dict[str, List[str]]:
        """Load industry mappings for analysis."""
        return {
            "Technology": ["software", "tech", "computer", "internet", "cloud"],
            "Finance": ["bank", "financial", "fintech", "investment", "insurance"],
            "Healthcare": ["health", "medical", "pharma", "biotech", "hospital"],
            "E-commerce": ["ecommerce", "retail", "marketplace", "shopping"],
            "Gaming": ["game", "gaming", "entertainment", "media"],
            "Education": ["education", "learning", "university", "school"],
            "Consulting": ["consulting", "advisory", "services"]
        }

    def _load_skill_categories(self) -> Dict[str, List[str]]:
        """Load skill category mappings."""
        return {
            "Programming": ["Python", "JavaScript", "Java", "C++", "Go", "Rust"],
            "Web Development": ["React", "Vue.js", "Angular", "Node.js", "HTML", "CSS"],
            "Cloud & DevOps": ["AWS", "Azure", "Docker", "Kubernetes", "Terraform"],
            "Data Science": ["Machine Learning", "TensorFlow", "PyTorch", "Pandas"],
            "Databases": ["PostgreSQL", "MongoDB", "Redis", "MySQL"],
            "Mobile": ["React Native", "Flutter", "iOS", "Android"]
        }

    def _load_location_tiers(self) -> Dict[str, Dict[str, Any]]:
        """Load location tier information."""
        return {
            "Tier 1": {
                "locations": ["San Francisco", "New York", "Seattle", "Boston"],
                "cost_multiplier": 1.4,
                "demand_multiplier": 1.3
            },
            "Tier 2": {
                "locations": ["Austin", "Denver", "Chicago", "Atlanta"],
                "cost_multiplier": 1.1,
                "demand_multiplier": 1.1
            },
            "Tier 3": {
                "locations": ["Remote"],
                "cost_multiplier": 1.0,
                "demand_multiplier": 1.2
            }
        }

    def analyze_geographic_trends(
            self,
            jobs_df: pd.DataFrame,
            skills_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze geographic trends and opportunities."""

        with self.performance_monitor.track_operation("geographic_analysis"):
            logger.info("Analyzing geographic trends...")

            if jobs_df.empty:
                return {"error": "No job data available"}

            geographic_insights = {
                "location_demand": {},
                "salary_by_location": {},
                "skill_concentration": {},
                "growth_markets": [],
                "remote_trends": {}
            }

            # Location demand analysis
            if "location" in jobs_df.columns:
                location_counts = jobs_df["location"].value_counts().head(10)
                geographic_insights["location_demand"] = {
                    location: int(count) for location, count in location_counts.items()
                }

                # Salary analysis by location
                if "salary_median" in jobs_df.columns:
                    salary_by_loc = jobs_df.groupby("location")["salary_median"].agg([
                        "mean", "median", "count"
                    ]).round(0)

                    # Filter locations with sufficient data
                    salary_by_loc = salary_by_loc[salary_by_loc["count"] >= 5]

                    geographic_insights["salary_by_location"] = {
                        location: {
                            "avg_salary": int(row["mean"]),
                            "median_salary": int(row["median"]),
                            "job_count": int(row["count"])
                        }
                        for location, row in salary_by_loc.iterrows()
                    }

                # Remote work trends
                remote_jobs = jobs_df[
                    jobs_df["location"].str.contains("Remote", case=False, na=False)
                ]

                if not remote_jobs.empty:
                    geographic_insights["remote_trends"] = {
                        "remote_job_percentage": len(remote_jobs) / len(jobs_df) * 100,
                        "remote_avg_salary": int(
                            remote_jobs["salary_median"].mean()) if "salary_median" in remote_jobs.columns else 0,
                        "remote_job_count": len(remote_jobs)
                    }

            # Skill concentration by location
            if not skills_df.empty and "job_id" in skills_df.columns:
                job_locations = jobs_df.set_index("job_id")["location"]
                skills_with_location = skills_df.merge(
                    job_locations, left_on="job_id", right_index=True, how="left"
                )

                skill_location_counts = skills_with_location.groupby([
                    "location", "skill_name"
                ]).size().reset_index(name="count")

                # Find top skill concentrations
                top_concentrations = skill_location_counts.nlargest(15, "count")

                geographic_insights["skill_concentration"] = [
                    {
                        "location": row["location"],
                        "skill": row["skill_name"],
                        "concentration": int(row["count"])
                    }
                    for _, row in top_concentrations.iterrows()
                ]

            return geographic_insights

    def analyze_industry_trends(
            self,
            jobs_df: pd.DataFrame,
            skills_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze industry-specific trends and patterns."""

        with self.performance_monitor.track_operation("industry_analysis"):
            logger.info("Analyzing industry trends...")

            industry_insights = {
                "industry_demand": {},
                "industry_skills": {},
                "salary_by_industry": {},
                "growth_industries": []
            }

            if jobs_df.empty:
                return industry_insights

            # Classify jobs by industry based on company names and descriptions
            jobs_df = self._classify_industries(jobs_df)

            # Industry demand
            if "industry_classified" in jobs_df.columns:
                industry_counts = jobs_df["industry_classified"].value_counts()
                industry_insights["industry_demand"] = {
                    industry: int(count) for industry, count in industry_counts.items()
                }

                # Salary by industry
                if "salary_median" in jobs_df.columns:
                    industry_salaries = jobs_df.groupby("industry_classified")["salary_median"].agg([
                        "mean", "median", "count"
                    ]).round(0)

                    industry_insights["salary_by_industry"] = {
                        industry: {
                            "avg_salary": int(row["mean"]),
                            "median_salary": int(row["median"]),
                            "job_count": int(row["count"])
                        }
                        for industry, row in industry_salaries.iterrows()
                        if row["count"] >= 3
                    }

            # Skills by industry
            if not skills_df.empty and "job_id" in skills_df.columns:
                job_industries = jobs_df.set_index("job_id")["industry_classified"]
                skills_with_industry = skills_df.merge(
                    job_industries, left_on="job_id", right_index=True, how="left"
                )

                industry_skills = skills_with_industry.groupby([
                    "industry_classified", "skill_name"
                ]).size().reset_index(name="count")

                # Top skills per industry
                industry_insights["industry_skills"] = {}
                for industry in industry_skills["industry_classified"].unique():
                    if pd.notna(industry):
                        top_skills = industry_skills[
                            industry_skills["industry_classified"] == industry
                            ].nlargest(5, "count")

                        industry_insights["industry_skills"][industry] = [
                            {"skill": row["skill_name"], "demand": int(row["count"])}
                            for _, row in top_skills.iterrows()
                        ]

            return industry_insights

    def _classify_industries(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Classify jobs into industries based on keywords."""

        jobs_df = jobs_df.copy()

        def classify_job(row):
            text_to_analyze = " ".join([
                str(row.get("company_name", "")),
                str(row.get("title", "")),
                str(row.get("description", ""))[:200]  # First 200 chars of description
            ]).lower()

            # Score each industry
            industry_scores = {}
            for industry, keywords in self.industry_mappings.items():
                score = sum(1 for keyword in keywords if keyword in text_to_analyze)
                if score > 0:
                    industry_scores[industry] = score

            # Return industry with highest score, or "Other"
            if industry_scores:
                return max(industry_scores, key=industry_scores.get)
            else:
                return "Other"

        jobs_df["industry_classified"] = jobs_df.apply(classify_job, axis=1)
        return jobs_df

    def analyze_skill_correlations(self, skills_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze skill correlations and combinations."""

        with self.performance_monitor.track_operation("skill_correlation_analysis"):
            logger.info("Analyzing skill correlations...")

            if skills_df.empty or "job_id" not in skills_df.columns:
                return {"error": "Insufficient skill data"}

            # Create skill co-occurrence matrix
            job_skills = skills_df.groupby("job_id")["skill_name"].apply(list).to_dict()

            skill_combinations = defaultdict(int)
            skill_counts = defaultdict(int)

            # Count skill occurrences and combinations
            for job_id, skills in job_skills.items():
                unique_skills = list(set(skills))  # Remove duplicates within job

                for skill in unique_skills:
                    skill_counts[skill] += 1

                # Count pairwise combinations
                for i, skill1 in enumerate(unique_skills):
                    for skill2 in unique_skills[i + 1:]:
                        pair = tuple(sorted([skill1, skill2]))
                        skill_combinations[pair] += 1

            # Calculate correlation scores
            correlations = []
            min_co_occurrence = 3  # Minimum times skills appear together

            for (skill1, skill2), co_count in skill_combinations.items():
                if co_count >= min_co_occurrence:
                    # Calculate lift (how much more likely they appear together)
                    total_jobs = len(job_skills)
                    skill1_prob = skill_counts[skill1] / total_jobs
                    skill2_prob = skill_counts[skill2] / total_jobs
                    joint_prob = co_count / total_jobs

                    expected_prob = skill1_prob * skill2_prob
                    lift = joint_prob / expected_prob if expected_prob > 0 else 0

                    correlations.append({
                        "skill1": skill1,
                        "skill2": skill2,
                        "co_occurrence_count": co_count,
                        "lift": round(lift, 2),
                        "correlation_strength": "strong" if lift > 2 else "moderate" if lift > 1.5 else "weak"
                    })

            # Sort by lift score
            correlations.sort(key=lambda x: x["lift"], reverse=True)

            return {
                "top_correlations": correlations[:20],
                "total_skills_analyzed": len(skill_counts),
                "total_combinations_found": len(skill_combinations),
                "analysis_summary": {
                    "strong_correlations": len([c for c in correlations if c["lift"] > 2]),
                    "moderate_correlations": len([c for c in correlations if 1.5 < c["lift"] <= 2]),
                    "avg_lift_score": np.mean([c["lift"] for c in correlations]) if correlations else 0
                }
            }

    def analyze_salary_trends(
            self,
            jobs_df: pd.DataFrame,
            skills_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze salary trends and skill impact on compensation."""

        with self.performance_monitor.track_operation("salary_analysis"):
            logger.info("Analyzing salary trends...")

            salary_insights = {
                "overall_statistics": {},
                "skill_salary_impact": {},
                "experience_premiums": {},
                "location_premiums": {},
                "trend_analysis": {}
            }

            if jobs_df.empty or "salary_median" not in jobs_df.columns:
                return salary_insights

            # Overall salary statistics
            salary_data = jobs_df["salary_median"].dropna()
            if not salary_data.empty:
                salary_insights["overall_statistics"] = {
                    "median_salary": int(salary_data.median()),
                    "mean_salary": int(salary_data.mean()),
                    "percentile_25": int(salary_data.quantile(0.25)),
                    "percentile_75": int(salary_data.quantile(0.75)),
                    "min_salary": int(salary_data.min()),
                    "max_salary": int(salary_data.max()),
                    "sample_size": len(salary_data)
                }

            # Skill salary impact
            if not skills_df.empty and "job_id" in skills_df.columns:
                job_salaries = jobs_df.set_index("job_id")["salary_median"]
                skills_with_salary = skills_df.merge(
                    job_salaries, left_on="job_id", right_index=True, how="inner"
                )

                skill_salary_impact = skills_with_salary.groupby("skill_name")["salary_median"].agg([
                    "mean", "median", "count"
                ]).round(0)

                # Filter skills with sufficient data
                skill_salary_impact = skill_salary_impact[skill_salary_impact["count"] >= 5]

                # Calculate premium vs market average
                market_median = salary_data.median()

                skill_premiums = []
                for skill, row in skill_salary_impact.iterrows():
                    premium = ((row["median"] - market_median) / market_median) * 100
                    skill_premiums.append({
                        "skill": skill,
                        "median_salary": int(row["median"]),
                        "sample_size": int(row["count"]),
                        "premium_percentage": round(premium, 1)
                    })

                # Sort by premium
                skill_premiums.sort(key=lambda x: x["premium_percentage"], reverse=True)
                salary_insights["skill_salary_impact"] = skill_premiums[:15]

            # Experience level premiums
            if "experience_level" in jobs_df.columns:
                exp_salaries = jobs_df.groupby("experience_level")["salary_median"].agg([
                    "mean", "median", "count"
                ]).round(0)

                salary_insights["experience_premiums"] = {
                    level: {
                        "median_salary": int(row["median"]),
                        "mean_salary": int(row["mean"]),
                        "sample_size": int(row["count"])
                    }
                    for level, row in exp_salaries.iterrows()
                    if row["count"] >= 3
                }

            return salary_insights

    def calculate_market_opportunities(
            self,
            forecasts: Dict[str, Any],
            salary_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate market opportunities based on growth and salary potential."""

        opportunities = []

        # Extract skill salary data
        skill_salaries = {}
        if "skill_salary_impact" in salary_insights:
            for skill_data in salary_insights["skill_salary_impact"]:
                skill_salaries[skill_data["skill"]] = {
                    "median_salary": skill_data["median_salary"],
                    "premium": skill_data["premium_percentage"]
                }

        # Analyze each forecasted skill
        for skill, forecast_data in forecasts.items():
            try:
                if "forecast_data" in forecast_data and "values" in forecast_data["forecast_data"]:
                    forecast_values = forecast_data["forecast_data"]["values"]

                    if len(forecast_values) >= 2:
                        # Calculate growth
                        start_value = forecast_values[0]
                        end_value = forecast_values[-1]
                        growth_rate = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0

                        # Get salary info
                        salary_info = skill_salaries.get(skill, {})
                        median_salary = salary_info.get("median_salary", 0)
                        salary_premium = salary_info.get("premium", 0)

                        # Calculate opportunity score (growth + salary premium)
                        opportunity_score = (growth_rate * 0.6) + (salary_premium * 0.4)

                        opportunities.append({
                            "skill": skill,
                            "growth_rate": round(growth_rate, 1),
                            "salary_premium": round(salary_premium, 1),
                            "median_salary": median_salary,
                            "opportunity_score": round(opportunity_score, 1),
                            "recommendation": self._get_opportunity_recommendation(
                                growth_rate, salary_premium, opportunity_score
                            )
                        })

            except Exception as e:
                logger.warning(f"Error calculating opportunity for {skill}: {e}")
                continue

        # Sort by opportunity score
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)

        return opportunities[:10]  # Top 10 opportunities

    def _get_opportunity_recommendation(
            self,
            growth_rate: float,
            salary_premium: float,
            opportunity_score: float
    ) -> str:
        """Generate recommendation based on opportunity metrics."""

        if opportunity_score > 30:
            return "Excellent opportunity - High growth and salary potential"
        elif opportunity_score > 15:
            if growth_rate > 20:
                return "Strong growth opportunity - Consider for rapid career advancement"
            elif salary_premium > 15:
                return "High-value skill - Excellent for salary optimization"
            else:
                return "Solid opportunity - Good balance of growth and compensation"
        elif opportunity_score > 5:
            return "Moderate opportunity - Consider as secondary skill development"
        else:
            return "Limited opportunity - Focus on higher-impact skills"

    def generate_executive_summary(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from all analytics."""

        summary = {
            "key_insights": [],
            "market_overview": {},
            "strategic_recommendations": [],
            "growth_opportunities": [],
            "risk_factors": []
        }

        # Extract key insights
        if "geographic" in analytics_data:
            geo_data = analytics_data["geographic"]
            if "remote_trends" in geo_data:
                remote_pct = geo_data["remote_trends"].get("remote_job_percentage", 0)
                summary["key_insights"].append(
                    f"Remote work represents {remote_pct:.1f}% of all job postings"
                )

        if "industry" in analytics_data:
            industry_data = analytics_data["industry"]
            if "industry_demand" in industry_data:
                top_industry = max(industry_data["industry_demand"].items(), key=lambda x: x[1])
                summary["key_insights"].append(
                    f"{top_industry[0]} sector leads with {top_industry[1]} job postings"
                )

        if "salary" in analytics_data:
            salary_data = analytics_data["salary"]
            if "overall_statistics" in salary_data:
                median_salary = salary_data["overall_statistics"].get("median_salary", 0)
                summary["key_insights"].append(
                    f"Market median salary is ${median_salary:,}"
                )

        # Market overview
        summary["market_overview"] = {
            "market_health": "Strong" if len(summary["key_insights"]) > 2 else "Stable",
            "dominant_trends": ["Remote work adoption", "Cloud technology growth", "AI/ML expansion"],
            "analysis_date": datetime.now().strftime("%Y-%m-%d")
        }

        # Strategic recommendations
        summary["strategic_recommendations"] = [
            "Focus on cloud technologies for maximum growth potential",
            "Develop AI/ML skills to access premium salary opportunities",
            "Consider remote-first career strategies for flexibility",
            "Build full-stack capabilities for market versatility"
        ]

        return summary

    # API helper methods

    def enhance_job_features(self, job_features: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance job features with market intelligence."""

        enhanced = job_features.copy()

        # Add location tier information
        location = job_features.get("location", "")
        enhanced["location_tier"] = self._get_location_tier(location)

        # Add skill category analysis
        skills = job_features.get("skills", [])
        enhanced["skill_categories"] = self._categorize_job_skills(skills)

        # Add market positioning
        enhanced["market_position"] = self._assess_market_position(job_features)

        return enhanced

    def _get_location_tier(self, location: str) -> str:
        """Get location tier for a given location."""
        location_lower = location.lower()

        for tier, data in self.location_tiers.items():
            if any(loc.lower() in location_lower for loc in data["locations"]):
                return tier

        return "Tier 3"  # Default

    def _categorize_job_skills(self, skills: List[str]) -> Dict[str, int]:
        """Categorize skills and count by category."""

        category_counts = defaultdict(int)

        for skill in skills:
            for category, category_skills in self.skill_categories.items():
                if skill in category_skills:
                    category_counts[category] += 1
                    break

        return dict(category_counts)

    def _assess_market_position(self, job_features: Dict[str, Any]) -> str:
        """Assess market position based on job features."""

        skills = job_features.get("skills", [])
        experience = job_features.get("experience_level", "")
        location = job_features.get("location", "")

        # Simple scoring
        score = 0

        # High-value skills
        high_value_skills = ["Python", "AWS", "Kubernetes", "React", "Machine Learning"]
        score += len([s for s in skills if s in high_value_skills]) * 2

        # Experience level
        if "senior" in experience.lower() or "lead" in experience.lower():
            score += 3
        elif "principal" in experience.lower():
            score += 4

        # Location tier
        if "san francisco" in location.lower() or "new york" in location.lower():
            score += 2

        # Return position assessment
        if score >= 8:
            return "Premium"
        elif score >= 5:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Entry"

    def get_market_context(self, job_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get market context for job features."""

        return {
            "demand_level": "High",  # Would calculate based on actual data
            "competition_level": "Medium",
            "growth_trajectory": "Positive",
            "market_saturation": "Low",
            "recommendation": "Favorable market conditions for this role profile"
        }

    def analyze_skill_gaps(
            self,
            current_skills: List[str],
            target_role: str,
            experience_level: str
    ) -> Dict[str, Any]:
        """Analyze skill gaps for career advancement."""

        # Role requirements mapping
        role_requirements = {
            "Data Scientist": {
                "required": ["Python", "SQL", "Statistics", "Machine Learning"],
                "preferred": ["R", "TensorFlow", "AWS", "Tableau"],
                "experience_multiplier": {"Entry Level": 0.7, "Mid Level": 1.0, "Senior Level": 1.3}
            },
            "Full Stack Developer": {
                "required": ["JavaScript", "React", "Node.js", "SQL"],
                "preferred": ["TypeScript", "AWS", "Docker", "MongoDB"],
                "experience_multiplier": {"Entry Level": 0.8, "Mid Level": 1.0, "Senior Level": 1.2}
            },
            "DevOps Engineer": {
                "required": ["Docker", "Kubernetes", "AWS", "Linux"],
                "preferred": ["Terraform", "Python", "Jenkins", "Monitoring"],
                "experience_multiplier": {"Entry Level": 0.6, "Mid Level": 1.0, "Senior Level": 1.4}
            }
        }

        role_data = role_requirements.get(target_role, {
            "required": ["Python", "SQL"],
            "preferred": ["AWS", "Docker"],
            "experience_multiplier": {"Entry Level": 0.8, "Mid Level": 1.0, "Senior Level": 1.2}
        })

        required_skills = role_data["required"]
        preferred_skills = role_data["preferred"]

        # Calculate gaps
        missing_required = [s for s in required_skills if s not in current_skills]
        missing_preferred = [s for s in preferred_skills if s not in current_skills]

        # Calculate readiness score
        required_coverage = (len(required_skills) - len(missing_required)) / len(required_skills)
        preferred_coverage = (len(preferred_skills) - len(missing_preferred)) / len(preferred_skills)

        exp_multiplier = role_data["experience_multiplier"].get(experience_level, 1.0)
        readiness_score = (required_coverage * 0.7 + preferred_coverage * 0.3) * exp_multiplier

        return {
            "readiness_score": round(readiness_score * 100, 1),
            "missing_required_skills": missing_required,
            "missing_preferred_skills": missing_preferred,
            "skills_you_have": [s for s in current_skills if s in required_skills + preferred_skills],
            "priority_recommendations": missing_required[:3],
            "estimated_learning_time_weeks": len(missing_required) * 8 + len(missing_preferred) * 4
        }

    async def generate_learning_path(
            self,
            skill_gaps: Dict[str, Any],
            include_market_trends: bool = True
    ) -> Dict[str, Any]:
        """Generate optimized learning path."""

        missing_skills = skill_gaps.get("missing_required_skills", []) + skill_gaps.get("missing_preferred_skills", [])

        # Skill learning data with market trends
        skill_learning_data = {
            "Python": {"weeks": 8, "difficulty": "Medium", "market_growth": 15.2, "priority": "High"},
            "JavaScript": {"weeks": 6, "difficulty": "Medium", "market_growth": 8.5, "priority": "High"},
            "React": {"weeks": 4, "difficulty": "Medium", "market_growth": 22.1, "priority": "High"},
            "AWS": {"weeks": 12, "difficulty": "Hard", "market_growth": 28.3, "priority": "Very High"},
            "Docker": {"weeks": 3, "difficulty": "Easy", "market_growth": 18.7, "priority": "High"},
            "Kubernetes": {"weeks": 6, "difficulty": "Hard", "market_growth": 35.4, "priority": "Very High"},
            "Machine Learning": {"weeks": 16, "difficulty": "Hard", "market_growth": 31.2, "priority": "Very High"},
            "SQL": {"weeks": 4, "difficulty": "Easy", "market_growth": 5.2, "priority": "Medium"}
        }

        learning_path = []

        for skill in missing_skills:
            skill_data = skill_learning_data.get(skill, {
                "weeks": 6, "difficulty": "Medium", "market_growth": 10.0, "priority": "Medium"
            })

            # Calculate priority score
            priority_score = skill_data["market_growth"] * 0.4
            if skill_data["priority"] == "Very High":
                priority_score += 40
            elif skill_data["priority"] == "High":
                priority_score += 30
            elif skill_data["priority"] == "Medium":
                priority_score += 20

            learning_path.append({
                "skill": skill,
                "estimated_weeks": skill_data["weeks"],
                "difficulty": skill_data["difficulty"],
                "market_growth_rate": skill_data["market_growth"],
                "priority_score": round(priority_score, 1),
                "recommended_resources": self._get_learning_resources(skill)
            })

        # Sort by priority score
        learning_path.sort(key=lambda x: x["priority_score"], reverse=True)

        return {
            "recommended_skills": learning_path,
            "total_learning_time_weeks": sum(item["estimated_weeks"] for item in learning_path),
            "high_priority_skills": [item["skill"] for item in learning_path if item["priority_score"] > 40],
            "learning_strategy": self._generate_learning_strategy(learning_path)
        }

    def _get_learning_resources(self, skill: str) -> List[str]:
        """Get learning resources for a skill."""

        resource_mapping = {
            "Python": ["Python.org Tutorial", "Automate the Boring Stuff", "Real Python"],
            "JavaScript": ["MDN Web Docs", "JavaScript.info", "freeCodeCamp"],
            "React": ["React Documentation", "React Tutorial", "Full Stack Open"],
            "AWS": ["AWS Training", "A Cloud Guru", "AWS Solutions Architect Course"],
            "Docker": ["Docker Documentation", "Docker Mastery Course", "Play with Docker"],
            "Kubernetes": ["Kubernetes Documentation", "Kubernetes the Hard Way", "CKAD Course"],
            "Machine Learning": ["Andrew Ng Course", "Hands-On ML Book", "Fast.ai"],
            "SQL": ["SQLBolt", "W3Schools SQL", "SQL Bootcamp"]
        }

        return resource_mapping.get(skill, ["Online courses", "Documentation", "Practice projects"])

    def _generate_learning_strategy(self, learning_path: List[Dict[str, Any]]) -> str:
        """Generate learning strategy recommendation."""

        if not learning_path:
            return "No additional skills needed"

        total_weeks = sum(item["estimated_weeks"] for item in learning_path)
        high_priority_count = len([item for item in learning_path if item["priority_score"] > 40])

        if total_weeks <= 12:
            return "Focused sprint approach - Complete all skills within 3 months"
        elif total_weeks <= 24:
            return "Balanced learning - Spread over 6 months with parallel skill development"
        elif high_priority_count >= 3:
            return "Priority-first approach - Focus on high-impact skills first, then expand"
        else:
            return "Long-term development - Plan 12+ month learning journey with milestones"

    def calculate_skill_roi(
            self,
            current_skills: List[str],
            target_skills: List[str]
    ) -> Dict[str, Any]:
        """Calculate ROI for skill development."""

        # Sample ROI data (would be calculated from real market data)
        skill_roi_data = {
            "Python": {"salary_increase": 12000, "job_opportunities": 85, "learning_investment": 200},
            "AWS": {"salary_increase": 15000, "job_opportunities": 90, "learning_investment": 400},
            "Kubernetes": {"salary_increase": 18000, "job_opportunities": 75, "learning_investment": 300},
            "React": {"salary_increase": 10000, "job_opportunities": 80, "learning_investment": 150},
            "Machine Learning": {"salary_increase": 20000, "job_opportunities": 70, "learning_investment": 500}
        }

        skills_to_learn = [s for s in target_skills if s not in current_skills]
        roi_analysis = []

        for skill in skills_to_learn:
            roi_data = skill_roi_data.get(skill, {
                "salary_increase": 8000, "job_opportunities": 60, "learning_investment": 250
            })

            # Calculate ROI score
            roi_score = (roi_data["salary_increase"] * 0.4 +
                         roi_data["job_opportunities"] * 0.4 +
                         (1000 - roi_data["learning_investment"]) * 0.2) / 10

            roi_analysis.append({
                "skill": skill,
                "expected_salary_increase": roi_data["salary_increase"],
                "job_opportunity_score": roi_data["job_opportunities"],
                "learning_investment": roi_data["learning_investment"],
                "roi_score": round(roi_score, 1),
                "payback_months": round(roi_data["learning_investment"] / (roi_data["salary_increase"] / 12), 1)
            })

        # Sort by ROI score
        roi_analysis.sort(key=lambda x: x["roi_score"], reverse=True)

        return {
            "skill_roi_analysis": roi_analysis,
            "total_investment": sum(item["learning_investment"] for item in roi_analysis),
            "total_expected_increase": sum(item["expected_salary_increase"] for item in roi_analysis),
            "average_payback_months": np.mean([item["payback_months"] for item in roi_analysis]) if roi_analysis else 0
        }

    def generate_career_timeline(
            self,
            current_skills: List[str],
            target_role: str,
            learning_path: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate career progression timeline."""

        timeline_milestones = []
        current_date = datetime.now()

        # Skill development milestones
        cumulative_weeks = 0
        for skill_data in learning_path.get("recommended_skills", []):
            cumulative_weeks += skill_data["estimated_weeks"]
            milestone_date = current_date + timedelta(weeks=cumulative_weeks)

            timeline_milestones.append({
                "date": milestone_date.strftime("%Y-%m-%d"),
                "milestone": f"Complete {skill_data['skill']} certification",
                "type": "skill_development",
                "estimated_effort": f"{skill_data['estimated_weeks']} weeks"
            })

        # Career progression milestones
        total_learning_weeks = learning_path.get("total_learning_time_weeks", 24)

        timeline_milestones.extend([
            {
                "date": (current_date + timedelta(weeks=total_learning_weeks + 4)).strftime("%Y-%m-%d"),
                "milestone": "Begin job search for target role",
                "type": "career_action",
                "estimated_effort": "4-8 weeks"
            },
            {
                "date": (current_date + timedelta(weeks=total_learning_weeks + 12)).strftime("%Y-%m-%d"),
                "milestone": f"Transition to {target_role} role",
                "type": "career_milestone",
                "estimated_effort": "Ongoing"
            },
            {
                "date": (current_date + timedelta(weeks=total_learning_weeks + 64)).strftime("%Y-%m-%d"),
                "milestone": "Senior-level role progression",
                "type": "career_milestone",
                "estimated_effort": "1-2 years experience"
            }
        ])

        return {
            "timeline_milestones": timeline_milestones,
            "total_timeline_months": round((total_learning_weeks + 64) / 4, 1),
            "critical_path": [m["milestone"] for m in timeline_milestones if m["type"] == "skill_development"][:3],
            "success_factors": [
                "Consistent learning schedule",
                "Practical project experience",
                "Network building in target industry",
                "Regular skill assessment and adjustment"
            ]
        }

    def get_role_market_outlook(self, target_role: str) -> Dict[str, Any]:
        """Get market outlook for specific role."""

        # Sample market outlook data
        role_outlooks = {
            "Data Scientist": {
                "growth_rate": "22% (Much faster than average)",
                "job_availability": "High",
                "competition_level": "Medium-High",
                "salary_trend": "Increasing",
                "remote_friendly": "Very High",
                "automation_risk": "Low",
                "key_trends": ["AI/ML adoption", "Big data analytics", "Business intelligence"]
            },
            "Full Stack Developer": {
                "growth_rate": "13% (Faster than average)",
                "job_availability": "Very High",
                "competition_level": "High",
                "salary_trend": "Stable",
                "remote_friendly": "Very High",
                "automation_risk": "Medium",
                "key_trends": ["JAMstack architecture", "Microservices", "Cloud-native development"]
            },
            "DevOps Engineer": {
                "growth_rate": "18% (Much faster than average)",
                "job_availability": "High",
                "competition_level": "Medium",
                "salary_trend": "Increasing",
                "remote_friendly": "High",
                "automation_risk": "Low",
                "key_trends": ["Infrastructure as Code", "Container orchestration", "GitOps"]
            }
        }

        return role_outlooks.get(target_role, {
            "growth_rate": "10% (Average)",
            "job_availability": "Medium",
            "competition_level": "Medium",
            "salary_trend": "Stable",
            "remote_friendly": "Medium",
            "automation_risk": "Medium",
            "key_trends": ["Digital transformation", "Cloud adoption", "Skills evolution"]
        })
