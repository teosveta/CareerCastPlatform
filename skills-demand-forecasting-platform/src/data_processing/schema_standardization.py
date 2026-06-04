"""Schema standardization utilities for unifying different data sources."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SchemaStandardizer:
    """Standardizes schemas across different data sources."""

    def __init__(self):
        self.unified_job_schema = self._define_unified_job_schema()
        self.unified_skill_schema = self._define_unified_skill_schema()
        self.unified_company_schema = self._define_unified_company_schema()

    def _define_unified_job_schema(self) -> Dict[str, str]:
        """Define the unified job posting schema."""
        return {
            "job_id": "string",
            "title": "string",
            "description": "string",
            "company_name": "string",
            "company_id": "string",
            "location": "string",
            "location_standardized": "string",
            "posting_date": "datetime",
            "salary_min": "float",
            "salary_max": "float",
            "salary_median": "float",
            "salary_currency": "string",
            "experience_level": "string",
            "employment_type": "string",
            "remote_allowed": "boolean",
            "skills_list": "string",
            "industries_list": "string",
            "benefits_list": "string",
            "source_dataset": "string"
        }

    def _define_unified_skill_schema(self) -> Dict[str, str]:
        """Define the unified skills schema."""
        return {
            "skill_id": "string",
            "skill_name": "string",
            "skill_category": "string",
            "skill_type": "string",  # technical, soft, industry
            "frequency": "int",
            "avg_salary_impact": "float",
            "growth_trend": "float",
            "source_dataset": "string"
        }

    def _define_unified_company_schema(self) -> Dict[str, str]:
        """Define the unified company schema."""
        return {
            "company_id": "string",
            "company_name": "string",
            "industry": "string",
            "company_size": "string",
            "location": "string",
            "description": "string",
            "specialties": "string",
            "source_dataset": "string"
        }

    def standardize_linkedin_124k_jobs(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Standardize LinkedIn 124k dataset to unified schema."""
        logger.info("Standardizing LinkedIn 124k jobs to unified schema...")

        # Start with main postings
        jobs = datasets["postings"].copy()

        # Create unified structure
        unified_jobs = pd.DataFrame()

        # Map columns to unified schema
        column_mapping = {
            "job_id": "job_id",
            "title": "title",
            "description": "description",
            "company_id": "company_id",
            "location": "location",
            "posted_date": "posting_date",
            "formatted_work_type": "employment_type",
            "formatted_experience_level": "experience_level"
        }

        # Apply basic mappings
        for old_col, new_col in column_mapping.items():
            if old_col in jobs.columns:
                unified_jobs[new_col] = jobs[old_col]

        # Add company information
        if "companies" in datasets:
            company_info = datasets["companies"][["company_id", "name", "description"]]
            company_info.columns = ["company_id", "company_name", "company_description"]
            unified_jobs = unified_jobs.merge(company_info, on="company_id", how="left")

        # Add salary information
        if "salaries" in datasets:
            salary_info = datasets["salaries"][["job_id", "min_salary", "max_salary", "med_salary", "pay_period"]]
            salary_info.columns = ["job_id", "salary_min", "salary_max", "salary_median", "pay_period"]
            unified_jobs = unified_jobs.merge(salary_info, on="job_id", how="left")

            # Standardize to annual salaries
            unified_jobs = self._standardize_salary_periods(unified_jobs)

        # Add skills as comma-separated list
        if "job_skills" in datasets:
            skills_grouped = datasets["job_skills"].groupby("job_id")["skill_abr"].apply(
                lambda x: ",".join(x.dropna().astype(str))
            ).reset_index()
            skills_grouped.columns = ["job_id", "skills_list"]
            unified_jobs = unified_jobs.merge(skills_grouped, on="job_id", how="left")

        # Add industries as comma-separated list
        if "job_industries" in datasets:
            industries_grouped = datasets["job_industries"].groupby("job_id")["industry_id"].apply(
                lambda x: ",".join(x.dropna().astype(str))
            ).reset_index()
            industries_grouped.columns = ["job_id", "industries_list"]
            unified_jobs = unified_jobs.merge(industries_grouped, on="job_id", how="left")

        # Add benefits as comma-separated list
        if "benefits" in datasets:
            benefits_grouped = datasets["benefits"].groupby("job_id")["type"].apply(
                lambda x: ",".join(x.dropna().astype(str))
            ).reset_index()
            benefits_grouped.columns = ["job_id", "benefits_list"]
            unified_jobs = unified_jobs.merge(benefits_grouped, on="job_id", how="left")

        # Add metadata
        unified_jobs["source_dataset"] = "linkedin_124k"
        unified_jobs["salary_currency"] = "USD"  # Assume USD for LinkedIn data

        # Ensure all schema columns exist
        for col, dtype in self.unified_job_schema.items():
            if col not in unified_jobs.columns:
                if dtype == "string":
                    unified_jobs[col] = ""
                elif dtype == "float":
                    unified_jobs[col] = np.nan
                elif dtype == "int":
                    unified_jobs[col] = 0
                elif dtype == "boolean":
                    unified_jobs[col] = False
                elif dtype == "datetime":
                    unified_jobs[col] = pd.NaT

        logger.info(f"Standardized {len(unified_jobs)} LinkedIn 124k jobs")
        return unified_jobs

    def standardize_linkedin_1_3m_jobs(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Standardize LinkedIn 1.3M dataset to unified schema."""
        logger.info("Standardizing LinkedIn 1.3M jobs to unified schema...")

        # Start with job postings
        jobs = datasets["job_postings"].copy()

        # Create unified structure
        unified_jobs = pd.DataFrame()

        # Map columns to unified schema
        column_mapping = {
            "job_id": "job_id",
            "title": "title",
            "company": "company_name",
            "location": "location",
            "posted_date": "posting_date"
        }

        # Apply basic mappings
        for old_col, new_col in column_mapping.items():
            if old_col in jobs.columns:
                unified_jobs[new_col] = jobs[old_col]

        # Add job descriptions from job_summary
        if "job_summary" in datasets:
            summary_info = datasets["job_summary"][["job_id", "description"]]
            unified_jobs = unified_jobs.merge(summary_info, on="job_id", how="left")

        # Add skills from job_skills
        if "job_skills" in datasets:
            skills_grouped = datasets["job_skills"].groupby("job_id")["skill_name"].apply(
                lambda x: ",".join(x.dropna().astype(str))
            ).reset_index()
            skills_grouped.columns = ["job_id", "skills_list"]
            unified_jobs = unified_jobs.merge(skills_grouped, on="job_id", how="left")

        # Add metadata
        unified_jobs["source_dataset"] = "linkedin_1_3m"
        unified_jobs["salary_currency"] = "USD"  # Assume USD

        # Ensure all schema columns exist
        for col, dtype in self.unified_job_schema.items():
            if col not in unified_jobs.columns:
                if dtype == "string":
                    unified_jobs[col] = ""
                elif dtype == "float":
                    unified_jobs[col] = np.nan
                elif dtype == "int":
                    unified_jobs[col] = 0
                elif dtype == "boolean":
                    unified_jobs[col] = False
                elif dtype == "datetime":
                    unified_jobs[col] = pd.NaT

        logger.info(f"Standardized {len(unified_jobs)} LinkedIn 1.3M jobs")
        return unified_jobs

    def _standardize_salary_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all salaries to annual equivalents."""
        if "pay_period" not in df.columns:
            return df

        # Conversion factors to annual salary
        period_multipliers = {
            "YEARLY": 1,
            "MONTHLY": 12,
            "WEEKLY": 52,
            "DAILY": 252,  # Assuming ~252 working days per year
            "HOURLY": 2080  # Assuming 40 hours/week * 52 weeks
        }

        for col in ["salary_min", "salary_max", "salary_median"]:
            if col in df.columns:
                df[f"{col}_annual"] = df.apply(
                    lambda row: row[col] * period_multipliers.get(row["pay_period"], 1)
                    if pd.notna(row[col]) and pd.notna(row["pay_period"]) else row[col],
                    axis=1
                )

        return df

    def create_unified_skills_dataset(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create unified skills dataset from multiple sources."""
        logger.info("Creating unified skills dataset...")

        unified_skills = []

        # Process LinkedIn 124k skills
        if "job_skills" in datasets:
            skills_124k = datasets["job_skills"].copy()
            if "skills_mapping" in datasets:
                skills_124k = skills_124k.merge(
                    datasets["skills_mapping"],
                    left_on="skill_abr",
                    right_on="skill_abr",
                    how="left"
                )

            skills_124k["source_dataset"] = "linkedin_124k"
            unified_skills.append(skills_124k)

        # Combine all skills datasets
        if unified_skills:
            combined_skills = pd.concat(unified_skills, ignore_index=True)

            # Aggregate by skill name
            skill_summary = combined_skills.groupby("skill_name").agg({
                "job_id": "count",
                "source_dataset": "first"
            }).reset_index()

            skill_summary.columns = ["skill_name", "frequency", "source_dataset"]
            skill_summary["skill_id"] = range(len(skill_summary))

            logger.info(f"Created unified skills dataset with {len(skill_summary)} unique skills")
            return skill_summary

        return pd.DataFrame()