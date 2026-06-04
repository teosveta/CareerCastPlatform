"""Data ingestion utilities for loading and initial processing of raw datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import config
from src.utils.file_utils import FileHandler
from src.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles loading and initial processing of raw job market datasets."""

    def __init__(self):
        self.data_config = config.get_data_config()
        self.file_handler = FileHandler()
        self.validator = DataValidator()

    def load_linkedin_124k_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load the LinkedIn 124k job postings dataset."""
        logger.info("Loading LinkedIn 124k dataset...")

        base_path = Path(self.data_config.get("processed_data_path", "data/processed")) / "linkedin_124k"

        datasets = {}

        try:
            # Core job postings
            postings_file = base_path / "postings.csv"
            if postings_file.exists():
                datasets["postings"] = self.file_handler.load_csv(postings_file)
                logger.info(f"Loaded {len(datasets['postings'])} job postings")
            else:
                logger.warning(f"Postings file not found: {postings_file}")
                return {}

            # Company information
            files_to_load = {
                "companies": "companies.csv",
                "company_industries": "company_industries.csv",
                "company_specialities": "company_specialities.csv",
                "employee_counts": "employee_counts.csv",
                "benefits": "benefits.csv",
                "job_industries": "job_industries.csv",
                "job_skills": "job_skills.csv",
                "salaries": "salaries.csv",
                "industries_mapping": "industries_mapping.csv",
                "skills_mapping": "skills_mapping.csv"
            }

            for key, filename in files_to_load.items():
                file_path = base_path / filename
                if file_path.exists():
                    datasets[key] = self.file_handler.load_csv(file_path)
                    logger.info(f"Loaded {key}: {len(datasets[key])} records")
                else:
                    logger.warning(f"Optional file not found: {file_path}")

            logger.info("Successfully loaded LinkedIn 124k dataset files")

        except Exception as e:
            logger.error(f"Error loading LinkedIn 124k dataset: {e}")
            raise

        return datasets

    def load_linkedin_1_3m_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load the LinkedIn 1.3M job postings dataset."""
        logger.info("Loading LinkedIn 1.3M dataset...")

        base_path = Path(self.data_config.get("processed_data_path", "data/processed")) / "linkedin_1_3m"

        datasets = {}

        try:
            # Check if files exist
            job_postings_file = base_path / "job_postings.csv"
            job_skills_file = base_path / "job_skills.csv"
            job_summary_file = base_path / "job_summary.csv"

            if job_postings_file.exists():
                datasets["job_postings"] = self.file_handler.load_csv(job_postings_file)
                logger.info(f"Loaded job_postings: {len(datasets['job_postings'])} records")

            if job_skills_file.exists():
                datasets["job_skills"] = self.file_handler.load_csv(job_skills_file)
                logger.info(f"Loaded job_skills: {len(datasets['job_skills'])} records")

            if job_summary_file.exists():
                # Load job_summary in chunks due to large size
                logger.info("Loading large job_summary file...")
                try:
                    datasets["job_summary"] = self.file_handler.load_csv(job_summary_file)
                    logger.info(f"Loaded job_summary: {len(datasets['job_summary'])} records")
                except MemoryError:
                    logger.warning("job_summary file too large, loading first 100k records")
                    datasets["job_summary"] = pd.read_csv(job_summary_file, nrows=100000)

            if not datasets:
                logger.warning("No LinkedIn 1.3M dataset files found")

        except Exception as e:
            logger.error(f"Error loading LinkedIn 1.3M dataset: {e}")
            raise

        return datasets

    def load_stackoverflow_dataset(self, years: List[int] = [2022, 2023, 2024]) -> Dict[str, pd.DataFrame]:
        """Load Stack Overflow Developer Survey datasets."""
        logger.info(f"Loading Stack Overflow datasets for years: {years}")

        base_path = Path(self.data_config.get("processed_data_path", "data/processed")) / "stackoverflow"
        datasets = {}

        for year in years:
            try:
                year_file = base_path / f"survey_{year}.csv"
                if year_file.exists():
                    datasets[f"stackoverflow_{year}"] = self.file_handler.load_csv(year_file)
                    logger.info(f"Loaded Stack Overflow {year} dataset: {len(datasets[f'stackoverflow_{year}'])} responses")
                else:
                    logger.warning(f"Stack Overflow {year} dataset not found at {year_file}")
            except Exception as e:
                logger.error(f"Error loading Stack Overflow {year} dataset: {e}")

        return datasets

    def validate_loaded_data(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate loaded datasets for quality and completeness."""
        logger.info("Validating loaded datasets...")

        validation_results = {}

        for name, df in datasets.items():
            logger.info(f"Validating dataset: {name}")

            # Basic data quality check
            quality_report = self.validator.validate_data_quality(df)

            # Dataset-specific validation
            if "posting" in name.lower() or "job" in name.lower():
                job_validation = self.validator.validate_job_posting_data(df)
                validation_results[name] = {
                    "quality": quality_report,
                    "job_specific": job_validation
                }
            elif "salary" in name.lower():
                salary_validation = self.validator.validate_salary_data(df)
                validation_results[name] = {
                    "quality": quality_report,
                    "salary_specific": salary_validation
                }
            else:
                validation_results[name] = {
                    "quality": quality_report
                }

        return validation_results

    def create_unified_job_dataset(self, linkedin_124k: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a unified job dataset from LinkedIn 124k data."""
        logger.info("Creating unified job dataset...")

        if "postings" not in linkedin_124k:
            logger.error("No postings data found in LinkedIn 124k dataset")
            return pd.DataFrame()

        # Start with main postings
        unified_jobs = linkedin_124k["postings"].copy()

        # Add company information
        if "companies" in linkedin_124k:
            unified_jobs = unified_jobs.merge(
                linkedin_124k["companies"],
                on="company_id",
                how="left",
                suffixes=("", "_company")
            )

        # Add salary information
        if "salaries" in linkedin_124k:
            unified_jobs = unified_jobs.merge(
                linkedin_124k["salaries"],
                on="job_id",
                how="left"
            )

        # Add aggregated skills
        if "job_skills" in linkedin_124k:
            skills_agg = linkedin_124k["job_skills"].groupby("job_id")["skill_abr"].apply(
                lambda x: ",".join(x.astype(str))
            ).reset_index()
            skills_agg.columns = ["job_id", "skills_list"]

            unified_jobs = unified_jobs.merge(skills_agg, on="job_id", how="left")

        # Add industry information
        if "job_industries" in linkedin_124k:
            industries_agg = linkedin_124k["job_industries"].groupby("job_id")["industry_id"].apply(
                lambda x: ",".join(x.astype(str))
            ).reset_index()
            industries_agg.columns = ["job_id", "industries_list"]

            unified_jobs = unified_jobs.merge(industries_agg, on="job_id", how="left")

        logger.info(f"Created unified dataset with {len(unified_jobs)} jobs and {len(unified_jobs.columns)} columns")

        return unified_jobs