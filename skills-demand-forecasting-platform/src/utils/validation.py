"""Data validation utilities for the Skills Demand Forecasting Platform."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation utilities."""

    @staticmethod
    def validate_dataframe_schema(
        df: pd.DataFrame,
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate DataFrame schema against required and optional columns."""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }

        # Check required columns
        missing_required = set(required_columns) - set(df.columns)
        if missing_required:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Missing required columns: {list(missing_required)}"
            )

        # Check for unexpected columns
        expected_columns = set(required_columns + (optional_columns or []))
        unexpected_columns = set(df.columns) - expected_columns
        if unexpected_columns:
            validation_result["warnings"].append(
                f"Unexpected columns found: {list(unexpected_columns)}"
            )

        validation_result["summary"] = {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "required_columns_present": len(set(required_columns) & set(df.columns)),
            "missing_required": list(missing_required)
        }

        return validation_result

    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_data": {},
            "data_types": {},
            "duplicates": 0,
            "quality_score": 0.0
        }

        # Missing data analysis
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            quality_report["missing_data"][column] = {
                "count": int(missing_count),
                "percentage": round(missing_percentage, 2)
            }

        # Data types
        quality_report["data_types"] = df.dtypes.astype(str).to_dict()

        # Duplicates
        quality_report["duplicates"] = int(df.duplicated().sum())

        # Calculate quality score (0-100)
        total_missing = sum(info["count"] for info in quality_report["missing_data"].values())
        total_cells = len(df) * len(df.columns)
        duplicate_penalty = quality_report["duplicates"] / len(df) if len(df) > 0 else 0

        quality_score = max(0, (1 - (total_missing / total_cells) - duplicate_penalty) * 100)
        quality_report["quality_score"] = round(quality_score, 2)

        return quality_report

    @staticmethod
    def validate_job_posting_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Specific validation for job posting data."""
        required_columns = ["job_id", "title", "company"]
        optional_columns = ["description", "location", "salary", "posting_date", "skills"]

        # Basic schema validation
        schema_validation = DataValidator.validate_dataframe_schema(
            df, required_columns, optional_columns
        )

        # Job-specific validations
        job_validation = {
            "schema": schema_validation,
            "business_rules": []
        }

        # Validate job IDs are unique
        if "job_id" in df.columns:
            duplicate_jobs = df["job_id"].duplicated().sum()
            if duplicate_jobs > 0:
                job_validation["business_rules"].append({
                    "rule": "unique_job_ids",
                    "status": "failed",
                    "message": f"Found {duplicate_jobs} duplicate job IDs"
                })
            else:
                job_validation["business_rules"].append({
                    "rule": "unique_job_ids",
                    "status": "passed",
                    "message": "All job IDs are unique"
                })

        # Validate posting dates if present
        if "posting_date" in df.columns:
            try:
                pd.to_datetime(df["posting_date"], errors="coerce")
                invalid_dates = df["posting_date"].isnull().sum()
                job_validation["business_rules"].append({
                    "rule": "valid_posting_dates",
                    "status": "passed" if invalid_dates == 0 else "warning",
                    "message": f"Found {invalid_dates} invalid dates" if invalid_dates > 0 else "All dates valid"
                })
            except Exception as e:
                job_validation["business_rules"].append({
                    "rule": "valid_posting_dates",
                    "status": "failed",
                    "message": f"Date validation failed: {str(e)}"
                })

        return job_validation

    @staticmethod
    def validate_salary_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate salary data for consistency and reasonableness."""
        salary_validation = {
            "valid_ranges": True,
            "issues": [],
            "statistics": {}
        }

        salary_columns = [col for col in df.columns if 'salary' in col.lower()]

        for col in salary_columns:
            if col in df.columns:
                salary_series = pd.to_numeric(df[col], errors='coerce')

                # Basic statistics
                salary_validation["statistics"][col] = {
                    "min": float(salary_series.min()) if not salary_series.empty else None,
                    "max": float(salary_series.max()) if not salary_series.empty else None,
                    "median": float(salary_series.median()) if not salary_series.empty else None,
                    "mean": float(salary_series.mean()) if not salary_series.empty else None
                }

                # Reasonable range checks (assuming USD)
                unreasonably_low = (salary_series < 10000).sum()
                unreasonably_high = (salary_series > 1000000).sum()

                if unreasonably_low > 0:
                    salary_validation["issues"].append(
                        f"{col}: {unreasonably_low} values below $10,000"
                    )

                if unreasonably_high > 0:
                    salary_validation["issues"].append(
                        f"{col}: {unreasonably_high} values above $1,000,000"
                    )

        salary_validation["valid_ranges"] = len(salary_validation["issues"]) == 0

        return salary_validation