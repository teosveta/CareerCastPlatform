"""Data cleaning and preprocessing functions."""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataCleaner:
    """Comprehensive data cleaning and preprocessing utilities."""

    def __init__(self):
        self.location_mappings = self._load_location_mappings()
        self.currency_rates = {"USD": 1.0, "EUR": 1.1, "GBP": 1.25, "CAD": 0.75}  # Simplified rates

    def _load_location_mappings(self) -> Dict[str, str]:
        """Load location standardization mappings."""
        # Common location standardizations
        return {
            # US States
            "CA": "California, US", "NY": "New York, US", "TX": "Texas, US",
            "WA": "Washington, US", "FL": "Florida, US", "IL": "Illinois, US",

            # Major cities
            "NYC": "New York, US", "SF": "San Francisco, US", "LA": "Los Angeles, US",
            "Seattle": "Seattle, US", "Austin": "Austin, US", "Boston": "Boston, US",

            # International
            "London": "London, UK", "Berlin": "Berlin, Germany", "Paris": "Paris, France",
            "Toronto": "Toronto, Canada", "Sydney": "Sydney, Australia",
            "Amsterdam": "Amsterdam, Netherlands", "Stockholm": "Stockholm, Sweden",

            # Remote work indicators
            "Remote": "Remote", "Work from home": "Remote", "WFH": "Remote",
            "Distributed": "Remote", "Virtual": "Remote"
        }

    def clean_job_postings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize job posting data."""
        logger.info("Cleaning job postings data...")

        cleaned_df = df.copy()

        # Standardize column names
        cleaned_df = self._standardize_column_names(cleaned_df)

        # Clean text fields
        text_columns = ["title", "description", "company_name"]
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = self._clean_text_field(cleaned_df[col])

        # Standardize locations
        if "location" in cleaned_df.columns:
            cleaned_df["location_standardized"] = self._standardize_locations(cleaned_df["location"])

        # Clean and standardize posting dates
        if "posted_date" in cleaned_df.columns or "posting_date" in cleaned_df.columns:
            date_col = "posted_date" if "posted_date" in cleaned_df.columns else "posting_date"
            cleaned_df["posting_date_clean"] = self._clean_dates(cleaned_df[date_col])

        # Remove duplicates based on job_id or content similarity
        cleaned_df = self._remove_duplicates(cleaned_df)

        logger.info(f"Cleaned job postings: {len(df)} -> {len(cleaned_df)} rows")

        return cleaned_df

    def clean_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize salary information."""
        logger.info("Cleaning salary data...")

        cleaned_df = df.copy()

        # Identify salary columns
        salary_columns = [col for col in df.columns if any(
            term in col.lower() for term in ["salary", "pay", "wage", "compensation"]
        )]

        for col in salary_columns:
            if col in cleaned_df.columns:
                # Extract numeric values and standardize currency
                cleaned_df[f"{col}_clean"] = self._clean_salary_column(cleaned_df[col])

        # Create standardized salary ranges
        if "min_salary" in cleaned_df.columns and "max_salary" in cleaned_df.columns:
            cleaned_df["salary_range_clean"] = self._create_salary_ranges(
                cleaned_df["min_salary"], cleaned_df["max_salary"]
            )

        return cleaned_df

    def clean_skills_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize skills data."""
        logger.info("Cleaning skills data...")

        cleaned_df = df.copy()

        # Standardize skill names
        if "skill_name" in cleaned_df.columns:
            cleaned_df["skill_name_clean"] = self._standardize_skill_names(cleaned_df["skill_name"])

        if "skill_abr" in cleaned_df.columns:
            cleaned_df["skill_abr_clean"] = self._standardize_skill_names(cleaned_df["skill_abr"])

        return cleaned_df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across datasets."""
        column_mappings = {
            # Common variations
            "job_title": "title",
            "job_description": "description",
            "company": "company_name",
            "posted_date": "posting_date",
            "min_salary": "salary_min",
            "max_salary": "salary_max",
            "med_salary": "salary_median",
            "location": "location",
        }

        # Apply mappings
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Convert column names to lowercase with underscores
        df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]

        return df

    def _clean_text_field(self, series: pd.Series) -> pd.Series:
        """Clean text fields by removing extra whitespace and special characters."""
        return series.astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    def _standardize_locations(self, locations: pd.Series) -> pd.Series:
        """Standardize location names using predefined mappings."""
        def standardize_location(location):
            if pd.isna(location):
                return "Unknown"

            location = str(location).strip()

            # Check exact mappings first
            if location in self.location_mappings:
                return self.location_mappings[location]

            # Pattern-based standardization
            location_lower = location.lower()

            # Remote work patterns
            if any(term in location_lower for term in ["remote", "work from home", "virtual"]):
                return "Remote"

            # US state patterns
            us_state_pattern = r"^([A-Z]{2})$|^([A-Z]{2}),?\s*USA?$"
            if re.match(us_state_pattern, location):
                state = re.match(us_state_pattern, location).group(1) or re.match(us_state_pattern, location).group(2)
                return self.location_mappings.get(state, f"{state}, US")

            # City, Country pattern
            if "," in location:
                parts = [part.strip() for part in location.split(",")]
                if len(parts) >= 2:
                    city, country = parts[0], parts[-1]
                    return f"{city}, {country}"

            return location

        return locations.apply(standardize_location)

    def _clean_dates(self, dates: pd.Series) -> pd.Series:
        """Clean and standardize date formats."""
        return pd.to_datetime(dates, errors='coerce', infer_datetime_format=True)

    def _clean_salary_column(self, salary_series: pd.Series) -> pd.Series:
        """Extract and standardize salary values."""
        def extract_salary(salary_str):
            if pd.isna(salary_str):
                return np.nan

            salary_str = str(salary_str).replace(",", "").replace("$", "")

            # Extract numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', salary_str)
            if not numbers:
                return np.nan

            # Take the largest number (usually the main salary figure)
            return float(max(numbers, key=float))

        return salary_series.apply(extract_salary)

    def _create_salary_ranges(self, min_salaries: pd.Series, max_salaries: pd.Series) -> pd.Series:
        """Create standardized salary range strings."""
        def create_range(min_sal, max_sal):
            if pd.isna(min_sal) and pd.isna(max_sal):
                return "Not specified"
            elif pd.isna(min_sal):
                return f"Up to ${int(max_sal):,}"
            elif pd.isna(max_sal):
                return f"${int(min_sal):,}+"
            else:
                return f"${int(min_sal):,} - ${int(max_sal):,}"

        return pd.Series([create_range(min_sal, max_sal)
                         for min_sal, max_sal in zip(min_salaries, max_salaries)])

    def _standardize_skill_names(self, skills: pd.Series) -> pd.Series:
        """Standardize skill names for consistency."""
        skill_mappings = {
            # Programming languages
            "js": "JavaScript", "javascript": "JavaScript",
            "py": "Python", "python": "Python",
            "java": "Java", "c++": "C++", "cpp": "C++",
            "c#": "C#", "csharp": "C#",
            "r": "R", "r-lang": "R",

            # Frameworks
            "react": "React", "reactjs": "React",
            "vue": "Vue.js", "vuejs": "Vue.js",
            "angular": "Angular", "angularjs": "Angular",
            "node": "Node.js", "nodejs": "Node.js",

            # Databases
            "sql": "SQL", "mysql": "MySQL",
            "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
            "mongo": "MongoDB", "mongodb": "MongoDB",

            # Cloud platforms
            "aws": "AWS", "amazon web services": "AWS",
            "gcp": "Google Cloud", "google cloud platform": "Google Cloud",
            "azure": "Microsoft Azure",

            # Tools
            "git": "Git", "github": "GitHub",
            "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes"
        }

        def standardize_skill(skill):
            if pd.isna(skill):
                return "Unknown"

            skill_lower = str(skill).lower().strip()
            return skill_mappings.get(skill_lower, skill.strip())

        return skills.apply(standardize_skill)

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate job postings based on multiple criteria."""
        initial_count = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates()

        # Remove duplicates based on job_id if present
        if "job_id" in df.columns:
            df = df.drop_duplicates(subset=["job_id"])

        # Remove near-duplicates based on title and company
        if "title" in df.columns and "company_name" in df.columns:
            df = df.drop_duplicates(subset=["title", "company_name"])

        logger.info(f"Removed {initial_count - len(df)} duplicate records")

        return df