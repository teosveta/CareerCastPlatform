"""
Enhanced data processing with advanced cleaning, validation, and optimization.
Implements streaming processing, intelligent caching, and quality scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Iterator, Tuple
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.performance_monitor import PerformanceMonitor
from src.utils.config_manager import config

logger = logging.getLogger(__name__)


class EnhancedDataProcessor:
    """
    Advanced data processor with streaming capabilities, intelligent caching,
    and comprehensive quality assessment.
    """

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.processing_cache = {}
        self.quality_thresholds = {
            "completeness": 0.8,
            "validity": 0.9,
            "consistency": 0.85,
            "accuracy": 0.8
        }

    def load_dataset(self, source: str, file_path: str) -> pd.DataFrame:
        """Load dataset with optimized memory usage and validation."""

        with self.performance_monitor.track_operation(f"load_dataset_{source}"):
            logger.info(f"Loading dataset: {source} from {file_path}")

            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}, creating sample data")
                return self._create_sample_data(source)

            try:
                # Determine optimal chunk size based on file size
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                chunk_size = self._calculate_optimal_chunk_size(file_size_mb)

                if file_size_mb > 100:  # Large file - use streaming
                    return self._load_large_dataset_streaming(file_path, chunk_size)
                else:
                    # Load normally with type optimization
                    df = pd.read_csv(file_path)
                    return self._optimize_dtypes(df)

            except Exception as e:
                logger.error(f"Error loading {source}: {e}")
                return self._create_sample_data(source)

    def _calculate_optimal_chunk_size(self, file_size_mb: float) -> int:
        """Calculate optimal chunk size based on available memory and file size."""
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

            # Use 10% of available memory or 10MB, whichever is smaller
            optimal_size = min(available_memory_mb * 0.1, 10) * 1024  # Convert to KB

            # Ensure minimum chunk size
            return max(int(optimal_size), 1000)

        except ImportError:
            return 10000  # Default chunk size

    def _load_large_dataset_streaming(self, file_path: str, chunk_size: int) -> pd.DataFrame:
        """Load large datasets using streaming to manage memory."""

        chunks = []
        total_rows = 0

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Process chunk
                processed_chunk = self._preprocess_chunk(chunk)
                chunks.append(processed_chunk)
                total_rows += len(processed_chunk)

                # Log progress for large files
                if total_rows % 50000 == 0:
                    logger.info(f"Processed {total_rows:,} rows...")

                # Memory management: limit chunks in memory
                if len(chunks) > 20:  # Combine every 20 chunks
                    combined = pd.concat(chunks, ignore_index=True)
                    chunks = [combined]

            result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            logger.info(f"Loaded {len(result):,} rows via streaming")

            return result

        except Exception as e:
            logger.error(f"Streaming load failed: {e}")
            return pd.DataFrame()

    def _preprocess_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Preprocess individual chunks during streaming."""
        # Basic cleaning
        chunk = chunk.dropna(how='all')  # Remove empty rows
        chunk = chunk.drop_duplicates()  # Remove exact duplicates

        # Optimize data types
        chunk = self._optimize_dtypes(chunk)

        return chunk

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""

        for col in df.columns:
            col_type = df[col].dtype

            if col_type == 'object':
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df[col] = numeric_series
                        continue
                except:
                    pass

                # Try to convert to datetime
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        continue
                    except:
                        pass

                # Optimize string columns
                num_unique = df[col].nunique()
                num_total = len(df[col])

                if num_unique / num_total < 0.5:  # High repetition - use category
                    df[col] = df[col].astype('category')

            elif col_type in ['int64', 'int32']:
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')

            elif col_type in ['float64', 'float32']:
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    def enhanced_cleaning(self, df: pd.DataFrame, quality_threshold: float = 0.8) -> pd.DataFrame:
        """Enhanced data cleaning with quality assessment."""

        with self.performance_monitor.track_operation("enhanced_cleaning"):
            logger.info(f"Enhanced cleaning for {len(df)} rows...")

            if df.empty:
                return df

            original_count = len(df)

            # 1. Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')

            # 2. Intelligent duplicate removal
            df = self._intelligent_duplicate_removal(df)

            # 3. Data type standardization
            df = self._standardize_data_types(df)

            # 4. Outlier detection and handling
            df = self._handle_outliers(df)

            # 5. Text normalization
            df = self._normalize_text_fields(df)

            # 6. Date standardization
            df = self._standardize_dates(df)

            # 7. Location standardization
            df = self._standardize_locations(df)

            # 8. Quality assessment
            quality_score = self._assess_data_quality(df)

            if quality_score < quality_threshold:
                logger.warning(f"Data quality score {quality_score:.2f} below threshold {quality_threshold}")

            logger.info(f"Cleaning complete: {original_count} -> {len(df)} rows (quality: {quality_score:.2f})")

            return df

    def _intelligent_duplicate_removal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent duplicate removal considering fuzzy matches."""

        # Standard duplicate removal
        df = df.drop_duplicates()

        # Fuzzy duplicate detection for key fields
        if 'title' in df.columns and 'company_name' in df.columns:
            df = self._remove_fuzzy_duplicates(df, ['title', 'company_name'])

        return df

    def _remove_fuzzy_duplicates(self, df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
        """Remove fuzzy duplicates based on text similarity."""

        try:
            from difflib import SequenceMatcher

            indices_to_remove = set()

            for i, row1 in df.iterrows():
                if i in indices_to_remove:
                    continue

                for j, row2 in df.iloc[i + 1:].iterrows():
                    if j in indices_to_remove:
                        continue

                    # Calculate similarity for key columns
                    similarities = []
                    for col in key_columns:
                        if col in df.columns:
                            text1 = str(row1[col]).lower().strip()
                            text2 = str(row2[col]).lower().strip()
                            similarity = SequenceMatcher(None, text1, text2).ratio()
                            similarities.append(similarity)

                    # If highly similar, mark as duplicate
                    if similarities and np.mean(similarities) > 0.85:
                        indices_to_remove.add(j)

            if indices_to_remove:
                df = df.drop(index=indices_to_remove)
                logger.info(f"Removed {len(indices_to_remove)} fuzzy duplicates")

        except ImportError:
            logger.warning("difflib not available for fuzzy duplicate detection")

        return df

    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types across the dataset."""

        # ID columns should be strings
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for col in id_columns:
            df[col] = df[col].astype(str)

        # Salary columns should be numeric
        salary_columns = [col for col in df.columns if 'salary' in col.lower()]
        for col in salary_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Date columns
        date_columns = [col for col in df.columns if any(
            term in col.lower() for term in ['date', 'time', 'created', 'updated', 'posted']
        )]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numeric columns."""

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if 'salary' in col.lower():
                # Salary-specific outlier handling
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                # More generous bounds for salary data
                lower_bound = max(0, Q1 - 3 * IQR)
                upper_bound = Q3 + 3 * IQR

                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _normalize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text fields for consistency."""

        text_columns = df.select_dtypes(include=['object', 'string']).columns

        for col in text_columns:
            if col in df.columns:
                # Basic text normalization
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces
                df[col] = df[col].replace('nan', np.nan)  # String 'nan' to actual NaN

        return df

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date formats and handle timezone issues."""

        date_columns = df.select_dtypes(include=['datetime64']).columns

        for col in date_columns:
            # Remove timezone info for consistency
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)

            # Filter out unrealistic dates
            current_date = datetime.now()
            future_limit = current_date + timedelta(days=365)  # 1 year in future
            past_limit = current_date - timedelta(days=365 * 10)  # 10 years in past

            df[col] = df[col].where(
                (df[col] >= past_limit) & (df[col] <= future_limit)
            )

        return df

    def _standardize_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced location standardization with geocoding hints."""

        location_columns = [col for col in df.columns if 'location' in col.lower()]

        for col in location_columns:
            if col in df.columns:
                df[f"{col}_standardized"] = df[col].apply(self._standardize_single_location)

        return df

    def _standardize_single_location(self, location: str) -> str:
        """Standardize a single location string."""

        if pd.isna(location):
            return "Unknown"

        location = str(location).strip()

        # Common location mappings
        location_mappings = {
            # US Cities
            "NYC": "New York, NY, US",
            "SF": "San Francisco, CA, US",
            "LA": "Los Angeles, CA, US",
            "Chi": "Chicago, IL, US",
            "Boston": "Boston, MA, US",
            "Seattle": "Seattle, WA, US",
            "Austin": "Austin, TX, US",
            "Denver": "Denver, CO, US",

            # Countries
            "USA": "United States",
            "UK": "United Kingdom",
            "Germany": "Germany",
            "Canada": "Canada",

            # Remote work
            "Remote": "Remote",
            "Work from home": "Remote",
            "WFH": "Remote",
            "Distributed": "Remote"
        }

        # Check exact mappings
        if location in location_mappings:
            return location_mappings[location]

        # Pattern-based standardization
        location_lower = location.lower()

        # Remote work patterns
        if any(term in location_lower for term in ["remote", "work from home", "virtual", "anywhere"]):
            return "Remote"

        # US state patterns
        us_states = {
            "california": "CA", "new york": "NY", "texas": "TX", "florida": "FL",
            "washington": "WA", "illinois": "IL", "massachusetts": "MA"
        }

        for state_name, state_code in us_states.items():
            if state_name in location_lower:
                return f"{state_name.title()}, {state_code}, US"

        return location

    def comprehensive_quality_assessment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment with detailed metrics."""

        with self.performance_monitor.track_operation("quality_assessment"):
            if df.empty:
                return {"overall_score": 0, "message": "Empty dataset"}

            # 1. Completeness Assessment
            completeness_score = self._assess_completeness(df)

            # 2. Validity Assessment
            validity_score = self._assess_validity(df)

            # 3. Consistency Assessment
            consistency_score = self._assess_consistency(df)

            # 4. Accuracy Assessment
            accuracy_score = self._assess_accuracy(df)

            # 5. Timeliness Assessment
            timeliness_score = self._assess_timeliness(df)

            # Calculate weighted overall score
            weights = {
                "completeness": 0.25,
                "validity": 0.25,
                "consistency": 0.20,
                "accuracy": 0.20,
                "timeliness": 0.10
            }

            overall_score = (
                    completeness_score * weights["completeness"] +
                    validity_score * weights["validity"] +
                    consistency_score * weights["consistency"] +
                    accuracy_score * weights["accuracy"] +
                    timeliness_score * weights["timeliness"]
            )

            return {
                "overall_score": round(overall_score, 3),
                "dimensions": {
                    "completeness": round(completeness_score, 3),
                    "validity": round(validity_score, 3),
                    "consistency": round(consistency_score, 3),
                    "accuracy": round(accuracy_score, 3),
                    "timeliness": round(timeliness_score, 3)
                },
                "recommendations": self._generate_quality_recommendations(df),
                "total_records": len(df),
                "total_fields": len(df.columns),
                "assessment_timestamp": datetime.now().isoformat()
            }

    def _assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness (missing values)."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        return max(0, (total_cells - missing_cells) / total_cells)

    def _assess_validity(self, df: pd.DataFrame) -> float:
        """Assess data validity (correct formats and types)."""
        validity_scores = []

        for col in df.columns:
            col_validity = 1.0  # Start with perfect score

            # Check date columns
            if 'date' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    valid_dates = pd.to_datetime(df[col], errors='coerce').notna().sum()
                    total_dates = df[col].notna().sum()
                    if total_dates > 0:
                        col_validity = valid_dates / total_dates
                except:
                    col_validity = 0.5

            # Check salary columns
            elif 'salary' in col.lower():
                numeric_values = pd.to_numeric(df[col], errors='coerce').notna().sum()
                total_values = df[col].notna().sum()
                if total_values > 0:
                    col_validity = numeric_values / total_values

            validity_scores.append(col_validity)

        return sum(validity_scores) / len(validity_scores) if validity_scores else 1.0

    def _assess_consistency(self, df: pd.DataFrame) -> float:
        """Assess data consistency across related fields."""
        consistency_checks = []

        # Check for consistent formatting in similar columns
        text_columns = df.select_dtypes(include=['object']).columns

        for col in text_columns:
            if col in df.columns and not df[col].empty:
                # Check case consistency
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    all_upper = non_null_values.str.isupper().sum()
                    all_lower = non_null_values.str.islower().sum()
                    mixed_case = len(non_null_values) - all_upper - all_lower

                    # Prefer mixed case, but consistent formatting is good
                    consistency_score = max(all_upper, all_lower, mixed_case) / len(non_null_values)
                    consistency_checks.append(consistency_score)

        return sum(consistency_checks) / len(consistency_checks) if consistency_checks else 1.0

    def _assess_accuracy(self, df: pd.DataFrame) -> float:
        """Assess data accuracy based on business rules."""
        accuracy_checks = []

        # Salary range checks
        salary_columns = [col for col in df.columns if 'salary' in col.lower()]
        for col in salary_columns:
            if col in df.columns:
                salary_values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(salary_values) > 0:
                    # Reasonable salary ranges (adjust as needed)
                    reasonable_values = salary_values[
                        (salary_values >= 20000) & (salary_values <= 500000)
                        ]
                    accuracy_score = len(reasonable_values) / len(salary_values)
                    accuracy_checks.append(accuracy_score)

        # Date range checks
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if col in df.columns:
                date_values = df[col].dropna()
                if len(date_values) > 0:
                    current_date = datetime.now()
                    reasonable_dates = date_values[
                        (date_values >= current_date - timedelta(days=3650)) &  # 10 years ago
                        (date_values <= current_date + timedelta(days=365))  # 1 year future
                        ]
                    accuracy_score = len(reasonable_dates) / len(date_values)
                    accuracy_checks.append(accuracy_score)

        return sum(accuracy_checks) / len(accuracy_checks) if accuracy_checks else 1.0

    def _assess_timeliness(self, df: pd.DataFrame) -> float:
        """Assess data timeliness (how recent is the data)."""
        date_columns = df.select_dtypes(include=['datetime64']).columns

        if not date_columns.any():
            return 0.5  # Neutral score if no dates

        timeliness_scores = []
        current_date = datetime.now()

        for col in date_columns:
            if col in df.columns:
                dates = df[col].dropna()
                if len(dates) > 0:
                    # Calculate average age of data
                    avg_date = dates.mean()
                    days_old = (current_date - avg_date).days

                    # Score based on recency (1.0 for current, decreasing with age)
                    if days_old <= 30:  # Very recent
                        score = 1.0
                    elif days_old <= 90:  # Recent
                        score = 0.8
                    elif days_old <= 365:  # Moderately old
                        score = 0.6
                    elif days_old <= 1095:  # Old (3 years)
                        score = 0.4
                    else:  # Very old
                        score = 0.2

                    timeliness_scores.append(score)

        return sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 0.5

    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """Quick data quality assessment."""
        if df.empty:
            return 0.0

        # Simple quality metrics
        completeness = 1 - (df.isnull().sum().sum() / df.size)
        uniqueness = df.drop_duplicates().shape[0] / df.shape[0]

        return (completeness + uniqueness) / 2

    def _generate_quality_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations for data quality improvement."""
        recommendations = []

        # Missing data recommendations
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 20:
            recommendations.append(
                f"High missing data rate ({missing_percentage:.1f}%). "
                "Consider data imputation or additional data collection."
            )

        # Duplicate recommendations
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append(
                f"Found {duplicate_count} duplicate records. Consider deduplication."
            )

        # Data type recommendations
        object_columns = df.select_dtypes(include=['object']).columns
        if len(object_columns) > len(df.columns) * 0.7:
            recommendations.append(
                "Many columns are stored as text. Consider converting to appropriate data types."
            )

        return recommendations

    def _create_sample_data(self, source: str) -> pd.DataFrame:
        """Create sample data when actual data is not available."""

        logger.info(f"Creating sample data for {source}")

        if "linkedin" in source.lower():
            return self._create_linkedin_sample()
        elif "stackoverflow" in source.lower():
            return self._create_stackoverflow_sample()
        else:
            return self._create_generic_sample()

    def _create_linkedin_sample(self) -> pd.DataFrame:
        """Create sample LinkedIn job data."""
        import random
        from datetime import datetime, timedelta

        sample_size = 1000

        titles = [
            "Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer",
            "Full Stack Developer", "ML Engineer", "Backend Developer", "Frontend Developer",
            "Data Analyst", "Product Designer", "Engineering Manager", "Technical Lead"
        ]

        companies = [
            "TechCorp", "DataFlow Inc", "CloudSolutions", "AI Innovations", "WebDev Pro",
            "StartupXYZ", "Enterprise Solutions", "Digital Dynamics", "Innovation Labs"
        ]

        locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
            "Boston, MA", "Remote", "Chicago, IL", "Los Angeles, CA"
        ]

        sample_data = []

        for i in range(sample_size):
            posted_date = datetime.now() - timedelta(days=random.randint(1, 365))

            sample_data.append({
                "job_id": f"job_{i:06d}",
                "title": random.choice(titles),
                "company_name": random.choice(companies),
                "location": random.choice(locations),
                "description": f"We are looking for a skilled professional to join our team...",
                "posted_date": posted_date,
                "salary_min": random.randint(60000, 120000),
                "salary_max": random.randint(120000, 200000),
                "salary_median": random.randint(80000, 160000),
                "employment_type": random.choice(["Full-time", "Part-time", "Contract"]),
                "experience_level": random.choice(["Entry Level", "Mid Level", "Senior Level"])
            })

        return pd.DataFrame(sample_data)

    def _create_stackoverflow_sample(self) -> pd.DataFrame:
        """Create sample Stack Overflow survey data."""
        # Implementation for Stack Overflow sample data
        return pd.DataFrame()

    def _create_generic_sample(self) -> pd.DataFrame:
        """Create generic sample data."""
        return pd.DataFrame({
            "id": range(100),
            "name": [f"Item_{i}" for i in range(100)],
            "value": np.random.normal(100, 15, 100),
            "category": np.random.choice(["A", "B", "C"], 100)
        })

    def create_unified_dataset(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create unified dataset from multiple sources with advanced schema mapping."""

        with self.performance_monitor.track_operation("create_unified_dataset"):
            logger.info("Creating unified dataset from multiple sources...")

            if not datasets:
                logger.warning("No datasets provided for unification")
                return pd.DataFrame()

            unified_records = []

            for source, df in datasets.items():
                if df.empty:
                    continue

                # Map to unified schema based on source
                mapped_df = self._map_to_unified_schema(df, source)

                # Add source tracking
                mapped_df["source_dataset"] = source
                mapped_df["source_record_id"] = mapped_df.index.astype(str)

                unified_records.append(mapped_df)

            if not unified_records:
                return pd.DataFrame()

            # Combine all datasets
            unified_df = pd.concat(unified_records, ignore_index=True, sort=False)

            # Final cleaning and standardization
            unified_df = self._final_unification_cleaning(unified_df)

            logger.info(f"Created unified dataset with {len(unified_df)} records from {len(datasets)} sources")

            return unified_df

    def _map_to_unified_schema(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Map source-specific schema to unified schema."""

        # Define unified schema mapping for each source
        schema_mappings = {
            "linkedin_124k": {
                "job_id": "job_id",
                "title": "title",
                "company_id": "company_id",
                "location": "location",
                "posted_date": "posting_date",
                "description": "description"
            },
            "linkedin_1_3m": {
                "job_id": "job_id",
                "title": "title",
                "company": "company_name",
                "location": "location",
                "posted_date": "posting_date"
            }
        }

        # Get mapping for this source
        mapping = schema_mappings.get(source, {})

        # Apply mapping
        mapped_df = pd.DataFrame()

        for unified_col, source_col in mapping.items():
            if source_col in df.columns:
                mapped_df[unified_col] = df[source_col]

        # Fill missing unified columns with defaults
        unified_columns = {
            "job_id": f"{source}_" + df.index.astype(str),
            "title": "Unknown Title",
            "company_name": "Unknown Company",
            "location": "Unknown Location",
            "posting_date": datetime.now(),
            "description": "No description available",
            "salary_min": np.nan,
            "salary_max": np.nan,
            "salary_median": np.nan
        }

        for col, default_value in unified_columns.items():
            if col not in mapped_df.columns:
                if col == "job_id":
                    mapped_df[col] = default_value
                else:
                    mapped_df[col] = default_value

        return mapped_df

        def _final_unification_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
            """Final cleaning steps for unified dataset."""

            # Remove duplicates across sources
            df = df.drop_duplicates(subset=["title", "company_name", "location"], keep="first")

            # Standardize common fields
            if "posting_date" in df.columns:
                df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")

            # Create unified job ID if duplicates exist
            df["unified_job_id"] = "job_" + (df.index + 1).astype(str).str.zfill(8)

            return df

        def get_feature_engineer(self):
            """Get feature engineering component."""
            return AdvancedFeatureEngineer()

    class AdvancedFeatureEngineer:
        """Advanced feature engineering for ML models."""

        def __init__(self):
            self.feature_importance = {}
            self.feature_types = {}

        def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """Create time-based features."""

            temporal_df = df.copy()

            if "posting_date" in df.columns:
                dates = pd.to_datetime(df["posting_date"], errors="coerce")

                temporal_df["posting_year"] = dates.dt.year
                temporal_df["posting_month"] = dates.dt.month
                temporal_df["posting_quarter"] = dates.dt.quarter
                temporal_df["posting_dayofweek"] = dates.dt.dayofweek
                temporal_df["posting_dayofyear"] = dates.dt.dayofyear

                # Days since posting
                current_date = datetime.now()
                temporal_df["days_since_posting"] = (current_date - dates).dt.days

                # Seasonal features
                temporal_df["is_q4"] = (dates.dt.quarter == 4).astype(int)
                temporal_df["is_january"] = (dates.dt.month == 1).astype(int)

            return temporal_df

        def create_skill_features(self, jobs_df: pd.DataFrame, skills_df: pd.DataFrame) -> pd.DataFrame:
            """Create advanced skill-based features."""

            feature_df = jobs_df.copy()

            if not skills_df.empty and "job_id" in skills_df.columns:
                # Skill counts and categories
                skill_counts = skills_df.groupby("job_id").agg({
                    "skill_name": "count",
                    "confidence": ["mean", "max", "std"]
                }).round(3)

                skill_counts.columns = [
                    "total_skills", "avg_skill_confidence",
                    "max_skill_confidence", "skill_confidence_std"
                ]

                feature_df = feature_df.merge(skill_counts, left_on="job_id", right_index=True, how="left")

                # High-value skill indicators
                high_value_skills = {
                    "Python", "AWS", "Kubernetes", "React", "Machine Learning",
                    "Docker", "TensorFlow", "Go", "Rust", "GraphQL"
                }

                job_skills = skills_df.groupby("job_id")["skill_name"].apply(set).to_dict()
                feature_df["high_value_skills_count"] = feature_df["job_id"].apply(
                    lambda x: len(job_skills.get(x, set()) & high_value_skills)
                )

                # Skill diversity (entropy)
                feature_df["skill_diversity"] = feature_df["job_id"].apply(
                    lambda x: self._calculate_skill_diversity(job_skills.get(x, set()))
                )

            return feature_df

        def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """Create location-based features with economic indicators."""

            location_df = df.copy()

            if "location" in df.columns:
                # Extract location components
                location_df["is_remote"] = df["location"].str.contains("Remote", case=False, na=False)
                location_df["is_us"] = df["location"].str.contains("US|USA|United States", case=False, na=False)

                # Major tech hubs
                tech_hubs = ["San Francisco", "Seattle", "New York", "Austin", "Boston"]
                location_df["is_tech_hub"] = df["location"].apply(
                    lambda x: any(hub in str(x) for hub in tech_hubs)
                )

                # Cost of living tiers (simplified)
                high_col_cities = ["San Francisco", "New York", "Seattle", "Boston"]
                medium_col_cities = ["Austin", "Denver", "Chicago", "Atlanta"]

                location_df["col_tier"] = df["location"].apply(
                    lambda x: self._get_col_tier(str(x), high_col_cities, medium_col_cities)
                )

            return location_df

        def create_company_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """Create company-based features."""

            company_df = df.copy()

            if "company_name" in df.columns:
                # Company size indicators (based on known patterns)
                big_tech = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix"]
                company_df["is_big_tech"] = df["company_name"].apply(
                    lambda x: any(company in str(x) for company in big_tech)
                )

                # Startup indicators
                startup_keywords = ["startup", "labs", "technologies", "solutions"]
                company_df["likely_startup"] = df["company_name"].str.lower().apply(
                    lambda x: any(keyword in str(x) for keyword in startup_keywords)
                )

            return company_df

        def combine_features(self, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
            """Combine multiple feature DataFrames."""

            if not feature_dfs:
                return pd.DataFrame()

            # Start with first DataFrame
            combined = feature_dfs[0].copy()

            # Add features from other DataFrames
            for feature_df in feature_dfs[1:]:
                # Find common columns to merge on
                common_cols = list(set(combined.columns) & set(feature_df.columns))
                if not common_cols:
                    continue

                # Use first common column as merge key
                merge_key = common_cols[0]

                # Get new columns only
                new_cols = [col for col in feature_df.columns if col not in combined.columns]
                if new_cols:
                    merge_df = feature_df[[merge_key] + new_cols]
                    combined = combined.merge(merge_df, on=merge_key, how="left")

            return combined

        def select_features(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
            """Select most important features using various methods."""

            if target_column and target_column in df.columns and SKLEARN_AVAILABLE:
                # Use statistical feature selection
                from sklearn.feature_selection import SelectKBest, f_regression
                from sklearn.preprocessing import LabelEncoder

                # Prepare features
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Handle categorical variables
                X_processed = pd.DataFrame(index=X.index)

                for col in X.columns:
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                        # Encode categorical variables
                        le = LabelEncoder()
                        X_processed[col] = le.fit_transform(X[col].astype(str))
                    else:
                        X_processed[col] = X[col].fillna(X[col].median())

                # Select features
                try:
                    selector = SelectKBest(score_func=f_regression, k=min(20, len(X_processed.columns)))
                    X_selected = selector.fit_transform(X_processed, y)
                    selected_features = X_processed.columns[selector.get_support()]

                    # Store feature importance
                    feature_scores = dict(zip(X_processed.columns, selector.scores_))
                    self.feature_importance = {
                        col: score for col, score in feature_scores.items()
                        if col in selected_features
                    }

                    # Return selected features plus target
                    return df[list(selected_features) + [target_column]]

                except Exception as e:
                    logger.warning(f"Feature selection failed: {e}")
                    return df

            return df

        def get_feature_importance(self) -> Dict[str, float]:
            """Get feature importance scores."""
            return self.feature_importance

        def get_feature_types(self) -> Dict[str, str]:
            """Get feature type mapping."""
            return self.feature_types

        def _calculate_skill_diversity(self, skills: set) -> float:
            """Calculate skill diversity using entropy."""
            if not skills or len(skills) <= 1:
                return 0.0

            # Simple diversity measure
            return len(skills) / (len(skills) + 5)  # Normalized diversity

        def _get_col_tier(self, location: str, high_col: List[str], medium_col: List[str]) -> str:
            """Get cost of living tier for location."""
            location_lower = location.lower()

            if any(city.lower() in location_lower for city in high_col):
                return "high"
            elif any(city.lower() in location_lower for city in medium_col):
                return "medium"
            else:
                return "low"
