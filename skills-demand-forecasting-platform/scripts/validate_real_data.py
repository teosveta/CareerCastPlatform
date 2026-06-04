# File: scripts/validate_real_data.py (NEW - REAL DATA VALIDATION)
"""
Comprehensive validation script to ensure the platform works with real data.
Tests all components with actual datasets and validates outputs.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.platform_orchestrator import PlatformOrchestrator, PipelineConfig
from src.data_processing.enhanced_data_processor import EnhancedDataProcessor
from src.nlp.skill_extractor import SkillExtractor
from src.ml_models.ensemble_forecaster import EnsembleForecaster
from src.ml_models.regression.salary_predictor import SalaryPredictor
from src.utils.validation import DataValidator
from src.utils.performance_monitor import PerformanceMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealDataValidator:
    """
    Comprehensive validator to ensure the platform works with real data.
    Tests data loading, processing, ML models, and API endpoints.
    """

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.validation_results = {}
        self.test_results = []

    def validate_data_paths(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Validate that data paths exist and contain real data."""

        logger.info("🔍 Validating data paths and file existence...")

        validation_results = {
            "files_found": {},
            "file_sizes": {},
            "data_samples": {},
            "issues": []
        }

        for source, path in data_paths.items():
            logger.info(f"Checking {source}: {path}")

            if not Path(path).exists():
                validation_results["issues"].append(f"❌ Path not found: {path}")
                continue

            # Check if it's a directory with CSV files
            if Path(path).is_dir():
                csv_files = list(Path(path).glob("*.csv"))
                validation_results["files_found"][source] = [str(f) for f in csv_files]

                if not csv_files:
                    validation_results["issues"].append(f"❌ No CSV files found in {path}")
                else:
                    logger.info(f"✅ Found {len(csv_files)} CSV files in {source}")

                    # Check file sizes
                    for csv_file in csv_files:
                        size_mb = csv_file.stat().st_size / (1024 * 1024)
                        validation_results["file_sizes"][str(csv_file)] = f"{size_mb:.2f} MB"

                        # Sample data to verify it's not empty
                        try:
                            sample_df = pd.read_csv(csv_file, nrows=5)
                            validation_results["data_samples"][str(csv_file)] = {
                                "columns": list(sample_df.columns),
                                "shape": f"{len(sample_df)} rows × {len(sample_df.columns)} cols",
                                "sample_row": sample_df.iloc[0].to_dict() if not sample_df.empty else {}
                            }
                            logger.info(f"✅ {csv_file.name}: {len(sample_df.columns)} columns, sample data looks good")
                        except Exception as e:
                            validation_results["issues"].append(f"❌ Cannot read {csv_file}: {e}")

            # Check if it's a single CSV file
            elif Path(path).is_file() and path.endswith('.csv'):
                validation_results["files_found"][source] = [path]
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                validation_results["file_sizes"][path] = f"{size_mb:.2f} MB"

                try:
                    sample_df = pd.read_csv(path, nrows=5)
                    validation_results["data_samples"][path] = {
                        "columns": list(sample_df.columns),
                        "shape": f"{len(sample_df)} rows × {len(sample_df.columns)} cols",
                        "sample_row": sample_df.iloc[0].to_dict() if not sample_df.empty else {}
                    }
                    logger.info(f"✅ {Path(path).name}: {len(sample_df.columns)} columns, sample data looks good")
                except Exception as e:
                    validation_results["issues"].append(f"❌ Cannot read {path}: {e}")

        return validation_results

    def test_data_loading(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Test actual data loading with the enhanced data processor."""

        logger.info("📊 Testing data loading with real data...")

        processor = EnhancedDataProcessor()
        loading_results = {
            "datasets_loaded": {},
            "loading_errors": [],
            "data_quality": {},
            "processing_time": {}
        }

        for source, path in data_paths.items():
            try:
                start_time = datetime.now()

                # Load the dataset
                dataset = processor.load_dataset(source, path)

                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()

                if not dataset.empty:
                    loading_results["datasets_loaded"][source] = {
                        "rows": len(dataset),
                        "columns": len(dataset.columns),
                        "column_names": list(dataset.columns),
                        "memory_usage_mb": dataset.memory_usage(deep=True).sum() / (1024 * 1024),
                        "data_types": dataset.dtypes.astype(str).to_dict()
                    }

                    loading_results["processing_time"][source] = f"{processing_time:.2f} seconds"

                    # Test data quality assessment
                    quality_report = processor.comprehensive_quality_assessment(dataset)
                    loading_results["data_quality"][source] = quality_report

                    logger.info(f"✅ {source}: {len(dataset):,} rows loaded successfully")
                    logger.info(f"   Quality Score: {quality_report.get('overall_score', 0):.3f}")

                else:
                    loading_results["loading_errors"].append(f"❌ {source}: Dataset is empty")

            except Exception as e:
                loading_results["loading_errors"].append(f"❌ {source}: {str(e)}")
                logger.error(f"Failed to load {source}: {e}")

        return loading_results

    def test_skill_extraction(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Test skill extraction with real job description data."""

        logger.info("🧠 Testing skill extraction with real data...")

        processor = EnhancedDataProcessor()
        skill_extractor = SkillExtractor()

        extraction_results = {
            "skills_extracted": {},
            "extraction_errors": [],
            "skill_quality": {},
            "sample_extractions": {}
        }

        for source, path in data_paths.items():
            try:
                # Load a sample of data for testing
                dataset = processor.load_dataset(source, path)

                if dataset.empty:
                    continue

                # Look for text columns that might contain job descriptions
                text_columns = []
                for col in dataset.columns:
                    if any(keyword in col.lower() for keyword in ['description', 'summary', 'detail', 'text']):
                        text_columns.append(col)

                if not text_columns:
                    # If no obvious text columns, try any string columns
                    text_columns = dataset.select_dtypes(include=['object']).columns.tolist()

                if text_columns:
                    # Test with a sample of 10 rows
                    sample_data = dataset.head(10).copy()

                    # Use the first text column for extraction
                    text_column = text_columns[0]

                    extracted_skills = skill_extractor.extract_skills_from_dataframe(
                        sample_data,
                        text_column=text_column,
                        method="hybrid"
                    )

                    if not extracted_skills.empty:
                        extraction_results["skills_extracted"][source] = {
                            "total_skill_mentions": len(extracted_skills),
                            "unique_skills": extracted_skills['skill_name'].nunique(),
                            "avg_confidence": extracted_skills['confidence'].mean(),
                            "text_column_used": text_column,
                            "sample_skills": extracted_skills['skill_name'].value_counts().head(10).to_dict()
                        }

                        # Create skill taxonomy from extracted skills
                        skill_taxonomy = skill_extractor.create_skill_taxonomy(extracted_skills)

                        extraction_results["skill_quality"][source] = {
                            "skills_in_taxonomy": len(skill_taxonomy),
                            "avg_frequency": skill_taxonomy['frequency'].mean() if not skill_taxonomy.empty else 0,
                            "categories_found": skill_taxonomy['category'].nunique() if not skill_taxonomy.empty else 0
                        }

                        # Sample extractions for manual verification
                        sample_extractions = []
                        for _, row in sample_data.head(3).iterrows():
                            text = str(row[text_column])[:200] + "..." if len(str(row[text_column])) > 200 else str(
                                row[text_column])
                            job_skills = extracted_skills[extracted_skills['job_id'] == row.name]['skill_name'].tolist()
                            sample_extractions.append({
                                "text_sample": text,
                                "extracted_skills": job_skills
                            })

                        extraction_results["sample_extractions"][source] = sample_extractions

                        logger.info(f"✅ {source}: Extracted {len(extracted_skills)} skill mentions")
                        logger.info(f"   Unique skills: {extracted_skills['skill_name'].nunique()}")
                        logger.info(f"   Avg confidence: {extracted_skills['confidence'].mean():.3f}")
                    else:
                        extraction_results["extraction_errors"].append(f"❌ {source}: No skills extracted")
                else:
                    extraction_results["extraction_errors"].append(f"❌ {source}: No text columns found for extraction")

            except Exception as e:
                extraction_results["extraction_errors"].append(f"❌ {source}: {str(e)}")
                logger.error(f"Skill extraction failed for {source}: {e}")

        return extraction_results

    def test_forecasting_models(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Test forecasting models with real skill data."""

        logger.info("🔮 Testing forecasting models with real data...")

        processor = EnhancedDataProcessor()
        skill_extractor = SkillExtractor()
        forecaster = EnsembleForecaster()

        forecasting_results = {
            "forecasts_generated": {},
            "forecasting_errors": [],
            "model_performance": {},
            "trend_analysis": {}
        }

        for source, path in data_paths.items():
            try:
                # Load and process data
                dataset = processor.load_dataset(source, path)

                if dataset.empty:
                    continue

                # Extract skills first
                text_columns = [col for col in dataset.columns
                                if any(keyword in col.lower()
                                       for keyword in ['description', 'summary', 'detail', 'text'])]

                if text_columns:
                    # Use sample data for faster testing
                    sample_data = dataset.head(50).copy()

                    extracted_skills = skill_extractor.extract_skills_from_dataframe(
                        sample_data,
                        text_column=text_columns[0],
                        method="hybrid"
                    )

                    if not extracted_skills.empty:
                        # Prepare time series data
                        time_series_data = forecaster.prepare_enhanced_time_series(extracted_skills)

                        if time_series_data:
                            # Generate forecasts for top 3 skills
                            top_skills = list(time_series_data.keys())[:3]

                            forecasts = {}
                            for skill in top_skills:
                                try:
                                    forecast = forecaster._generate_single_skill_forecast(
                                        skill, time_series_data[skill], horizon=6, confidence_intervals=True
                                    )
                                    if forecast:
                                        forecasts[skill] = forecast
                                except Exception as e:
                                    logger.warning(f"Forecast failed for {skill}: {e}")

                            if forecasts:
                                forecasting_results["forecasts_generated"][source] = {
                                    "skills_forecasted": list(forecasts.keys()),
                                    "forecast_horizon": 6,
                                    "models_used": [f.get("models_used", []) for f in forecasts.values()],
                                    "sample_forecast": {
                                        skill: {
                                            "predicted_values": data["forecast_data"]["values"][:3],
                                            "confidence_bounds": [
                                                data["forecast_data"].get("lower_bound", [])[:3],
                                                data["forecast_data"].get("upper_bound", [])[:3]
                                            ]
                                        }
                                        for skill, data in list(forecasts.items())[:1]
                                    }
                                }

                                # Analyze trends
                                trend_analysis = forecaster.analyze_market_trends(forecasts)
                                forecasting_results["trend_analysis"][source] = trend_analysis

                                logger.info(f"✅ {source}: Generated forecasts for {len(forecasts)} skills")
                            else:
                                forecasting_results["forecasting_errors"].append(f"❌ {source}: No forecasts generated")
                        else:
                            forecasting_results["forecasting_errors"].append(
                                f"❌ {source}: No time series data prepared")
                    else:
                        forecasting_results["forecasting_errors"].append(
                            f"❌ {source}: No skills extracted for forecasting")
                else:
                    forecasting_results["forecasting_errors"].append(
                        f"❌ {source}: No text columns for skill extraction")

            except Exception as e:
                forecasting_results["forecasting_errors"].append(f"❌ {source}: {str(e)}")
                logger.error(f"Forecasting failed for {source}: {e}")

        return forecasting_results

    def test_salary_prediction(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Test salary prediction models with real data."""

        logger.info("💰 Testing salary prediction with real data...")

        processor = EnhancedDataProcessor()
        skill_extractor = SkillExtractor()
        salary_predictor = SalaryPredictor()

        salary_results = {
            "models_trained": {},
            "prediction_errors": [],
            "model_performance": {},
            "sample_predictions": {}
        }

        for source, path in data_paths.items():
            try:
                # Load data
                dataset = processor.load_dataset(source, path)

                if dataset.empty:
                    continue

                # Look for salary columns
                salary_columns = [col for col in dataset.columns
                                  if any(keyword in col.lower()
                                         for keyword in ['salary', 'pay', 'wage', 'compensation'])]

                if salary_columns:
                    # Extract skills if possible
                    text_columns = [col for col in dataset.columns
                                    if any(keyword in col.lower()
                                           for keyword in ['description', 'summary', 'detail', 'text'])]

                    skills_df = pd.DataFrame()
                    if text_columns:
                        sample_data = dataset.head(30).copy()
                        skills_df = skill_extractor.extract_skills_from_dataframe(
                            sample_data,
                            text_column=text_columns[0],
                            method="hybrid"
                        )

                    # Prepare features for salary prediction
                    features_df = salary_predictor.prepare_features(dataset.head(30), skills_df)

                    if not features_df.empty and len(features_df) > 10:
                        # Train models
                        model_results = salary_predictor.train_models(features_df)

                        if model_results:
                            salary_results["models_trained"][source] = {
                                "models_available": list(model_results.keys()),
                                "training_data_size": len(features_df),
                                "features_used": len(features_df.columns),
                                "salary_column_used": salary_columns[0]
                            }

                            # Test predictions
                            if "best" in salary_predictor.models:
                                sample_features = {
                                    "skills": ["Python", "SQL"],
                                    "location": "San Francisco, CA",
                                    "experience_level": "Mid Level"
                                }

                                prediction = salary_predictor.predict_salary(sample_features)

                                if prediction:
                                    salary_results["sample_predictions"][source] = prediction

                            # Store model performance
                            performance_summary = {}
                            for model_name, results in model_results.items():
                                if isinstance(results, dict) and "test_metrics" in results:
                                    performance_summary[model_name] = results["test_metrics"]

                            salary_results["model_performance"][source] = performance_summary

                            logger.info(f"✅ {source}: Trained {len(model_results)} salary models")
                        else:
                            salary_results["prediction_errors"].append(f"❌ {source}: Model training failed")
                    else:
                        salary_results["prediction_errors"].append(f"❌ {source}: Insufficient data for training")
                else:
                    salary_results["prediction_errors"].append(f"❌ {source}: No salary columns found")

            except Exception as e:
                salary_results["prediction_errors"].append(f"❌ {source}: {str(e)}")
                logger.error(f"Salary prediction failed for {source}: {e}")

        return salary_results

    def test_full_pipeline(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Test the complete pipeline with real data."""

        logger.info("🚀 Testing complete pipeline with real data...")

        orchestrator = PlatformOrchestrator()
        pipeline_config = PipelineConfig(
            batch_size=500,
            parallel_processing=True,
            max_workers=2,
            cache_results=False,  # Disable caching for testing
            enable_monitoring=True,
            quality_threshold=0.7
        )

        try:
            # Run the full pipeline
            import asyncio

            async def run_pipeline():
                return await orchestrator.execute_full_pipeline(data_paths, pipeline_config)

            # Run pipeline
            pipeline_results = asyncio.run(run_pipeline())

            if pipeline_results.get("status") == "completed":
                logger.info("✅ Full pipeline completed successfully!")

                # Extract key metrics
                pipeline_summary = {
                    "status": "success",
                    "execution_time": pipeline_results.get("end_time", "") + " - " + pipeline_results.get("start_time",
                                                                                                          ""),
                    "stages_completed": list(pipeline_results.get("stages", {}).keys()),
                    "performance_metrics": pipeline_results.get("performance_metrics", {}),
                    "data_processed": {},
                    "models_trained": {},
                    "forecasts_generated": {},
                    "analytics_completed": {}
                }

                # Extract stage summaries
                stages = pipeline_results.get("stages", {})

                if "data_processing" in stages:
                    pipeline_summary["data_processed"] = {
                        "datasets_loaded": stages["data_processing"].get("datasets_loaded", 0),
                        "total_jobs": stages["data_processing"].get("processing_summary", {}).get("total_jobs", 0),
                        "quality_score": stages["data_processing"].get("processing_summary", {}).get(
                            "data_quality_score", 0)
                    }

                if "skill_extraction" in stages:
                    pipeline_summary["skills_extracted"] = stages["skill_extraction"].get("extraction_summary", {})

                if "model_training" in stages:
                    pipeline_summary["models_trained"] = stages["model_training"].get("model_performance_summary", {})

                if "forecasting" in stages:
                    pipeline_summary["forecasts_generated"] = stages["forecasting"].get("forecasting_summary", {})

                if "analytics" in stages:
                    pipeline_summary["analytics_completed"] = {
                        "geographic_insights": bool(stages["analytics"].get("geographic_insights")),
                        "industry_insights": bool(stages["analytics"].get("industry_insights")),
                        "market_opportunities": bool(stages["analytics"].get("market_opportunities"))
                    }

                return pipeline_summary
            else:
                logger.error("❌ Pipeline failed!")
                return {
                    "status": "failed",
                    "error": pipeline_results.get("error", "Unknown error"),
                    "stages_completed": list(pipeline_results.get("stages", {}).keys())
                }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "stages_completed": []
            }

    def run_comprehensive_validation(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Run comprehensive validation of the platform with real data."""

        logger.info("🔍 Starting comprehensive validation with real data...")
        logger.info(f"Data sources to test: {list(data_paths.keys())}")

        with self.performance_monitor.track_operation("comprehensive_validation"):
            # 1. Validate data paths and files
            path_validation = self.validate_data_paths(data_paths)

            # 2. Test data loading
            loading_results = self.test_data_loading(data_paths)

            # 3. Test skill extraction
            extraction_results = self.test_skill_extraction(data_paths)

            # 4. Test forecasting
            forecasting_results = self.test_forecasting_models(data_paths)

            # 5. Test salary prediction
            salary_results = self.test_salary_prediction(data_paths)

            # 6. Test full pipeline
            pipeline_results = self.test_full_pipeline(data_paths)

            # Compile comprehensive report
            validation_report = {
                "validation_timestamp": datetime.now().isoformat(),
                "data_sources_tested": list(data_paths.keys()),
                "validation_summary": {
                    "data_paths_valid": len(path_validation.get("issues", [])) == 0,
                    "data_loading_successful": len(loading_results.get("datasets_loaded", {})) > 0,
                    "skill_extraction_working": len(extraction_results.get("skills_extracted", {})) > 0,
                    "forecasting_functional": len(forecasting_results.get("forecasts_generated", {})) > 0,
                    "salary_prediction_working": len(salary_results.get("models_trained", {})) > 0,
                    "full_pipeline_successful": pipeline_results.get("status") == "success"
                },
                "detailed_results": {
                    "path_validation": path_validation,
                    "data_loading": loading_results,
                    "skill_extraction": extraction_results,
                    "forecasting": forecasting_results,
                    "salary_prediction": salary_results,
                    "full_pipeline": pipeline_results
                },
                "performance_metrics": self.performance_monitor.get_summary(),
                "recommendations": self._generate_recommendations(
                    path_validation, loading_results, extraction_results,
                    forecasting_results, salary_results, pipeline_results
                )
            }

        return validation_report

    def _generate_recommendations(self, *results) -> List[str]:
        """Generate recommendations based on validation results."""

        recommendations = []

        path_validation, loading_results, extraction_results, forecasting_results, salary_results, pipeline_results = results

        # Data quality recommendations
        if path_validation.get("issues"):
            recommendations.append("⚠️ Fix data path issues before production deployment")

        if loading_results.get("loading_errors"):
            recommendations.append("⚠️ Resolve data loading errors for all datasets")

        # Performance recommendations
        total_datasets = len(loading_results.get("datasets_loaded", {}))
        if total_datasets > 0:
            recommendations.append(f"✅ Data loading working for {total_datasets} datasets")

        # Feature recommendations
        skills_extracted = len(extraction_results.get("skills_extracted", {}))
        if skills_extracted > 0:
            recommendations.append(f"✅ Skill extraction working for {skills_extracted} datasets")
        else:
            recommendations.append("⚠️ Skill extraction needs improvement - check text column detection")

        forecasts_generated = len(forecasting_results.get("forecasts_generated", {}))
        if forecasts_generated > 0:
            recommendations.append(f"✅ Forecasting working for {forecasts_generated} datasets")
        else:
            recommendations.append("⚠️ Forecasting may need more historical data or better time series preparation")

        models_trained = len(salary_results.get("models_trained", {}))
        if models_trained > 0:
            recommendations.append(f"✅ Salary prediction working for {models_trained} datasets")
        else:
            recommendations.append("⚠️ Salary prediction needs datasets with salary columns")

        # Pipeline recommendations
        if pipeline_results.get("status") == "success":
            recommendations.append("✅ Full pipeline working end-to-end with real data")
        else:
            recommendations.append("⚠️ Full pipeline needs debugging - check individual component results")

        return recommendations


def main():
    """Main validation function."""

    print("🔍 Real Data Validation for Skills Demand Forecasting Platform")
    print("=" * 70)

    # Define your actual data paths here
    data_paths = {
        # UPDATE THESE PATHS WITH YOUR ACTUAL DATA LOCATIONS
        "linkedin_124k": "data/processed/linkedin_124k",
        "linkedin_1_3m": "data/processed/linkedin_1_3m",
        "stackoverflow": "data/processed/stackoverflow"
    }

    print(f"📂 Data paths to validate:")
    for source, path in data_paths.items():
        print(f"   {source}: {path}")
    print()

    # Create validator
    validator = RealDataValidator()

    # Run comprehensive validation
    validation_report = validator.run_comprehensive_validation(data_paths)

    # Print summary
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)

    summary = validation_report["validation_summary"]
    for check, passed in summary.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check.replace('_', ' ').title()}")

    print(f"\n⏱️  Total validation time: {validation_report['performance_metrics']['total_duration']} seconds")

    # Print recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    for rec in validation_report["recommendations"]:
        print(f"   {rec}")

    # Save detailed report
    report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {report_file}")

    # Final assessment
    all_passed = all(summary.values())
    if all_passed:
        print(f"\n🎉 SUCCESS: Platform validated with real data!")
        print(f"✅ Your platform is ready for production use.")
    else:
        print(f"\n⚠️  ISSUES FOUND: Some components need attention.")
        print(f"📝 Review the detailed report and recommendations above.")

    return validation_report


if __name__ == "__main__":
    main()






