"""
Centralized orchestrator for the Skills Demand Forecasting Platform.
Coordinates all components and provides unified API for complex operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import config
from src.utils.performance_monitor import PerformanceMonitor
from src.data_processing.enhanced_data_processor import EnhancedDataProcessor
from src.nlp.advanced_skill_extractor import AdvancedSkillExtractor
from src.ml_models.ensemble_forecaster import EnsembleForecaster
from src.ml_models.advanced_salary_predictor import AdvancedSalaryPredictor
from src.analytics.market_intelligence import MarketIntelligence
from src.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages for the platform."""
    DATA_INGESTION = "data_ingestion"
    DATA_CLEANING = "data_cleaning"
    SKILL_EXTRACTION = "skill_extraction"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    FORECASTING = "forecasting"
    ANALYTICS = "analytics"
    CACHING = "caching"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    batch_size: int = 1000
    parallel_processing: bool = True
    max_workers: int = 4
    cache_results: bool = True
    enable_monitoring: bool = True
    quality_threshold: float = 0.8


class PlatformOrchestrator:
    """
    Centralized orchestrator that coordinates all platform components.
    Provides high-level API for complex operations with optimized performance.
    """

    def __init__(self, config_override: Optional[Dict] = None):
        self.config = config_override or {}
        self.performance_monitor = PerformanceMonitor()
        self.cache = RedisCache() if self._redis_available() else None

        # Initialize components
        self.data_processor = EnhancedDataProcessor()
        self.skill_extractor = AdvancedSkillExtractor()
        self.forecaster = EnsembleForecaster()
        self.salary_predictor = AdvancedSalaryPredictor()
        self.market_intelligence = MarketIntelligence()

        # Pipeline state
        self.pipeline_state = {}
        self.processing_history = []

    def _redis_available(self) -> bool:
        """Check if Redis is available for caching."""
        try:
            import redis
            return True
        except ImportError:
            return False

    async def execute_full_pipeline(
            self,
            data_sources: Dict[str, str],
            pipeline_config: Optional[PipelineConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete data processing and analysis pipeline.

        Args:
            data_sources: Dictionary mapping dataset names to file paths
            pipeline_config: Configuration for pipeline execution

        Returns:
            Comprehensive results including all models, forecasts, and analytics
        """
        config = pipeline_config or PipelineConfig()

        with self.performance_monitor.track_operation("full_pipeline"):
            logger.info("🚀 Starting comprehensive platform pipeline...")

            results = {
                "pipeline_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "start_time": datetime.now().isoformat(),
                "config": config.__dict__,
                "stages": {},
                "performance_metrics": {}
            }

            try:
                # Stage 1: Data Ingestion and Processing
                results["stages"]["data_processing"] = await self._execute_data_processing_stage(
                    data_sources, config
                )

                # Stage 2: Advanced Skill Extraction
                results["stages"]["skill_extraction"] = await self._execute_skill_extraction_stage(
                    results["stages"]["data_processing"]["unified_jobs"], config
                )

                # Stage 3: Feature Engineering
                results["stages"]["feature_engineering"] = await self._execute_feature_engineering_stage(
                    results["stages"]["data_processing"]["unified_jobs"],
                    results["stages"]["skill_extraction"]["skills_data"],
                    config
                )

                # Stage 4: Model Training (Parallel)
                results["stages"]["model_training"] = await self._execute_model_training_stage(
                    results["stages"]["feature_engineering"]["features"], config
                )

                # Stage 5: Forecasting and Predictions
                results["stages"]["forecasting"] = await self._execute_forecasting_stage(
                    results["stages"]["skill_extraction"]["skills_data"], config
                )

                # Stage 6: Market Intelligence Analytics
                results["stages"]["analytics"] = await self._execute_analytics_stage(
                    results["stages"]["data_processing"]["unified_jobs"],
                    results["stages"]["skill_extraction"]["skills_data"],
                    results["stages"]["forecasting"]["forecasts"],
                    config
                )

                # Stage 7: Cache Results
                if config.cache_results and self.cache:
                    await self._cache_pipeline_results(results)

                results["end_time"] = datetime.now().isoformat()
                results["performance_metrics"] = self.performance_monitor.get_summary()
                results["status"] = "completed"

                logger.info("✅ Pipeline completed successfully!")
                return results

            except Exception as e:
                logger.error(f"❌ Pipeline failed: {e}")
                results["status"] = "failed"
                results["error"] = str(e)
                results["end_time"] = datetime.now().isoformat()
                return results

    async def _execute_data_processing_stage(
            self,
            data_sources: Dict[str, str],
            config: PipelineConfig
    ) -> Dict[str, Any]:
        """Execute optimized data processing stage."""

        with self.performance_monitor.track_operation("data_processing"):
            logger.info("📊 Executing data processing stage...")

            # Load and validate data in parallel
            datasets = {}

            if config.parallel_processing:
                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    future_to_source = {
                        executor.submit(self.data_processor.load_dataset, source, path): source
                        for source, path in data_sources.items()
                    }

                    for future in as_completed(future_to_source):
                        source = future_to_source[future]
                        try:
                            datasets[source] = future.result()
                        except Exception as e:
                            logger.warning(f"Failed to load {source}: {e}")
            else:
                for source, path in data_sources.items():
                    try:
                        datasets[source] = self.data_processor.load_dataset(source, path)
                    except Exception as e:
                        logger.warning(f"Failed to load {source}: {e}")

            # Enhanced data cleaning and validation
            cleaned_datasets = {}
            for source, data in datasets.items():
                cleaned_data = self.data_processor.enhanced_cleaning(
                    data,
                    quality_threshold=config.quality_threshold
                )
                if cleaned_data is not None and not cleaned_data.empty:
                    cleaned_datasets[source] = cleaned_data

            # Create unified dataset with advanced schema mapping
            unified_jobs = self.data_processor.create_unified_dataset(cleaned_datasets)

            # Advanced data quality assessment
            quality_report = self.data_processor.comprehensive_quality_assessment(unified_jobs)

            return {
                "datasets_loaded": len(datasets),
                "datasets_cleaned": len(cleaned_datasets),
                "unified_jobs": unified_jobs,
                "quality_report": quality_report,
                "processing_summary": {
                    "total_jobs": len(unified_jobs),
                    "data_quality_score": quality_report.get("overall_score", 0),
                    "duplicate_removal_rate": quality_report.get("duplicate_removal_rate", 0)
                }
            }

    async def _execute_skill_extraction_stage(
            self,
            jobs_df: pd.DataFrame,
            config: PipelineConfig
    ) -> Dict[str, Any]:
        """Execute advanced skill extraction with NLP optimization."""

        with self.performance_monitor.track_operation("skill_extraction"):
            logger.info("🧠 Executing advanced skill extraction...")

            # Batch processing for memory efficiency
            skills_data = []
            batch_size = config.batch_size

            for i in range(0, len(jobs_df), batch_size):
                batch = jobs_df.iloc[i:i + batch_size]
                batch_skills = await self.skill_extractor.extract_skills_batch(
                    batch,
                    use_advanced_nlp=True,
                    parallel_processing=config.parallel_processing
                )
                skills_data.append(batch_skills)

                logger.info(f"Processed batch {i // batch_size + 1}/{(len(jobs_df) // batch_size) + 1}")

            # Combine all skill extractions
            combined_skills = pd.concat(skills_data, ignore_index=True)

            # Advanced skill standardization and deduplication
            standardized_skills = self.skill_extractor.advanced_skill_standardization(
                combined_skills
            )

            # Create enhanced skill taxonomy with market intelligence
            skill_taxonomy = self.skill_extractor.create_enhanced_taxonomy(
                standardized_skills
            )

            # Skill relationship mapping
            skill_relationships = self.skill_extractor.build_skill_relationship_graph(
                standardized_skills
            )

            return {
                "skills_data": standardized_skills,
                "skill_taxonomy": skill_taxonomy,
                "skill_relationships": skill_relationships,
                "extraction_summary": {
                    "total_skill_mentions": len(combined_skills),
                    "unique_skills": len(skill_taxonomy),
                    "extraction_accuracy": skill_taxonomy["avg_confidence"].mean() if not skill_taxonomy.empty else 0,
                    "skill_categories": skill_taxonomy["category"].nunique() if not skill_taxonomy.empty else 0
                }
            }

    async def _execute_feature_engineering_stage(
            self,
            jobs_df: pd.DataFrame,
            skills_df: pd.DataFrame,
            config: PipelineConfig
    ) -> Dict[str, Any]:
        """Execute advanced feature engineering."""

        with self.performance_monitor.track_operation("feature_engineering"):
            logger.info("⚙️ Executing feature engineering...")

            # Advanced feature creation
            feature_engineer = self.data_processor.get_feature_engineer()

            # Create temporal features
            temporal_features = feature_engineer.create_temporal_features(jobs_df)

            # Create skill-based features
            skill_features = feature_engineer.create_skill_features(jobs_df, skills_df)

            # Create location features with geocoding
            location_features = feature_engineer.create_location_features(jobs_df)

            # Create company features
            company_features = feature_engineer.create_company_features(jobs_df)

            # Combine all features
            combined_features = feature_engineer.combine_features([
                temporal_features, skill_features, location_features, company_features
            ])

            # Feature selection and importance ranking
            selected_features = feature_engineer.select_features(
                combined_features,
                target_column="target_salary" if "target_salary" in combined_features.columns else None
            )

            return {
                "features": selected_features,
                "feature_importance": feature_engineer.get_feature_importance(),
                "feature_summary": {
                    "total_features": len(selected_features.columns),
                    "feature_types": feature_engineer.get_feature_types(),
                    "missing_data_rate": selected_features.isnull().mean().mean()
                }
            }

    async def _execute_model_training_stage(
            self,
            features_df: pd.DataFrame,
            config: PipelineConfig
    ) -> Dict[str, Any]:
        """Execute parallel model training for all prediction tasks."""

        with self.performance_monitor.track_operation("model_training"):
            logger.info("🤖 Executing model training...")

            training_results = {}

            if config.parallel_processing:
                # Train models in parallel
                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    futures = {}

                    # Submit salary prediction training
                    if "target_salary" in features_df.columns:
                        futures["salary_prediction"] = executor.submit(
                            self.salary_predictor.train_advanced_models, features_df
                        )

                    # Submit demand forecasting preparation
                    futures["forecasting_prep"] = executor.submit(
                        self.forecaster.prepare_forecasting_models, features_df
                    )

                    # Collect results
                    for model_type, future in futures.items():
                        try:
                            training_results[model_type] = future.result()
                        except Exception as e:
                            logger.error(f"Model training failed for {model_type}: {e}")
                            training_results[model_type] = {"status": "failed", "error": str(e)}

            return {
                "models_trained": list(training_results.keys()),
                "training_results": training_results,
                "model_performance_summary": self._summarize_model_performance(training_results)
            }

    async def _execute_forecasting_stage(
            self,
            skills_df: pd.DataFrame,
            config: PipelineConfig
    ) -> Dict[str, Any]:
        """Execute advanced demand forecasting."""

        with self.performance_monitor.track_operation("forecasting"):
            logger.info("🔮 Executing demand forecasting...")

            # Prepare time series data with enhanced preprocessing
            time_series_data = self.forecaster.prepare_enhanced_time_series(skills_df)

            # Generate forecasts using ensemble methods
            forecasts = await self.forecaster.generate_ensemble_forecasts(
                time_series_data,
                forecast_horizon=12,
                confidence_intervals=True,
                parallel_processing=config.parallel_processing
            )

            # Generate trend analysis
            trend_analysis = self.forecaster.analyze_market_trends(forecasts)

            # Identify emerging technologies
            emerging_techs = self.forecaster.identify_emerging_technologies(
                forecasts, growth_threshold=20.0
            )

            return {
                "forecasts": forecasts,
                "trend_analysis": trend_analysis,
                "emerging_technologies": emerging_techs,
                "forecasting_summary": {
                    "skills_forecasted": len(forecasts),
                    "avg_forecast_accuracy": self._calculate_forecast_accuracy(forecasts),
                    "emerging_tech_count": len(emerging_techs)
                }
            }

    async def _execute_analytics_stage(
            self,
            jobs_df: pd.DataFrame,
            skills_df: pd.DataFrame,
            forecasts: Dict[str, Any],
            config: PipelineConfig
    ) -> Dict[str, Any]:
        """Execute comprehensive market intelligence analytics."""

        with self.performance_monitor.track_operation("analytics"):
            logger.info("📈 Executing market intelligence analytics...")

            # Geographic analysis
            geographic_insights = self.market_intelligence.analyze_geographic_trends(
                jobs_df, skills_df
            )

            # Industry analysis
            industry_insights = self.market_intelligence.analyze_industry_trends(
                jobs_df, skills_df
            )

            # Skill correlation analysis
            skill_correlations = self.market_intelligence.analyze_skill_correlations(
                skills_df
            )

            # Salary analytics
            salary_insights = self.market_intelligence.analyze_salary_trends(
                jobs_df, skills_df
            )

            # Market opportunity scoring
            opportunities = self.market_intelligence.calculate_market_opportunities(
                forecasts, salary_insights
            )

            # Generate executive summary
            executive_summary = self.market_intelligence.generate_executive_summary({
                "geographic": geographic_insights,
                "industry": industry_insights,
                "salary": salary_insights,
                "opportunities": opportunities
            })

            return {
                "geographic_insights": geographic_insights,
                "industry_insights": industry_insights,
                "skill_correlations": skill_correlations,
                "salary_insights": salary_insights,
                "market_opportunities": opportunities,
                "executive_summary": executive_summary
            }

    async def _cache_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Cache pipeline results for quick access."""
        if not self.cache:
            return

        try:
            # Cache key components with TTL
            cache_ttl = 3600 * 24  # 24 hours

            await self.cache.set(
                f"pipeline_results:{results['pipeline_id']}",
                results,
                ttl=cache_ttl
            )

            # Cache frequently accessed data separately
            if "forecasts" in results.get("stages", {}).get("forecasting", {}):
                await self.cache.set(
                    "latest_forecasts",
                    results["stages"]["forecasting"]["forecasts"],
                    ttl=cache_ttl
                )

            if "skill_taxonomy" in results.get("stages", {}).get("skill_extraction", {}):
                await self.cache.set(
                    "skill_taxonomy",
                    results["stages"]["skill_extraction"]["skill_taxonomy"],
                    ttl=cache_ttl
                )

            logger.info("✅ Pipeline results cached successfully")

        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    def _summarize_model_performance(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize model performance across all trained models."""
        summary = {
            "total_models": len(training_results),
            "successful_models": 0,
            "failed_models": 0,
            "best_performers": {},
            "avg_performance": {}
        }

        for model_type, results in training_results.items():
            if results.get("status") == "failed":
                summary["failed_models"] += 1
            else:
                summary["successful_models"] += 1

                # Extract best performance metrics
                if "test_metrics" in results:
                    best_metric = max(results["test_metrics"].values()) if results["test_metrics"] else 0
                    summary["best_performers"][model_type] = best_metric

        return summary

    def _calculate_forecast_accuracy(self, forecasts: Dict[str, Any]) -> float:
        """Calculate average forecast accuracy across all skills."""
        accuracies = []

        for skill, forecast_data in forecasts.items():
            if "model_performance" in forecast_data:
                perf = forecast_data["model_performance"]
                if "mape" in perf:
                    accuracy = max(0, 100 - perf["mape"])  # Convert MAPE to accuracy
                    accuracies.append(accuracy)

        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    # High-level API methods for common operations

    async def get_skill_forecast(
            self,
            skill_name: str,
            horizon_months: int = 12
    ) -> Dict[str, Any]:
        """Get forecast for a specific skill with caching."""

        cache_key = f"skill_forecast:{skill_name}:{horizon_months}"

        if self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result

        # Generate forecast if not cached
        forecast = await self.forecaster.forecast_single_skill(skill_name, horizon_months)

        if self.cache and forecast:
            await self.cache.set(cache_key, forecast, ttl=3600)

        return forecast

    async def predict_salary_optimized(
            self,
            job_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized salary prediction with feature enhancement."""

        # Enhance features with market intelligence
        enhanced_features = self.market_intelligence.enhance_job_features(job_features)

        # Get prediction with confidence intervals
        prediction = await self.salary_predictor.predict_with_confidence(enhanced_features)

        # Add market context
        market_context = self.market_intelligence.get_market_context(enhanced_features)
        prediction["market_context"] = market_context

        return prediction

    async def get_career_roadmap(
            self,
            current_skills: List[str],
            target_role: str,
            experience_level: str
    ) -> Dict[str, Any]:
        """Generate comprehensive career roadmap with market intelligence."""

        # Get skill gap analysis
        skill_gaps = self.market_intelligence.analyze_skill_gaps(
            current_skills, target_role, experience_level
        )

        # Get learning recommendations with market trends
        learning_path = await self.market_intelligence.generate_learning_path(
            skill_gaps, include_market_trends=True
        )

        # Calculate ROI for skill development
        skill_roi = self.market_intelligence.calculate_skill_roi(
            current_skills, learning_path["recommended_skills"]
        )

        # Generate timeline with milestones
        career_timeline = self.market_intelligence.generate_career_timeline(
            current_skills, target_role, learning_path
        )

        return {
            "skill_gaps": skill_gaps,
            "learning_path": learning_path,
            "skill_roi": skill_roi,
            "career_timeline": career_timeline,
            "market_outlook": self.market_intelligence.get_role_market_outlook(target_role)
        }
