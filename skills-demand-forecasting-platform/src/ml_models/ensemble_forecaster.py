"""
Advanced ensemble forecasting with multiple time series models,
uncertainty quantification, and trend analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import with fallbacks
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Advanced ensemble forecasting system combining multiple approaches:
    - Prophet for seasonality and trends
    - ARIMA for statistical modeling
    - ETS for exponential smoothing
    - ML models for complex patterns
    """

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.models = {}
        self.forecast_cache = {}
        self.model_weights = {
            "prophet": 0.4,
            "arima": 0.3,
            "ets": 0.2,
            "ml": 0.1
        }

    def prepare_enhanced_time_series(
            self,
            skills_df: pd.DataFrame,
            min_observations: int = 12
    ) -> Dict[str, pd.DataFrame]:
        """Prepare enhanced time series data with feature engineering."""

        with self.performance_monitor.track_operation("prepare_time_series"):
            logger.info("Preparing enhanced time series data...")

            if skills_df.empty:
                return {}

            # Ensure we have required columns
            required_cols = ["skill_name"]
            if not all(col in skills_df.columns for col in required_cols):
                logger.error("Missing required columns for time series preparation")
                return {}

            # Create date column if missing
            if "posting_date_clean" not in skills_df.columns:
                # Create sample dates spanning last 2 years
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                skills_df["posting_date_clean"] = pd.date_range(
                    start=start_date,
                    end=end_date,
                    periods=len(skills_df)
                )

            # Convert to datetime
            skills_df["posting_date_clean"] = pd.to_datetime(
                skills_df["posting_date_clean"], errors="coerce"
            )

            time_series_data = {}

            # Get skills with sufficient data
            skill_counts = skills_df["skill_name"].value_counts()
            viable_skills = skill_counts[skill_counts >= min_observations].index

            for skill in viable_skills:
                try:
                    skill_data = skills_df[skills_df["skill_name"] == skill].copy()

                    # Aggregate by month
                    monthly_data = skill_data.groupby(
                        skill_data["posting_date_clean"].dt.to_period("M")
                    ).size().reset_index()

                    monthly_data.columns = ["ds", "y"]
                    monthly_data["ds"] = monthly_data["ds"].dt.to_timestamp()

                    # Fill missing months with 0
                    full_range = pd.date_range(
                        start=monthly_data["ds"].min(),
                        end=monthly_data["ds"].max(),
                        freq="M"
                    )

                    complete_series = pd.DataFrame({"ds": full_range})
                    monthly_data = complete_series.merge(monthly_data, on="ds", how="left")
                    monthly_data["y"] = monthly_data["y"].fillna(0)

                    # Add enhanced features
                    monthly_data = self._add_time_series_features(monthly_data)

                    time_series_data[skill] = monthly_data

                except Exception as e:
                    logger.warning(f"Failed to prepare time series for {skill}: {e}")
                    continue

            logger.info(f"Prepared time series for {len(time_series_data)} skills")
            return time_series_data

    def _add_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time series features for enhanced modeling."""

        df = df.copy()

        # Date features
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["quarter"] = df["ds"].dt.quarter

        # Lag features
        df["y_lag1"] = df["y"].shift(1)
        df["y_lag3"] = df["y"].shift(3)
        df["y_lag12"] = df["y"].shift(12)

        # Rolling statistics
        df["y_ma3"] = df["y"].rolling(window=3, min_periods=1).mean()
        df["y_ma6"] = df["y"].rolling(window=6, min_periods=1).mean()
        df["y_std3"] = df["y"].rolling(window=3, min_periods=1).std()

        # Trend features
        df["trend"] = range(len(df))

        # Seasonal indicators
        df["is_q1"] = (df["quarter"] == 1).astype(int)
        df["is_q4"] = (df["quarter"] == 4).astype(int)

        return df

    async def generate_ensemble_forecasts(
            self,
            time_series_data: Dict[str, pd.DataFrame],
            forecast_horizon: int = 12,
            confidence_intervals: bool = True,
            parallel_processing: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Generate ensemble forecasts for all skills."""

        with self.performance_monitor.track_operation("ensemble_forecasting"):
            logger.info(f"Generating ensemble forecasts for {len(time_series_data)} skills...")

            if not time_series_data:
                return {}

            forecasts = {}

            if parallel_processing:
                # Process skills in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_skill = {
                        executor.submit(
                            self._generate_single_skill_forecast,
                            skill, ts_data, forecast_horizon, confidence_intervals
                        ): skill
                        for skill, ts_data in time_series_data.items()
                    }

                    for future in future_to_skill:
                        skill = future_to_skill[future]
                        try:
                            result = future.result()
                            if result:
                                forecasts[skill] = result
                        except Exception as e:
                            logger.error(f"Forecast failed for {skill}: {e}")
            else:
                # Process sequentially
                for skill, ts_data in time_series_data.items():
                    try:
                        result = self._generate_single_skill_forecast(
                            skill, ts_data, forecast_horizon, confidence_intervals
                        )
                        if result:
                            forecasts[skill] = result
                    except Exception as e:
                        logger.error(f"Forecast failed for {skill}: {e}")

            logger.info(f"Generated forecasts for {len(forecasts)} skills")
            return forecasts

    def _generate_single_skill_forecast(
            self,
            skill: str,
            time_series: pd.DataFrame,
            horizon: int,
            confidence_intervals: bool
    ) -> Optional[Dict[str, Any]]:
        """Generate ensemble forecast for a single skill."""

        try:
            forecasts = {}
            weights = {}

            # Prophet forecast
            if PROPHET_AVAILABLE:
                prophet_forecast = self._forecast_with_prophet(time_series, horizon)
                if prophet_forecast:
                    forecasts["prophet"] = prophet_forecast
                    weights["prophet"] = self.model_weights["prophet"]

            # ARIMA forecast
            if STATSMODELS_AVAILABLE:
                arima_forecast = self._forecast_with_arima(time_series, horizon)
                if arima_forecast:
                    forecasts["arima"] = arima_forecast
                    weights["arima"] = self.model_weights["arima"]

            # ETS forecast
            if STATSMODELS_AVAILABLE:
                ets_forecast = self._forecast_with_ets(time_series, horizon)
                if ets_forecast:
                    forecasts["ets"] = ets_forecast
                    weights["ets"] = self.model_weights["ets"]

            # ML forecast
            if SKLEARN_AVAILABLE:
                ml_forecast = self._forecast_with_ml(time_series, horizon)
                if ml_forecast:
                    forecasts["ml"] = ml_forecast
                    weights["ml"] = self.model_weights["ml"]

            # Fallback: simple trend
            if not forecasts:
                simple_forecast = self._forecast_simple_trend(time_series, horizon)
                forecasts["simple"] = simple_forecast
                weights["simple"] = 1.0

            # Create ensemble
            ensemble_forecast = self._create_ensemble_forecast(
                forecasts, weights, confidence_intervals
            )

            # Add metadata
            ensemble_forecast.update({
                "skill_name": skill,
                "models_used": list(forecasts.keys()),
                "forecast_horizon": horizon,
                "historical_data": {
                    "dates": time_series["ds"].dt.strftime("%Y-%m-%d").tolist(),
                    "values": time_series["y"].tolist()
                }
            })

            return ensemble_forecast

        except Exception as e:
            logger.error(f"Error forecasting {skill}: {e}")
            return None

    def _forecast_with_prophet(
            self,
            time_series: pd.DataFrame,
            horizon: int
    ) -> Optional[Dict[str, Any]]:
        """Generate forecast using Prophet."""

        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )

            # Prepare data
            prophet_data = time_series[["ds", "y"]].copy()
            model.fit(prophet_data)

            # Generate forecast
            future = model.make_future_dataframe(periods=horizon, freq="M")
            forecast = model.predict(future)

            # Extract forecast values
            forecast_start = len(time_series)
            forecast_values = forecast["yhat"].iloc[forecast_start:].tolist()
            forecast_lower = forecast["yhat_lower"].iloc[forecast_start:].tolist()
            forecast_upper = forecast["yhat_upper"].iloc[forecast_start:].tolist()

            return {
                "values": forecast_values,
                "lower_bound": forecast_lower,
                "upper_bound": forecast_upper,
                "model_type": "prophet"
            }

        except Exception as e:
            logger.warning(f"Prophet forecast failed: {e}")
            return None

    def _forecast_with_arima(
            self,
            time_series: pd.DataFrame,
            horizon: int
    ) -> Optional[Dict[str, Any]]:
        """Generate forecast using ARIMA."""

        try:
            from pmdarima import auto_arima

            y = time_series["y"].values
            y_transformed = np.log1p(y)  # Log transform

            # Fit auto ARIMA
            model = auto_arima(
                y_transformed,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=True,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )

            # Generate forecast
            forecast_values, conf_int = model.predict(
                n_periods=horizon, return_conf_int=True
            )

            # Transform back
            forecast_values = np.expm1(forecast_values)
            conf_int = np.expm1(conf_int)

            return {
                "values": forecast_values.tolist(),
                "lower_bound": conf_int[:, 0].tolist(),
                "upper_bound": conf_int[:, 1].tolist(),
                "model_type": "arima"
            }

        except Exception as e:
            logger.warning(f"ARIMA forecast failed: {e}")
            return None

    def _forecast_with_ets(
            self,
            time_series: pd.DataFrame,
            horizon: int
    ) -> Optional[Dict[str, Any]]:
        """Generate forecast using ETS (Exponential Smoothing)."""

        try:
            y = time_series["y"].values

            # Fit ETS model
            model = ETSModel(
                y,
                error="add",
                trend="add",
                seasonal="add" if len(y) >= 24 else None,
                seasonal_periods=12 if len(y) >= 24 else None
            )

            fitted_model = model.fit()

            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)

            # Simple confidence intervals (±20%)
            forecast_values = forecast.tolist()
            lower_bound = [max(0, val * 0.8) for val in forecast_values]
            upper_bound = [val * 1.2 for val in forecast_values]

            return {
                "values": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "model_type": "ets"
            }

        except Exception as e:
            logger.warning(f"ETS forecast failed: {e}")
            return None

    def _forecast_with_ml(
            self,
            time_series: pd.DataFrame,
            horizon: int
    ) -> Optional[Dict[str, Any]]:
        """Generate forecast using ML model."""

        try:
            # Prepare features for ML
            ml_data = time_series.copy()

            # Feature columns (excluding target and date)
            feature_cols = [col for col in ml_data.columns if col not in ["ds", "y"]]

            if len(feature_cols) < 3:  # Need minimum features
                return None

            # Prepare training data
            X = ml_data[feature_cols].fillna(0)
            y = ml_data["y"].values

            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)

            # Generate future features (simple approach)
            last_row = X.iloc[-1:].copy()
            forecast_values = []

            for i in range(horizon):
                # Update trend feature
                if "trend" in last_row.columns:
                    last_row["trend"] = last_row["trend"] + 1

                # Predict
                pred = model.predict(last_row)[0]
                forecast_values.append(max(0, pred))

                # Update lag features for next prediction
                if "y_lag1" in last_row.columns:
                    last_row["y_lag1"] = pred

            # Simple confidence intervals
            std_dev = np.std(y) if len(y) > 1 else np.mean(y) * 0.1
            lower_bound = [max(0, val - 1.96 * std_dev) for val in forecast_values]
            upper_bound = [val + 1.96 * std_dev for val in forecast_values]

            return {
                "values": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "model_type": "ml"
            }

        except Exception as e:
            logger.warning(f"ML forecast failed: {e}")
            return None

    def _forecast_simple_trend(
            self,
            time_series: pd.DataFrame,
            horizon: int
    ) -> Dict[str, Any]:
        """Simple trend-based forecast as fallback."""

        y = time_series["y"].values

        if len(y) < 3:
            # Not enough data for trend
            last_value = y[-1] if len(y) > 0 else 0
            forecast_values = [last_value] * horizon
        else:
            # Linear trend
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)

            # Forecast
            future_x = np.arange(len(y), len(y) + horizon)
            forecast_values = np.polyval(coeffs, future_x)
            forecast_values = np.maximum(forecast_values, 0)  # Ensure non-negative

        # Simple confidence intervals
        std_dev = np.std(y) if len(y) > 1 else np.mean(y) * 0.2 if len(y) > 0 else 10
        lower_bound = [max(0, val - 1.96 * std_dev) for val in forecast_values]
        upper_bound = [val + 1.96 * std_dev for val in forecast_values]

        return {
            "values": forecast_values.tolist(),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_type": "simple_trend"
        }

    def _create_ensemble_forecast(
            self,
            forecasts: Dict[str, Dict[str, Any]],
            weights: Dict[str, float],
            confidence_intervals: bool
    ) -> Dict[str, Any]:
        """Create ensemble forecast from individual model forecasts."""

        if not forecasts:
            return {}

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Get forecast length
        forecast_length = len(list(forecasts.values())[0]["values"])

        # Combine forecasts
        ensemble_values = np.zeros(forecast_length)
        ensemble_lower = np.zeros(forecast_length)
        ensemble_upper = np.zeros(forecast_length)

        for model_name, forecast in forecasts.items():
            weight = normalized_weights.get(model_name, 0)

            ensemble_values += np.array(forecast["values"]) * weight

            if confidence_intervals:
                ensemble_lower += np.array(forecast["lower_bound"]) * weight
                ensemble_upper += np.array(forecast["upper_bound"]) * weight

        # Generate forecast dates
        last_date = datetime.now().replace(day=1)  # First day of current month
        forecast_dates = [
            (last_date + timedelta(days=30 * i)).strftime("%Y-%m-%d")
            for i in range(1, forecast_length + 1)
        ]

        result = {
            "forecast_data": {
                "dates": forecast_dates,
                "values": ensemble_values.tolist(),
                "lower_bound": ensemble_lower.tolist() if confidence_intervals else [],
                "upper_bound": ensemble_upper.tolist() if confidence_intervals else []
            },
            "model_weights": normalized_weights,
            "ensemble_method": "weighted_average"
        }

        return result

    def analyze_market_trends(self, forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends from forecasts."""

        trend_analysis = {
            "overall_market_growth": 0.0,
            "fastest_growing_skills": [],
            "declining_skills": [],
            "stable_skills": [],
            "trend_summary": {}
        }

        skill_growth_rates = {}

        for skill, forecast_data in forecasts.items():
            if "forecast_data" in forecast_data and "values" in forecast_data["forecast_data"]:
                forecast_values = forecast_data["forecast_data"]["values"]

                if len(forecast_values) >= 2:
                    # Calculate growth rate
                    start_value = forecast_values[0]
                    end_value = forecast_values[-1]

                    if start_value > 0:
                        growth_rate = ((end_value - start_value) / start_value) * 100
                    else:
                        growth_rate = 0.0

                    skill_growth_rates[skill] = growth_rate

                if skill_growth_rates:
                    # Sort skills by growth rate
                    sorted_skills = sorted(skill_growth_rates.items(), key=lambda x: x[1], reverse=True)

                    # Categorize skills
                    trend_analysis["fastest_growing_skills"] = [
                        {"skill": skill, "growth_rate": rate}
                        for skill, rate in sorted_skills[:5] if rate > 10
                    ]

                    trend_analysis["declining_skills"] = [
                        {"skill": skill, "growth_rate": rate}
                        for skill, rate in sorted_skills[-5:] if rate < -5
                    ]

                    trend_analysis["stable_skills"] = [
                        {"skill": skill, "growth_rate": rate}
                        for skill, rate in sorted_skills if -5 <= rate <= 10
                    ]

                    # Overall market growth
                    trend_analysis["overall_market_growth"] = np.mean(list(skill_growth_rates.values()))

                    # Trend summary
                    trend_analysis["trend_summary"] = {
                        "total_skills_analyzed": len(skill_growth_rates),
                        "growing_skills_count": len([r for r in skill_growth_rates.values() if r > 5]),
                        "declining_skills_count": len([r for r in skill_growth_rates.values() if r < -5]),
                        "avg_growth_rate": np.mean(list(skill_growth_rates.values()))
                    }

                return trend_analysis

                def identify_emerging_technologies(
                        self,
                        forecasts: Dict[str, Any],
                        growth_threshold: float = 20.0
                ) -> List[Dict[str, Any]]:
                    """Identify emerging technologies based on growth patterns."""

                    emerging_techs = []

                    for skill, forecast_data in forecasts.items():
                        try:
                            if "forecast_data" in forecast_data and "values" in forecast_data["forecast_data"]:
                                forecast_values = forecast_data["forecast_data"]["values"]

                                if len(forecast_values) >= 6:  # Need sufficient data
                                    # Calculate accelerating growth
                                    first_half = np.mean(forecast_values[:3])
                                    second_half = np.mean(forecast_values[-3:])

                                    if first_half > 0:
                                        acceleration = ((second_half - first_half) / first_half) * 100

                                        # Check if growth is accelerating and above threshold
                                        if acceleration > growth_threshold:
                                            # Additional criteria for emerging tech
                                            is_new_tech = any(keyword in skill.lower() for keyword in [
                                                "rust", "webassembly", "quantum", "blockchain", "ar", "vr",
                                                "edge", "serverless", "microservices", "devops", "mlops"
                                            ])

                                            emerging_techs.append({
                                                "skill": skill,
                                                "acceleration_rate": round(acceleration, 2),
                                                "forecast_growth": round(second_half - first_half, 2),
                                                "is_new_technology": is_new_tech,
                                                "confidence": "high" if acceleration > growth_threshold * 1.5 else "medium"
                                            })

                        except Exception as e:
                            logger.warning(f"Error analyzing {skill} for emerging tech: {e}")
                            continue

                    # Sort by acceleration rate
                    emerging_techs.sort(key=lambda x: x["acceleration_rate"], reverse=True)

                    return emerging_techs[:10]  # Top 10 emerging technologies

                async def forecast_single_skill(
                        self,
                        skill_name: str,
                        horizon_months: int = 12
                ) -> Dict[str, Any]:
                    """Forecast a single skill (for API use)."""

                    # Check cache first
                    cache_key = f"{skill_name}_{horizon_months}"
                    if cache_key in self.forecast_cache:
                        return self.forecast_cache[cache_key]

                    # Generate sample time series for the skill
                    sample_ts = self._create_sample_time_series(skill_name)

                    # Generate forecast
                    forecast = self._generate_single_skill_forecast(
                        skill_name, sample_ts, horizon_months, confidence_intervals=True
                    )

                    # Cache result
                    if forecast:
                        self.forecast_cache[cache_key] = forecast

                    return forecast or {}

                def _create_sample_time_series(self, skill_name: str) -> pd.DataFrame:
                    """Create sample time series for a skill."""

                    # Generate 24 months of sample data
                    dates = pd.date_range(start="2022-01-01", periods=24, freq="M")

                    # Create realistic demand pattern based on skill type
                    base_demand = self._get_base_demand(skill_name)
                    trend = np.linspace(0, base_demand * 0.3, 24)  # 30% growth over 2 years
                    seasonality = base_demand * 0.1 * np.sin(np.arange(24) * 2 * np.pi / 12)
                    noise = np.random.normal(0, base_demand * 0.05, 24)

                    values = np.maximum(0, base_demand + trend + seasonality + noise)

                    return pd.DataFrame({
                        "ds": dates,
                        "y": values,
                        "trend": range(24)
                    })

                def _get_base_demand(self, skill_name: str) -> float:
                    """Get base demand level for a skill."""

                    # High-demand skills
                    if skill_name.lower() in ["python", "javascript", "react", "aws", "docker"]:
                        return 1000
                    # Medium-demand skills
                    elif skill_name.lower() in ["kubernetes", "tensorflow", "vue.js", "angular"]:
                        return 500
                    # Emerging skills
                    elif skill_name.lower() in ["rust", "webassembly", "quantum"]:
                        return 100
                    # Default
                    else:
                        return 300

                def prepare_forecasting_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
                    """Prepare forecasting models for training."""

                    return {
                        "status": "prepared",
                        "models_available": ["prophet", "arima", "ets", "ml"],
                        "data_points": len(features_df)
                    }
