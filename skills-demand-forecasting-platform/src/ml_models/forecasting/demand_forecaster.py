"""Skills demand forecasting using time series analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import with fallbacks
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Forecasting will use simple trend analysis.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. ARIMA forecasting disabled.")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Model evaluation limited.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from src.utils.config_manager import config
from src.utils.file_utils import FileHandler

logger = logging.getLogger(__name__)

class SkillDemandForecaster:
    """Advanced time series forecasting for skill demand prediction with fallbacks."""

    def __init__(self):
        self.config = config.get_model_config("forecasting")
        self.forecast_horizon = self.config.get("forecast_horizon", 12)
        self.models = {}
        self.forecasts_cache = {}

    def prepare_time_series_data(
        self,
        skills_df: pd.DataFrame,
        date_column: str = "posting_date_clean",
        skill_column: str = "skill_name",
        aggregation: str = "monthly"
    ) -> Dict[str, pd.DataFrame]:
        """Prepare time series data for each skill."""
        logger.info("Preparing time series data for skill demand forecasting...")

        if skills_df.empty:
            logger.warning("Empty skills DataFrame provided")
            return {}

        # Check if date column exists, create sample dates if not
        if date_column not in skills_df.columns:
            logger.warning(f"Date column {date_column} not found, creating sample dates")
            # Create sample dates for the past year
            start_date = datetime.now() - timedelta(days=365)
            skills_df[date_column] = [
                start_date + timedelta(days=i % 365)
                for i in range(len(skills_df))
            ]

        # Ensure date column is datetime
        try:
            skills_df[date_column] = pd.to_datetime(skills_df[date_column], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting dates: {e}")
            return {}

        # Remove rows with invalid dates
        skills_df = skills_df.dropna(subset=[date_column])

        if skills_df.empty:
            logger.warning("No valid dates found in data")
            return {}

        # Set aggregation frequency
        freq_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q"
        }
        freq = freq_map.get(aggregation, "M")

        # Create time series for each skill
        skill_time_series = {}

        # Get skills with sufficient data points
        skill_counts = skills_df[skill_column].value_counts()
        min_occurrences = 10  # Minimum occurrences for forecasting

        for skill in skill_counts[skill_counts >= min_occurrences].index:
            try:
                skill_data = skills_df[skills_df[skill_column] == skill].copy()

                # Aggregate by time period
                time_series = skill_data.groupby(
                    skill_data[date_column].dt.to_period(freq)
                ).size().reset_index()

                time_series.columns = ['ds', 'y']
                time_series['ds'] = time_series['ds'].dt.to_timestamp()

                # Fill missing periods with 0
                if len(time_series) > 2:  # Need minimum data points
                    full_date_range = pd.date_range(
                        start=time_series['ds'].min(),
                        end=time_series['ds'].max(),
                        freq=freq
                    )

                    full_series = pd.DataFrame({'ds': full_date_range})
                    time_series = full_series.merge(time_series, on='ds', how='left')
                    time_series['y'] = time_series['y'].fillna(0)

                    # Add additional features
                    time_series['skill_name'] = skill
                    time_series['trend'] = range(len(time_series))
                    time_series['month'] = time_series['ds'].dt.month
                    time_series['quarter'] = time_series['ds'].dt.quarter
                    time_series['year'] = time_series['ds'].dt.year

                    skill_time_series[skill] = time_series

            except Exception as e:
                logger.warning(f"Error processing skill {skill}: {e}")
                continue

        logger.info(f"Prepared time series data for {len(skill_time_series)} skills")
        return skill_time_series

    def fit_prophet_model(self, time_series: pd.DataFrame, skill_name: str):
        """Fit Prophet model for a specific skill with fallback."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, using simple trend forecasting")
            return None

        try:
            # Configure Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=10.0
            )

            # Fit the model
            prophet_data = time_series[['ds', 'y']].copy()
            model.fit(prophet_data)

            logger.info(f"Successfully fitted Prophet model for {skill_name}")
            return model

        except Exception as e:
            logger.error(f"Error fitting Prophet model for {skill_name}: {e}")
            return None

    def _simple_trend_forecast(self, time_series: pd.DataFrame, skill_name: str) -> Dict[str, Any]:
        """Simple trend-based forecasting when advanced models are not available."""
        try:
            values = time_series['y'].values
            dates = time_series['ds'].values

            # Calculate simple trend
            if len(values) < 3:
                # Not enough data for trend
                forecast_values = [values[-1]] * self.forecast_horizon
            else:
                # Linear trend
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)

                # Forecast future values
                future_x = np.arange(len(values), len(values) + self.forecast_horizon)
                forecast_values = np.polyval(coeffs, future_x)
                forecast_values = np.maximum(forecast_values, 0)  # Ensure non-negative

            # Generate future dates
            last_date = pd.to_datetime(dates[-1])
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_horizon,
                freq='M'
            )

            # Create confidence intervals (simple approach)
            std_dev = np.std(values) if len(values) > 1 else np.mean(values) * 0.1
            lower_bound = forecast_values - 1.96 * std_dev
            upper_bound = forecast_values + 1.96 * std_dev

            forecast_data = {
                "model_type": "simple_trend",
                "skill_name": skill_name,
                "historical_data": {
                    "dates": [d.isoformat() for d in dates],
                    "values": values.tolist()
                },
                "forecast_data": {
                    "dates": [d.isoformat() for d in future_dates],
                    "values": forecast_values.tolist(),
                    "lower_bound": lower_bound.tolist(),
                    "upper_bound": upper_bound.tolist()
                },
                "model_performance": {
                    "mae": 0.0,
                    "mse": 0.0,
                    "rmse": 0.0,
                    "mape": 0.0
                },
                "trend_analysis": {
                    "overall_direction": "increasing" if len(values) > 1 and values[-1] > values[0] else "stable",
                    "trend_slope": float(coeffs[0]) if len(values) >= 3 else 0.0,
                    "volatility": float(std_dev),
                    "growth_rate": 0.0
                }
            }

            return forecast_data

        except Exception as e:
            logger.error(f"Error in simple trend forecast for {skill_name}: {e}")
            return None

    def create_ensemble_forecast(
        self,
        skill_time_series: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Create forecasts using available methods."""
        logger.info("Creating forecasts...")

        forecasts = {}

        for skill, time_series in skill_time_series.items():
            logger.info(f"Forecasting demand for: {skill}")

            try:
                if PROPHET_AVAILABLE:
                    # Try Prophet first
                    model = self.fit_prophet_model(time_series, skill)
                    if model:
                        forecast_result = self._forecast_with_prophet(time_series, skill, model)
                    else:
                        forecast_result = self._simple_trend_forecast(time_series, skill)
                else:
                    # Fallback to simple trend
                    forecast_result = self._simple_trend_forecast(time_series, skill)

                if forecast_result:
                    forecasts[skill] = forecast_result

            except Exception as e:
                logger.error(f"Error forecasting {skill}: {e}")
                continue

        logger.info(f"Generated forecasts for {len(forecasts)} skills")
        return forecasts

    def _forecast_with_prophet(self, time_series: pd.DataFrame, skill_name: str, model) -> Dict[str, Any]:
        """Generate Prophet forecast for a single skill."""
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=self.forecast_horizon, freq='M')

            # Generate forecast
            forecast = model.predict(future)

            # Extract historical and forecast data
            historical_end = len(time_series)

            forecast_data = {
                "model_type": "prophet",
                "skill_name": skill_name,
                "historical_data": {
                    "dates": [d.isoformat() for d in time_series['ds']],
                    "values": time_series['y'].tolist()
                },
                "forecast_data": {
                    "dates": [d.isoformat() for d in forecast['ds'].iloc[historical_end:]],
                    "values": forecast['yhat'].iloc[historical_end:].tolist(),
                    "lower_bound": forecast['yhat_lower'].iloc[historical_end:].tolist(),
                    "upper_bound": forecast['yhat_upper'].iloc[historical_end:].tolist()
                },
                "model_performance": self._evaluate_model_performance(
                    time_series['y'], forecast['yhat'][:historical_end]
                ),
                "trend_analysis": self._analyze_trend(forecast)
            }

            return forecast_data

        except Exception as e:
            logger.error(f"Error in Prophet forecast for {skill_name}: {e}")
            return self._simple_trend_forecast(time_series, skill_name)

    def _evaluate_model_performance(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Evaluate model performance metrics."""
        if not SKLEARN_AVAILABLE:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mape": 0.0}

        try:
            # Align series lengths
            min_length = min(len(actual), len(predicted))
            actual = actual[:min_length]
            predicted = predicted[:min_length]

            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)

            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100

            return {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape)
            }
        except Exception:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mape": 0.0}

    def _analyze_trend(self, forecast: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend components from forecast."""
        try:
            trend_values = forecast['trend'].values if 'trend' in forecast.columns else forecast['yhat'].values

            # Calculate trend direction
            recent_trend = np.polyfit(range(len(trend_values[-12:])), trend_values[-12:], 1)[0]

            trend_analysis = {
                "overall_direction": "increasing" if recent_trend > 0 else "decreasing",
                "trend_slope": float(recent_trend),
                "trend_strength": float(abs(recent_trend)),
                "volatility": float(np.std(np.diff(trend_values))),
                "growth_rate": float((trend_values[-1] - trend_values[0]) / trend_values[0] * 100) if trend_values[0] != 0 else 0.0
            }

            return trend_analysis
        except Exception:
            return {
                "overall_direction": "stable",
                "trend_slope": 0.0,
                "trend_strength": 0.0,
                "volatility": 0.0,
                "growth_rate": 0.0
            }

    def save_forecasts(self, forecasts: Dict[str, Any], file_path: str) -> None:
        """Save forecasts to file."""
        try:
            FileHandler.save_json(forecasts, file_path)
            logger.info(f"Saved forecasts to {file_path}")
        except Exception as e:
            logger.error(f"Error saving forecasts: {e}")

    def load_forecasts(self, file_path: str) -> Dict[str, Any]:
        """Load forecasts from file."""
        try:
            forecasts = FileHandler.load_json(file_path)
            logger.info(f"Loaded forecasts from {file_path}")
            return forecasts
        except Exception as e:
            logger.error(f"Error loading forecasts: {e}")
            return {}
