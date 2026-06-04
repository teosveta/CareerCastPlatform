"""Configuration management utilities for the Skills Demand Forecasting Platform."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, config_path: Optional[str] = None):
        # Look for config file in project root
        if config_path is None:
            # Try to find config.yaml starting from current file location
            current_dir = Path(__file__).parent
            # Go up to project root
            project_root = current_dir.parent.parent
            config_path = project_root / "config.yaml"

            # If not found, try current working directory
            if not config_path.exists():
                config_path = Path.cwd() / "config.yaml"

            # If still not found, use default config
            if not config_path.exists():
                logging.warning(f"Configuration file not found. Using default config.")
                self._config = self._get_default_config()
                return

        self.config_path = str(config_path)
        self._config = None
        self._load_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Provide default configuration when config file is not found."""
        return {
            "app": {
                "name": "Skills Demand Forecasting Platform",
                "version": "1.0.0",
                "debug": True
            },
            "data": {
                "raw_data_path": "data/raw",
                "processed_data_path": "data/processed",
                "unified_data_path": "data/processed/unified",
                "sample_data_path": "data/processed/sample"
            },
            "nlp": {
                "skill_extraction": {
                    "method": "hybrid",
                    "confidence_threshold": 0.7,
                    "max_skills_per_job": 50
                },
                "models": {
                    "spacy_model": "en_core_web_sm",
                    "sentence_transformer": "all-MiniLM-L6-v2"
                }
            },
            "forecasting": {
                "granularity": "monthly",
                "forecast_horizon": 12,
                "models": {
                    "primary": "prophet",
                    "fallback": "arima",
                    "ensemble": True
                }
            },
            "ml_models": {
                "salary_prediction": {
                    "model_type": "xgboost",
                    "features": ["skills", "location", "experience", "company_size"],
                    "target": "median_salary"
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "timeout": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/app.log"
            }
        }

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Configuration file not found: {self.config_path}")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            self._config = self._get_default_config()

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'data.processed_data_path')."""
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.get('data', {})

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get specific model configuration."""
        return self.get(f'{model_type}', {})

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get('database', {})

# Initialize global config manager
config = ConfigManager()