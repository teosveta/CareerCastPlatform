"""File handling utilities for the Skills Demand Forecasting Platform."""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """Centralized file handling operations."""

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if it doesn't."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise

    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save DataFrame to CSV with directory creation."""
        file_path = Path(file_path)
        FileHandler.ensure_directory(file_path.parent)

        try:
            df.to_csv(file_path, index=False, **kwargs)
            logger.info(f"Saved CSV to {file_path}")
        except Exception as e:
            logger.error(f"Error saving CSV to {file_path}: {e}")
            raise

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {e}")
            raise

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save data to JSON file with directory creation."""
        file_path = Path(file_path)
        FileHandler.ensure_directory(file_path.parent)

        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved JSON to {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {e}")
            raise

    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get file information including size and modification time."""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"exists": False}

        stat = file_path.stat()
        return {
            "exists": True,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "is_file": file_path.is_file(),
            "suffix": file_path.suffix
        }

    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
        """List files in directory matching pattern."""
        directory = Path(directory)
        if not directory.exists():
            return []

        return list(directory.glob(pattern))