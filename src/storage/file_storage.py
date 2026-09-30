import json
import logging
import os
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class FileStorage:
    """Storage utilities for reading and writing Parquet, CSV, and JSON data files."""

    @staticmethod
    def ensure_dir(filepath: str) -> None:
        """Ensures that the directory containing filepath exists."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def save_dataframe(
        self, df: pd.DataFrame, filepath_without_ext: str, save_parquet: bool = True, save_csv: bool = True
    ) -> None:
        """Saves a DataFrame to Parquet and/or CSV."""
        if df.empty:
            logger.warning(f"Attempted to save empty DataFrame to {filepath_without_ext}")
            return

        if save_parquet:
            parquet_path = f"{filepath_without_ext}.parquet"
            self.ensure_dir(parquet_path)
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved Parquet dataset to {parquet_path}")

        if save_csv:
            csv_path = f"{filepath_without_ext}.csv"
            self.ensure_dir(csv_path)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV dataset to {csv_path}")

    @classmethod
    def load_dataframe(self, filepath: str) -> pd.DataFrame:
        """Loads a DataFrame from a Parquet or CSV file."""
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        if filepath.endswith(".parquet"):
            return pd.read_parquet(filepath)
        elif filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format for path: {filepath}")

    @classmethod
    def save_json(self, data: Dict[str, Any], filepath: str) -> None:
        """Saves a Python dictionary to JSON."""
        self.ensure_dir(filepath)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON data to {filepath}")

    @classmethod
    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Loads JSON data from file."""
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return {}
        with open(filepath, "r") as f:
            return json.load(f)
