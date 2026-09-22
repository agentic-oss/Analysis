"""
Data storage manager for handling Parquet, CSV, and JSON formats.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DataStorage:
    """Manages reading and writing data across parquet, csv, and json formats."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir

    def save_dataframe(self, df: pd.DataFrame, path_relative: str, formats: list = ["parquet", "csv"]):
        """Save dataframe into specified relative path in multiple formats."""
        full_path_without_ext = os.path.join(self.base_dir, path_relative)
        dir_name = os.path.dirname(full_path_without_ext)
        os.makedirs(dir_name, exist_ok=True)

        if "parquet" in formats:
            parquet_path = f"{full_path_without_ext}.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved parquet to {parquet_path}")

        if "csv" in formats:
            csv_path = f"{full_path_without_ext}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved csv to {csv_path}")

    def save_json(self, data: Dict[str, Any], path_relative: str):
        """Save dict into specified relative json path."""
        full_path = os.path.join(self.base_dir, path_relative)
        if not full_path.endswith(".json"):
            full_path = f"{full_path}.json"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved json to {full_path}")

    def load_dataframe(self, path_relative: str, prefer_format: str = "parquet") -> pd.DataFrame:
        """Load dataframe from parquet or csv if available."""
        base_path = os.path.join(self.base_dir, path_relative)
        parquet_path = f"{base_path}.parquet" if not base_path.endswith(".parquet") else base_path
        csv_path = f"{base_path}.csv" if not base_path.endswith(".csv") else base_path

        if prefer_format == "parquet" and os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        elif os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        else:
            logger.warning(f"File not found: {base_path}")
            return pd.DataFrame()
