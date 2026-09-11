"""
Data Storage Layer.
Manages Parquet, CSV, and JSON persistence for raw, processed, feature, and report datasets.
"""

import json
import os
from typing import Dict, Any, Optional
import pandas as pd


class StorageManager:
    """Manages reading and writing data across raw, processed, feature, and analysis directories."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir

    def get_raw_dir(self, date_str: str) -> str:
        """Returns partition path: data/raw/YYYY/MM/YYYY-MM-DD/"""
        dt_parts = date_str.split("-")
        if len(dt_parts) == 3:
            year, month, _ = dt_parts
            path = os.path.join(self.base_dir, "raw", year, month, date_str)
        else:
            path = os.path.join(self.base_dir, "raw", date_str)
        os.makedirs(path, exist_ok=True)
        return path

    def get_processed_dir(self, category: str = "metals") -> str:
        path = os.path.join(self.base_dir, "processed", category)
        os.makedirs(path, exist_ok=True)
        return path

    def save_dataframe(self, df: pd.DataFrame, file_path: str, save_parquet: bool = True, save_csv: bool = True):
        """Saves DataFrame as Parquet and/or CSV."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        base_path, _ = os.path.splitext(file_path)

        if save_parquet:
            parquet_path = f"{base_path}.parquet"
            df.to_parquet(parquet_path, index=False)

        if save_csv:
            csv_path = f"{base_path}.csv"
            df.to_csv(csv_path, index=False)

    def load_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        """Loads DataFrame from Parquet or CSV."""
        base_path, _ = os.path.splitext(file_path)
        parquet_path = f"{base_path}.parquet"
        csv_path = f"{base_path}.csv"

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def save_json(self, data: Dict[str, Any], file_path: str):
        """Saves dictionary as JSON."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Loads JSON file."""
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None
