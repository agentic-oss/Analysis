"""Storage module for time-series, features, reports, and models persistence."""

import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages Parquet, CSV, and JSON persistence with deduplication and append safety."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir

    def save_parquet(self, df: pd.DataFrame, relative_path: str, key_cols: Optional[list] = None) -> str:
        """Saves or updates a Parquet file safely, merging with existing data if present."""
        full_path = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        if os.path.exists(full_path) and key_cols:
            try:
                existing_df = pd.read_parquet(full_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=key_cols, keep='last')
                df = combined_df
            except Exception as e:
                logger.warning(f"Could not merge with existing parquet file at {full_path}: {e}")

        df.to_parquet(full_path, index=False)
        logger.info(f"Saved Parquet data to {full_path}")
        return full_path

    def read_parquet(self, relative_path: str) -> pd.DataFrame:
        """Reads Parquet file, returning empty DataFrame if file missing."""
        full_path = os.path.join(self.base_dir, relative_path)
        if not os.path.exists(full_path):
            return pd.DataFrame()
        return pd.read_parquet(full_path)

    def save_csv(self, df: pd.DataFrame, relative_path: str) -> str:
        """Saves a CSV file."""
        full_path = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=False)
        logger.info(f"Saved CSV data to {full_path}")
        return full_path

    def save_json(self, data: Dict[str, Any], relative_path: str) -> str:
        """Saves a JSON file."""
        full_path = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON data to {full_path}")
        return full_path

    def read_json(self, relative_path: str) -> Dict[str, Any]:
        """Reads a JSON file, returning empty dict if missing."""
        full_path = os.path.join(self.base_dir, relative_path)
        if not os.path.exists(full_path):
            return {}
        with open(full_path, "r") as f:
            return json.load(f)
