import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages reading, writing, and merging Parquet, CSV, and JSON data files."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = base_data_dir

    def get_raw_dir(self, dt: Union[str, datetime]) -> str:
        if isinstance(dt, str):
            dt = datetime.strptime(dt, "%Y-%m-%d")
        path = os.path.join(
            self.base_data_dir, "raw", dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%Y-%m-%d")
        )
        os.makedirs(path, exist_ok=True)
        return path

    def save_df(
        self,
        df: pd.DataFrame,
        filepath_without_ext: str,
        formats: Optional[list] = None,
    ):
        """Save a DataFrame as Parquet and/or CSV."""
        if formats is None:
            formats = ["parquet", "csv"]

        os.makedirs(os.path.dirname(filepath_without_ext), exist_ok=True)

        if "parquet" in formats:
            parquet_path = f"{filepath_without_ext}.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved Parquet: {parquet_path}")

        if "csv" in formats:
            csv_path = f"{filepath_without_ext}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")

    def load_df(self, filepath_without_ext: str) -> pd.DataFrame:
        """Load DataFrame preferring Parquet, then CSV."""
        parquet_path = f"{filepath_without_ext}.parquet"
        csv_path = f"{filepath_without_ext}.csv"

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            logger.warning(f"File not found: {filepath_without_ext}")
            return pd.DataFrame()

    def save_json(self, data: Dict[str, Any], filepath: str):
        """Save dictionary as JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON: {filepath}")

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON file if exists, else empty dict."""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def append_or_update_df(
        self,
        new_df: pd.DataFrame,
        filepath_without_ext: str,
        dedup_cols: Optional[list] = None,
    ):
        """Append new observations to an existing time series dataset without duplicating."""
        existing_df = self.load_df(filepath_without_ext)
        if existing_df.empty:
            combined_df = new_df
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            if dedup_cols:
                combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep="last")

        if "Date" in combined_df.columns:
            combined_df["Date"] = pd.to_datetime(combined_df["Date"])
            combined_df = combined_df.sort_values("Date").reset_index(drop=True)

        self.save_df(combined_df, filepath_without_ext)
        return combined_df
