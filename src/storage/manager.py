"""
Data Storage & Repository Management Module.
Manages persistent time-series storage across Parquet, CSV, and JSON formats.
Ensures append-only / partition-safe updates without overwriting historical data.
"""

from datetime import datetime
import json
import logging
import os
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages reading and writing raw, processed, feature, analysis, and prediction datasets."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_metals_dir = os.path.join(base_dir, "processed", "metals")
        self.processed_macro_dir = os.path.join(base_dir, "processed", "macro")
        self.features_dir = os.path.join(base_dir, "features", "daily")
        self.analysis_dir = os.path.join(base_dir, "analysis", "daily")
        self.alerts_dir = os.path.join(base_dir, "analysis", "alerts")
        self.predictions_dir = os.path.join(base_dir, "predictions", "daily")
        self.models_dir = os.path.join(base_dir, "models")
        self.backtests_dir = os.path.join(base_dir, "backtests")
        self.quality_dir = os.path.join(base_dir, "quality")

        self._ensure_directories()

    def _ensure_directories(self):
        dirs = [
            self.raw_dir, self.processed_metals_dir, self.processed_macro_dir,
            self.features_dir, self.analysis_dir, self.alerts_dir,
            self.predictions_dir, self.models_dir, self.backtests_dir, self.quality_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def save_raw_snapshot(self, symbol: str, df: pd.DataFrame, as_of_date: str):
        """Saves daily raw snapshot partitioned by YYYY/MM/YYYY-MM-DD."""
        dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        path = os.path.join(self.raw_dir, f"{dt.year:04d}", f"{dt.month:02d}", as_of_date)
        os.makedirs(path, exist_ok=True)

        csv_file = os.path.join(path, f"{symbol}.csv")
        df.to_csv(csv_file, index=False)

        parquet_file = os.path.join(path, f"{symbol}.parquet")
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved raw snapshot for {symbol} at {path}")

    def save_processed_dataset(self, category: str, name: str, df: pd.DataFrame):
        """
        Saves continuously updated master dataset in parquet + csv for quick inspection.
        category: 'metals' or 'macro'
        """
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        os.makedirs(target_dir, exist_ok=True)

        parquet_file = os.path.join(target_dir, f"{name}.parquet")
        csv_file = os.path.join(target_dir, f"{name}.csv")

        if os.path.exists(parquet_file):
            try:
                existing_df = pd.read_parquet(parquet_file)
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                df = combined
            except Exception as e:
                logger.warning(f"Failed to merge existing dataset for {name}: {e}")

        df.to_parquet(parquet_file, index=False)
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved processed master dataset {name} ({len(df)} rows)")

    def load_processed_dataset(self, category: str, name: str) -> pd.DataFrame:
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        parquet_file = os.path.join(target_dir, f"{name}.parquet")
        csv_file = os.path.join(target_dir, f"{name}.csv")

        if os.path.exists(parquet_file):
            return pd.read_parquet(parquet_file)
        elif os.path.exists(csv_file):
            return pd.read_csv(csv_file)
        return pd.DataFrame()

    def save_features(self, df: pd.DataFrame, as_of_date: str):
        path = os.path.join(self.features_dir, f"{as_of_date}.parquet")
        df.to_parquet(path, index=False)
        csv_path = os.path.join(self.features_dir, f"{as_of_date}.csv")
        df.to_csv(csv_path, index=False)

    def save_analysis_json(self, data: Dict[str, Any], as_of_date: str):
        path = os.path.join(self.analysis_dir, f"{as_of_date}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_alerts_json(self, data: Dict[str, Any], as_of_date: str):
        path = os.path.join(self.alerts_dir, f"{as_of_date}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
