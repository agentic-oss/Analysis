import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages reading and writing precious metals time-series data, feature stores, and artifacts."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = base_data_dir
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_metals_dir = os.path.join(self.base_dir, "processed/metals")
        self.processed_macro_dir = os.path.join(self.base_dir, "processed/macro")
        self.features_dir = os.path.join(self.base_dir, "features/daily")
        self.analysis_dir = os.path.join(self.base_dir, "analysis/daily")
        self.predictions_dir = os.path.join(self.base_dir, "predictions/daily")

        for d in [
            self.raw_dir,
            self.processed_metals_dir,
            self.processed_macro_dir,
            self.features_dir,
            self.analysis_dir,
            self.predictions_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def save_raw_snapshot(self, df: pd.DataFrame, symbol: str, date_str: str) -> str:
        """Saves daily raw snapshot as CSV / Parquet."""
        year, month = date_str.split("-")[0], date_str.split("-")[1]
        target_dir = os.path.join(self.raw_dir, year, month, date_str)
        os.makedirs(target_dir, exist_ok=True)

        file_base = os.path.join(target_dir, f"{symbol.lower()}")
        df.to_csv(f"{file_base}.csv", index=False)
        try:
            df.to_parquet(f"{file_base}.parquet", index=False)
        except Exception as e:
            logger.warning(f"Failed to write parquet for {symbol}: {e}")

        return file_base

    def save_processed(self, df: pd.DataFrame, symbol: str, category: str = "metals") -> str:
        """Saves cumulated processed dataset for an instrument."""
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        os.makedirs(target_dir, exist_ok=True)

        file_base = os.path.join(target_dir, f"{symbol.lower()}")
        df.to_csv(f"{file_base}.csv", index=False)
        try:
            df.to_parquet(f"{file_base}.parquet", index=False)
        except Exception as e:
            logger.warning(f"Failed to save parquet for {symbol}: {e}")

        return file_base

    def load_processed(self, symbol: str, category: str = "metals") -> pd.DataFrame:
        """Loads processed dataset for an instrument."""
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        parquet_path = os.path.join(target_dir, f"{symbol.lower()}.parquet")
        csv_path = os.path.join(target_dir, f"{symbol.lower()}.csv")

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            return pd.DataFrame()

    def save_features(self, df: pd.DataFrame, date_str: str) -> str:
        """Saves feature store dataset for a specific date."""
        path_csv = os.path.join(self.features_dir, f"{date_str}.csv")
        path_parquet = os.path.join(self.features_dir, f"{date_str}.parquet")
        df.to_csv(path_csv, index=False)
        try:
            df.to_parquet(path_parquet, index=False)
        except Exception as e:
            logger.warning(f"Failed to save feature parquet: {e}")
        return path_csv
