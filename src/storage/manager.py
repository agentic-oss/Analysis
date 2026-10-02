import os
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages reading and writing data in Parquet, JSON, and CSV across system directories."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = base_data_dir
        self.raw_dir = os.path.join(base_data_dir, "raw")
        self.processed_metals_dir = os.path.join(base_data_dir, "processed", "metals")
        self.processed_macro_dir = os.path.join(base_data_dir, "processed", "macro")
        self.features_dir = os.path.join(base_data_dir, "features", "daily")
        self.analysis_dir = os.path.join(base_data_dir, "analysis", "daily")
        self.alerts_dir = os.path.join(base_data_dir, "analysis", "alerts")
        self.predictions_dir = os.path.join(base_data_dir, "predictions", "daily")
        self.quality_dir = os.path.join(base_data_dir, "quality")
        self.models_dir = os.path.join(base_data_dir, "models")
        self.backtests_dir = os.path.join(base_data_dir, "backtests")

        # Ensure all directories exist
        for path in [self.raw_dir, self.processed_metals_dir, self.processed_macro_dir,
                     self.features_dir, self.analysis_dir, self.alerts_dir, self.predictions_dir,
                     self.quality_dir, self.models_dir, self.backtests_dir]:
            os.makedirs(path, exist_ok=True)

    def save_raw_data(self, df: pd.DataFrame, date_str: str, symbol: str) -> str:
        """Save raw observation dataframe to data/raw/YYYY/MM/YYYY-MM-DD/symbol.parquet"""
        if df.empty:
            return ""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            path_dir = os.path.join(self.raw_dir, f"{dt.year:04d}", f"{dt.month:02d}", date_str)
            os.makedirs(path_dir, exist_ok=True)
            filepath = os.path.join(path_dir, f"{symbol}.parquet")
            df.to_parquet(filepath, index=False)
            return filepath
        except Exception as e:
            logger.error(f"Error saving raw data for {symbol} on {date_str}: {e}")
            return ""

    def save_processed_data(self, df: pd.DataFrame, category: str, symbol: str) -> str:
        """Save processed historical time series dataset for a symbol."""
        if df.empty:
            return ""
        try:
            sub_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
            filepath_parquet = os.path.join(sub_dir, f"{symbol}.parquet")
            filepath_csv = os.path.join(sub_dir, f"{symbol}.csv")

            df.to_parquet(filepath_parquet, index=False)
            df.to_csv(filepath_csv, index=False)
            return filepath_parquet
        except Exception as e:
            logger.error(f"Error saving processed data for {symbol}: {e}")
            return ""

    def load_processed_data(self, category: str, symbol: str) -> pd.DataFrame:
        """Load processed time series dataframe for a symbol."""
        sub_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        filepath = os.path.join(sub_dir, f"{symbol}.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        filepath_csv = os.path.join(sub_dir, f"{symbol}.csv")
        if os.path.exists(filepath_csv):
            return pd.read_csv(filepath_csv)
        return pd.DataFrame()

    def save_features(self, df: pd.DataFrame, date_str: str) -> str:
        """Save daily feature dataset."""
        if df.empty:
            return ""
        filepath = os.path.join(self.features_dir, f"{date_str}.parquet")
        df.to_parquet(filepath, index=False)
        csv_path = os.path.join(self.features_dir, f"{date_str}.csv")
        df.to_csv(csv_path, index=False)
        return filepath

    def load_features(self, date_str: str) -> pd.DataFrame:
        filepath = os.path.join(self.features_dir, f"{date_str}.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return pd.DataFrame()

    def save_analysis_json(self, data: dict, date_str: str) -> str:
        """Save daily machine-readable analysis JSON."""
        filepath = os.path.join(self.analysis_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def save_alerts_json(self, data: dict, date_str: str) -> str:
        """Save daily alerts JSON."""
        filepath = os.path.join(self.alerts_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def save_predictions_json(self, data: dict, date_str: str) -> str:
        """Save daily prediction outputs JSON."""
        filepath = os.path.join(self.predictions_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def save_quality_json(self, data: dict, date_str: str) -> str:
        """Save data quality report JSON."""
        filepath = os.path.join(self.quality_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
