import os
import json
import logging
import pandas as pd
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class DataStorage:
    """Manages saving and loading data following Section 6 scalable directory structure."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_metals_dir = os.path.join(base_dir, "processed", "metals")
        self.processed_macro_dir = os.path.join(base_dir, "processed", "macro")
        self.features_dir = os.path.join(base_dir, "features", "daily")
        self.analysis_dir = os.path.join(base_dir, "analysis", "daily")
        self.predictions_dir = os.path.join(base_dir, "predictions", "daily")
        self.backtests_dir = os.path.join(base_dir, "backtests")
        self.models_dir = os.path.join(base_dir, "models")
        self.quality_dir = os.path.join(base_dir, "quality")
        self.alerts_dir = os.path.join(base_dir, "analysis", "alerts")

        for d in [
            self.raw_dir, self.processed_metals_dir, self.processed_macro_dir,
            self.features_dir, self.analysis_dir, self.predictions_dir,
            self.backtests_dir, self.models_dir, self.quality_dir, self.alerts_dir
        ]:
            os.makedirs(d, exist_ok=True)

    def save_raw_data(self, df: pd.DataFrame, date_str: str, symbol: str) -> str:
        """Saves raw market data under data/raw/YYYY/MM/YYYY-MM-DD/{symbol}.parquet."""
        parts = date_str.split("-")
        year, month = parts[0], parts[1]
        target_dir = os.path.join(self.raw_dir, year, month, date_str)
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, f"{symbol}.parquet")
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved raw data for {symbol} to {file_path}")
        return file_path

    def save_processed_data(self, df: pd.DataFrame, category: str, symbol: str) -> str:
        """Saves continuous processed time-series to data/processed/{category}/{symbol}.parquet."""
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, f"{symbol}.parquet")

        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"], keep="last")
            combined_df = combined_df.sort_values("date")
            combined_df.to_parquet(file_path, index=False)
        else:
            df.sort_values("date").to_parquet(file_path, index=False)

        logger.info(f"Updated processed dataset for {symbol} at {file_path}")
        return file_path

    def load_processed_data(self, category: str, symbol: str) -> pd.DataFrame:
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        file_path = os.path.join(target_dir, f"{symbol}.parquet")
        if not os.path.exists(file_path):
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def save_features(self, df: pd.DataFrame, date_str: str) -> str:
        """Saves daily feature snapshot to data/features/daily/{date_str}.parquet."""
        file_path = os.path.join(self.features_dir, f"{date_str}.parquet")
        df.to_parquet(file_path, index=False)
        return file_path

    def load_features(self, date_str: str) -> pd.DataFrame:
        file_path = os.path.join(self.features_dir, f"{date_str}.parquet")
        if not os.path.exists(file_path):
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def save_daily_analysis_json(self, analysis_dict: Dict[str, Any], date_str: str) -> str:
        """Saves daily analysis JSON to data/analysis/daily/{date_str}.json."""
        file_path = os.path.join(self.analysis_dir, f"{date_str}.json")
        with open(file_path, "w") as f:
            json.dump(analysis_dict, f, indent=2)
        return file_path

    def save_daily_predictions_json(self, predictions_dict: Dict[str, Any], date_str: str) -> str:
        """Saves predictions to data/predictions/daily/{date_str}.json."""
        file_path = os.path.join(self.predictions_dir, f"{date_str}.json")
        with open(file_path, "w") as f:
            json.dump(predictions_dict, f, indent=2)
        return file_path
