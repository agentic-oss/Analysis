import os
import json
import logging
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger("StorageManager")


class StorageManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.features_dir = os.path.join(base_dir, "features")
        self.analysis_dir = os.path.join(base_dir, "analysis")
        self.predictions_dir = os.path.join(base_dir, "predictions")
        self.models_dir = os.path.join(base_dir, "models")
        self.backtests_dir = os.path.join(base_dir, "backtests")

        for d in [self.raw_dir, self.processed_dir, self.features_dir, self.analysis_dir, self.predictions_dir, self.models_dir, self.backtests_dir]:
            os.makedirs(d, exist_ok=True)

    def save_raw_data(self, df: pd.DataFrame, instrument_id: str, date_str: str) -> str:
        # data/raw/YYYY/MM/YYYY-MM-DD/
        year, month = date_str.split("-")[0], date_str.split("-")[1]
        target_dir = os.path.join(self.raw_dir, year, month, date_str)
        os.makedirs(target_dir, exist_ok=True)
        filepath = os.path.join(target_dir, f"{instrument_id}.parquet")
        df.to_parquet(filepath, index=False)
        return filepath

    def save_processed_data(self, df: pd.DataFrame, category: str, symbol: str) -> str:
        # data/processed/<category>/<symbol>.parquet
        cat_dir = os.path.join(self.processed_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        filepath = os.path.join(cat_dir, f"{symbol}.parquet")

        if os.path.exists(filepath):
            existing_df = pd.read_parquet(filepath)
            df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        df.to_parquet(filepath, index=False)
        return filepath

    def load_processed_data(self, category: str, symbol: str) -> Optional[pd.DataFrame]:
        filepath = os.path.join(self.processed_dir, category, f"{symbol}.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def save_features(self, df: pd.DataFrame, date_str: str) -> str:
        daily_dir = os.path.join(self.features_dir, "daily")
        os.makedirs(daily_dir, exist_ok=True)
        filepath = os.path.join(daily_dir, f"{date_str}.parquet")
        df.to_parquet(filepath, index=False)
        return filepath

    def save_daily_analysis(self, analysis_data: Dict[str, Any], date_str: str) -> str:
        daily_dir = os.path.join(self.analysis_dir, "daily")
        os.makedirs(daily_dir, exist_ok=True)
        filepath = os.path.join(daily_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(analysis_data, f, indent=2)
        return filepath

    def save_daily_alerts(self, alerts_data: Dict[str, Any], date_str: str) -> str:
        alerts_dir = os.path.join(self.analysis_dir, "alerts")
        os.makedirs(alerts_dir, exist_ok=True)
        filepath = os.path.join(alerts_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(alerts_data, f, indent=2)
        return filepath

    def save_predictions(self, predictions_data: Dict[str, Any], date_str: str) -> str:
        daily_dir = os.path.join(self.predictions_dir, "daily")
        os.makedirs(daily_dir, exist_ok=True)
        filepath = os.path.join(daily_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(predictions_data, f, indent=2)
        return filepath
