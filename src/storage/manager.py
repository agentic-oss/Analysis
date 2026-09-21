import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """Storage manager handling multi-format (Parquet, CSV, JSON) file I/O across dataset domains."""

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

    def save_raw_batch(self, data_dict: Dict[str, pd.DataFrame], date_str: str) -> List[str]:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year_str = dt.strftime("%Y")
        month_str = dt.strftime("%m")
        day_dir = os.path.join(self.raw_dir, year_str, month_str, date_str)
        os.makedirs(day_dir, exist_ok=True)

        saved_files = []
        for symbol, df in data_dict.items():
            if df.empty:
                continue
            pq_path = os.path.join(day_dir, f"{symbol}.parquet")
            csv_path = os.path.join(day_dir, f"{symbol}.csv")
            df.to_parquet(pq_path, index=False)
            df.to_csv(csv_path, index=False)
            saved_files.extend([pq_path, csv_path])

        return saved_files

    def save_processed_dataset(self, symbol: str, df: pd.DataFrame, category: str = "metals") -> str:
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        os.makedirs(target_dir, exist_ok=True)

        pq_path = os.path.join(target_dir, f"{symbol}.parquet")

        if os.path.exists(pq_path):
            try:
                existing_df = pd.read_parquet(pq_path)
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                df = combined
            except Exception as e:
                logger.warning(f"Failed to read existing processed dataset for {symbol}: {e}")

        df.to_parquet(pq_path, index=False)
        csv_path = os.path.join(target_dir, f"{symbol}.csv")
        df.to_csv(csv_path, index=False)
        return pq_path

    def load_processed_dataset(self, symbol: str, category: str = "metals") -> pd.DataFrame:
        target_dir = self.processed_metals_dir if category == "metals" else self.processed_macro_dir
        pq_path = os.path.join(target_dir, f"{symbol}.parquet")
        if os.path.exists(pq_path):
            return pd.read_parquet(pq_path)
        csv_path = os.path.join(target_dir, f"{symbol}.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def save_quality_report(self, report: Dict[str, Any], date_str: str) -> str:
        filepath = os.path.join(self.quality_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        return filepath

    def save_features(self, df: pd.DataFrame, date_str: str) -> str:
        pq_path = os.path.join(self.features_dir, f"features_{date_str}.parquet")
        csv_path = os.path.join(self.features_dir, f"features_{date_str}.csv")
        df.to_parquet(pq_path, index=False)
        df.to_csv(csv_path, index=False)
        return pq_path

    def save_daily_analysis(self, analysis_json: Dict[str, Any], date_str: str) -> str:
        filepath = os.path.join(self.analysis_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(analysis_json, f, indent=2)
        return filepath

    def save_alerts(self, alerts_json: Dict[str, Any], date_str: str) -> str:
        filepath = os.path.join(self.alerts_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(alerts_json, f, indent=2)
        return filepath

    def save_daily_predictions(self, pred_json: Dict[str, Any], date_str: str) -> str:
        filepath = os.path.join(self.predictions_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(pred_json, f, indent=2)
        return filepath
