"""Storage Manager for persisting raw, processed, feature, analysis, quality, and prediction outputs."""

import json
import os
from typing import Dict, Any, Optional
import pandas as pd


class StorageManager:
    """Manages file storage and directory structures across Parquet, CSV, and JSON formats."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_metals_dir = os.path.join(base_dir, "processed", "metals")
        self.processed_macro_dir = os.path.join(base_dir, "processed", "macro")
        self.features_dir = os.path.join(base_dir, "features", "daily")
        self.analysis_dir = os.path.join(base_dir, "analysis", "daily")
        self.alerts_dir = os.path.join(base_dir, "analysis", "alerts")
        self.predictions_dir = os.path.join(base_dir, "predictions", "daily")
        self.quality_dir = os.path.join(base_dir, "quality")
        self.models_dir = os.path.join(base_dir, "models")
        self.backtests_dir = os.path.join(base_dir, "backtests")

        # Ensure all directories exist
        for d in [
            self.raw_dir, self.processed_metals_dir, self.processed_macro_dir,
            self.features_dir, self.analysis_dir, self.alerts_dir,
            self.predictions_dir, self.quality_dir, self.models_dir, self.backtests_dir
        ]:
            os.makedirs(d, exist_ok=True)

    def save_raw_data(self, df: pd.DataFrame, year: str, month: str, date_str: str, filename: str) -> str:
        """Save raw market data to data/raw/YYYY/MM/YYYY-MM-DD/filename."""
        target_dir = os.path.join(self.raw_dir, year, month, date_str)
        os.makedirs(target_dir, exist_ok=True)
        csv_path = os.path.join(target_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def save_processed_metals(self, df: pd.DataFrame) -> str:
        """Save processed precious metals dataset as Parquet and CSV."""
        parquet_path = os.path.join(self.processed_metals_dir, "metals_processed.parquet")
        csv_path = os.path.join(self.processed_metals_dir, "metals_processed.csv")
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        return parquet_path

    def save_processed_macro(self, df: pd.DataFrame) -> str:
        """Save processed macro dataset as Parquet and CSV."""
        parquet_path = os.path.join(self.processed_macro_dir, "macro_processed.parquet")
        csv_path = os.path.join(self.processed_macro_dir, "macro_processed.csv")
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        return parquet_path

    def save_quality_report(self, report: Dict[str, Any], date_str: str) -> str:
        """Save quality report to data/quality/YYYY-MM-DD.json."""
        filepath = os.path.join(self.quality_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        return filepath

    def save_daily_analysis(self, analysis_dict: Dict[str, Any], date_str: str) -> str:
        """Save daily analysis JSON report."""
        filepath = os.path.join(self.analysis_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(analysis_dict, f, indent=2)
        return filepath

    def save_alerts(self, alerts_dict: Dict[str, Any], date_str: str) -> str:
        """Save daily alerts JSON report."""
        filepath = os.path.join(self.alerts_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(alerts_dict, f, indent=2)
        return filepath

    def save_daily_features(self, df: pd.DataFrame, date_str: str) -> str:
        """Save daily feature dataset."""
        parquet_path = os.path.join(self.features_dir, f"{date_str}.parquet")
        csv_path = os.path.join(self.features_dir, f"{date_str}.csv")
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        return parquet_path

    def save_predictions(self, predictions_dict: Dict[str, Any], date_str: str) -> str:
        """Save predictions output for a date."""
        filepath = os.path.join(self.predictions_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(predictions_dict, f, indent=2)
        return filepath
