"""
Data Storage Repository for time-series, features, reports, quality metrics, models, and predictions.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class DataRepository:
    """
    Manages reading and writing data across raw, processed, features, analysis, predictions, quality, models, and backtests directories.
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.features_dir = self.base_dir / "features"
        self.analysis_dir = self.base_dir / "analysis"
        self.predictions_dir = self.base_dir / "predictions"
        self.quality_dir = self.base_dir / "quality"
        self.models_dir = self.base_dir / "models"
        self.backtests_dir = self.base_dir / "backtests"

        # Ensure directories exist
        for d in [
            self.raw_dir,
            self.processed_dir,
            self.features_dir,
            self.analysis_dir,
            self.predictions_dir,
            self.quality_dir,
            self.models_dir,
            self.backtests_dir,
            self.analysis_dir / "alerts",
            self.analysis_dir / "daily",
            self.features_dir / "daily",
            self.predictions_dir / "daily",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def save_raw_data(self, df: pd.DataFrame, symbol: str, date_str: str):
        """
        Save raw observation batch in data/raw/YYYY/MM/YYYY-MM-DD/{symbol}.parquet
        """
        dt = pd.to_datetime(date_str)
        path = self.raw_dir / f"{dt.year:04d}" / f"{dt.month:02d}" / date_str / f"{symbol}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def save_processed_data(self, df: pd.DataFrame, symbol: str):
        """
        Save merged processed time-series for an instrument in data/processed/{symbol}.parquet
        """
        path = self.processed_dir / f"{symbol}.parquet"
        df.to_parquet(path, index=False)

    def load_processed_data(self, symbol: str) -> pd.DataFrame:
        """
        Load processed DataFrame for symbol.
        """
        path = self.processed_dir / f"{symbol}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def save_quality_report(self, report: Dict[str, Any], date_str: str):
        """
        Save quality report in data/quality/{YYYY-MM-DD}.json
        """
        path = self.quality_dir / f"{date_str}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def save_daily_features(self, df: pd.DataFrame, date_str: str):
        """
        Save daily feature store batch.
        """
        path = self.features_dir / "daily" / f"{date_str}.parquet"
        df.to_parquet(path, index=False)

    def load_daily_features(self, date_str: str) -> pd.DataFrame:
        """
        Load daily features dataframe for date.
        """
        path = self.features_dir / "daily" / f"{date_str}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def save_daily_analysis(self, analysis: Dict[str, Any], date_str: str):
        """
        Save daily analysis JSON in data/analysis/daily/{YYYY-MM-DD}.json
        """
        path = self.analysis_dir / "daily" / f"{date_str}.json"
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2)

    def save_alerts(self, alerts: Dict[str, Any], date_str: str):
        """
        Save daily alerts in data/analysis/alerts/{YYYY-MM-DD}.json
        """
        path = self.analysis_dir / "alerts" / f"{date_str}.json"
        with open(path, "w") as f:
            json.dump(alerts, f, indent=2)

    def save_daily_predictions(self, predictions: Dict[str, Any], date_str: str):
        """
        Save daily prediction results.
        """
        path = self.predictions_dir / "daily" / f"{date_str}.json"
        with open(path, "w") as f:
            json.dump(predictions, f, indent=2)
