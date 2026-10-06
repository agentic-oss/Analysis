import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DataStorageManager:
    """Manages persistent storage for raw, processed, feature, analysis, report, and model data."""

    def __init__(self, base_dir: str = "data", reports_dir: str = "reports"):
        self.base_dir = base_dir
        self.reports_dir = reports_dir

        self.dirs = {
            "raw": os.path.join(self.base_dir, "raw"),
            "processed_metals": os.path.join(self.base_dir, "processed", "metals"),
            "processed_macro": os.path.join(self.base_dir, "processed", "macro"),
            "features_daily": os.path.join(self.base_dir, "features", "daily"),
            "analysis_daily": os.path.join(self.base_dir, "analysis", "daily"),
            "alerts": os.path.join(self.base_dir, "analysis", "alerts"),
            "predictions_daily": os.path.join(self.base_dir, "predictions", "daily"),
            "models": os.path.join(self.base_dir, "models"),
            "backtests": os.path.join(self.base_dir, "backtests"),
            "quality": os.path.join(self.base_dir, "quality"),
            "reports_daily": os.path.join(self.reports_dir, "daily")
        }

        self._ensure_directories()

    def _ensure_directories(self):
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    def save_raw_data(self, date_str: str, dataset_name: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        parts = date_str.split("-")
        if len(parts) == 3:
            year, month = parts[0], parts[1]
            target_dir = os.path.join(self.dirs["raw"], year, month, date_str)
        else:
            target_dir = os.path.join(self.dirs["raw"], date_str)
        os.makedirs(target_dir, exist_ok=True)

        parquet_path = os.path.join(target_dir, f"{dataset_name}.parquet")
        csv_path = os.path.join(target_dir, f"{dataset_name}.csv")

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved raw dataset {dataset_name} to {target_dir}")

    def save_processed_dataset(self, category: str, name: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        target_dir = self.dirs["processed_metals"] if category == "metals" else self.dirs["processed_macro"]
        os.makedirs(target_dir, exist_ok=True)

        parquet_path = os.path.join(target_dir, f"{name}.parquet")
        csv_path = os.path.join(target_dir, f"{name}.csv")

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed dataset {name} to {target_dir}")

    def load_processed_dataset(self, category: str, name: str) -> Optional[pd.DataFrame]:
        target_dir = self.dirs["processed_metals"] if category == "metals" else self.dirs["processed_macro"]
        parquet_path = os.path.join(target_dir, f"{name}.parquet")
        csv_path = os.path.join(target_dir, f"{name}.csv")

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def save_quality_report(self, date_str: str, report: Dict[str, Any]):
        path = os.path.join(self.dirs["quality"], f"{date_str}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved quality report to {path}")

    def save_analysis_json(self, date_str: str, data: Dict[str, Any]):
        path = os.path.join(self.dirs["analysis_daily"], f"{date_str}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved daily analysis JSON to {path}")

    def save_alerts_json(self, date_str: str, data: Dict[str, Any]):
        path = os.path.join(self.dirs["alerts"], f"{date_str}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved alerts JSON to {path}")

    def save_daily_report_md(self, date_str: str, content: str):
        path = os.path.join(self.dirs["reports_daily"], f"{date_str}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved daily markdown report to {path}")
