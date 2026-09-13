import os
import json
import yaml
import pandas as pd
from typing import Any, Dict, Optional

class StorageManager:
    """Manages reading/writing raw, processed, feature, and report datasets."""

    def __init__(self, base_data_dir: str = "data", base_reports_dir: str = "reports"):
        self.base_data_dir = base_data_dir
        self.base_reports_dir = base_reports_dir
        self._ensure_directories()

    def _ensure_directories(self):
        subdirs = [
            "raw",
            "processed/metals",
            "processed/macro",
            "processed/currencies",
            "features/daily",
            "analysis/daily",
            "analysis/alerts",
            "predictions/daily",
            "models",
            "backtests",
            "quality",
            "events"
        ]
        for sd in subdirs:
            os.makedirs(os.path.join(self.base_data_dir, sd), exist_ok=True)
        os.makedirs(os.path.join(self.base_reports_dir, "daily"), exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, relative_path: str, format: str = "parquet") -> str:
        filepath = os.path.join(self.base_data_dir, relative_path)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format == "parquet":
            if not relative_path.endswith(".parquet"):
                filepath += ".parquet"
            df.to_parquet(filepath, index=True)
        elif format == "csv":
            if not relative_path.endswith(".csv"):
                filepath += ".csv"
            df.to_csv(filepath, index=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return filepath

    def load_dataframe(self, relative_path: str, format: str = "parquet") -> pd.DataFrame:
        filepath = os.path.join(self.base_data_dir, relative_path)
        if not os.path.exists(filepath):
            # check extension extensions
            if format == "parquet" and not filepath.endswith(".parquet"):
                filepath += ".parquet"
            elif format == "csv" and not filepath.endswith(".csv"):
                filepath += ".csv"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        if format == "parquet":
            return pd.read_parquet(filepath)
        elif format == "csv":
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_json(self, data: Dict[str, Any], relative_path: str) -> str:
        filepath = os.path.join(self.base_data_dir, relative_path)
        if not filepath.endswith(".json"):
            filepath += ".json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def load_json(self, relative_path: str) -> Dict[str, Any]:
        filepath = os.path.join(self.base_data_dir, relative_path)
        if not filepath.endswith(".json"):
            filepath += ".json"
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_markdown_report(self, content: str, filename: str) -> str:
        filepath = os.path.join(self.base_reports_dir, "daily", filename)
        if not filepath.endswith(".md"):
            filepath += ".md"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_instruments(instruments_path: str = "config/instruments.json") -> Dict[str, Any]:
    with open(instruments_path, "r", encoding="utf-8") as f:
        return json.load(f)
