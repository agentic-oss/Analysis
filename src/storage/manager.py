import os
import json
import pandas as pd
from typing import Dict, Any, Optional

class StorageManager:
    """
    Manages structured file persistence (Parquet, CSV, JSON) across data directory hierarchy:
    data/raw/YYYY/MM/YYYY-MM-DD/
    data/processed/metals/
    data/processed/macro/
    data/features/daily/
    data/analysis/daily/
    data/predictions/daily/
    data/backtests/
    data/models/
    """
    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = base_data_dir

    def save_raw_observation(self, df: pd.DataFrame, instrument: str, date_str: str) -> str:
        dt = pd.to_datetime(date_str)
        year_str = dt.strftime("%Y")
        month_str = dt.strftime("%m")
        day_dir = os.path.join(self.base_data_dir, "raw", year_str, month_str, date_str)
        os.makedirs(day_dir, exist_ok=True)

        file_prefix = os.path.join(day_dir, f"{instrument.lower()}")

        # Save Parquet
        parquet_path = f"{file_prefix}.parquet"
        df.to_parquet(parquet_path, index=False)

        # Save CSV for inspection
        csv_path = f"{file_prefix}.csv"
        df.to_csv(csv_path, index=False)

        return parquet_path

    def save_processed_data(self, df: pd.DataFrame, category: str, name: str) -> str:
        category_dir = os.path.join(self.base_data_dir, "processed", category)
        os.makedirs(category_dir, exist_ok=True)

        parquet_path = os.path.join(category_dir, f"{name.lower()}.parquet")
        csv_path = os.path.join(category_dir, f"{name.lower()}.csv")

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        return parquet_path

    def load_processed_data(self, category: str, name: str) -> pd.DataFrame:
        category_dir = os.path.join(self.base_data_dir, "processed", category)
        parquet_path = os.path.join(category_dir, f"{name.lower()}.parquet")
        csv_path = os.path.join(category_dir, f"{name.lower()}.csv")

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            return pd.DataFrame()

    def save_json(self, data: Dict[str, Any], sub_dir: str, filename: str) -> str:
        target_dir = os.path.join(self.base_data_dir, sub_dir)
        os.makedirs(target_dir, exist_ok=True)

        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        filepath = os.path.join(target_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def load_json(self, sub_dir: str, filename: str) -> Dict[str, Any]:
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        filepath = os.path.join(self.base_data_dir, sub_dir, filename)
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "r") as f:
            return json.load(f)
