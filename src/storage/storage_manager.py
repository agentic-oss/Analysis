import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Manages reading and writing data across raw, processed, features, macro, analysis, and models directories.
    """
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.features_dir = os.path.join(base_dir, "features")
        self.macro_dir = os.path.join(base_dir, "macro")
        self.analysis_dir = os.path.join(base_dir, "analysis")
        self.predictions_dir = os.path.join(base_dir, "predictions")
        self.models_dir = os.path.join(base_dir, "models")
        self.backtests_dir = os.path.join(base_dir, "backtests")
        self.quality_dir = os.path.join(base_dir, "quality")

        for d in [self.raw_dir, self.processed_dir, self.features_dir, self.macro_dir,
                  self.analysis_dir, self.predictions_dir, self.models_dir, self.backtests_dir, self.quality_dir]:
            os.makedirs(d, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, relative_path: str, format: str = "parquet") -> str:
        """
        Saves a DataFrame as Parquet or CSV.
        """
        full_path = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if format.lower() == "parquet":
            if not full_path.endswith(".parquet"):
                full_path += ".parquet"
            df.to_parquet(full_path, index=True)
        elif format.lower() == "csv":
            if not full_path.endswith(".csv"):
                full_path += ".csv"
            df.to_csv(full_path, index=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info(f"Saved DataFrame to {full_path}")
        return full_path

    def load_dataframe(self, relative_path: str, format: str = "parquet") -> pd.DataFrame:
        """
        Loads a DataFrame from Parquet or CSV.
        """
        full_path = os.path.join(self.base_dir, relative_path)
        if format.lower() == "parquet":
            if not full_path.endswith(".parquet"):
                full_path += ".parquet"
            return pd.read_parquet(full_path)
        elif format.lower() == "csv":
            if not full_path.endswith(".csv"):
                full_path += ".csv"
            return pd.read_csv(full_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_json(self, data: Dict[str, Any], relative_path: str) -> str:
        """
        Saves a dictionary as JSON.
        """
        full_path = os.path.join(self.base_dir, relative_path)
        if not full_path.endswith(".json"):
            full_path += ".json"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON to {full_path}")
        return full_path

    def load_json(self, relative_path: str) -> Dict[str, Any]:
        """
        Loads JSON file.
        """
        full_path = os.path.join(self.base_dir, relative_path)
        if not full_path.endswith(".json"):
            full_path += ".json"
        with open(full_path, "r") as f:
            return json.load(f)
