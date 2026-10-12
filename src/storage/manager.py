import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages reading and writing Parquet, CSV, and JSON files across data directories."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = base_data_dir

    def get_path(self, category: str, filename: str) -> str:
        path = os.path.join(self.base_dir, category)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        category: str,
        filename_prefix: str,
        formats: list = ["parquet", "csv"]
    ) -> Dict[str, str]:
        saved_paths = {}
        for fmt in formats:
            if fmt == "parquet":
                p = self.get_path(category, f"{filename_prefix}.parquet")
                df.to_parquet(p, index=False)
                saved_paths["parquet"] = p
            elif fmt == "csv":
                p = self.get_path(category, f"{filename_prefix}.csv")
                df.to_csv(p, index=False)
                saved_paths["csv"] = p
        return saved_paths

    def load_dataframe(self, category: str, filename: str) -> pd.DataFrame:
        p = os.path.join(self.base_dir, category, filename)
        if not os.path.exists(p):
            logger.warning(f"File not found: {p}")
            return pd.DataFrame()
        if filename.endswith(".parquet"):
            return pd.read_parquet(p)
        elif filename.endswith(".csv"):
            return pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported format for file {filename}")

    def save_json(self, data: Any, category: str, filename: str) -> str:
        p = self.get_path(category, filename)
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        return p

    def load_json(self, category: str, filename: str) -> Any:
        p = os.path.join(self.base_dir, category, filename)
        if not os.path.exists(p):
            return {}
        with open(p, "r") as f:
            return json.load(f)
