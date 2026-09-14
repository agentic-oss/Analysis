"""
Storage Manager for persisting raw, processed, feature, and report datasets in Parquet, CSV, and JSON formats.
"""
import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages structured storage for raw, processed, feature, prediction, model, and analysis datasets.
    Prefers Parquet for large time-series while supporting CSV and JSON outputs for inspection.
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir

    def save_dataframe(
        self,
        df: pd.DataFrame,
        relative_path: str,
        save_csv: bool = True
    ) -> str:
        """
        Saves DataFrame to Parquet format (and optionally CSV).
        `relative_path` should be relative to `base_dir` (e.g. 'processed/metals/gold.parquet').
        """
        full_path = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Remove extension to get base path
        base_path, _ = os.path.splitext(full_path)

        parquet_path = f"{base_path}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet dataset: {parquet_path}")

        if save_csv:
            csv_path = f"{base_path}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV dataset: {csv_path}")

        return parquet_path

    def load_dataframe(self, relative_path: str) -> pd.DataFrame:
        """
        Loads DataFrame from Parquet or fallback CSV.
        """
        full_path = os.path.join(self.base_dir, relative_path)
        base_path, ext = os.path.splitext(full_path)

        parquet_path = f"{base_path}.parquet"
        csv_path = f"{base_path}.csv"

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        elif os.path.exists(full_path):
            if ext == ".parquet":
                return pd.read_parquet(full_path)
            else:
                return pd.read_csv(full_path)
        else:
            logger.warning(f"File not found: {full_path}")
            return pd.DataFrame()

    def save_json(self, data: Dict[str, Any], relative_path: str) -> str:
        """
        Saves dictionary to JSON file.
        """
        full_path = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved JSON report: {full_path}")
        return full_path

    def load_json(self, relative_path: str) -> Dict[str, Any]:
        """
        Loads dictionary from JSON file.
        """
        full_path = os.path.join(self.base_dir, relative_path)
        if not os.path.exists(full_path):
            logger.warning(f"JSON file not found: {full_path}")
            return {}

        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
