import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages file storage and data persistence for raw, processed, and quality datasets."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = base_data_dir

    def save_df(
        self,
        df: pd.DataFrame,
        relative_path: str,
        formats: Optional[list] = None
    ):
        """Save DataFrame as parquet and/or csv."""
        if formats is None:
            formats = ['parquet', 'csv']

        full_path_no_ext = os.path.join(self.base_data_dir, relative_path)
        os.makedirs(os.path.dirname(full_path_no_ext), exist_ok=True)

        if 'parquet' in formats:
            try:
                parquet_path = f"{full_path_no_ext}.parquet"
                df.to_parquet(parquet_path, index=False)
            except Exception as e:
                logger.warning(f"Could not save parquet file to {full_path_no_ext}.parquet: {e}")

        if 'csv' in formats:
            csv_path = f"{full_path_no_ext}.csv"
            df.to_csv(csv_path, index=False)

    def load_df(self, relative_path: str) -> pd.DataFrame:
        """Load DataFrame trying parquet first, then csv."""
        full_path_no_ext = os.path.join(self.base_data_dir, relative_path)
        parquet_path = f"{full_path_no_ext}.parquet"
        csv_path = f"{full_path_no_ext}.csv"

        if os.path.exists(parquet_path):
            try:
                return pd.read_parquet(parquet_path)
            except Exception as e:
                logger.warning(f"Failed to load Parquet {parquet_path}: {e}")

        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)

        logger.info(f"File not found: {relative_path}")
        return pd.DataFrame()

    def save_json(self, data: Dict[str, Any], relative_path: str):
        """Save dictionary as JSON."""
        full_path = os.path.join(self.base_data_dir, relative_path)
        if not full_path.endswith('.json'):
            full_path += '.json'
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_json(self, relative_path: str) -> Dict[str, Any]:
        """Load JSON file into dict."""
        full_path = os.path.join(self.base_data_dir, relative_path)
        if not full_path.endswith('.json'):
            full_path += '.json'
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
