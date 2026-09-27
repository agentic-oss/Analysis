import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages reading/writing of raw, processed, feature, prediction, and backtest datasets."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = base_data_dir
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.features_dir = os.path.join(self.base_dir, "features")
        self.analysis_dir = os.path.join(self.base_dir, "analysis")
        self.predictions_dir = os.path.join(self.base_dir, "predictions")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.backtests_dir = os.path.join(self.base_dir, "backtests")
        self.quality_dir = os.path.join(self.base_dir, "quality")

        for d in [self.raw_dir, self.processed_dir, self.features_dir, self.analysis_dir,
                  self.predictions_dir, self.models_dir, self.backtests_dir, self.quality_dir]:
            os.makedirs(d, exist_ok=True)

    def save_processed_dataset(self, df: pd.DataFrame, category: str, name: str) -> str:
        """Saves a processed DataFrame in Parquet format and CSV format for inspection."""
        cat_dir = os.path.join(self.processed_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        parquet_path = os.path.join(cat_dir, f"{name}.parquet")
        csv_path = os.path.join(cat_dir, f"{name}.csv")

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed dataset {name} to {parquet_path}")
        return parquet_path

    def load_processed_dataset(self, category: str, name: str) -> pd.DataFrame:
        """Loads a processed DataFrame from Parquet if available, otherwise CSV."""
        cat_dir = os.path.join(self.processed_dir, category)
        parquet_path = os.path.join(cat_dir, f"{name}.parquet")
        csv_path = os.path.join(cat_dir, f"{name}.csv")

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            logger.warning(f"File {name} not found in {cat_dir}")
            return pd.DataFrame()

    def save_quality_report(self, date_str: str, metrics: Dict[str, Any]) -> str:
        """Saves daily data quality metric summary to JSON."""
        file_path = os.path.join(self.quality_dir, f"{date_str}.json")
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=2)
        return file_path
