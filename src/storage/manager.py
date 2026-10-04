import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages reading and writing precious metals data across raw, processed, feature, and analysis tiers."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir

    def save_raw(self, date_str: str, symbol: str, df: pd.DataFrame) -> str:
        """Save raw downloaded observation batch."""
        year, month = date_str.split("-")[0], date_str.split("-")[1]
        raw_dir = os.path.join(self.base_dir, "raw", year, month, date_str)
        os.makedirs(raw_dir, exist_ok=True)

        filepath = os.path.join(raw_dir, f"{symbol}.parquet")
        df.to_parquet(filepath, index=False)
        return filepath

    def save_processed(self, category: str, symbol: str, df: pd.DataFrame) -> str:
        """Save normalized and validated dataset in data/processed/{category}/."""
        proc_dir = os.path.join(self.base_dir, "processed", category)
        os.makedirs(proc_dir, exist_ok=True)

        filepath = os.path.join(proc_dir, f"{symbol}.parquet")
        df.to_parquet(filepath, index=False)
        return filepath

    def load_processed(self, category: str, symbol: str) -> Optional[pd.DataFrame]:
        """Load processed dataset if exists."""
        filepath = os.path.join(self.base_dir, "processed", category, f"{symbol}.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def save_features(self, df: pd.DataFrame, filename: str = "daily_features.parquet") -> str:
        """Save feature dataset."""
        feat_dir = os.path.join(self.base_dir, "features", "daily")
        os.makedirs(feat_dir, exist_ok=True)
        filepath = os.path.join(feat_dir, filename)
        df.to_parquet(filepath, index=False)
        return filepath

    def load_features(self, filename: str = "daily_features.parquet") -> Optional[pd.DataFrame]:
        filepath = os.path.join(self.base_dir, "features", "daily", filename)
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def save_daily_analysis_json(self, date_str: str, analysis_data: Dict[str, Any]) -> str:
        """Save machine-readable daily research report JSON."""
        analysis_dir = os.path.join(self.base_dir, "analysis", "daily")
        os.makedirs(analysis_dir, exist_ok=True)
        filepath = os.path.join(analysis_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(analysis_data, f, indent=2)
        return filepath

    def save_daily_alerts_json(self, date_str: str, alerts_data: Dict[str, Any]) -> str:
        """Save daily alert report JSON."""
        alerts_dir = os.path.join(self.base_dir, "analysis", "alerts")
        os.makedirs(alerts_dir, exist_ok=True)
        filepath = os.path.join(alerts_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(alerts_data, f, indent=2)
        return filepath


def enrich_with_inr_conversions(
    df: pd.DataFrame,
    usdinr_df: pd.DataFrame,
    unit_conversion_factor: float
) -> pd.DataFrame:
    """Enrich USD price time series with explicit converted INR prices, USD/INR exchange rate, and timestamp."""
    df = df.copy()
    if "date" not in df.columns or "close" not in df.columns:
        return df

    fx_slice = usdinr_df[["date", "close"]].rename(columns={"close": "usdinr_rate"})
    merged = pd.merge(df, fx_slice, on="date", how="left")
    merged["usdinr_rate"] = merged["usdinr_rate"].ffill().bfill()

    # Convert USD price per unit (e.g. USD/oz) to INR per unit (e.g. INR/10g or INR/kg)
    # price_inr = close_usd * usdinr_rate * unit_conversion_factor
    merged["close_inr"] = merged["close"] * merged["usdinr_rate"] * unit_conversion_factor
    if "open" in merged.columns:
        merged["open_inr"] = merged["open"] * merged["usdinr_rate"] * unit_conversion_factor
    if "high" in merged.columns:
        merged["high_inr"] = merged["high"] * merged["usdinr_rate"] * unit_conversion_factor
    if "low" in merged.columns:
        merged["low_inr"] = merged["low"] * merged["usdinr_rate"] * unit_conversion_factor

    merged["conversion_factor_used"] = unit_conversion_factor
    merged["currency_original"] = merged.get("currency", "USD")
    merged["currency_converted"] = "INR"

    return merged
