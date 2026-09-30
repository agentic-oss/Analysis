import json
import logging
import os
from typing import Dict, Any, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates daily market ingestion batches and classifies records as valid, suspicious, or invalid."""

    def __init__(self, jump_threshold_pct: float = 10.0, max_volume_zscore: float = 5.0):
        self.jump_threshold_pct = jump_threshold_pct
        self.max_volume_zscore = max_volume_zscore

    def validate_dataset(
        self, df: pd.DataFrame, instrument_symbol: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLCV DataFrame for a specific instrument.
        Returns:
            - df with classification column 'validation_status' ('valid', 'suspicious', 'invalid')
            - summary dictionary with errors and warnings
        """
        summary = {
            "symbol": instrument_symbol,
            "total_records": len(df),
            "valid_count": 0,
            "suspicious_count": 0,
            "invalid_count": 0,
            "errors": [],
            "warnings": [],
        }

        if df.empty:
            summary["errors"].append("Dataset is empty.")
            return df, summary

        df = df.copy()
        df["validation_status"] = "valid"
        status = df["validation_status"].values

        # 1. Missing timestamps or duplicates
        if df["date"].isnull().any():
            summary["errors"].append("Missing timestamp values found.")
            df.loc[df["date"].isnull(), "validation_status"] = "invalid"

        duplicate_dates = df[df.duplicated(subset=["date"], keep=False)]
        if not duplicate_dates.empty:
            summary["errors"].append(f"Found {len(duplicate_dates)} duplicate date entries.")
            df.loc[df.duplicated(subset=["date"]), "validation_status"] = "invalid"

        # 2. Non-positive prices
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            invalid_mask = df[col] <= 0
            if invalid_mask.any():
                summary["errors"].append(f"Non-positive values found in column '{col}'.")
                df.loc[invalid_mask, "validation_status"] = "invalid"

        # 3. OHLC relationship check: low <= open, high, close and high >= open, low, close
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            invalid_ohlc = (
                (df["low"] > df["open"])
                | (df["low"] > df["high"])
                | (df["low"] > df["close"])
                | (df["high"] < df["open"])
                | (df["high"] < df["low"])
                | (df["high"] < df["close"])
            )
            if invalid_ohlc.any():
                summary["errors"].append(f"Found {invalid_ohlc.sum()} invalid OHLC relationships.")
                df.loc[invalid_ohlc, "validation_status"] = "invalid"

        # 4. Abnormal price jumps (Suspicious)
        if "close" in df.columns and len(df) > 1:
            price_change = df["close"].pct_change().abs() * 100.0
            suspicious_jumps = price_change > self.jump_threshold_pct
            if suspicious_jumps.any():
                count = suspicious_jumps.sum()
                summary["warnings"].append(
                    f"Found {count} price jumps exceeding {self.jump_threshold_pct}%."
                )
                # Mark as suspicious unless already invalid
                df.loc[suspicious_jumps & (df["validation_status"] == "valid"), "validation_status"] = "suspicious"

        # 5. Volume check
        if "volume" in df.columns:
            negative_vol = df["volume"] < 0
            if negative_vol.any():
                summary["errors"].append(f"Found {negative_vol.sum()} negative volume records.")
                df.loc[negative_vol, "validation_status"] = "invalid"

        # Counts summary
        summary["valid_count"] = int((df["validation_status"] == "valid").sum())
        summary["suspicious_count"] = int((df["validation_status"] == "suspicious").sum())
        summary["invalid_count"] = int((df["validation_status"] == "invalid").sum())

        return df, summary

    def generate_quality_report(
        self, date_str: str, summaries: List[Dict[str, Any]], output_path: str
    ) -> Dict[str, Any]:
        """Saves quality validation report to JSON."""
        report = {
            "date": date_str,
            "instruments_validated": len(summaries),
            "summaries": summaries,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return report
