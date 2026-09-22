"""
Data validation module for verifying ingestion batches.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates raw market data and classifies records into valid, suspicious, invalid."""

    def __init__(self, jump_threshold_pct: float = 0.20):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_series(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate OHLCV dataframe.
        Adds 'quality_status' column with values: 'valid', 'suspicious', 'invalid'.
        Returns updated dataframe and audit summary dict.
        """
        audit = {
            "symbol": symbol,
            "total_records": len(df),
            "valid_records": 0,
            "suspicious_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": [],
        }

        if df.empty:
            audit["errors"].append(f"No records found for {symbol}")
            return df, audit

        df = df.copy()
        if "quality_status" not in df.columns:
            df["quality_status"] = "valid"

        # Check required columns
        req_cols = ["date", "close"]
        missing_cols = [col for col in req_cols if col not in df.columns]
        if missing_cols:
            audit["errors"].append(f"Missing required columns: {missing_cols}")
            df["quality_status"] = "invalid"
            audit["invalid_records"] = len(df)
            return df, audit

        # Null or negative / zero close price check
        invalid_mask = df["close"].isna() | (df["close"] <= 0)
        df.loc[invalid_mask, "quality_status"] = "invalid"

        # Check high / low logic if available
        if "high" in df.columns and "low" in df.columns:
            ohlc_invalid = (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
            suspicious_ohlc = ohlc_invalid & (df["quality_status"] != "invalid")
            df.loc[suspicious_ohlc, "quality_status"] = "suspicious"
            if suspicious_ohlc.sum() > 0:
                audit["warnings"].append(f"Found {suspicious_ohlc.sum()} records with abnormal OHLC relationships.")

        # Check abnormal jumps (> threshold daily change)
        returns = df["close"].pct_change().abs()
        jump_mask = (returns > self.jump_threshold_pct) & (df["quality_status"] == "valid")
        df.loc[jump_mask, "quality_status"] = "suspicious"
        if jump_mask.sum() > 0:
            audit["warnings"].append(f"Found {jump_mask.sum()} price jumps exceeding {self.jump_threshold_pct*100}% threshold.")

        # Check for missing timestamps / duplicate dates
        dup_dates = df["date"].duplicated().sum()
        if dup_dates > 0:
            audit["warnings"].append(f"Found {dup_dates} duplicate date entries.")

        audit["valid_records"] = int((df["quality_status"] == "valid").sum())
        audit["suspicious_records"] = int((df["quality_status"] == "suspicious").sum())
        audit["invalid_records"] = int((df["quality_status"] == "invalid").sum())

        return df, audit
