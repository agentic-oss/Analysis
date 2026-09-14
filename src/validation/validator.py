"""
Data Validation and Quality Assurance Module.
Validates ingestion batches and logs quality metrics.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates precious-metals and macro market datasets.
    Classifies observations into: valid, suspicious, or invalid.
    Does NOT delete unusual observations, but tags and reports them.
    """

    def __init__(self, max_daily_jump_pct: float = 15.0):
        self.max_daily_jump_pct = max_daily_jump_pct

    def validate_dataset(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLCV dataset for missing timestamps, duplicate records, invalid OHLC,
        zero/negative prices, and abnormal price jumps.
        Returns:
            - df_validated: DataFrame with 'quality_status' column
            - quality_report: Dict summarising data quality analysis
        """
        if df.empty:
            return df, {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "symbol": symbol,
                "record_count": 0,
                "valid_count": 0,
                "suspicious_count": 0,
                "invalid_count": 0,
                "warnings": [f"Empty dataset received for symbol {symbol}"],
                "errors": [f"No data available for symbol {symbol}"]
            }

        df_val = df.copy()
        warnings: List[str] = []
        errors: List[str] = []
        quality_status = pd.Series("valid", index=df_val.index)

        # 1. Missing Timestamps & Duplicates
        if df_val["date"].duplicated().any():
            dup_dates = df_val[df_val["date"].duplicated()]["date"].tolist()
            warnings.append(f"Duplicate dates found for {symbol}: {dup_dates[:5]}")
            quality_status[df_val["date"].duplicated()] = "suspicious"

        # 2. Check for missing values in core price columns
        for col in ["open", "high", "low", "close"]:
            if df_val[col].isnull().any():
                null_cnt = df_val[col].isnull().sum()
                errors.append(f"Found {null_cnt} missing values in {col} column for {symbol}")
                quality_status[df_val[col].isnull()] = "invalid"

        # 3. Negative or zero prices
        for col in ["open", "high", "low", "close"]:
            invalid_price_mask = df_val[col] <= 0
            if invalid_price_mask.any():
                invalid_cnt = invalid_price_mask.sum()
                errors.append(f"Found {invalid_cnt} zero/negative values in {col} column for {symbol}")
                quality_status[invalid_price_mask] = "invalid"

        # 4. Invalid OHLC Relationships (High < Low, High < Open/Close, Low > Open/Close)
        invalid_ohlc = (
            (df_val["high"] < df_val["low"]) |
            (df_val["high"] < df_val["open"]) |
            (df_val["high"] < df_val["close"]) |
            (df_val["low"] > df_val["open"]) |
            (df_val["low"] > df_val["close"])
        )
        if invalid_ohlc.any():
            invalid_cnt = invalid_ohlc.sum()
            errors.append(f"Found {invalid_cnt} invalid OHLC relationship records for {symbol}")
            quality_status[invalid_ohlc] = "invalid"

        # 5. Abnormal Price Jumps (e.g. > 15% daily change)
        if "close" in df_val.columns and len(df_val) > 1:
            price_pct_change = df_val["close"].pct_change().abs() * 100
            abnormal_jumps = price_pct_change > self.max_daily_jump_pct
            if abnormal_jumps.any():
                jump_dates = df_val.loc[abnormal_jumps, "date"].tolist()
                warnings.append(f"Abnormal daily price jump (> {self.max_daily_jump_pct}%) detected on dates: {jump_dates[:5]}")
                # Mark as suspicious unless already invalid
                suspicious_mask = abnormal_jumps & (quality_status != "invalid")
                quality_status[suspicious_mask] = "suspicious"

        df_val["quality_status"] = quality_status

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "symbol": symbol,
            "record_count": len(df_val),
            "valid_count": int((quality_status == "valid").sum()),
            "suspicious_count": int((quality_status == "suspicious").sum()),
            "invalid_count": int((quality_status == "invalid").sum()),
            "warnings": warnings,
            "errors": errors
        }

        return df_val, report
