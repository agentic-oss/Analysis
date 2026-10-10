import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates OHLCV time-series data for gold, silver, currencies, and macro series.
    Classifies records and generates data quality report JSON.
    """

    def __init__(self, max_jump_pct: float = 20.0):
        self.max_jump_pct = max_jump_pct

    def validate_series(self, symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLC DataFrame.
        Adds a 'quality_status' column ('valid', 'suspicious', 'invalid').
        Returns (validated_df, summary_report).
        """
        if df.empty:
            return df, {
                "symbol": symbol,
                "total_records": 0,
                "valid_records": 0,
                "suspicious_records": 0,
                "invalid_records": 0,
                "warnings": ["DataFrame is empty"],
                "errors": []
            }

        df_val = df.copy()
        warnings = []
        errors = []

        status = np.full(len(df_val), "valid", dtype=object)

        # 1. Missing timestamps / NaN prices
        close_col = "Close" if "Close" in df_val.columns else df_val.columns[0]
        nan_mask = df_val[close_col].isna()
        if nan_mask.any():
            status[nan_mask] = "invalid"
            errors.append(f"Found {nan_mask.sum()} missing Close values in {symbol}")

        # 2. Non-positive prices
        non_pos = (df_val[close_col] <= 0) & (~nan_mask)
        if non_pos.any():
            status[non_pos] = "invalid"
            errors.append(f"Found {non_pos.sum()} non-positive price values in {symbol}")

        # 3. Invalid High/Low/Open OHLC logic
        if all(c in df_val.columns for c in ["Open", "High", "Low", "Close"]):
            invalid_ohlc = (df_val["Low"] > df_val["High"]) | \
                           (df_val["Close"] > df_val["High"] * 1.0001) | \
                           (df_val["Close"] < df_val["Low"] * 0.9999) | \
                           (df_val["Open"] > df_val["High"] * 1.0001) | \
                           (df_val["Open"] < df_val["Low"] * 0.9999)
            invalid_ohlc = invalid_ohlc & (~nan_mask)
            if invalid_ohlc.any():
                status[invalid_ohlc] = "suspicious"
                warnings.append(f"Found {invalid_ohlc.sum()} abnormal OHLC relationships in {symbol}")

        # 4. Abnormal Price Jumps
        pct_change = df_val[close_col].pct_change().abs() * 100
        jump_mask = (pct_change > self.max_jump_pct) & (~nan_mask)
        if jump_mask.any():
            status[jump_mask] = "suspicious"
            warnings.append(f"Found {jump_mask.sum()} price jumps exceeding {self.max_jump_pct}% in {symbol}")

        df_val["quality_status"] = status

        summary = {
            "symbol": symbol,
            "total_records": len(df_val),
            "valid_records": int((df_val["quality_status"] == "valid").sum()),
            "suspicious_records": int((df_val["quality_status"] == "suspicious").sum()),
            "invalid_records": int((df_val["quality_status"] == "invalid").sum()),
            "warnings": warnings,
            "errors": errors
        }

        return df_val, summary

    def build_quality_report(self, date_str: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines instrument summaries into daily data quality report schema.
        """
        total_valid = sum(s["valid_records"] for s in summaries)
        total_records = sum(s["total_records"] for s in summaries)
        all_warnings = [w for s in summaries for w in s["warnings"]]
        all_errors = [e for s in summaries for e in s["errors"]]

        return {
            "date": date_str,
            "total_instruments_checked": len(summaries),
            "total_records_checked": total_records,
            "total_valid_records": total_valid,
            "overall_status": "PASSED" if len(all_errors) == 0 else "WARNINGS_OR_ERRORS",
            "warnings": all_warnings,
            "errors": all_errors,
            "instrument_details": summaries
        }
