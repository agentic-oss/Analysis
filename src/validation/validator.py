"""
Data Quality Validator for market data.
"""
import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd


class DataValidator:
    """
    Validates ingested batch market data for missing fields, OHLC anomalies, price spikes, zero/negative prices, etc.
    """

    def __init__(self, max_daily_jump_pct: float = 0.15, allow_zero_volume: bool = True):
        self.max_daily_jump_pct = max_daily_jump_pct
        self.allow_zero_volume = allow_zero_volume

    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates a dataframe for a given instrument.
        Returns:
            - Validated DataFrame with a 'quality_status' column ('valid', 'suspicious', 'invalid')
            - Validation summary dict with counts, warnings, and errors.
        """
        summary = {
            "symbol": symbol,
            "total_records": len(df),
            "valid_records": 0,
            "suspicious_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": [],
        }

        if df.empty:
            summary["errors"].append("DataFrame is empty.")
            return df, summary

        df = df.copy()
        df["quality_status"] = "valid"
        df["quality_reason"] = ""

        # Check missing required columns
        req_cols = ["date", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            summary["errors"].append(f"Missing required columns: {missing_cols}")
            df["quality_status"] = "invalid"
            df["quality_reason"] = "missing_columns"
            summary["invalid_records"] = len(df)
            return df, summary

        # 1. Zero or negative prices
        price_cols = ["open", "high", "low", "close"]
        invalid_mask = (df[price_cols] <= 0).any(axis=1)
        if invalid_mask.any():
            invalid_cnt = invalid_mask.sum()
            summary["errors"].append(f"Found {invalid_cnt} records with zero or negative price.")
            df.loc[invalid_mask, "quality_status"] = "invalid"
            df.loc[invalid_mask, "quality_reason"] += "invalid_price;"

        # 2. Invalid OHLC relationships (e.g. low > high, close > high, close < low, open > high, open < low)
        ohlc_invalid = (
            (df["low"] > df["high"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
            | (df["open"] > df["high"])
            | (df["open"] < df["low"])
        )
        if ohlc_invalid.any():
            cnt = ohlc_invalid.sum()
            summary["errors"].append(f"Found {cnt} records with invalid OHLC relationships.")
            df.loc[ohlc_invalid, "quality_status"] = "invalid"
            df.loc[ohlc_invalid, "quality_reason"] += "invalid_ohlc;"

        # 3. Negative volume
        neg_vol = df["volume"] < 0
        if neg_vol.any():
            cnt = neg_vol.sum()
            summary["errors"].append(f"Found {cnt} records with negative volume.")
            df.loc[neg_vol, "quality_status"] = "invalid"
            df.loc[neg_vol, "quality_reason"] += "negative_volume;"

        # 4. Price jump anomaly (suspicious)
        df["close_pct_change"] = df["close"].pct_change().abs()
        suspicious_jump = (df["close_pct_change"] > self.max_daily_jump_pct) & (df["quality_status"] != "invalid")
        if suspicious_jump.any():
            cnt = suspicious_jump.sum()
            summary["warnings"].append(f"Found {cnt} records with daily price jump > {self.max_daily_jump_pct * 100}%.")
            df.loc[suspicious_jump, "quality_status"] = "suspicious"
            df.loc[suspicious_jump, "quality_reason"] += "abnormal_jump;"

        # Drop temporary column
        df.drop(columns=["close_pct_change"], inplace=True, errors="ignore")

        # Counts
        summary["valid_records"] = int((df["quality_status"] == "valid").sum())
        summary["suspicious_records"] = int((df["quality_status"] == "suspicious").sum())
        summary["invalid_records"] = int((df["quality_status"] == "invalid").sum())

        return df, summary

    def create_daily_quality_report(self, date_str: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidates quality summaries for all instruments on a given date.
        """
        total_valid = sum(s["valid_records"] for s in summaries)
        total_suspicious = sum(s["suspicious_records"] for s in summaries)
        total_invalid = sum(s["invalid_records"] for s in summaries)
        all_warnings = [f"{s['symbol']}: {w}" for s in summaries for w in s["warnings"]]
        all_errors = [f"{s['symbol']}: {e}" for s in summaries for e in s["errors"]]

        return {
            "date": date_str,
            "total_instruments": len(summaries),
            "total_valid_records": total_valid,
            "total_suspicious_records": total_suspicious,
            "total_invalid_records": total_invalid,
            "warnings": all_warnings,
            "errors": all_errors,
            "instrument_breakdown": summaries,
        }
