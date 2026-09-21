import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Batch data validator for precious metals and cross-asset market datasets."""

    def __init__(self, jump_threshold_pct: float = 15.0):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataframe(self, symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df is None or df.empty:
            return df, {
                "symbol": symbol,
                "records": 0,
                "valid": 0,
                "suspicious": 0,
                "invalid": 0,
                "warnings": [f"Empty dataset for {symbol}"],
                "errors": []
            }

        df = df.copy()
        if "quality_status" not in df.columns:
            df["quality_status"] = "valid"
        if "quality_issues" not in df.columns:
            df["quality_issues"] = ""

        warnings = []
        errors = []

        # 1. Duplicate check
        dup_mask = df.duplicated(subset=["date"], keep=False)
        if dup_mask.any():
            dup_dates = df.loc[dup_mask, "date"].unique().tolist()
            warnings.append(f"Duplicate dates found for {symbol}: {dup_dates}")
            df.loc[dup_mask, "quality_status"] = "suspicious"
            df.loc[dup_mask, "quality_issues"] += "; Duplicate timestamp"

        # 2. OHLC validation
        ohlc_cols = ["open", "high", "low", "close"]
        existing_ohlc = [c for c in ohlc_cols if c in df.columns]

        if len(existing_ohlc) == 4:
            # Check for negative or zero prices
            zero_neg_mask = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
            if zero_neg_mask.any():
                errors.append(f"Zero or negative prices detected in {symbol}")
                df.loc[zero_neg_mask, "quality_status"] = "invalid"
                df.loc[zero_neg_mask, "quality_issues"] += "; Zero or negative price"

            # Check OHLC logical bounds
            ohlc_invalid = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["low"] > df["open"]) | (df["low"] > df["close"])
            if ohlc_invalid.any():
                errors.append(f"Invalid High/Low/Open/Close bounds in {symbol}")
                df.loc[ohlc_invalid, "quality_status"] = "invalid"
                df.loc[ohlc_invalid, "quality_issues"] += "; Invalid OHLC geometry"

        # 3. Abnormal price jump check
        if "close" in df.columns and len(df) > 1:
            price_returns = df["close"].pct_change().abs() * 100.0
            jump_mask = price_returns > self.jump_threshold_pct
            if jump_mask.any():
                jump_dates = df.loc[jump_mask, "date"].tolist()
                warnings.append(f"Abnormal price jump (> {self.jump_threshold_pct}%) in {symbol} on dates: {jump_dates}")
                # Mark jump rows as suspicious, not invalid
                df.loc[jump_mask & (df["quality_status"] != "invalid"), "quality_status"] = "suspicious"
                df.loc[jump_mask, "quality_issues"] += f"; Abnormal jump (> {self.jump_threshold_pct}%)"

        # 4. Volume check
        if "volume" in df.columns:
            neg_vol = df["volume"] < 0
            if neg_vol.any():
                errors.append(f"Negative volume detected in {symbol}")
                df.loc[neg_vol, "quality_status"] = "invalid"
                df.loc[neg_vol, "quality_issues"] += "; Negative volume"

        # Clean quality_issues formatting
        df["quality_issues"] = df["quality_issues"].str.lstrip("; ")

        status_counts = df["quality_status"].value_counts().to_dict()

        report = {
            "symbol": symbol,
            "records": len(df),
            "valid": int(status_counts.get("valid", 0)),
            "suspicious": int(status_counts.get("suspicious", 0)),
            "invalid": int(status_counts.get("invalid", 0)),
            "warnings": warnings,
            "errors": errors
        }

        return df, report

    def validate_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        validated_dict = {}
        batch_reports = {}
        total_records = 0
        all_warnings = []
        all_errors = []

        gold_records = 0
        silver_records = 0

        for symbol, df in data_dict.items():
            val_df, rep = self.validate_dataframe(symbol, df)
            validated_dict[symbol] = val_df
            batch_reports[symbol] = rep

            total_records += rep["records"]
            all_warnings.extend(rep["warnings"])
            all_errors.extend(rep["errors"])

            if "GOLD" in symbol:
                gold_records += rep["records"]
            elif "SILVER" in symbol:
                silver_records += rep["records"]

        summary = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": total_records,
            "gold_records": gold_records,
            "silver_records": silver_records,
            "symbols_validated": list(data_dict.keys()),
            "warnings": all_warnings,
            "errors": all_errors,
            "symbol_details": batch_reports
        }

        return validated_dict, summary
