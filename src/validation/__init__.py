import json
import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("DataValidator")


class DataValidator:
    def __init__(self, quality_dir: str = "data/quality"):
        self.quality_dir = quality_dir
        os.makedirs(self.quality_dir, exist_ok=True)

    def validate_df(self, df: pd.DataFrame, instrument_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        warnings = []
        errors = []

        if df.empty:
            errors.append(f"Empty dataframe for {instrument_id}")
            return df, {"instrument_id": instrument_id, "valid_records": 0, "warnings": warnings, "errors": errors}

        df = df.copy()

        # 1. Missing timestamps / dates
        null_dates = df["date"].isnull().sum()
        if null_dates > 0:
            errors.append(f"{null_dates} missing date timestamps found.")
            df = df.dropna(subset=["date"])

        # 2. Duplicate dates
        dups = df.duplicated(subset=["date"]).sum()
        if dups > 0:
            warnings.append(f"{dups} duplicate date records found. Deduplicating.")
            df = df.drop_duplicates(subset=["date"], keep="last")

        # 3. OHLC validity
        invalid_ohlc = (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])
        if invalid_ohlc.sum() > 0:
            warnings.append(f"{invalid_ohlc.sum()} records with invalid High/Low relationships detected.")

        # 4. Negative/zero prices
        zero_or_neg = (df["close"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["open"] <= 0)
        if zero_or_neg.sum() > 0:
            warnings.append(f"{zero_or_neg.sum()} records with zero or negative price detected.")

        # 5. Classification
        df["quality_status"] = "valid"

        suspicious_mask = invalid_ohlc | zero_or_neg
        df.loc[suspicious_mask, "quality_status"] = "suspicious"

        # 6. Abnormal price jumps (>20% in 1 day)
        pct_change = df["close"].pct_change().abs()
        jumps = pct_change > 0.20
        if jumps.sum() > 0:
            warnings.append(f"{jumps.sum()} abnormal price jumps (>20%) detected.")
            df.loc[jumps & (df["quality_status"] == "valid"), "quality_status"] = "suspicious"

        summary = {
            "instrument_id": instrument_id,
            "total_records": len(df),
            "valid_records": int((df["quality_status"] == "valid").sum()),
            "suspicious_records": int((df["quality_status"] == "suspicious").sum()),
            "warnings": warnings,
            "errors": errors
        }
        return df, summary

    def save_quality_report(self, date_str: str, summaries: Dict[str, Any]) -> str:
        filepath = os.path.join(self.quality_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(summaries, f, indent=2)
        return filepath
