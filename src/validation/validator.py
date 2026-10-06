import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates incoming ingestion batches and classifies records/datasets as valid, suspicious, or invalid."""

    def __init__(self, jump_threshold_pct: float = 15.0):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataset(self, df: pd.DataFrame, instrument_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df is None or df.empty:
            return df, {"records": 0, "status": "empty", "warnings": ["Dataset is empty"], "errors": []}

        df = df.copy()
        warnings = []
        errors = []

        # Ensure status column
        df["validation_status"] = "valid"

        # 1. Check required columns
        req_cols = ["date", "open", "high", "low", "close"]
        for col in req_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
                return df, {"records": len(df), "status": "invalid", "warnings": warnings, "errors": errors}

        # 2. Check for missing timestamps / NaNs
        null_dates = df["date"].isnull().sum()
        if null_dates > 0:
            errors.append(f"Found {null_dates} null dates")

        # 3. Check for duplicates
        dups = df.duplicated(subset=["date"]).sum()
        if dups > 0:
            warnings.append(f"Found {dups} duplicate date records; keeping first")
            df = df.drop_duplicates(subset=["date"], keep="first")

        # 4. OHLC relationship check & negative/zero prices
        for idx, row in df.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            date_str = str(row["date"])

            if pd.isnull(c) or c <= 0:
                df.at[idx, "validation_status"] = "invalid"
                errors.append(f"Invalid price {c} on {date_str}")
                continue

            if not (l <= o <= h and l <= c <= h):
                df.at[idx, "validation_status"] = "suspicious"
                warnings.append(f"Suspicious OHLC relationship on {date_str}: O={o}, H={h}, L={l}, C={c}")

        # 5. Price jump check
        if len(df) > 1:
            pct_change = df["close"].pct_change().abs() * 100.0
            jump_indices = pct_change[pct_change > self.jump_threshold_pct].index
            for j_idx in jump_indices:
                date_str = str(df.at[j_idx, "date"])
                chg = pct_change.at[j_idx]
                if df.at[j_idx, "validation_status"] == "valid":
                    df.at[j_idx, "validation_status"] = "suspicious"
                warnings.append(f"Abnormal price jump of {chg:.2f}% on {date_str}")

        valid_count = (df["validation_status"] == "valid").sum()
        suspicious_count = (df["validation_status"] == "suspicious").sum()
        invalid_count = (df["validation_status"] == "invalid").sum()

        summary = {
            "records": len(df),
            "valid_records": int(valid_count),
            "suspicious_records": int(suspicious_count),
            "invalid_records": int(invalid_count),
            "warnings": warnings,
            "errors": errors
        }

        return df, summary

    def generate_quality_report(self, date_str: str, validation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        total_gold = validation_results.get("gold_usd", {}).get("records", 0)
        total_silver = validation_results.get("silver_usd", {}).get("records", 0)

        all_warnings = []
        all_errors = []

        for inst, res in validation_results.items():
            for w in res.get("warnings", []):
                all_warnings.append(f"[{inst}] {w}")
            for e in res.get("errors", []):
                all_errors.append(f"[{inst}] {e}")

        return {
            "date": date_str,
            "gold_records": total_gold,
            "silver_records": total_silver,
            "datasets_validated": list(validation_results.keys()),
            "warnings": all_warnings,
            "errors": all_errors
        }
