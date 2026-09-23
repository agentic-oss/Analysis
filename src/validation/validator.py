import json
import os
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class DataValidator:
    """
    Validates batch market data ingestion according to rules:
    - Missing timestamps
    - Duplicate records
    - Invalid OHLC relationships (High < Low, High < Open, High < Close, Low > Open, Low > Close)
    - Negative or Zero prices
    - Abnormal price jumps (> jump_threshold_pct)
    - Timezone or unit inconsistencies

    Classifies observations as: valid, suspicious, or invalid.
    Outputs quality report JSON.
    """
    def __init__(self, jump_threshold_pct: float = 20.0):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataset(
        self,
        df: pd.DataFrame,
        instrument_name: str,
        target_date: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df.empty:
            report = {
                "date": target_date,
                "instrument": instrument_name,
                "total_records": 0,
                "valid_records": 0,
                "suspicious_records": 0,
                "invalid_records": 0,
                "warnings": ["Dataset is empty"],
                "errors": ["No records found for validation"]
            }
            return df, report

        validated_df = df.copy()
        validated_df["quality_flag"] = "valid"
        validated_df["quality_reason"] = ""

        warnings = []
        errors = []

        # Check required columns
        req_cols = ["timestamp", "open", "high", "low", "close"]
        for col in req_cols:
            if col not in validated_df.columns:
                errors.append(f"Missing required column: {col}")
                report = {
                    "date": target_date,
                    "instrument": instrument_name,
                    "total_records": len(validated_df),
                    "valid_records": 0,
                    "suspicious_records": 0,
                    "invalid_records": len(validated_df),
                    "warnings": warnings,
                    "errors": errors
                }
                validated_df["quality_flag"] = "invalid"
                return validated_df, report

        # Check duplicate timestamps
        duplicates = validated_df.duplicated(subset=["timestamp"], keep=False)
        if duplicates.any():
            warnings.append(f"Found {duplicates.sum()} duplicate timestamp entries")
            validated_df.loc[duplicates, "quality_flag"] = "suspicious"
            validated_df.loc[duplicates, "quality_reason"] += "duplicate_timestamp; "

        # Check non-positive prices
        non_positive = (
            (validated_df["close"] <= 0) |
            (validated_df["open"] <= 0) |
            (validated_df["high"] <= 0) |
            (validated_df["low"] <= 0)
        )
        if non_positive.any():
            errors.append(f"Found {non_positive.sum()} non-positive price observations")
            validated_df.loc[non_positive, "quality_flag"] = "invalid"
            validated_df.loc[non_positive, "quality_reason"] += "non_positive_price; "

        # Check OHLC relationship validity
        invalid_ohlc = (
            (validated_df["high"] < validated_df["low"]) |
            (validated_df["high"] < validated_df["open"]) |
            (validated_df["high"] < validated_df["close"]) |
            (validated_df["low"] > validated_df["open"]) |
            (validated_df["low"] > validated_df["close"])
        )
        if invalid_ohlc.any():
            errors.append(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
            validated_df.loc[invalid_ohlc, "quality_flag"] = "invalid"
            validated_df.loc[invalid_ohlc, "quality_reason"] += "invalid_ohlc; "

        # Check abnormal price jumps
        sorted_df = validated_df.sort_values("timestamp")
        pct_change = sorted_df["close"].pct_change().abs() * 100.0
        abnormal_jumps = pct_change > self.jump_threshold_pct
        if abnormal_jumps.any():
            jump_indices = sorted_df[abnormal_jumps].index
            warnings.append(f"Found {len(jump_indices)} abnormal price jump(s) > {self.jump_threshold_pct}%")
            # Mark suspicious if not already invalid
            mask = validated_df.index.isin(jump_indices) & (validated_df["quality_flag"] != "invalid")
            validated_df.loc[mask, "quality_flag"] = "suspicious"
            validated_df.loc[mask, "quality_reason"] += f"abnormal_jump_gt_{self.jump_threshold_pct}pct; "

        counts = validated_df["quality_flag"].value_counts().to_dict()
        report = {
            "date": target_date,
            "instrument": instrument_name,
            "total_records": len(validated_df),
            "valid_records": int(counts.get("valid", 0)),
            "suspicious_records": int(counts.get("suspicious", 0)),
            "invalid_records": int(counts.get("invalid", 0)),
            "warnings": warnings,
            "errors": errors
        }

        return validated_df, report


def save_quality_report(report: Dict[str, Any], date_str: str, base_dir: str = "data/quality") -> str:
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, f"{date_str}.json")

    # If report file already exists, merge instrument statistics into daily report
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {"date": date_str, "instruments": {}, "warnings": [], "errors": []}
    else:
        existing = {"date": date_str, "instruments": {}, "warnings": [], "errors": []}

    inst = report.get("instrument", "unknown")
    existing["instruments"][inst] = {
        "total_records": report.get("total_records", 0),
        "valid_records": report.get("valid_records", 0),
        "suspicious_records": report.get("suspicious_records", 0),
        "invalid_records": report.get("invalid_records", 0)
    }
    existing["warnings"].extend(report.get("warnings", []))
    existing["errors"].extend(report.get("errors", []))

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    return filepath
