import json
import os
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates incoming financial market data batches and flags anomalies."""

    def __init__(self, jump_threshold_pct: float = 10.0):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataset(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate OHLCV data, add validation status column, and produce report summary."""
        report = {
            "symbol": symbol,
            "record_count": len(df),
            "warnings": [],
            "errors": [],
            "valid_count": 0,
            "suspicious_count": 0,
            "invalid_count": 0
        }

        if df.empty:
            report["errors"].append("Dataset is empty")
            return df, report

        df = df.copy().reset_index(drop=True)
        df["validation_status"] = "valid"
        df["validation_flags"] = ""

        flags_list = [[] for _ in range(len(df))]

        # Check required columns
        req_cols = ["date", "open", "high", "low", "close"]
        missing_req = [c for c in req_cols if c not in df.columns]
        if missing_req:
            report["errors"].append(f"Missing required columns: {missing_req}")
            return df, report

        # 1. Missing or duplicate timestamps
        if df["date"].isnull().any():
            report["errors"].append("Contains null dates")

        dup_dates = df[df["date"].duplicated()]["date"].tolist()
        if dup_dates:
            report["warnings"].append(f"Duplicate dates found: {dup_dates[:5]}")

        # 2. Price <= 0 checks
        neg_zero_mask = (df["close"] <= 0) | (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0)
        for idx in df[neg_zero_mask].index:
            flags_list[idx].append("non_positive_price")
            df.at[idx, "validation_status"] = "invalid"

        # 3. Invalid OHLC logic: High >= Low, High >= max(Open, Close), Low <= min(Open, Close)
        ohlc_invalid_mask = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        for idx in df[ohlc_invalid_mask].index:
            flags_list[idx].append("invalid_ohlc_relationship")
            df.at[idx, "validation_status"] = "invalid"

        # 4. Abnormal price jumps (daily % change)
        returns = df["close"].pct_change().abs() * 100.0
        jump_mask = returns > self.jump_threshold_pct
        for idx in df[jump_mask].index:
            flags_list[idx].append("abnormal_price_jump")
            # Mark suspicious rather than invalid so historical unusual moves aren't deleted
            if df.at[idx, "validation_status"] != "invalid":
                df.at[idx, "validation_status"] = "suspicious"

        # 5. Volume check
        if "volume" in df.columns:
            neg_vol_mask = df["volume"] < 0
            for idx in df[neg_vol_mask].index:
                flags_list[idx].append("negative_volume")
                df.at[idx, "validation_status"] = "invalid"

        # Aggregate flags
        df["validation_flags"] = [",".join(f) for f in flags_list]

        # Summarize counts
        status_counts = df["validation_status"].value_counts().to_dict()
        report["valid_count"] = int(status_counts.get("valid", 0))
        report["suspicious_count"] = int(status_counts.get("suspicious", 0))
        report["invalid_count"] = int(status_counts.get("invalid", 0))

        if report["invalid_count"] > 0:
            report["errors"].append(f"Found {report['invalid_count']} invalid records")
        if report["suspicious_count"] > 0:
            report["warnings"].append(f"Found {report['suspicious_count']} suspicious records")

        return df, report

    def generate_quality_report(
        self,
        date_str: str,
        reports_dict: Dict[str, Dict[str, Any]],
        output_dir: str = "data/quality"
    ) -> str:
        """Write daily aggregated data quality JSON report."""
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            "date": date_str,
            "instruments": reports_dict,
            "total_instruments_validated": len(reports_dict),
            "total_errors": sum(len(r["errors"]) for r in reports_dict.values()),
            "total_warnings": sum(len(r["warnings"]) for r in reports_dict.values())
        }

        output_path = os.path.join(output_dir, f"{date_str}.json")
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        return output_path
