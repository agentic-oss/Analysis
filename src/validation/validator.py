import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates raw and processed market data batches."""

    def __init__(self, jump_threshold_pct: float = 15.0):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataset(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLC time series dataset.
        Classifies records and returns (validated_df, quality_report).
        """
        report = {
            "symbol": symbol,
            "total_records": len(df),
            "valid_records": 0,
            "suspicious_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": [],
        }

        if df is None or df.empty:
            report["errors"].append(f"Empty dataset provided for {symbol}")
            return pd.DataFrame(), report

        df = df.copy()

        # Check duplicate timestamps
        if "Date" in df.columns:
            dups = df.duplicated(subset=["Date"], keep="first")
            if dups.any():
                dup_count = int(dups.sum())
                report["warnings"].append(f"Found {dup_count} duplicate timestamps. Deduplicating...")
                df = df.drop_duplicates(subset=["Date"], keep="first")

            # Check timestamp ordering
            df = df.sort_values("Date").reset_index(drop=True)

        # Initialize status
        df["validation_status"] = "valid"
        df["validation_notes"] = ""

        # Validate OHLC presence and numerical values
        required_cols = ["Open", "High", "Low", "Close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report["errors"].append(f"Missing required OHLC columns: {missing_cols}")
            return df, report

        for idx, row in df.iterrows():
            notes = []
            status = "valid"

            # Check zero or negative prices
            for col in required_cols:
                val = row[col]
                if pd.isna(val) or val <= 0:
                    status = "invalid"
                    notes.append(f"Invalid price in {col}: {val}")

            # Check OHLC relationship logic
            if status != "invalid":
                open_p, high_p, low_p, close_p = row["Open"], row["High"], row["Low"], row["Close"]
                if high_p < low_p:
                    status = "invalid"
                    notes.append(f"High ({high_p}) < Low ({low_p})")
                if high_p < open_p or high_p < close_p:
                    status = "invalid"
                    notes.append(f"High ({high_p}) < Open/Close ({open_p}/{close_p})")
                if low_p > open_p or low_p > close_p:
                    status = "invalid"
                    notes.append(f"Low ({low_p}) > Open/Close ({open_p}/{close_p})")

            # Check price jump
            if idx > 0 and status != "invalid":
                prev_close = df.at[idx - 1, "Close"]
                if prev_close > 0:
                    pct_change = abs((row["Close"] - prev_close) / prev_close) * 100.0
                    if pct_change > self.jump_threshold_pct:
                        status = "suspicious" if status == "valid" else status
                        notes.append(f"Price jump of {pct_change:.2f}% exceeds threshold {self.jump_threshold_pct}%")

            df.at[idx, "validation_status"] = status
            df.at[idx, "validation_notes"] = "; ".join(notes)

        report["valid_records"] = int((df["validation_status"] == "valid").sum())
        report["suspicious_records"] = int((df["validation_status"] == "suspicious").sum())
        report["invalid_records"] = int((df["validation_status"] == "invalid").sum())

        return df, report

    def create_daily_quality_report(self, date_str: str, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quality reports for all instruments into a single daily summary."""
        total_valid = sum(r.get("valid_records", 0) for r in reports)
        total_suspicious = sum(r.get("suspicious_records", 0) for r in reports)
        total_invalid = sum(r.get("invalid_records", 0) for r in reports)
        all_warnings = [w for r in reports for w in r.get("warnings", [])]
        all_errors = [e for r in reports for e in r.get("errors", [])]

        return {
            "date": date_str,
            "instruments_evaluated": len(reports),
            "total_valid_records": total_valid,
            "total_suspicious_records": total_suspicious,
            "total_invalid_records": total_invalid,
            "warnings": all_warnings,
            "errors": all_errors,
            "details": reports,
        }
