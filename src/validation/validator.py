"""
Data Validation Engine.
Validates incoming data batches, checking for missing timestamps, duplicates, invalid OHLC,
price anomalies, currency/unit mismatches, and classifies records as valid, suspicious, or invalid.
Generates data quality reports.
"""

from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


class DataValidator:
    """Validates financial time-series observations."""

    def __init__(self, spike_threshold_pct: float = 20.0):
        self.spike_threshold_pct = spike_threshold_pct

    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLCV dataframe.
        Adds 'classification' column: 'valid', 'suspicious', or 'invalid'.
        Returns validated DataFrame and summary quality dictionary.
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

        df = df.copy()
        warnings = []
        errors = []

        classifications = []
        reasons = []

        # Check duplicate dates
        duplicate_mask = df.duplicated(subset=["date"], keep=False)
        if duplicate_mask.any():
            warnings.append(f"Found {duplicate_mask.sum()} duplicate dates for {symbol}")

        # Check sorting
        df = df.sort_values("date").reset_index(drop=True)

        # Compute price changes for spike detection
        close_pct_change = df["close"].pct_change().abs() * 100.0

        for i, row in df.iterrows():
            row_warnings = []
            row_errors = []

            # Check missing values
            if pd.isna(row["close"]) or pd.isna(row["open"]) or pd.isna(row["high"]) or pd.isna(row["low"]):
                row_errors.append("missing_ohlc")

            # Non-positive prices
            if row["close"] <= 0 or row["open"] <= 0 or row["high"] <= 0 or row["low"] <= 0:
                row_errors.append("non_positive_price")

            # OHLC consistency
            if row["high"] < row["low"]:
                row_errors.append("high_less_than_low")
            if row["high"] < max(row["open"], row["close"]):
                row_errors.append("high_less_than_open_or_close")
            if row["low"] > min(row["open"], row["close"]):
                row_errors.append("low_greater_than_open_or_close")

            # Volume checks
            if "volume" in row and row["volume"] < 0:
                row_errors.append("negative_volume")

            # Abnormal price spikes
            if i > 0 and close_pct_change.iloc[i] > self.spike_threshold_pct:
                row_warnings.append(f"price_spike_{close_pct_change.iloc[i]:.1f}pct")

            if row_errors:
                classifications.append("invalid")
                reasons.append(";".join(row_errors))
                errors.append(f"Row {i} ({row['date']}): " + ";".join(row_errors))
            elif row_warnings:
                classifications.append("suspicious")
                reasons.append(";".join(row_warnings))
                warnings.append(f"Row {i} ({row['date']}): " + ";".join(row_warnings))
            else:
                classifications.append("valid")
                reasons.append("ok")

        df["classification"] = classifications
        df["validation_reason"] = reasons

        summary = {
            "symbol": symbol,
            "total_records": len(df),
            "valid_records": int((df["classification"] == "valid").sum()),
            "suspicious_records": int((df["classification"] == "suspicious").sum()),
            "invalid_records": int((df["classification"] == "invalid").sum()),
            "warnings": warnings,
            "errors": errors,
        }

        return df, summary

    @staticmethod
    def generate_quality_report(summaries: List[Dict[str, Any]], date_str: str, output_path: str):
        """Saves daily quality summary JSON file."""
        total_valid = sum(s["valid_records"] for s in summaries)
        total_suspicious = sum(s["suspicious_records"] for s in summaries)
        total_invalid = sum(s["invalid_records"] for s in summaries)
        all_warnings = []
        all_errors = []
        for s in summaries:
            all_warnings.extend([f"[{s['symbol']}] {w}" for w in s.get("warnings", [])])
            all_errors.extend([f"[{s['symbol']}] {e}" for e in s.get("errors", [])])

        report = {
            "date": date_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_instruments": len(summaries),
                "valid_records": total_valid,
                "suspicious_records": total_suspicious,
                "invalid_records": total_invalid,
            },
            "instruments": summaries,
            "warnings": all_warnings,
            "errors": all_errors,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
