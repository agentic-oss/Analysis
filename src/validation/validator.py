import json
import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates daily market ingestion batches without silently dropping data."""

    def __init__(self, quality_dir: str = "data/quality"):
        self.quality_dir = quality_dir
        os.makedirs(self.quality_dir, exist_ok=True)

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str, date_str: str) -> Dict[str, Any]:
        warnings = []
        errors = []
        status_flags = []

        if df.empty:
            errors.append(f"{dataset_name}: Dataset is empty")
            return {
                "date": date_str,
                "dataset": dataset_name,
                "records": 0,
                "warnings": warnings,
                "errors": errors,
                "classified_records": []
            }

        required_cols = ["date", "open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"{dataset_name}: Missing required column '{col}'")

        if errors:
            return {
                "date": date_str,
                "dataset": dataset_name,
                "records": len(df),
                "warnings": warnings,
                "errors": errors,
                "classified_records": []
            }

        # Validate record by record
        for idx, row in df.iterrows():
            rec_status = "valid"
            rec_notes = []

            # Check negative or zero
            for col in ["open", "high", "low", "close"]:
                val = row.get(col, 0)
                if pd.isna(val) or val <= 0:
                    rec_status = "invalid"
                    rec_notes.append(f"Non-positive or NaN price in {col}: {val}")

            # Check OHLC relationships
            if rec_status != "invalid":
                if row["high"] < max(row["open"], row["close"]) or row["low"] > min(row["open"], row["close"]):
                    rec_status = "suspicious"
                    rec_notes.append("Invalid OHLC structure (high/low bounds exceeded)")

            # Check for extreme price jumps relative to prior close if available
            if idx > 0:
                prev_close = df.loc[idx - 1, "close"]
                if prev_close > 0:
                    pct_change = abs((row["close"] - prev_close) / prev_close)
                    if pct_change > 0.15:  # > 15% move in 1 day
                        if rec_status == "valid":
                            rec_status = "suspicious"
                        rec_notes.append(f"Abnormal daily jump of {pct_change:.1%}")

            if rec_status == "invalid":
                errors.append(f"Row {idx} ({row.get('date')}): {', '.join(rec_notes)}")
            elif rec_status == "suspicious":
                warnings.append(f"Row {idx} ({row.get('date')}): {', '.join(rec_notes)}")

            status_flags.append(rec_status)

        df["quality_status"] = status_flags

        quality_report = {
            "date": date_str,
            "dataset": dataset_name,
            "records": len(df),
            "valid_records": int((df["quality_status"] == "valid").sum()),
            "suspicious_records": int((df["quality_status"] == "suspicious").sum()),
            "invalid_records": int((df["quality_status"] == "invalid").sum()),
            "warnings": warnings,
            "errors": errors
        }

        return quality_report

    def save_aggregated_quality_report(self, date_str: str, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Saves combined quality report to data/quality/YYYY-MM-DD.json as required by schema."""
        total_records = sum(r.get("records", 0) for r in reports)
        valid_records = sum(r.get("valid_records", 0) for r in reports)
        suspicious_records = sum(r.get("suspicious_records", 0) for r in reports)
        invalid_records = sum(r.get("invalid_records", 0) for r in reports)

        all_warnings = []
        all_errors = []
        for r in reports:
            all_warnings.extend(r.get("warnings", []))
            all_errors.extend(r.get("errors", []))

        agg_report = {
            "date": date_str,
            "gold_records": next((r.get("records", 0) for r in reports if r.get("dataset") == "gold"), 0),
            "silver_records": next((r.get("records", 0) for r in reports if r.get("dataset") == "silver"), 0),
            "total_records": total_records,
            "valid_records": valid_records,
            "suspicious_records": suspicious_records,
            "invalid_records": invalid_records,
            "warnings": all_warnings,
            "errors": all_errors
        }

        out_path = os.path.join(self.quality_dir, f"{date_str}.json")
        with open(out_path, "w") as f:
            json.dump(agg_report, f, indent=2)

        return agg_report
