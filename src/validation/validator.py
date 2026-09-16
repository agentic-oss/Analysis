import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Engine for checking incoming data quality and classifying records as valid, suspicious, or invalid."""

    def __init__(self, jump_threshold_pct: float = 20.0):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataset(self, symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validates dataframe for a given symbol and generates a quality summary report."""
        metrics = {
            "symbol": symbol,
            "total_records": len(df),
            "valid_records": 0,
            "suspicious_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": [],
        }

        if df.empty:
            metrics["errors"].append("Empty dataframe received")
            return df, metrics

        # Ensure date sorting & drop exact duplicates
        df = df.sort_values("date").reset_index(drop=True)
        dup_count = df.duplicated(subset=["date"]).sum()
        if dup_count > 0:
            metrics["warnings"].append(f"Found {dup_count} duplicate timestamps, keeping first")
            df = df.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

        status_list = []
        for idx, row in df.iterrows():
            row_status = "valid"
            reasons = []

            # Check prices
            for col in ["open", "high", "low", "close"]:
                val = row.get(col)
                if pd.isna(val):
                    row_status = "invalid"
                    reasons.append(f"Missing {col}")
                elif val <= 0:
                    row_status = "invalid"
                    reasons.append(f"Non-positive {col}: {val}")

            # OHLC integrity
            if row_status != "invalid" and all(col in row for col in ["open", "high", "low", "close"]):
                if not (row["low"] <= row["open"] <= row["high"]) or not (row["low"] <= row["close"] <= row["high"]):
                    row_status = "invalid"
                    reasons.append("OHLC relationship violation (low <= open/close <= high)")

            # Check price jump
            if idx > 0 and row_status != "invalid":
                prev_close = df.loc[idx - 1, "close"]
                if prev_close > 0:
                    pct_change = abs((row["close"] - prev_close) / prev_close) * 100.0
                    if pct_change > self.jump_threshold_pct:
                        row_status = "suspicious" if row_status != "invalid" else row_status
                        reasons.append(f"Large price jump: {pct_change:.2f}%")

            if row_status == "valid":
                metrics["valid_records"] += 1
            elif row_status == "suspicious":
                metrics["suspicious_records"] += 1
                metrics["warnings"].append(f"Row {idx} ({row.get('date')}): {', '.join(reasons)}")
            else:
                metrics["invalid_records"] += 1
                metrics["errors"].append(f"Row {idx} ({row.get('date')}): {', '.join(reasons)}")

            status_list.append(row_status)

        df["data_quality_status"] = status_list
        return df, metrics

    def generate_daily_quality_report(self, all_metrics: List[Dict[str, Any]], date_str: str) -> Dict[str, Any]:
        """Combines quality metrics across instruments and saves to data/quality/YYYY-MM-DD.json."""
        summary = {
            "date": date_str,
            "timestamp": datetime.utcnow().isoformat(),
            "instruments_analyzed": len(all_metrics),
            "details": all_metrics,
            "overall_warnings_count": sum(len(m["warnings"]) for m in all_metrics),
            "overall_errors_count": sum(len(m["errors"]) for m in all_metrics),
        }
        os.makedirs("data/quality", exist_ok=True)
        report_path = f"data/quality/{date_str}.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved quality report to {report_path}")
        return summary
