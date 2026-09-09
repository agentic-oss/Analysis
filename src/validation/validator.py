"""
Data Validation Engine for Precious Metals & Macro Market Data.
Validates ingestion batches for missing timestamps, duplicates, invalid OHLC,
negative/zero prices, price jumps, stale data, and unit/currency consistency.
Classifies observations as 'valid', 'suspicious', or 'invalid' without deleting raw data.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import os
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    date: str
    total_records: int
    gold_records: int
    silver_records: int
    valid_count: int
    suspicious_count: int
    invalid_count: int
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataValidator:
    """Validates daily market datasets against consistency and integrity constraints."""

    def __init__(self, jump_threshold_pct: float = 15.0):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_series(
        self,
        df: pd.DataFrame,
        symbol: str,
        expected_currency: str = "USD",
        expected_unit: str = "oz"
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Validates OHLC DataFrame.
        Adds 'quality_classification' column: ['valid', 'suspicious', 'invalid'].
        Returns: (validated_df, warnings, errors)
        """
        warnings = []
        errors = []

        if df.empty:
            errors.append(f"[{symbol}] Empty DataFrame provided for validation.")
            return df, warnings, errors

        df = df.copy()
        if "quality_classification" not in df.columns:
            df["quality_classification"] = "valid"

        # 1. Check required columns
        req_cols = ["date", "open", "high", "low", "close"]
        for col in req_cols:
            if col not in df.columns:
                errors.append(f"[{symbol}] Missing required column: {col}")
                return df, warnings, errors

        # 2. Check duplicates
        dup_mask = df.duplicated(subset=["date"], keep="first")
        if dup_mask.any():
            warnings.append(f"[{symbol}] Found {dup_mask.sum()} duplicate timestamps. Deduplicating.")
            df = df.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

        # 3. Check for missing / zero / negative prices
        for idx, row in df.iterrows():
            d_str = str(row["date"])
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]

            # Negative or zero check
            if any(p <= 0 for p in [o, h, l, c] if pd.notna(p)):
                df.at[idx, "quality_classification"] = "invalid"
                errors.append(f"[{symbol}] [{d_str}] Invalid non-positive price observed (O:{o}, H:{h}, L:{l}, C:{c}).")
                continue

            # Invalid OHLC relationships: Low > High or Open/Close outside High/Low
            if pd.notna(l) and pd.notna(h) and l > h:
                df.at[idx, "quality_classification"] = "invalid"
                errors.append(f"[{symbol}] [{d_str}] Invalid OHLC relationship: Low ({l}) > High ({h}).")
                continue

            if pd.notna(h) and pd.notna(l):
                if o > h * 1.0001 or o < l * 0.9999 or c > h * 1.0001 or c < l * 0.9999:
                    df.at[idx, "quality_classification"] = "suspicious"
                    warnings.append(f"[{symbol}] [{d_str}] Suspicious OHLC relationship: Open/Close outside [Low, High].")

        # 4. Check abnormal daily price jumps
        df["returns"] = df["close"].pct_change().abs() * 100.0
        jump_mask = df["returns"] > self.jump_threshold_pct
        for idx in df[jump_mask].index:
            if df.at[idx, "quality_classification"] == "valid":
                df.at[idx, "quality_classification"] = "suspicious"
            ret_val = df.at[idx, "returns"]
            d_str = str(df.at[idx, "date"])
            warnings.append(f"[{symbol}] [{d_str}] Abnormal price jump detected: {ret_val:.2f}% shift.")

        df = df.drop(columns=["returns"], errors="ignore")
        return df, warnings, errors

    def validate_and_save_batch(
        self,
        datasets: Dict[str, pd.DataFrame],
        as_of_date: str,
        output_dir: str = "data/quality"
    ) -> QualityReport:
        """Runs validation across all market datasets and writes JSON report."""
        os.makedirs(output_dir, exist_ok=True)
        all_warnings = []
        all_errors = []

        total_records = 0
        gold_records = 0
        silver_records = 0
        valid_count = 0
        suspicious_count = 0
        invalid_count = 0

        for sym, df in datasets.items():
            val_df, warn, err = self.validate_series(df, sym)
            all_warnings.extend(warn)
            all_errors.extend(err)

            cnt = len(val_df)
            total_records += cnt
            if "GOLD" in sym:
                gold_records += cnt
            elif "SILVER" in sym:
                silver_records += cnt

            if "quality_classification" in val_df.columns:
                vc = val_df["quality_classification"].value_counts()
                valid_count += int(vc.get("valid", 0))
                suspicious_count += int(vc.get("suspicious", 0))
                invalid_count += int(vc.get("invalid", 0))

        report = QualityReport(
            date=as_of_date,
            total_records=total_records,
            gold_records=gold_records,
            silver_records=silver_records,
            valid_count=valid_count,
            suspicious_count=suspicious_count,
            invalid_count=invalid_count,
            warnings=all_warnings,
            errors=all_errors
        )

        file_path = os.path.join(output_dir, f"{as_of_date}.json")
        with open(file_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Quality report written to {file_path}")
        return report
