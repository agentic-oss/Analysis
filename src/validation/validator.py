"""Data validation module to inspect incoming market data batches for data quality issues."""

import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates daily market observations and classifies records."""

    def __init__(self, jump_threshold_pct: float = 0.15):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataset(self, df: pd.DataFrame, date_str: str) -> Dict[str, Any]:
        """
        Validates OHLC data for missing timestamps, duplicates, negative/zero prices,
        OHLC inconsistency, abnormal jumps, etc.
        Returns a quality summary dictionary.
        """
        report = {
            "date": date_str,
            "total_records": len(df),
            "warnings": [],
            "errors": [],
            "classifications": {
                "valid": 0,
                "suspicious": 0,
                "invalid": 0
            }
        }

        if df.empty:
            report["errors"].append("Dataset is empty.")
            return report

        # Check for duplicates
        if df.duplicated(subset=['date', 'symbol']).any():
            dup_count = int(df.duplicated(subset=['date', 'symbol']).sum())
            report["warnings"].append(f"Found {dup_count} duplicate date/symbol pairs.")

        # Classify each row
        classifications = []
        for idx, row in df.iterrows():
            row_status = "valid"
            symbol = row.get('symbol', 'UNKNOWN')

            # Check null values
            if pd.isna(row.get('close')) or pd.isna(row.get('open')) or pd.isna(row.get('high')) or pd.isna(row.get('low')):
                report["errors"].append(f"Null price found in record for symbol {symbol} on {row.get('date')}")
                row_status = "invalid"

            # Check zero or negative prices
            elif row.get('close', 0) <= 0 or row.get('open', 0) <= 0 or row.get('high', 0) <= 0 or row.get('low', 0) <= 0:
                report["errors"].append(f"Zero/Negative price found for symbol {symbol} on {row.get('date')}")
                row_status = "invalid"

            # Check OHLC relationship: Low <= Open, Close <= High
            elif not (row['low'] <= row['open'] <= row['high'] and row['low'] <= row['close'] <= row['high']):
                report["warnings"].append(f"OHLC relationship violation for symbol {symbol} on {row.get('date')}")
                row_status = "suspicious"

            classifications.append(row_status)

        # Check abnormal price jump per symbol
        for symbol, group in df.groupby('symbol'):
            if len(group) > 1:
                group = group.sort_values('date')
                pct_change = group['close'].pct_change().abs()
                large_jumps = pct_change[pct_change > self.jump_threshold_pct]
                if not large_jumps.empty:
                    for jump_idx, jump_val in large_jumps.items():
                        j_date = group.loc[jump_idx, 'date']
                        report["warnings"].append(
                            f"Abnormal price jump ({jump_val:.2%}) detected for {symbol} on {j_date}"
                        )

        report["classifications"]["valid"] = classifications.count("valid")
        report["classifications"]["suspicious"] = classifications.count("suspicious")
        report["classifications"]["invalid"] = classifications.count("invalid")

        return report
