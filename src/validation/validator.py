import datetime
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates ingestion batches, flags errors/warnings, classifies records, and generates report."""

    def __init__(self, max_allowed_jump_pct: float = 20.0):
        self.max_allowed_jump_pct = max_allowed_jump_pct

    def validate_series(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLCV time-series dataframe for a symbol.
        Adds 'quality_classification' column: valid, suspicious, or invalid.
        Returns classified DataFrame and validation summary dictionary.
        """
        if df.empty:
            return df, {
                "symbol": symbol,
                "records_count": 0,
                "status": "empty",
                "warnings": [f"No records found for {symbol}"],
                "errors": []
            }

        warnings = []
        errors = []
        classifications = []

        # Check duplicated timestamps
        if df.index.duplicated().any():
            dups = df.index[df.index.duplicated()].tolist()
            errors.append(f"Duplicate timestamps found: {dups}")

        # Missing timestamps check
        if isinstance(df.index, pd.DatetimeIndex):
            expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
            missing_dates = expected_range.difference(df.index)
            if len(missing_dates) > 0:
                warnings.append(f"Missing {len(missing_dates)} business days in time series.")

        # Iterate rows for record-level classification
        prev_close = None
        for idx, row in df.iterrows():
            row_status = "valid"
            c_errors = []
            c_warnings = []

            # 1. Negative or Zero prices
            for col in ['Open', 'High', 'Low', 'Close']:
                val = row.get(col)
                if pd.isna(val):
                    c_errors.append(f"Missing value in {col}")
                    row_status = "invalid"
                elif val <= 0:
                    c_errors.append(f"Non-positive price in {col}: {val}")
                    row_status = "invalid"

            # 2. Invalid OHLC relationships
            if row_status != "invalid":
                open_p, high_p, low_p, close_p = row['Open'], row['High'], row['Low'], row['Close']
                if not (low_p <= open_p <= high_p and low_p <= close_p <= high_p):
                    c_errors.append(f"Invalid OHLC relationship: Low={low_p}, Open={open_p}, High={high_p}, Close={close_p}")
                    row_status = "invalid"

            # 3. Abnormal price jumps
            if row_status != "invalid" and prev_close is not None and prev_close > 0:
                pct_change = abs(row['Close'] - prev_close) / prev_close * 100.0
                if pct_change > self.max_allowed_jump_pct:
                    c_warnings.append(f"Abnormal price jump: {pct_change:.2f}% from previous close {prev_close} to {row['Close']}")
                    if row_status == "valid":
                        row_status = "suspicious"

            if c_errors:
                errors.extend([f"Row {idx}: {e}" for e in c_errors])
            if c_warnings:
                warnings.extend([f"Row {idx}: {w}" for w in c_warnings])

            classifications.append(row_status)
            prev_close = row['Close'] if row_status != "invalid" else prev_close

        df_out = df.copy()
        df_out['quality_classification'] = classifications

        summary = {
            "symbol": symbol,
            "records_count": len(df),
            "valid_count": classifications.count("valid"),
            "suspicious_count": classifications.count("suspicious"),
            "invalid_count": classifications.count("invalid"),
            "warnings": warnings[:20],  # cap list length for clean JSON report
            "errors": errors[:20]
        }
        return df_out, summary

    def generate_quality_report(self, date_str: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates comprehensive daily quality report JSON structure."""
        total_records = sum(s.get("records_count", 0) for s in summaries)
        all_warnings = []
        all_errors = []

        for s in summaries:
            all_warnings.extend(s.get("warnings", []))
            all_errors.extend(s.get("errors", []))

        report = {
            "date": date_str,
            "total_records": total_records,
            "instruments_analyzed": len(summaries),
            "summaries": summaries,
            "warnings": all_warnings,
            "errors": all_errors,
            "generated_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        return report
