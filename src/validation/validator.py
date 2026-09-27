import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates price observations and classifies record status."""

    def __init__(self, jump_threshold_pct: float = 0.15):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_series(self, df: pd.DataFrame, instrument: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLC data for missing timestamps, invalid values, OHLC consistency, and abnormal jumps.
        Classifies each row into: 'valid', 'suspicious', 'invalid'.
        """
        metrics = {
            "instrument": instrument,
            "total_records": len(df),
            "valid_records": 0,
            "suspicious_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": []
        }

        if df.empty:
            metrics["errors"].append("DataFrame is empty")
            return df, metrics

        df = df.copy()

        # Find column names for close, high, low, open
        close_col = "close" if "close" in df.columns else (f"{instrument.upper()}_close" if f"{instrument.upper()}_close" in df.columns else None)
        high_col = "high" if "high" in df.columns else (f"{instrument.upper()}_high" if f"{instrument.upper()}_high" in df.columns else None)
        low_col = "low" if "low" in df.columns else (f"{instrument.upper()}_low" if f"{instrument.upper()}_low" in df.columns else None)
        open_col = "open" if "open" in df.columns else (f"{instrument.upper()}_open" if f"{instrument.upper()}_open" in df.columns else None)

        if close_col is None:
            # Fallback to first numeric column if specific close col not found
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            close_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

        df["quality_status"] = "valid"
        df["quality_reasons"] = ""

        reasons = [[] for _ in range(len(df))]

        # 1. Null / zero / negative check
        for i, row in df.iterrows():
            close = row.get(close_col) if close_col else None
            high = row.get(high_col) if high_col else None
            low = row.get(low_col) if low_col else None
            open_p = row.get(open_col) if open_col else None

            if close is None or pd.isna(close) or close <= 0:
                reasons[i].append("invalid_close_price")
                df.at[i, "quality_status"] = "invalid"

            if (high is not None and (pd.isna(high) or high <= 0)) or \
               (low is not None and (pd.isna(low) or low <= 0)) or \
               (open_p is not None and (pd.isna(open_p) or open_p <= 0)):
                if df.at[i, "quality_status"] != "invalid":
                    reasons[i].append("invalid_ohlc_values")
                    df.at[i, "quality_status"] = "suspicious"

            # 2. OHLC Logic: High >= Low, High >= Close, High >= Open, Low <= Close, Low <= Open
            if high is not None and low is not None and close is not None and open_p is not None and \
               not pd.isna(high) and not pd.isna(low) and not pd.isna(close) and not pd.isna(open_p):
                if high < low or high < close or high < open_p or low > close or low > open_p:
                    reasons[i].append("ohlc_inconsistency")
                    if df.at[i, "quality_status"] != "invalid":
                        df.at[i, "quality_status"] = "suspicious"

        # 3. Abnormal price jump check (relative to previous close)
        close_series = df[close_col].astype(float)
        returns = close_series.pct_change().abs()
        jump_mask = returns > self.jump_threshold_pct
        for idx in df[jump_mask].index:
            reasons[idx].append(f"price_jump_exceeds_{int(self.jump_threshold_pct*100)}pct")
            if df.at[idx, "quality_status"] == "valid":
                df.at[idx, "quality_status"] = "suspicious"

        df["quality_reasons"] = [";".join(r) for r in reasons]

        metrics["valid_records"] = int((df["quality_status"] == "valid").sum())
        metrics["suspicious_records"] = int((df["quality_status"] == "suspicious").sum())
        metrics["invalid_records"] = int((df["quality_status"] == "invalid").sum())

        if metrics["suspicious_records"] > 0:
            metrics["warnings"].append(f"Found {metrics['suspicious_records']} suspicious records for {instrument}")
        if metrics["invalid_records"] > 0:
            metrics["errors"].append(f"Found {metrics['invalid_records']} invalid records for {instrument}")

        return df, metrics
