import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """Engine for validating incoming market data batches, classifying observations, and generating quality reports."""

    def __init__(self, max_price_jump_pct: float = 0.20, max_stale_days: int = 5):
        self.max_price_jump_pct = max_price_jump_pct
        self.max_stale_days = max_stale_days

    def validate_dataframe(self, df: pd.DataFrame, expected_symbol: str = None) -> tuple[pd.DataFrame, dict]:
        """
        Validates a dataframe containing market observations.
        Classifies each row into status: 'valid', 'suspicious', or 'invalid'.
        Returns updated dataframe with 'validation_status' and 'validation_reasons', plus a report summary dictionary.
        """
        if df.empty:
            return df, {"date": datetime.now().strftime("%Y-%m-%d"), "warnings": ["Empty dataframe provided"], "errors": [], "summary": {}}

        df = df.copy()
        df["validation_status"] = "valid"
        df["validation_reasons"] = ""

        warnings = []
        errors = []

        # 1. Missing timestamp
        if "date" not in df.columns:
            errors.append("Missing required 'date' column.")
            df["validation_status"] = "invalid"
            df["validation_reasons"] += "missing_date;"
        else:
            null_dates = df["date"].isnull()
            if null_dates.any():
                df.loc[null_dates, "validation_status"] = "invalid"
                df.loc[null_dates, "validation_reasons"] += "null_date;"
                errors.append(f"Found {null_dates.sum()} null dates.")

        # 2. Duplicate records
        if "date" in df.columns and "symbol" in df.columns:
            dups = df.duplicated(subset=["date", "symbol"], keep=False)
            if dups.any():
                df.loc[dups, "validation_status"] = "suspicious"
                df.loc[dups, "validation_reasons"] += "duplicate_record;"
                warnings.append(f"Found {dups.sum()} duplicate records.")

        # 3. Price validation: Zero / Negative prices
        price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
        for col in price_cols:
            neg_mask = df[col] < 0
            zero_mask = df[col] == 0
            if neg_mask.any():
                df.loc[neg_mask, "validation_status"] = "invalid"
                df.loc[neg_mask, "validation_reasons"] += f"negative_{col};"
                errors.append(f"Found negative values in {col}.")
            if zero_mask.any():
                df.loc[zero_mask, "validation_status"] = "suspicious"
                df.loc[zero_mask, "validation_reasons"] += f"zero_{col};"
                warnings.append(f"Found zero values in {col}.")

        # 4. Invalid OHLC relationships
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            ohlc_invalid = (
                (df["high"] < df["low"]) |
                (df["open"] < df["low"]) |
                (df["open"] > df["high"]) |
                (df["close"] < df["low"]) |
                (df["close"] > df["high"])
            )
            if ohlc_invalid.any():
                df.loc[ohlc_invalid, "validation_status"] = "invalid"
                df.loc[ohlc_invalid, "validation_reasons"] += "invalid_ohlc_relationship;"
                errors.append(f"Found {ohlc_invalid.sum()} invalid OHLC relationship records.")

        # 5. Abnormal price jumps (> max_price_jump_pct)
        if "close" in df.columns and len(df) > 1:
            ret = df["close"].pct_change().abs()
            jump_mask = ret > self.max_price_jump_pct
            if jump_mask.any():
                df.loc[jump_mask, "validation_status"] = "suspicious"
                df.loc[jump_mask, "validation_reasons"] += "abnormal_price_jump;"
                warnings.append(f"Found {jump_mask.sum()} abnormal price jump observations (> {self.max_price_jump_pct*100}%).")

        # 6. Stale observations (identical close price for > max_stale_days)
        if "close" in df.columns and len(df) >= self.max_stale_days:
            close_diff = df["close"].diff().abs()
            stale_mask = (close_diff == 0).rolling(window=self.max_stale_days).sum() >= self.max_stale_days
            if stale_mask.any():
                df.loc[stale_mask, "validation_status"] = "suspicious"
                df.loc[stale_mask, "validation_reasons"] += "stale_observation;"
                warnings.append(f"Found stale price observations unchanged for > {self.max_stale_days} consecutive periods.")

        # Summary count
        status_counts = df["validation_status"].value_counts().to_dict()

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "expected_symbol": expected_symbol or "ALL",
            "warnings": warnings,
            "errors": errors,
            "summary": {
                "total_records": len(df),
                "valid_count": int(status_counts.get("valid", 0)),
                "suspicious_count": int(status_counts.get("suspicious", 0)),
                "invalid_count": int(status_counts.get("invalid", 0))
            }
        }

        return df, report
