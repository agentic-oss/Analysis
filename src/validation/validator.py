import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class DataValidator:
    """Batch data validation engine inspecting integrity, OHLC sanity, and anomalies."""

    @staticmethod
    def validate_df(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates DataFrame and classifies each record.
        Adds 'quality_status' column ('valid', 'suspicious', 'invalid').
        Returns (validated_df, summary_report).
        """
        warnings = []
        errors = []

        if df.empty:
            errors.append(f"Empty dataset for {symbol}")
            return df, {
                "records_count": 0,
                "valid_count": 0,
                "suspicious_count": 0,
                "invalid_count": 0,
                "warnings": warnings,
                "errors": errors
            }

        validated = df.copy()
        status_list = []

        # Ensure required columns exist
        req_cols = ['date', 'close']
        for col in req_cols:
            if col not in validated.columns:
                errors.append(f"Missing required column '{col}' for {symbol}")

        if errors:
            return validated, {
                "records_count": len(validated),
                "valid_count": 0,
                "suspicious_count": 0,
                "invalid_count": len(validated),
                "warnings": warnings,
                "errors": errors
            }

        # Check missing dates / duplicates
        if validated['date'].duplicated().any():
            dups = validated[validated['date'].duplicated()]['date'].tolist()
            warnings.append(f"Duplicate dates found for {symbol}: {dups}")

        # Compute price jump %
        validated['prev_close'] = validated['close'].shift(1)
        validated['price_change_pct'] = ((validated['close'] - validated['prev_close']) / validated['prev_close']).abs() * 100

        for idx, row in validated.iterrows():
            status = 'valid'
            close = row.get('close', None)
            open_p = row.get('open', None)
            high_p = row.get('high', None)
            low_p = row.get('low', None)
            price_jump = row.get('price_change_pct', 0.0)

            # Check invalid conditions
            if pd.isna(close) or close <= 0:
                status = 'invalid'
                errors.append(f"Invalid non-positive or NaN close price at date {row.get('date')}")

            # OHLC consistency check if available
            if status != 'invalid' and not pd.isna(high_p) and not pd.isna(low_p):
                if high_p < low_p:
                    status = 'invalid'
                    errors.append(f"High price < Low price at date {row.get('date')}")
                elif not pd.isna(open_p) and not pd.isna(close):
                    if open_p > high_p or open_p < low_p or close > high_p or close < low_p:
                        status = 'suspicious'
                        warnings.append(f"OHLC bounds anomaly at date {row.get('date')}")

            # Check abnormal price jumps (> 25% single day)
            if status == 'valid' and pd.notna(price_jump) and price_jump > 25.0:
                status = 'suspicious'
                warnings.append(f"Abnormal single-day price move ({price_jump:.2f}%) at date {row.get('date')}")

            status_list.append(status)

        validated['quality_status'] = status_list
        if 'prev_close' in validated.columns:
            validated.drop(columns=['prev_close', 'price_change_pct'], inplace=True, errors='ignore')

        summary = {
            "records_count": len(validated),
            "valid_count": int((validated['quality_status'] == 'valid').sum()),
            "suspicious_count": int((validated['quality_status'] == 'suspicious').sum()),
            "invalid_count": int((validated['quality_status'] == 'invalid').sum()),
            "warnings": warnings,
            "errors": errors
        }

        return validated, summary
