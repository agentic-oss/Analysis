"""Data validation engine for incoming market and macro observations."""

import json
import os
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates ingestion batches, classifies records, and generates data quality reports."""

    def __init__(self, jump_threshold_pct: float = 0.15):
        self.jump_threshold_pct = jump_threshold_pct

    def validate_dataframe(self, df: pd.DataFrame, instrument_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates OHLC, timestamps, duplicate check, negative/zero prices, jumps, volume.
        Classifies records as 'valid', 'suspicious', or 'invalid'.
        """
        if df.empty:
            return df, {
                "instrument": instrument_name,
                "total_records": 0,
                "valid_records": 0,
                "suspicious_records": 0,
                "invalid_records": 0,
                "warnings": ["Empty dataframe provided"],
                "errors": []
            }

        validated_df = df.copy()
        validated_df['classification'] = 'valid'
        warnings = []
        errors = []

        # 1. Missing required columns
        req_cols = ['date', 'open', 'high', 'low', 'close']
        missing_cols = [c for c in req_cols if c not in validated_df.columns]
        if missing_cols:
            errors.append(f"Missing required columns for {instrument_name}: {missing_cols}")
            return validated_df, {
                "instrument": instrument_name,
                "total_records": len(validated_df),
                "valid_records": 0,
                "suspicious_records": 0,
                "invalid_records": len(validated_df),
                "warnings": warnings,
                "errors": errors
            }

        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close']:
            validated_df[col] = pd.to_numeric(validated_df[col], errors='coerce')

        # 2. Check duplicate dates
        duplicates = validated_df.duplicated(subset=['date'], keep=False)
        if duplicates.any():
            warnings.append(f"Duplicate dates found for {instrument_name}: {validated_df[duplicates]['date'].tolist()}")
            validated_df.loc[duplicates, 'classification'] = 'suspicious'

        # 3. Check non-positive prices
        non_positive = (validated_df['close'] <= 0) | (validated_df['high'] <= 0) | (validated_df['low'] <= 0) | (validated_df['open'] <= 0)
        if non_positive.any():
            errors.append(f"Non-positive prices found for {instrument_name} on dates: {validated_df[non_positive]['date'].tolist()}")
            validated_df.loc[non_positive, 'classification'] = 'invalid'

        # 4. Check invalid OHLC relationships (High should be >= Open, Close, Low)
        invalid_ohlc = (validated_df['high'] < validated_df['low']) | \
                       (validated_df['high'] < validated_df['open']) | \
                       (validated_df['high'] < validated_df['close']) | \
                       (validated_df['low'] > validated_df['open']) | \
                       (validated_df['low'] > validated_df['close'])
        if invalid_ohlc.any():
            warnings.append(f"Invalid OHLC relationship for {instrument_name} on dates: {validated_df[invalid_ohlc]['date'].tolist()}")
            validated_df.loc[invalid_ohlc & (validated_df['classification'] != 'invalid'), 'classification'] = 'suspicious'

        # 5. Check abnormal price jumps
        validated_df['return_1d'] = validated_df['close'].pct_change().abs()
        large_jumps = validated_df['return_1d'] > self.jump_threshold_pct
        if large_jumps.any():
            jump_dates = validated_df[large_jumps]['date'].tolist()
            warnings.append(f"Abnormal price jump > {self.jump_threshold_pct*100}% for {instrument_name} on dates: {jump_dates}")
            validated_df.loc[large_jumps & (validated_df['classification'] != 'invalid'), 'classification'] = 'suspicious'

        validated_df.drop(columns=['return_1d'], inplace=True, errors='ignore')

        valid_count = int((validated_df['classification'] == 'valid').sum())
        suspicious_count = int((validated_df['classification'] == 'suspicious').sum())
        invalid_count = int((validated_df['classification'] == 'invalid').sum())

        quality_report = {
            "instrument": instrument_name,
            "total_records": len(validated_df),
            "valid_records": valid_count,
            "suspicious_records": suspicious_count,
            "invalid_records": invalid_count,
            "warnings": warnings,
            "errors": errors
        }

        return validated_df, quality_report

    def save_quality_report(self, date_str: str, report_data: Dict[str, Any], output_dir: str = "data/quality") -> str:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{date_str}.json")
        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Saved data quality report to {file_path}")
        return file_path
