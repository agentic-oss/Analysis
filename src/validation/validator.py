import json
import os
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates raw and processed market data batches, checks quality metrics, and classifies records/batches."""

    def __init__(self, max_allowed_jump_pct: float = 15.0):
        self.max_allowed_jump_pct = max_allowed_jump_pct

    def validate_dataset(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates a DataFrame for a given symbol and returns quality statistics and issues found."""
        warnings = []
        errors = []
        classification = "valid"
        record_count = len(df)

        if df.empty:
            errors.append(f"{symbol}: DataFrame is empty.")
            return {
                "symbol": symbol,
                "record_count": 0,
                "classification": "invalid",
                "warnings": warnings,
                "errors": errors
            }

        # Check required columns
        req_cols = ["date", "open", "high", "low", "close"]
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"{symbol}: Missing required columns {missing_cols}")
            classification = "invalid"

        # Check duplicates
        if "date" in df.columns:
            dups = df.duplicated(subset=["date"]).sum()
            if dups > 0:
                warnings.append(f"{symbol}: Found {dups} duplicate date records.")
                classification = "suspicious"

            # Check missing/null dates
            null_dates = df["date"].isnull().sum()
            if null_dates > 0:
                errors.append(f"{symbol}: Found {null_dates} null date values.")
                classification = "invalid"

        # OHLC relationship checks
        for idx, row in df.iterrows():
            o, h, l, c = row.get("open", 0), row.get("high", 0), row.get("low", 0), row.get("close", 0)
            date_str = str(row.get("date", idx))

            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                errors.append(f"{symbol} on {date_str}: Non-positive price detected (open={o}, high={h}, low={l}, close={c}).")
                classification = "invalid"

            if h < max(o, c) or l > min(o, c) or h < l:
                errors.append(f"{symbol} on {date_str}: Invalid OHLC relationship (open={o}, high={h}, low={l}, close={c}).")
                classification = "invalid"

        # Check extreme price jumps
        if "close" in df.columns and len(df) > 1:
            df_sorted = df.sort_values("date") if "date" in df.columns else df
            pct_change = df_sorted["close"].pct_change().abs() * 100.0
            spikes = df_sorted[pct_change > self.max_allowed_jump_pct]
            if not spikes.empty:
                for _, spike_row in spikes.iterrows():
                    dt = spike_row.get("date", "")
                    chg = pct_change.loc[spike_row.name]
                    warnings.append(f"{symbol} on {dt}: Abnormal price jump of {chg:.2f}%.")
                if classification != "invalid":
                    classification = "suspicious"

        return {
            "symbol": symbol,
            "record_count": record_count,
            "classification": classification,
            "warnings": warnings,
            "errors": errors
        }

    def generate_daily_quality_report(
        self,
        date_str: str,
        datasets: Dict[str, pd.DataFrame],
        output_dir: str = "data/quality"
    ) -> Dict[str, Any]:
        """Validates all instrument datasets and writes quality report data/quality/YYYY-MM-DD.json."""
        report = {
            "date": date_str,
            "instruments_evaluated": len(datasets),
            "symbol_quality": {},
            "all_warnings": [],
            "all_errors": [],
            "overall_status": "valid"
        }

        all_valid = True
        has_suspicious = False

        for symbol, df in datasets.items():
            res = self.validate_dataset(symbol, df)
            report["symbol_quality"][symbol] = res
            report["all_warnings"].extend(res["warnings"])
            report["all_errors"].extend(res["errors"])

            if res["classification"] == "invalid":
                all_valid = False
            elif res["classification"] == "suspicious":
                has_suspicious = True

        if not all_valid:
            report["overall_status"] = "invalid"
        elif has_suspicious:
            report["overall_status"] = "suspicious"

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{date_str}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved quality report to {out_path}")
        return report
