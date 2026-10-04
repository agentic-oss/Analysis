import pandas as pd
import numpy as np
from typing import Dict, List, Any

def calculate_rolling_correlations(
    target_series: pd.Series,
    benchmark_series: pd.Series,
    windows: List[int] = [20, 60, 120, 252]
) -> Dict[str, pd.Series]:
    """Calculate rolling correlations between target (e.g. Gold) and benchmark (e.g. DXY) across multiple horizons."""
    correlations = {}
    target_returns = target_series.pct_change()
    benchmark_returns = benchmark_series.pct_change()

    for w in windows:
        corr = target_returns.rolling(window=w, min_periods=max(5, w//4)).corr(benchmark_returns)
        correlations[f"corr_{w}d"] = corr

    return correlations

def analyze_cross_market_relationships(
    processed_dict: Dict[str, pd.DataFrame],
    windows: List[int] = [20, 60, 120, 252]
) -> Dict[str, pd.DataFrame]:
    """Calculate rolling correlations for Gold and Silver against key macro drivers (DXY, Rates, Equities, Oil, VIX)."""
    results = {}

    gold_df = processed_dict.get("GOLD")
    silver_df = processed_dict.get("SILVER")

    if gold_df is None or gold_df.empty:
        return results

    benchmarks = {
        "DXY": processed_dict.get("DXY"),
        "US10Y": processed_dict.get("US10Y"),
        "TIP": processed_dict.get("TIP"),
        "CRUDE_OIL": processed_dict.get("CRUDE_OIL"),
        "SP500": processed_dict.get("SP500"),
        "NIFTY50": processed_dict.get("NIFTY50"),
        "VIX": processed_dict.get("VIX")
    }

    for metal_key, metal_df in [("GOLD", gold_df), ("SILVER", silver_df)]:
        if metal_df is None or metal_df.empty:
            continue

        metal_close = metal_df.set_index("date")["close"]
        corr_df = pd.DataFrame(index=metal_close.index)

        for bm_key, bm_df in benchmarks.items():
            if bm_df is None or bm_df.empty:
                continue

            bm_close = bm_df.set_index("date")["close"]
            aligned = pd.concat([metal_close, bm_close], axis=1, keys=["metal", "bm"]).dropna()

            for w in windows:
                c = aligned["metal"].pct_change().rolling(w, min_periods=max(5, w//4)).corr(aligned["bm"].pct_change())
                corr_df[f"{metal_key}_vs_{bm_key}_corr_{w}d"] = c

        results[metal_key] = corr_df.reset_index()

    return results
