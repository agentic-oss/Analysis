import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze_gold_drivers(gold_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    gold_df: DataFrame with 'date' and 'close' (or technicals)
    macro_dfs: dict mapping key (e.g. 'DXY', 'US_REAL_YIELD', 'US10Y', 'BRENT', 'SP500', 'VIX') to DataFrame
    """
    if gold_df is None or gold_df.empty:
        return {}

    merged = gold_df[["date", "close"]].rename(columns={"close": "gold_price"})

    for key, df in macro_dfs.items():
        if df is not None and not df.empty and "close" in df.columns:
            sub_df = df[["date", "close"]].rename(columns={"close": key})
            merged = pd.merge_ordered(merged, sub_df, on="date", how="left").ffill()

    merged = merged.dropna(subset=["gold_price"])
    correlations = {}
    windows = [20, 60, 120, 252]

    for key in macro_dfs.keys():
        if key in merged.columns:
            correlations[key] = {}
            for w in windows:
                if len(merged) >= w:
                    corr = merged["gold_price"].iloc[-w:].corr(merged[key].iloc[-w:])
                    correlations[key][f"corr_{w}d"] = float(corr) if not np.isnan(corr) else 0.0
                else:
                    correlations[key][f"corr_{w}d"] = 0.0

    latest = merged.iloc[-1] if not merged.empty else {}

    return {
        "gold_latest_price": float(latest.get("gold_price", 0.0)),
        "driver_correlations": correlations,
        "sample_size": len(merged)
    }
