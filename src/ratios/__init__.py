import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze_gold_silver_ratio(gold_df: pd.DataFrame, silver_df: pd.DataFrame, usd_inr: float = 83.0) -> Dict[str, Any]:
    if gold_df is None or silver_df is None or gold_df.empty or silver_df.empty:
        return {}

    g_df = gold_df[["date", "close"]].rename(columns={"close": "gold_price"})
    s_df = silver_df[["date", "close"]].rename(columns={"close": "silver_price"})

    merged = pd.merge(g_df, s_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if merged.empty:
        return {}

    merged["ratio"] = merged["gold_price"] / merged["silver_price"]
    ratio_series = merged["ratio"]

    latest_ratio = float(ratio_series.iloc[-1])
    mean_ratio = float(ratio_series.mean())
    std_ratio = float(ratio_series.std()) if len(ratio_series) > 1 else 1.0
    z_score = float((latest_ratio - mean_ratio) / (std_ratio if std_ratio != 0 else 1.0))
    percentile = float((ratio_series <= latest_ratio).mean() * 100)

    # Indian Market Price Representations
    gold_usd_oz = float(merged["gold_price"].iloc[-1])
    silver_usd_oz = float(merged["silver_price"].iloc[-1])

    gold_inr_10g = (gold_usd_oz / 3.11034768) * usd_inr
    silver_inr_kg = (silver_usd_oz / 0.0311034768) * usd_inr

    return {
        "current_ratio": round(latest_ratio, 2),
        "long_term_average": round(mean_ratio, 2),
        "z_score": round(z_score, 2),
        "historical_percentile": round(percentile, 2),
        "gold_usd_oz": round(gold_usd_oz, 2),
        "silver_usd_oz": round(silver_usd_oz, 2),
        "gold_inr_10g": round(gold_inr_10g, 2),
        "silver_inr_kg": round(silver_inr_kg, 2),
        "usd_inr_rate_used": usd_inr
    }
