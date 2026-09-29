import pandas as pd
import numpy as np
from typing import Dict, Any, List


def find_historical_analogues(feature_df: pd.DataFrame, top_k: int = 25) -> Dict[str, Any]:
    """
    Finds historical analogues for current market state (last row of feature_df)
    using key normalized indicators. Strictly prevents look-ahead bias by only
    using past observations up to index N-60.
    """
    if feature_df is None or len(feature_df) < 100:
        return {"analogues": [], "sample_count": 0}

    df = feature_df.copy().reset_index(drop=True)

    # Feature columns to compute Euclidean distance
    feature_cols = ["rsi_14", "dist_sma_20", "dist_sma_50", "volatility_20d"]
    avail_cols = [c for c in feature_cols if c in df.columns]

    if not avail_cols:
        return {"analogues": [], "sample_count": 0}

    # Current state vector
    current_idx = len(df) - 1
    current_vec = df.loc[current_idx, avail_cols].values.astype(float)

    # Historical pool (excluding last 60 days to allow full forward return observation)
    hist_limit = max(0, len(df) - 61)
    hist_df = df.iloc[20:hist_limit].copy()

    if hist_df.empty:
        return {"analogues": [], "sample_count": 0}

    # Normalize vectors
    mean = df[avail_cols].mean().values.astype(float)
    std = df[avail_cols].std().replace(0, 1.0).values.astype(float)

    curr_norm = (current_vec - mean) / std
    hist_vecs = (hist_df[avail_cols].values.astype(float) - mean) / std

    # Euclidean distance
    distances = np.linalg.norm(hist_vecs - curr_norm, axis=1)
    hist_df["similarity_score"] = 100.0 / (1.0 + distances)

    top_analogues = hist_df.sort_values("similarity_score", ascending=False).head(top_k)

    horizons = [1, 3, 5, 10, 20, 60]
    outcomes = {h: [] for h in horizons}

    for idx, row in top_analogues.iterrows():
        for h in horizons:
            if idx + h < len(df):
                fwd_ret = (df.loc[idx + h, "close"] / df.loc[idx, "close"]) - 1.0
                outcomes[h].append(fwd_ret)

    stats_by_horizon = {}
    for h, rets in outcomes.items():
        if rets:
            arr = np.array(rets)
            stats_by_horizon[f"fwd_{h}d"] = {
                "sample_count": len(arr),
                "mean_return": float(np.mean(arr)),
                "median_return": float(np.median(arr)),
                "positive_prob": float((arr > 0).mean()),
                "negative_prob": float((arr < 0).mean()),
                "max_gain": float(np.max(arr)),
                "max_loss": float(np.min(arr))
            }

    analogues_list = []
    for idx, row in top_analogues.head(5).iterrows():
        analogues_list.append({
            "date": row["date"],
            "similarity_score": round(float(row["similarity_score"]), 2),
            "close": float(row["close"]),
            "rsi_14": float(row.get("rsi_14", 0.0))
        })

    return {
        "analogues": analogues_list,
        "sample_count": len(top_analogues),
        "horizon_statistics": stats_by_horizon
    }
