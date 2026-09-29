import pandas as pd
import numpy as np
from typing import Dict, Any


def build_daily_feature_dataset(
    instrument_df: pd.DataFrame,
    macro_dfs: Dict[str, pd.DataFrame],
    gold_df: pd.DataFrame = None,
    silver_df: pd.DataFrame = None,
    events_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Builds ML-ready daily feature store dataset.
    Features on date T contain ONLY information available by date T.
    Future targets are appended as separate columns to prevent look-ahead bias.
    """
    df = instrument_df.copy().sort_values("date").reset_index(drop=True)

    # 1. Macro features
    for key, m_df in macro_dfs.items():
        if m_df is not None and not m_df.empty and "close" in m_df.columns:
            m_sub = m_df[["date", "close"]].rename(columns={"close": f"macro_{key.lower()}"})
            df = pd.merge_ordered(df, m_sub, on="date", how="left").ffill()
            # Calculate 1d and 20d return/change for macro asset
            df[f"macro_{key.lower()}_ret1d"] = df[f"macro_{key.lower()}"].pct_change(1)
            df[f"macro_{key.lower()}_ret20d"] = df[f"macro_{key.lower()}"].pct_change(20)

    # 2. Gold/Silver ratio features if both available
    if gold_df is not None and silver_df is not None:
        gs = pd.merge(
            gold_df[["date", "close"]].rename(columns={"close": "g_close"}),
            silver_df[["date", "close"]].rename(columns={"close": "s_close"}),
            on="date", how="inner"
        )
        gs["gold_silver_ratio"] = gs["g_close"] / gs["s_close"]
        ratio_mean = gs["gold_silver_ratio"].rolling(252, min_periods=20).mean()
        ratio_std = gs["gold_silver_ratio"].rolling(252, min_periods=20).std().replace(0, 1.0)
        gs["gold_silver_ratio_zscore"] = (gs["gold_silver_ratio"] - ratio_mean) / ratio_std
        df = pd.merge_ordered(df, gs[["date", "gold_silver_ratio", "gold_silver_ratio_zscore"]], on="date", how="left").ffill()

    # 3. Macro Event features if events_df available
    if events_df is not None and not events_df.empty and "date" in events_df.columns:
        event_dates = pd.to_datetime(events_df["date"]).sort_values().drop_duplicates().tolist()
        dates_dt = pd.to_datetime(df["date"])

        days_to_next = []
        days_since_prev = []

        for d in dates_dt:
            future_evs = [ev for ev in event_dates if ev >= d]
            past_evs = [ev for ev in event_dates if ev <= d]

            days_to_next.append((future_evs[0] - d).days if future_evs else 999)
            days_since_prev.append((d - past_evs[-1]).days if past_evs else 999)

        df["days_to_event"] = days_to_next
        df["days_since_event"] = days_since_prev

    # 4. Target variables (Strictly future returns computed for horizon h)
    for h in [1, 3, 5, 10, 20, 60]:
        df[f"future_return_{h}d"] = df["close"].pct_change(h).shift(-h)
        df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

    return df
