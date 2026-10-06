import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DailyFeatureStoreGenerator:
    """Generates ML-ready feature datasets with zero look-ahead bias and separate future target metrics."""

    @staticmethod
    def generate_features(metal_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame] = None, instrument: str = "GOLD") -> pd.DataFrame:
        if metal_df.empty:
            return pd.DataFrame()

        df = metal_df.copy()

        # Ensure sorted chronologically
        df = df.sort_values(by="date").reset_index(drop=True)

        # Merge macro features on date T without look-ahead
        if macro_dfs:
            for key, m_df in macro_dfs.items():
                if m_df is not None and not m_df.empty and "close" in m_df.columns:
                    sub = m_df[["date", "close"]].rename(columns={"close": f"{key}_price"}).copy()
                    sub[f"{key}_return_1d"] = sub[f"{key}_price"].pct_change(1)
                    sub[f"{key}_change_5d"] = sub[f"{key}_price"].pct_change(5)
                    df = pd.merge(df, sub[["date", f"{key}_price", f"{key}_return_1d", f"{key}_change_5d"]], on="date", how="left")

        # Event-based features if events dataset provided in macro_dfs
        events_df = macro_dfs.get("events") if macro_dfs else None
        if events_df is not None and not events_df.empty and "date" in events_df.columns:
            event_dates = pd.to_datetime(events_df["date"]).sort_values()
            curr_dates = pd.to_datetime(df["date"])

            days_since = []
            days_to = []

            for d in curr_dates:
                past_ev = event_dates[event_dates <= d]
                fut_ev = event_dates[event_dates > d]

                days_since.append((d - past_ev.iloc[-1]).days if not past_ev.empty else 999)
                days_to.append((fut_ev.iloc[0] - d).days if not fut_ev.empty else 999)

            df["days_since_event"] = days_since
            df["days_to_event"] = days_to
        else:
            df["days_since_event"] = 999
            df["days_to_event"] = 999

        # Target definitions (FUTURE returns calculated on shift(-h) to separate targets from date T features)
        for h in [1, 3, 5, 10, 20, 60]:
            df[f"future_return_{h}d"] = df["close"].pct_change(h).shift(-h)
            df[f"future_dir_{h}d"] = np.where(df[f"future_return_{h}d"] > 0, 1, 0)

        df["instrument"] = instrument
        return df
