import pandas as pd
import numpy as np
from typing import Dict, Any, List

class FeatureEngine:
    """Builds ML-ready daily feature store and future targets strictly without look-ahead bias."""

    def __init__(self, horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = horizons

    def build_features(self, master_df: pd.DataFrame, events_df: pd.DataFrame = None) -> pd.DataFrame:
        df = master_df.copy()

        # Build return features
        for sym in ["GOLD", "SILVER", "DXY", "SP500", "NIFTY50", "CRUDE_OIL"]:
            col = f"{sym}_close"
            if col in df.columns:
                df[f"{sym}_ret_1d"] = df[col].pct_change(1)
                df[f"{sym}_ret_5d"] = df[col].pct_change(5)
                df[f"{sym}_ret_20d"] = df[col].pct_change(20)

        # Build rate & volatility change features
        for sym in ["US10Y", "VIX"]:
            col = f"{sym}_close"
            if col in df.columns:
                df[f"{sym}_chg_1d"] = df[col].diff(1)
                df[f"{sym}_chg_5d"] = df[col].diff(5)
                df[f"{sym}_chg_20d"] = df[col].diff(20)

        # Event-based features (e.g., days since / days to event)
        if events_df is not None and not events_df.empty:
            df["date_dt"] = pd.to_datetime(df["date"])
            events_df["event_date_dt"] = pd.to_datetime(events_df["date"])

            days_since = []
            days_to = []

            for dt in df["date_dt"]:
                past_events = events_df[events_df["event_date_dt"] <= dt]
                future_events = events_df[events_df["event_date_dt"] > dt]

                if not past_events.empty:
                    since = (dt - past_events["event_date_dt"].max()).days
                else:
                    since = 999
                days_since.append(since)

                if not future_events.empty:
                    to_evt = (future_events["event_date_dt"].min() - dt).days
                else:
                    to_evt = 999
                days_to.append(to_evt)

            df["days_since_macro_event"] = days_since
            df["days_to_macro_event"] = days_to
            df.drop(columns=["date_dt"], inplace=True)
        else:
            df["days_since_macro_event"] = 999
            df["days_to_macro_event"] = 999

        # Target variables (Stored separately, strictly future-looking)
        for sym in ["GOLD", "SILVER"]:
            col = f"{sym}_close"
            if col in df.columns:
                p = df[col].astype(float)
                for h in self.horizons:
                    df[f"target_{sym.lower()}_future_ret_{h}d"] = (p.shift(-h) - p) / p
                    df[f"target_{sym.lower()}_future_dir_{h}d"] = (df[f"target_{sym.lower()}_future_ret_{h}d"] > 0).astype(int)

        return df
