"""
ML Daily Feature Store & Macro Event Feature Generator.
Strictly prevents look-ahead bias by using only information available on date T.
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds ML-ready daily feature datasets and stores future return target columns separately.
    Prevents look-ahead bias by only utilizing info up to date T for features.
    """

    @staticmethod
    def build_daily_features(
        metal_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        relative_value_df: pd.DataFrame,
        macro_regime_str: str = "Neutral",
        market_regime_str: str = "Range / Normal Volatility",
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Combines technical indicators, macro correlations, relative value, and regime labels into a unified feature set.
        """
        if metal_df.empty:
            return pd.DataFrame()

        df_feat = metal_df.copy().sort_values("date").reset_index(drop=True)

        # Merge relative value features if available
        if not relative_value_df.empty and "gold_silver_ratio" in relative_value_df.columns:
            rv_cols = ["date", "gold_silver_ratio", "ratio_zscore_60d", "ratio_percentile_historical", "return_spread_1d"]
            rv_sub = relative_value_df[[c for c in rv_cols if c in relative_value_df.columns]]
            df_feat = pd.merge(df_feat, rv_sub, on="date", how="left")

        # Merge macro driver 1D returns
        for asset_name, m_df in macro_dfs.items():
            if m_df.empty or "close" not in m_df.columns:
                continue
            m_sub = m_df[["date", "close"]].rename(columns={"close": f"{asset_name}_close"})
            m_sub[f"{asset_name}_return_1d"] = m_sub[f"{asset_name}_close"].pct_change(1)
            df_feat = pd.merge(df_feat, m_sub[["date", f"{asset_name}_return_1d"]], on="date", how="left")

        # Encode Regime Categoricals
        df_feat["macro_regime"] = macro_regime_str
        df_feat["market_regime"] = market_regime_str

        # Forward Targets (Explicitly labeled as future_return_* and created via shift)
        for h in horizons:
            df_feat[f"future_return_{h}d"] = df_feat["close"].pct_change(h).shift(-h)
            df_feat[f"future_direction_{h}d"] = (df_feat[f"future_return_{h}d"] > 0).astype(int)

        return df_feat


class EventFeatureGenerator:
    """
    Generates event features for major macro events (FOMC decisions, CPI releases, RBI policy decisions).
    Computes days_to_event and days_since_event without look-ahead bias.
    """

    @staticmethod
    def attach_event_features(
        df_features: pd.DataFrame,
        event_calendar: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        `event_calendar` is a list of dicts: [{'date': 'YYYY-MM-DD', 'event_type': 'FOMC', 'importance': 'high'}]
        """
        if df_features.empty or not event_calendar:
            df_out = df_features.copy()
            df_out["days_to_event"] = 999
            df_out["days_since_event"] = 999
            df_out["next_event_type"] = "none"
            return df_out

        df_out = df_features.copy().sort_values("date").reset_index(drop=True)
        event_df = pd.DataFrame(event_calendar)
        event_df["event_date"] = pd.to_datetime(event_df["date"])

        feature_dates = pd.to_datetime(df_out["date"])

        days_to_evt = []
        days_since_evt = []
        next_evt_type = []

        for f_date in feature_dates:
            future_evts = event_df[event_df["event_date"] >= f_date]
            past_evts = event_df[event_df["event_date"] <= f_date]

            if not future_evts.empty:
                next_evt = future_evts.iloc[0]
                dt = (next_evt["event_date"] - f_date).days
                days_to_evt.append(dt)
                next_evt_type.append(next_evt.get("event_type", "macro_event"))
            else:
                days_to_evt.append(999)
                next_evt_type.append("none")

            if not past_evts.empty:
                last_evt = past_evts.iloc[-1]
                ds = (f_date - last_evt["event_date"]).days
                days_since_evt.append(ds)
            else:
                days_since_evt.append(999)

        df_out["days_to_event"] = days_to_evt
        df_out["days_since_event"] = days_since_evt
        df_out["next_event_type"] = next_evt_type

        return df_out
