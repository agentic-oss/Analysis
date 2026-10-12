import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.indicators.technical import compute_all_technical_indicators

class DailyFeatureStoreBuilder:
    """
    Constructs ML-ready daily feature datasets strictly preserving temporal order.
    Generates point-in-time features on date T and forward targets shift(-h) so that
    information on date T contains zero look-ahead bias.
    """

    HORIZONS = [1, 3, 5, 10, 20, 60]

    def build_feature_dataset(
        self,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        dxy_df: Optional[pd.DataFrame] = None,
        us10y_df: Optional[pd.DataFrame] = None,
        vix_df: Optional[pd.DataFrame] = None,
        oil_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Merges asset datasets, computes technical & macro features, and attaches future targets."""

        g_tech = compute_all_technical_indicators(gold_df).copy()
        s_tech = compute_all_technical_indicators(silver_df).copy()

        g_tech = g_tech.set_index("date")
        s_tech = s_tech.set_index("date")

        # Base frame on gold dates
        df = pd.DataFrame(index=g_tech.index)
        df["instrument"] = "GOLD"
        df["price"] = g_tech["close"]

        # Gold technical features
        df["return_1d"] = g_tech["close"].pct_change(1)
        df["return_5d"] = g_tech["close"].pct_change(5)
        df["return_20d"] = g_tech["close"].pct_change(20)
        df["return_60d"] = g_tech["close"].pct_change(60)

        df["rsi_14"] = g_tech["rsi_14"]
        df["macd"] = g_tech["macd"]
        df["macd_signal"] = g_tech["macd_signal"]
        df["macd_histogram"] = g_tech["macd_histogram"]
        df["atr_14"] = g_tech["atr_14"]
        df["hist_vol_20"] = g_tech["hist_vol_20"]
        df["sma_20"] = g_tech["sma_20"]
        df["sma_50"] = g_tech["sma_50"]
        df["sma_200"] = g_tech["sma_200"]
        df["dist_sma_20"] = g_tech["dist_sma_20"]
        df["dist_sma_50"] = g_tech["dist_sma_50"]
        df["dist_sma_200"] = g_tech["dist_sma_200"]

        # Gold/Silver ratio features
        common_idx = df.index.intersection(s_tech.index)
        df["silver_close"] = np.nan
        df.loc[common_idx, "silver_close"] = s_tech.loc[common_idx, "close"]
        df["gold_silver_ratio"] = df["price"] / df["silver_close"]

        ratio_mean = df["gold_silver_ratio"].expanding(min_periods=20).mean()
        ratio_std = df["gold_silver_ratio"].expanding(min_periods=20).std()
        df["gold_silver_ratio_zscore"] = (df["gold_silver_ratio"] - ratio_mean) / ratio_std.replace(0, np.nan)

        # Macro series features if provided
        for macro_df, name_prefix in [
            (dxy_df, "dxy"),
            (us10y_df, "us10y"),
            (vix_df, "vix"),
            (oil_df, "oil")
        ]:
            if macro_df is not None and not macro_df.empty:
                m_sub = macro_df.set_index("date")["close"]
                m_aligned = m_sub.reindex(df.index).ffill()
                df[f"{name_prefix}_price"] = m_aligned
                df[f"{name_prefix}_return_1d"] = m_aligned.pct_change(1)
                df[f"{name_prefix}_return_20d"] = m_aligned.pct_change(20)

        # Event features
        df["days_to_event"] = 999
        df["days_since_event"] = 999
        df["event_importance"] = 0

        if events_df is not None and not events_df.empty:
            event_dates = pd.to_datetime(events_df["date"]).sort_values().values
            current_dates = pd.to_datetime(df.index).values
            for i, cur_d in enumerate(current_dates):
                diffs = (event_dates - cur_d).astype("timedelta64[D]").astype(int)
                future_diffs = diffs[diffs >= 0]
                past_diffs = diffs[diffs < 0]
                if len(future_diffs) > 0:
                    df.iloc[i, df.columns.get_loc("days_to_event")] = int(future_diffs.min())
                if len(past_diffs) > 0:
                    df.iloc[i, df.columns.get_loc("days_since_event")] = int(abs(past_diffs.max()))

        # Fill missing values ONLY in feature columns
        target_cols = []
        for h in self.HORIZONS:
            target_cols.append(f"future_return_{h}d")
            target_cols.append(f"future_direction_{h}d")

        feature_cols = [c for c in df.columns if c not in target_cols]
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # Separate Forward Return Targets (zero look-ahead bias)
        for h in self.HORIZONS:
            f_ret = df["price"].pct_change(h).shift(-h)
            df[f"future_return_{h}d"] = f_ret
            df[f"future_direction_{h}d"] = np.where(f_ret.isna(), np.nan, (f_ret > 0).astype(float))

        df = df.reset_index().rename(columns={"index": "date"})
        return df
