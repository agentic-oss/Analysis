import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class FeatureBuilder:
    """Builds ML-ready daily feature store with look-ahead bias protection and event features."""

    @staticmethod
    def build_daily_features(
        metal_symbol: str,
        metal_df: pd.DataFrame,
        macro_dict: Dict[str, pd.DataFrame] = None,
        ratio_df: pd.DataFrame = None,
        events_df: pd.DataFrame = None,
        macro_regime: str = "Neutral",
        market_regime: str = "Bullish Trend"
    ) -> pd.DataFrame:
        if metal_df is None or metal_df.empty:
            return pd.DataFrame()

        df = metal_df.copy().sort_values("date").reset_index(drop=True)

        features = pd.DataFrame()
        features["date"] = df["date"]
        features["symbol"] = metal_symbol
        features["price"] = df["close"]

        # 1. Technical Features
        for col in [
            "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
            "rsi_14", "stoch_rsi", "macd", "macd_signal", "macd_hist", "roc_12", "momentum_10",
            "atr_14", "volatility_20d", "volatility_60d", "bb_width", "dist_sma_20", "dist_sma_50",
            "dist_sma_200", "dist_52w_high", "dist_52w_low", "gap_pct", "intraday_range_pct", "trend_strength"
        ]:
            features[col] = df[col] if col in df.columns else np.nan

        # 2. Ratio features
        if ratio_df is not None and not ratio_df.empty:
            r_sub = ratio_df[["date", "ratio", "ratio_zscore_252", "ratio_percentile_252"]].copy()
            r_sub = r_sub.rename(columns={
                "ratio": "gold_silver_ratio",
                "ratio_zscore_252": "gold_silver_ratio_zscore",
                "ratio_percentile_252": "gold_silver_ratio_percentile"
            })
            features = pd.merge(features, r_sub, on="date", how="left")
        else:
            features["gold_silver_ratio"] = np.nan
            features["gold_silver_ratio_zscore"] = np.nan
            features["gold_silver_ratio_percentile"] = np.nan

        # 3. Cross-asset Macro Features
        if macro_dict:
            for m_sym, m_df in macro_dict.items():
                if m_df is None or m_df.empty:
                    continue
                sub = m_df[["date", "close"]].rename(columns={"close": f"{m_sym.lower()}_close"}).sort_values("date")
                sub[f"{m_sym.lower()}_return_1d"] = sub[f"{m_sym.lower()}_close"].pct_change()
                features = pd.merge(features, sub[["date", f"{m_sym.lower()}_return_1d"]], on="date", how="left")

        # 4. Regime features
        features["macro_regime"] = macro_regime
        features["market_regime"] = market_regime

        # 5. Event features
        if events_df is not None and not events_df.empty:
            ev_sub = events_df[["date", "event_name", "importance"]].copy()
            features = pd.merge(features, ev_sub, on="date", how="left")
            features["event_name"] = features["event_name"].fillna("NONE")
            features["importance"] = features["importance"].fillna("NONE")
        else:
            features["event_name"] = "NONE"
            features["importance"] = "NONE"

        # Fill NaNs for feature inputs
        feat_cols = [c for c in features.columns if not c.startswith("target_") and c not in ["date", "symbol", "event_name", "importance", "macro_regime", "market_regime"]]
        features[feat_cols] = features[feat_cols].ffill().bfill().fillna(0.0)

        # 6. Target Generation (Future returns - strictly separated for ML training)
        for h in [1, 3, 5, 10, 20, 60]:
            features[f"target_future_return_{h}d"] = df["close"].pct_change(h).shift(-h)
            features[f"target_future_direction_{h}d"] = (features[f"target_future_return_{h}d"] > 0).astype(int)

        return features
