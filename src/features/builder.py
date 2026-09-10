import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.indicators.technical import TechnicalIndicators
from src.regimes.classifier import MacroRegimeClassifier, MarketRegimeClassifier

class FeatureStoreBuilder:
    """Builds daily ML-ready feature datasets with temporal correctness and macro event features."""

    def __init__(self, horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = horizons

    def create_features_for_instrument(
        self,
        symbol: str,
        df: pd.DataFrame,
        macro_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        events_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if df.empty or "close" not in df.columns:
            return pd.DataFrame()

        # Step 1: Calculate technical indicators
        feat_df = TechnicalIndicators.calculate_all(df)
        feat_df["instrument"] = symbol

        # Step 2: Merge macro features (lagged by 1 day or aligned on exact date without future leak)
        if macro_dfs:
            date_idx_feat = feat_df.set_index("date")
            for macro_name, m_df in macro_dfs.items():
                if not m_df.empty and "close" in m_df.columns:
                    m_close = m_df.set_index("date")["close"]
                    m_ret = m_df.set_index("date")["close"].pct_change(1)
                    date_idx_feat[f"{macro_name.lower()}_close"] = m_close
                    date_idx_feat[f"{macro_name.lower()}_return_1d"] = m_ret
            feat_df = date_idx_feat.reset_index()

        # Step 3: Event-based features (e.g. FOMC, RBI decision)
        if events_df is not None and not events_df.empty and "date" in events_df.columns:
            events = events_df.sort_values("date")
            event_dates = pd.to_datetime(events["date"]).values

            days_to_list = []
            days_since_list = []

            for dt_str in feat_df["date"]:
                dt = pd.to_datetime(dt_str)
                future_events = event_dates[event_dates >= dt]
                past_events = event_dates[event_dates <= dt]

                days_to = (future_events[0] - dt).days if len(future_events) > 0 else 999
                days_since = (dt - past_events[-1]).days if len(past_events) > 0 else 999

                days_to_list.append(days_to)
                days_since_list.append(days_since)

            feat_df["days_to_event"] = days_to_list
            feat_df["days_since_event"] = days_since_list

        # Step 4: Regimes
        macro_regimes = []
        market_regimes = []
        for _, row in feat_df.iterrows():
            m_res = MacroRegimeClassifier.classify_macro_regime(
                us10y_change_20d=row.get("us10y_return_1d", 0.0) * 20.0,
                dxy_return_20d=row.get("dxy_return_1d", 0.0) * 20.0,
                oil_return_20d=row.get("crude_oil_return_1d", 0.0) * 20.0,
                sp500_return_20d=row.get("sp500_return_1d", 0.0) * 20.0,
                vix_level=row.get("vix_close", 18.0)
            )
            macro_label = m_res["primary_macro_regime"]
            macro_regimes.append(macro_label)

            mkt_res = MarketRegimeClassifier.classify_market_regime(
                price=row["close"],
                sma_20=row.get("sma_20", row["close"]),
                sma_50=row.get("sma_50", row["close"]),
                sma_200=row.get("sma_200", row["close"]),
                volatility_20d=row.get("volatility_20d", 0.15),
                rsi_14=row.get("rsi_14", 50.0),
                macro_regime=macro_label
            )
            market_regimes.append(mkt_res["trend_regime"])

        feat_df["macro_regime"] = macro_regimes
        feat_df["market_regime"] = market_regimes

        # Step 5: Targets stored separately (Future returns computed looking forward)
        c = feat_df["close"]
        for h in self.horizons:
            feat_df[f"future_return_{h}d"] = (c.shift(-h) - c) / c.replace(0, 1e-9)

        return feat_df
