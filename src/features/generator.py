"""ML-ready feature store generator with strict look-ahead bias protection and event features."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class FeatureStoreGenerator:
    """Builds unified ML daily feature datasets with target lookups stored separately or as clearly prefixed columns."""

    @staticmethod
    def build_feature_dataset(
        symbol: str,
        metal_df: pd.DataFrame,
        ratio_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        events: List[Dict[str, Any]] = []
    ) -> pd.DataFrame:
        """Constructs temporally safe daily feature matrix."""
        if metal_df.empty or 'close' not in metal_df.columns:
            return pd.DataFrame()

        df = metal_df.copy().sort_values('date').reset_index(drop=True)
        df['symbol'] = symbol

        # 1. Price Returns & Volatility
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)
        df['return_60d'] = df['close'].pct_change(60)

        # 2. Ratio features if available
        if not ratio_df.empty and 'gold_silver_ratio' in ratio_df.columns:
            r_sub = ratio_df[['date', 'gold_silver_ratio', 'ratio_zscore_252d', 'return_spread_1d']]
            df = pd.merge(df, r_sub, on='date', how='left')

        # 3. Macro asset features (DXY, Yields, VIX, Oil, Equities)
        for m_name, m_df in macro_dfs.items():
            if not m_df.empty and 'close' in m_df.columns:
                sub_m = m_df[['date', 'close']].copy().rename(columns={'close': f'{m_name.lower()}_close'})
                sub_m[f'{m_name.lower()}_return_1d'] = sub_m[f'{m_name.lower()}_close'].pct_change(1)
                sub_m[f'{m_name.lower()}_return_20d'] = sub_m[f'{m_name.lower()}_close'].pct_change(20)
                df = pd.merge(df, sub_m, on='date', how='left')

        # Forward fill any missing macro aligned dates
        df.ffill(inplace=True)

        # 4. Macro Event Features
        if events:
            event_dates = [pd.to_datetime(e['date']) for e in events if 'date' in e]
            if event_dates:
                curr_dates = pd.to_datetime(df['date'])
                days_to_next = []
                days_since_prev = []

                for d in curr_dates:
                    future_events = [ed for ed in event_dates if ed >= d]
                    past_events = [ed for ed in event_dates if ed <= d]

                    to_next = (min(future_events) - d).days if future_events else 999
                    since_prev = (d - max(past_events)).days if past_events else 999

                    days_to_next.append(to_next)
                    days_since_prev.append(since_prev)

                df['days_to_event'] = days_to_next
                df['days_since_event'] = days_since_prev
            else:
                df['days_to_event'] = 999
                df['days_since_event'] = 999
        else:
            df['days_to_event'] = 999
            df['days_since_event'] = 999

        # 5. Future Target Horizons (Explicitly prefixed with future_ to prevent feature leakage)
        df['future_return_1d'] = df['close'].pct_change(1).shift(-1)
        df['future_return_3d'] = df['close'].pct_change(3).shift(-3)
        df['future_return_5d'] = df['close'].pct_change(5).shift(-5)
        df['future_return_10d'] = df['close'].pct_change(10).shift(-10)
        df['future_return_20d'] = df['close'].pct_change(20).shift(-20)
        df['future_return_60d'] = df['close'].pct_change(60).shift(-60)

        df['future_direction_5d'] = (df['future_return_5d'] > 0).astype(int)

        return df
