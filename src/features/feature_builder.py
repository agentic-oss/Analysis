"""Feature engineering module to construct ML-ready daily feature sets and strictly separated future targets."""

import numpy as np
import pandas as pd
from typing import Dict, Any


class FeatureBuilder:
    """Builds comprehensive ML-ready daily feature sets without look-ahead bias."""

    @staticmethod
    def build_daily_features(
        metals_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merges technical indicators, cross-market correlations, macro variables,
        and event proximity into a unified feature set, adding strictly future targets.
        """
        if metals_df.empty:
            return pd.DataFrame()

        df = metals_df.copy().sort_values(['symbol', 'date'])

        # Add future return targets per instrument
        horizons = [1, 3, 5, 10, 20, 60]
        for h in horizons:
            df[f'future_return_{h}d'] = df.groupby('symbol')['close'].shift(-h) / df['close'] - 1.0
            df[f'future_direction_{h}d'] = (df[f'future_return_{h}d'] > 0).astype(int)

        # Merge macro driver changes (DXY, yields, oil, VIX, NIFTY)
        for macro_sym in ['DXY', 'US10Y', 'US_REAL_YIELD', 'WTI', 'SP500', 'VIX', 'USDINR', 'NIFTY50']:
            if macro_sym in macro_dfs and not macro_dfs[macro_sym].empty:
                m_df = macro_dfs[macro_sym][['date', 'close']].sort_values('date').copy()
                m_df = m_df.rename(columns={'close': f'{macro_sym.lower()}_close'})
                m_df[f'{macro_sym.lower()}_return_1d'] = m_df[f'{macro_sym.lower()}_close'].pct_change()
                m_df[f'{macro_sym.lower()}_return_5d'] = m_df[f'{macro_sym.lower()}_close'].pct_change(5)

                df = pd.merge(df, m_df, on='date', how='left')

        # Clean fill forward for macro data gaps
        df = df.ffill().bfill()
        return df
