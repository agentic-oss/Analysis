"""Gold multi-factor correlation and macro driver analysis."""

import pandas as pd
from typing import Dict, Any


class GoldAnalyzer:
    """Analyzes Gold against DXY, yields, oil, equities, VIX, and calculates rolling correlations."""

    def __init__(self, lookbacks: list = [20, 60, 120, 252]):
        self.lookbacks = lookbacks

    def analyze_relationships(self, gold_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Computes rolling correlations between Gold daily returns and macro drivers.
        macro_dfs dictionary maps driver symbol -> DataFrame with 'date' and 'close'.
        """
        if gold_df.empty or 'close' not in gold_df.columns:
            return {}

        df = gold_df[['date', 'close']].copy().sort_values('date')
        df['gold_ret'] = df['close'].pct_change()

        correlations = {}
        drivers = ['DXY', 'US10Y', 'US_REAL_YIELD', 'WTI', 'SP500', 'VIX', 'USDINR', 'NIFTY50']

        for driver in drivers:
            if driver in macro_dfs and not macro_dfs[driver].empty:
                m_df = macro_dfs[driver][['date', 'close']].sort_values('date')
                m_df = m_df.rename(columns={'close': f'{driver}_close'})
                m_df[f'{driver}_ret'] = m_df[f'{driver}_close'].pct_change()

                merged = pd.merge(df, m_df, on='date', how='inner')
                if len(merged) > 20:
                    correlations[driver] = {}
                    for win in self.lookbacks:
                        corr_val = merged['gold_ret'].rolling(win).corr(merged[f'{driver}_ret']).iloc[-1]
                        correlations[driver][f'corr_{win}d'] = float(corr_val) if pd.notna(corr_val) else 0.0

        latest_row = gold_df.iloc[-1] if not gold_df.empty else {}
        return {
            "symbol": "GOLD",
            "price": float(latest_row.get('close', 0.0)),
            "return_1d": float(latest_row.get('return_1d', 0.0)) if pd.notna(latest_row.get('return_1d')) else 0.0,
            "return_5d": float(latest_row.get('return_5d', 0.0)) if pd.notna(latest_row.get('return_5d')) else 0.0,
            "return_20d": float(latest_row.get('return_20d', 0.0)) if pd.notna(latest_row.get('return_20d')) else 0.0,
            "correlations": correlations
        }
