"""Silver multi-factor correlation and driver analysis."""

import pandas as pd
from typing import Dict, Any


class SilverAnalyzer:
    """Analyzes Silver against Gold, copper, commodities, equities, rates, and DXY."""

    def __init__(self, lookbacks: list = [20, 60, 120, 252]):
        self.lookbacks = lookbacks

    def analyze_relationships(
        self,
        silver_df: pd.DataFrame,
        gold_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Computes rolling correlations for Silver."""
        if silver_df.empty or 'close' not in silver_df.columns:
            return {}

        df = silver_df[['date', 'close']].copy().sort_values('date')
        df['silver_ret'] = df['close'].pct_change()

        correlations = {}

        # Silver vs Gold
        if not gold_df.empty and 'close' in gold_df.columns:
            g_df = gold_df[['date', 'close']].rename(columns={'close': 'gold_close'}).sort_values('date')
            g_df['gold_ret'] = g_df['gold_close'].pct_change()
            merged = pd.merge(df, g_df, on='date', how='inner')
            if len(merged) > 20:
                correlations['GOLD'] = {}
                for win in self.lookbacks:
                    corr_val = merged['silver_ret'].rolling(win).corr(merged['gold_ret']).iloc[-1]
                    correlations['GOLD'][f'corr_{win}d'] = float(corr_val) if pd.notna(corr_val) else 0.0

        drivers = ['DXY', 'WTI', 'SP500', 'VIX', 'US10Y', 'USDINR']
        for driver in drivers:
            if driver in macro_dfs and not macro_dfs[driver].empty:
                m_df = macro_dfs[driver][['date', 'close']].sort_values('date')
                m_df = m_df.rename(columns={'close': f'{driver}_close'})
                m_df[f'{driver}_ret'] = m_df[f'{driver}_close'].pct_change()

                merged = pd.merge(df, m_df, on='date', how='inner')
                if len(merged) > 20:
                    correlations[driver] = {}
                    for win in self.lookbacks:
                        corr_val = merged['silver_ret'].rolling(win).corr(merged[f'{driver}_ret']).iloc[-1]
                        correlations[driver][f'corr_{win}d'] = float(corr_val) if pd.notna(corr_val) else 0.0

        latest_row = silver_df.iloc[-1] if not silver_df.empty else {}
        return {
            "symbol": "SILVER",
            "price": float(latest_row.get('close', 0.0)),
            "return_1d": float(latest_row.get('return_1d', 0.0)) if pd.notna(latest_row.get('return_1d')) else 0.0,
            "return_5d": float(latest_row.get('return_5d', 0.0)) if pd.notna(latest_row.get('return_5d')) else 0.0,
            "return_20d": float(latest_row.get('return_20d', 0.0)) if pd.notna(latest_row.get('return_20d')) else 0.0,
            "correlations": correlations
        }
