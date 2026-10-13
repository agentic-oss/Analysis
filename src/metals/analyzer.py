"""Dedicated Gold & Silver Multi-Factor and Cross-Market Correlation Analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class MetalAnalyzer:
    """Analyzes metal-specific price trends, multi-period rolling correlations against macro assets."""

    def __init__(self, lookback_windows: List[int] = [20, 60, 120, 252]):
        self.lookback_windows = lookback_windows

    def calculate_correlations(self, metal_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculates rolling correlations between metal returns and macro asset returns (DXY, Yields, VIX, Oil, Equities)
        for specified lookback windows.
        """
        if metal_df.empty or 'close' not in metal_df.columns:
            return {}

        metal_returns = metal_df.set_index('date')['close'].pct_change().dropna()
        correlations: Dict[str, Dict[str, float]] = {}

        for asset_name, df_macro in macro_dfs.items():
            if df_macro.empty or 'close' not in df_macro.columns:
                continue

            macro_returns = df_macro.set_index('date')['close'].pct_change().dropna()
            aligned = pd.concat([metal_returns, macro_returns], axis=1, keys=['metal', 'macro']).dropna()

            if len(aligned) < 20:
                continue

            asset_corrs = {}
            for w in self.lookback_windows:
                if len(aligned) >= w:
                    window_df = aligned.tail(w)
                    corr_val = float(window_df['metal'].corr(window_df['macro']))
                    asset_corrs[f"{w}d"] = round(corr_val, 4) if not np.isnan(corr_val) else 0.0
                else:
                    asset_corrs[f"{w}d"] = 0.0

            correlations[asset_name] = asset_corrs

        return correlations

    def analyze_metal_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extracts technical structure summary for gold or silver."""
        if df.empty:
            return {}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        close = float(latest.get('close', 0.0))
        prev_close = float(prev.get('close', close))
        return_1d = (close - prev_close) / prev_close if prev_close else 0.0

        return {
            "symbol": latest.get('symbol', 'UNKNOWN'),
            "date": str(latest.get('date', '')),
            "close": close,
            "return_1d_pct": round(return_1d * 100, 2),
            "rsi_14": round(float(latest.get('rsi_14', 50.0)), 2),
            "macd": round(float(latest.get('macd', 0.0)), 4),
            "sma_50": round(float(latest.get('sma_50', 0.0)), 2),
            "sma_200": round(float(latest.get('sma_200', 0.0)), 2),
            "dist_sma_200_pct": round(float(latest.get('dist_sma_200_pct', 0.0)), 2),
            "volatility_20d_pct": round(float(latest.get('volatility_20d', 0.0)) * 100, 2),
            "atr_14": round(float(latest.get('atr_14', 0.0)), 2)
        }
