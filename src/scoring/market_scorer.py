import pandas as pd
import numpy as np
from typing import Dict, Any

class MarketScorer:
    """Calculates modular component scores (0-100) and composite market health score."""

    @staticmethod
    def calculate_scores(df: pd.DataFrame, row_idx: int, prefix: str = "gold_") -> Dict[str, Any]:
        if row_idx >= len(df) or row_idx < 0:
            return {}

        row = df.iloc[row_idx]

        # 1. Technical Score
        rsi = row.get(f"{prefix}rsi_14", 50.0)
        dist_sma50 = row.get(f"{prefix}dist_sma_50", 0.0)
        dist_sma200 = row.get(f"{prefix}dist_sma_200", 0.0)

        tech_pts = 50.0
        if rsi > 50: tech_pts += min((rsi - 50) * 0.8, 20.0)
        else: tech_pts -= min((50 - rsi) * 0.8, 20.0)

        if dist_sma50 > 0: tech_pts += 15.0
        else: tech_pts -= 15.0

        if dist_sma200 > 0: tech_pts += 15.0
        else: tech_pts -= 15.0

        tech_score = int(np.clip(tech_pts, 0, 100))

        # 2. Macro Score
        vix = row.get("VIX_close", 20.0)
        dxy_ret = row.get("DXY_close_ret_20d", 0.0)

        macro_pts = 50.0
        if vix > 20: macro_pts += min((vix - 20) * 1.5, 25.0)  # Safe haven demand
        if dxy_ret < 0: macro_pts += min(abs(dxy_ret) * 500, 25.0) # Weaker dollar aids gold

        macro_score = int(np.clip(macro_pts, 0, 100))

        # 3. Momentum Score
        macd_hist = row.get(f"{prefix}macd_hist", 0.0)
        mom_pts = 50.0 + (25.0 if macd_hist > 0 else -25.0)
        momentum_score = int(np.clip(mom_pts, 0, 100))

        # 4. Volatility Score
        vol = row.get(f"{prefix}historical_vol_20", 0.15)
        vol_pts = 100.0 - min(vol * 200, 80.0)
        volatility_score = int(np.clip(vol_pts, 0, 100))

        # 5. Relative Value Score
        ratio_z = row.get("ratio_zscore_252d", 0.0)
        rv_pts = 50.0 - ratio_z * 15.0  # High ratio favors silver, lower z favors gold
        rv_score = int(np.clip(rv_pts, 0, 100))

        # Composite score
        composite = int(0.25 * tech_score + 0.25 * macro_score + 0.20 * momentum_score + 0.15 * volatility_score + 0.15 * rv_score)

        return {
            "technical_score": tech_score,
            "macro_score": macro_score,
            "momentum_score": momentum_score,
            "volatility_score": volatility_score,
            "relative_value_score": rv_score,
            "composite_score": composite
        }
