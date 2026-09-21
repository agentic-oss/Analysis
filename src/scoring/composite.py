import pandas as pd
import numpy as np
from typing import Dict, List, Any


class CompositeScoreCalculator:
    """Computes transparent, explainable sub-scores and composite multi-factor research scores (0-100)."""

    DEFAULT_WEIGHTS = {
        "technical": 0.25,
        "macro": 0.20,
        "momentum": 0.15,
        "volatility": 0.10,
        "relative_value": 0.15,
        "pattern": 0.15
    }

    @staticmethod
    def calculate_scores(
        metal_df: pd.DataFrame,
        macro_dict: Dict[str, pd.DataFrame] = None,
        ratio_df: pd.DataFrame = None,
        analogue_stats: Dict[str, Any] = None,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        if metal_df is None or metal_df.empty:
            return {}

        w = weights if weights else CompositeScoreCalculator.DEFAULT_WEIGHTS
        latest = metal_df.iloc[-1]

        # 1. Technical Score (MAs, Trend, Distance from 200 SMA)
        dist_200 = float(latest.get("dist_sma_200", 0.0) or 0.0)
        close = float(latest["close"])
        sma50 = float(latest.get("sma_50", close) or close)
        tech_base = 50.0
        if close > sma50:
            tech_base += 20.0
        if dist_200 > 0:
            tech_base += min(20.0, dist_200 * 2.0)
        elif dist_200 < 0:
            tech_base -= min(20.0, abs(dist_200) * 2.0)
        technical_score = max(0.0, min(100.0, tech_base))

        # 2. Momentum Score (RSI, MACD)
        rsi = float(latest.get("rsi_14", 50.0) or 50.0)
        macd_hist = float(latest.get("macd_hist", 0.0) or 0.0)
        mom_base = 50.0 + (rsi - 50.0)
        if macd_hist > 0:
            mom_base += 10.0
        elif macd_hist < 0:
            mom_base -= 10.0
        momentum_score = max(0.0, min(100.0, mom_base))

        # 3. Volatility Score (Stability score: low/normal vol = higher score, extreme spike = lower score)
        vol_20d = float(latest.get("volatility_20d", 15.0) or 15.0)
        vol_score = max(0.0, min(100.0, 100.0 - (vol_20d * 2.0)))

        # 4. Macro Score
        macro_score = 50.0
        if macro_dict:
            dxy_df = macro_dict.get("DXY")
            if dxy_df is not None and not dxy_df.empty:
                dxy_ret = float(dxy_df.iloc[-1].get("return_20d", 0.0) or 0.0)
                # Weak dollar favors metals
                macro_score -= (dxy_ret * 200.0)

            vix_df = macro_dict.get("VIX")
            if vix_df is not None and not vix_df.empty:
                vix_val = float(vix_df.iloc[-1]["close"])
                if vix_val > 20.0:
                    macro_score += 10.0  # Safe haven support
        macro_score = max(0.0, min(100.0, macro_score))

        # 5. Relative Value Score
        rv_score = 50.0
        if ratio_df is not None and not ratio_df.empty:
            r_z = float(ratio_df.iloc[-1].get("ratio_zscore_252", 0.0) or 0.0)
            # High ratio means silver cheap relative to gold
            rv_score = max(0.0, min(100.0, 50.0 - (r_z * 15.0)))

        # 6. Historical Pattern Score
        pattern_score = 50.0
        if analogue_stats and "10d" in analogue_stats:
            pos_prob = analogue_stats["10d"].get("positive_prob_pct", 50.0)
            pattern_score = pos_prob

        # Composite Score Calculation
        composite = (
            w.get("technical", 0.25) * technical_score +
            w.get("macro", 0.20) * macro_score +
            w.get("momentum", 0.15) * momentum_score +
            w.get("volatility", 0.10) * vol_score +
            w.get("relative_value", 0.15) * rv_score +
            w.get("pattern", 0.15) * pattern_score
        )

        return {
            "technical_score": round(technical_score, 1),
            "macro_score": round(macro_score, 1),
            "momentum_score": round(momentum_score, 1),
            "volatility_score": round(vol_score, 1),
            "relative_value_score": round(rv_score, 1),
            "historical_pattern_score": round(pattern_score, 1),
            "composite_score": round(composite, 1)
        }
