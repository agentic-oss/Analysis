"""
Transparent Scoring Engine (0-100).
Calculates sub-scores and composite score.
"""
from typing import Dict, Any


class CompositeScorer:
    """
    Computes transparent sub-scores (Technical, Macro, Momentum, Volatility, Relative Value, Historical Pattern)
    and an explainable composite score on a scale of 0 to 100.
    """

    @staticmethod
    def calculate_scores(
        latest_row: Dict[str, Any],
        macro_regime: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        analogue_stats: Dict[str, Any],
    ) -> Dict[str, Any]:

        # 1. Technical Score (0 - 100)
        tech_score = 50.0
        rsi = float(latest_row.get("rsi_14", 50.0))
        if 40 <= rsi <= 65:
            tech_score += 15.0
        elif rsi > 70:
            tech_score -= 10.0  # Overbought penalty
        elif rsi < 30:
            tech_score += 20.0  # Oversold bounce potential

        dist_sma_200 = float(latest_row.get("dist_sma_200", 0.0))
        if dist_sma_200 > 0:
            tech_score += 15.0
        else:
            tech_score -= 15.0

        if latest_row.get("golden_cross", False):
            tech_score += 15.0

        tech_score = max(0.0, min(100.0, tech_score))

        # 2. Momentum Score (0 - 100)
        mom_score = 50.0
        macd_hist = float(latest_row.get("macd_hist", 0.0))
        roc = float(latest_row.get("roc_10", 0.0))
        if macd_hist > 0:
            mom_score += 20.0
        else:
            mom_score -= 20.0
        if roc > 0:
            mom_score += 15.0
        else:
            mom_score -= 15.0
        mom_score = max(0.0, min(100.0, mom_score))

        # 3. Volatility Score (0 - 100)
        vol_score = 50.0
        vol20 = float(latest_row.get("volatility_20d", 0.15))
        if 0.10 <= vol20 <= 0.22:
            vol_score = 75.0  # Stable trending volatility
        elif vol20 > 0.30:
            vol_score = 30.0  # High unstable volatility
        vol_score = max(0.0, min(100.0, vol_score))

        # 4. Macro Score (0 - 100)
        macro_score = 50.0
        primary_m = macro_regime.get("primary_regime", "Neutral")
        if primary_m == "Risk-Off":
            macro_score = 80.0  # Bullish precious metals safe haven
        elif primary_m == "Inflationary / Rate-Tightening":
            macro_score = 65.0
        elif primary_m == "Risk-On":
            macro_score = 40.0
        elif primary_m == "Disinflationary / Easing":
            macro_score = 60.0

        # 5. Relative Value Score (0 - 100)
        rel_val_score = 50.0
        zscore = float(ratio_analysis.get("zscore", 0.0))
        symbol = latest_row.get("symbol", "GOLD")
        if symbol == "SILVER":
            if zscore > 1.5:
                rel_val_score = 85.0  # Silver is cheap relative to Gold
            elif zscore < -1.5:
                rel_val_score = 30.0
        else:  # GOLD
            if zscore < -1.5:
                rel_val_score = 85.0  # Gold is cheap relative to Silver
            elif zscore > 1.5:
                rel_val_score = 40.0
        rel_val_score = max(0.0, min(100.0, rel_val_score))

        # 6. Historical Pattern / Analogue Score (0 - 100)
        pattern_score = 50.0
        f_stats = analogue_stats.get("forward_statistics", {}).get("20d", {})
        if f_stats:
            pos_prob = float(f_stats.get("positive_probability_pct", 50.0))
            pattern_score = pos_prob
        pattern_score = max(0.0, min(100.0, pattern_score))

        # Composite score (Weighted average)
        composite = (
            tech_score * 0.25
            + mom_score * 0.15
            + vol_score * 0.10
            + macro_score * 0.20
            + rel_val_score * 0.15
            + pattern_score * 0.15
        )

        return {
            "technical_score": round(tech_score, 1),
            "momentum_score": round(mom_score, 1),
            "volatility_score": round(vol_score, 1),
            "macro_score": round(macro_score, 1),
            "relative_value_score": round(rel_val_score, 1),
            "pattern_score": round(pattern_score, 1),
            "composite_score": round(composite, 1),
        }
