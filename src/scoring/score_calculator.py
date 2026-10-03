import pandas as pd
import numpy as np
from typing import Dict, Any, List


class ScoreCalculator:
    """
    Calculates transparent 0-100 scores across individual pillars
    and builds an explainable composite score.
    """

    @staticmethod
    def calculate_technical_score(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 50.0

        latest = df.iloc[-1]
        c = float(latest["Close"])
        sma20 = float(latest.get("sma_20", c))
        sma50 = float(latest.get("sma_50", c))
        sma200 = float(latest.get("sma_200", c))

        score = 50.0
        if c > sma20: score += 10.0
        if c > sma50: score += 15.0
        if c > sma200: score += 15.0
        if sma50 > sma200: score += 10.0

        return float(np.clip(score, 0.0, 100.0))

    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 50.0

        latest = df.iloc[-1]
        rsi = float(latest.get("rsi_14", 50.0))
        macd_hist = float(latest.get("macd_hist", 0.0))
        roc = float(latest.get("roc_10", 0.0))

        score = 50.0
        # RSI contribution
        if 50.0 <= rsi <= 70.0:
            score += 15.0
        elif rsi > 70.0:
            score += 5.0
        elif 30.0 <= rsi < 50.0:
            score -= 10.0
        elif rsi < 30.0:
            score -= 20.0

        # MACD
        if macd_hist > 0:
            score += 15.0
        else:
            score -= 15.0

        # ROC
        if roc > 0:
            score += 10.0
        else:
            score -= 10.0

        return float(np.clip(score, 0.0, 100.0))

    @staticmethod
    def calculate_volatility_score(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 50.0

        latest = df.iloc[-1]
        vol20 = float(latest.get("volatility_20d", 0.15))
        vol252 = float(latest.get("volatility_252d", 0.15))

        score = 50.0
        if vol20 < vol252:
            score += 20.0 # Low volatility environment is constructive
        else:
            score -= 15.0 # High volatility adds uncertainty

        return float(np.clip(score, 0.0, 100.0))

    @staticmethod
    def calculate_macro_score(macro_summary: Dict[str, Any]) -> float:
        if not macro_summary:
            return 50.0

        score = 50.0
        dxy_pct = macro_summary.get("dxy", {}).get("pct_change", 0.0)
        yield_change = macro_summary.get("us10y", {}).get("abs_change", 0.0)
        vix_level = macro_summary.get("vix", {}).get("latest", 15.0)

        # Dollar weakness is bullish for metals
        if dxy_pct < -0.3:
            score += 15.0
        elif dxy_pct > 0.3:
            score -= 15.0

        # Falling yields bullish
        if yield_change < -0.03:
            score += 15.0
        elif yield_change > 0.03:
            score -= 15.0

        # Elevated VIX increases safe-haven demand
        if vix_level > 20.0:
            score += 10.0

        return float(np.clip(score, 0.0, 100.0))

    @staticmethod
    def calculate_composite_score(
        df: pd.DataFrame,
        macro_summary: Dict[str, Any],
        ratio_summary: Dict[str, Any] = None,
        weights: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        if weights is None:
            weights = {
                "technical": 0.25,
                "macro": 0.20,
                "momentum": 0.15,
                "volatility": 0.10,
                "relative_value": 0.15,
                "historical_pattern": 0.15,
            }

        tech_score = ScoreCalculator.calculate_technical_score(df)
        mom_score = ScoreCalculator.calculate_momentum_score(df)
        vol_score = ScoreCalculator.calculate_volatility_score(df)
        macro_score = ScoreCalculator.calculate_macro_score(macro_summary)

        # Relative Value score
        rv_score = 50.0
        if ratio_summary and "zscore" in ratio_summary:
            zs = ratio_summary["zscore"]
            if zs > 1.5:
                rv_score = 75.0 # Favors mean reversion
            elif zs < -1.5:
                rv_score = 30.0

        pattern_score = (tech_score + mom_score) / 2.0

        composite = (
            tech_score * weights["technical"]
            + macro_score * weights["macro"]
            + mom_score * weights["momentum"]
            + vol_score * weights["volatility"]
            + rv_score * weights["relative_value"]
            + pattern_score * weights["historical_pattern"]
        )

        return {
            "composite_score": round(float(composite), 2),
            "components": {
                "technical_score": round(tech_score, 2),
                "macro_score": round(macro_score, 2),
                "momentum_score": round(mom_score, 2),
                "volatility_score": round(vol_score, 2),
                "relative_value_score": round(rv_score, 2),
                "historical_pattern_score": round(pattern_score, 2),
            },
            "weights": weights,
        }
