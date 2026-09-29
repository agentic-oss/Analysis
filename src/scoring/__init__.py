import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_transparent_scores(
    tech_df: pd.DataFrame,
    macro_regime: str,
    ratio_stats: Dict[str, Any],
    analogue_stats: Dict[str, Any],
    model_preds: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculates transparent scores from 0 to 100 for each driver component
    and computes a weighted composite score.
    """
    if tech_df is None or tech_df.empty:
        return {}

    latest = tech_df.iloc[-1]

    # 1. Technical Score (0-100)
    rsi = float(latest.get("rsi_14", 50.0))
    dist_200 = float(latest.get("dist_sma_200", 0.0))

    # RSI score: neutral 50, >70 overbought, <30 oversold
    rsi_score = max(0.0, min(100.0, rsi))
    tech_score = max(0.0, min(100.0, 50.0 + (dist_200 * 2.5)))

    # 2. Macro Score (0-100)
    macro_map = {
        "Inflationary": 80.0,
        "Risk-Off": 75.0,
        "Easing": 70.0,
        "Neutral": 50.0,
        "Disinflationary": 40.0,
        "Tightening": 30.0
    }
    macro_score = macro_map.get(macro_regime, 50.0)

    # 3. Momentum Score (0-100)
    macd_hist = float(latest.get("macd_hist", 0.0))
    roc_12 = float(latest.get("roc_12", 0.0))
    momentum_score = max(0.0, min(100.0, 50.0 + (roc_12 * 5.0) + (macd_hist * 2.0)))

    # 4. Volatility Score (0-100)
    vol_20d = float(latest.get("volatility_20d", 15.0))
    vol_score = max(0.0, min(100.0, 100.0 - (vol_20d * 2.5)))

    # 5. Relative Value Score (0-100)
    z_score = ratio_stats.get("z_score", 0.0)
    # High gold/silver ratio z-score implies silver is cheap relative to gold
    rel_value_score = max(0.0, min(100.0, 50.0 + (z_score * 20.0)))

    # 6. Historical Pattern Score (0-100)
    fwd_10d_stat = analogue_stats.get("horizon_statistics", {}).get("fwd_10d", {})
    pos_prob = fwd_10d_stat.get("positive_prob", 0.5)
    pattern_score = pos_prob * 100.0

    # Composite Score
    weights = {
        "technical": 0.25,
        "macro": 0.20,
        "momentum": 0.15,
        "volatility": 0.10,
        "relative_value": 0.15,
        "historical_pattern": 0.15
    }

    composite = (
        tech_score * weights["technical"] +
        macro_score * weights["macro"] +
        momentum_score * weights["momentum"] +
        vol_score * weights["volatility"] +
        rel_value_score * weights["relative_value"] +
        pattern_score * weights["historical_pattern"]
    )

    return {
        "technical_score": round(tech_score, 2),
        "macro_score": round(macro_score, 2),
        "momentum_score": round(momentum_score, 2),
        "volatility_score": round(vol_score, 2),
        "relative_value_score": round(rel_value_score, 2),
        "historical_pattern_score": round(pattern_score, 2),
        "composite_score": round(composite, 2),
        "weights": weights
    }
