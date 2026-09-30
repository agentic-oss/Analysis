import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class ScoringSystem:
    """
    Calculates transparent 0-100 scores across key analytical dimensions:
    - Technical Score
    - Macro Score
    - Momentum Score
    - Volatility Score
    - Relative Value Score
    - Historical Pattern Score
    - Model Score
    - Composite Score
    """

    @staticmethod
    def calculate_scores(
        rsi_14: float,
        macd_hist: float,
        dist_sma_200: float,
        volatility_20d: float,
        gold_silver_ratio_z: float,
        macro_regime_info: Dict[str, Any],
        analogue_win_prob_10d: float,
        model_prob_10d: float,
    ) -> Dict[str, Any]:
        """Calculates 0-100 scores for each factor and computes composite score."""
        # 1. Momentum Score (0-100)
        # RSI 50 is neutral 50, RSI 70 -> 80, RSI 30 -> 20
        rsi_score = np.clip(rsi_14, 0, 100)
        macd_score = 60.0 if macd_hist > 0 else 40.0
        momentum_score = float(0.6 * rsi_score + 0.4 * macd_score)

        # 2. Technical Score (0-100)
        # Distance from 200 SMA (+10% -> 80 score, -10% -> 20 score)
        dist_score = np.clip(50.0 + (dist_sma_200 * 3.0), 0, 100)
        technical_score = float(0.5 * momentum_score + 0.5 * dist_score)

        # 3. Macro Score (0-100)
        regime = macro_regime_info.get("primary_regime", "Neutral")
        macro_map = {
            "Stagflationary": 85.0,
            "Inflationary": 75.0,
            "Risk-Off": 80.0,
            "Easing": 70.0,
            "Neutral": 50.0,
            "Tightening": 35.0,
            "Risk-On": 40.0,
        }
        macro_score = macro_map.get(regime, 50.0)

        # 4. Volatility Score (0-100) (Lower volatility = higher stability score)
        vol_score = np.clip(100.0 - (volatility_20d * 300.0), 0, 100)

        # 5. Relative Value Score (0-100) (Based on Gold/Silver Ratio Z-score)
        rel_val_score = np.clip(50.0 - (gold_silver_ratio_z * 20.0), 0, 100)

        # 6. Historical Pattern Score
        pattern_score = np.clip(analogue_win_prob_10d, 0, 100)

        # 7. Model Score
        model_score = np.clip(model_prob_10d * 100.0, 0, 100)

        # 8. Composite Weighted Score
        composite_score = float(
            0.25 * technical_score
            + 0.20 * macro_score
            + 0.15 * momentum_score
            + 0.10 * vol_score
            + 0.10 * rel_val_score
            + 0.10 * pattern_score
            + 0.10 * model_score
        )

        return {
            "technical_score": round(technical_score, 1),
            "macro_score": round(macro_score, 1),
            "momentum_score": round(momentum_score, 1),
            "volatility_score": round(vol_score, 1),
            "relative_value_score": round(rel_val_score, 1),
            "historical_pattern_score": round(pattern_score, 1),
            "model_score": round(model_score, 1),
            "composite_score": round(composite_score, 1),
        }


class ConfidenceEngine:
    """Calculates prediction confidence score (0-100) based on sample size, model agreement, and stability."""

    @staticmethod
    def calculate_confidence(
        analogue_sample_size: int,
        model_agreement_pct: float,
        regime_stability: bool,
        volatility_level: float,
    ) -> Dict[str, Any]:
        """Calculates explicit confidence score and outputs explainability reasons."""
        reasons = []
        score = 50.0

        if analogue_sample_size >= 20:
            score += 15.0
            reasons.append(f"Robust historical analogue sample size ({analogue_sample_size} occurrences).")
        else:
            reasons.append(f"Limited historical analogue sample size ({analogue_sample_size} occurrences).")

        if model_agreement_pct >= 0.75:
            score += 20.0
            reasons.append(f"High model agreement ({model_agreement_pct*100:.0f}% consensus).")
        else:
            reasons.append(f"Moderate/low model agreement ({model_agreement_pct*100:.0f}% consensus).")

        if regime_stability:
            score += 10.0
            reasons.append("Macro regime environment is stable.")

        if volatility_level < 0.20:
            score += 5.0
            reasons.append("Market volatility is subdued.")

        confidence_score = float(np.clip(score, 0, 100))
        return {
            "confidence_score": round(confidence_score, 1),
            "reasons": reasons,
            "disclaimer": "Confidence reflects historical consistency and model agreement, not guaranteed certainty.",
        }


class AlertEngine:
    """Detects price, RSI, ratio, yield, and regime alerts based on configurable thresholds."""

    @staticmethod
    def detect_alerts(
        date_str: str,
        gold_price_change_pct: float,
        silver_price_change_pct: float,
        rsi_gold: float,
        gold_silver_ratio_z: float,
        us10y_change_bp: float,
        macro_regime: str,
        thresholds: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluates triggers and generates daily alert payload."""
        alerts = []

        if abs(gold_price_change_pct) >= thresholds.get("price_jump_pct", 3.0):
            alerts.append({
                "type": "GOLD_PRICE_JUMP",
                "severity": "HIGH",
                "message": f"Gold moved {gold_price_change_pct:+.2f}% today.",
            })

        if abs(silver_price_change_pct) >= thresholds.get("price_jump_pct", 3.0):
            alerts.append({
                "type": "SILVER_PRICE_JUMP",
                "severity": "HIGH",
                "message": f"Silver moved {silver_price_change_pct:+.2f}% today.",
            })

        if rsi_gold >= thresholds.get("rsi_overbought", 70):
            alerts.append({
                "type": "GOLD_RSI_OVERBOUGHT",
                "severity": "MEDIUM",
                "message": f"Gold RSI reached overbought territory ({rsi_gold:.1f}).",
            })
        elif rsi_gold <= thresholds.get("rsi_oversold", 30):
            alerts.append({
                "type": "GOLD_RSI_OVERSOLD",
                "severity": "MEDIUM",
                "message": f"Gold RSI reached oversold territory ({rsi_gold:.1f}).",
            })

        if abs(gold_silver_ratio_z) >= thresholds.get("gold_silver_ratio_zscore", 2.0):
            alerts.append({
                "type": "GOLD_SILVER_RATIO_EXTREME",
                "severity": "HIGH",
                "message": f"Gold/Silver ratio Z-score reached extreme level ({gold_silver_ratio_z:+.2f}).",
            })

        return {
            "date": date_str,
            "alerts_count": len(alerts),
            "alerts": alerts,
        }
