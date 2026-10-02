import logging
import numpy as np
import pandas as pd
from src.models.baseline_models import BaselineForecastingModels

logger = logging.getLogger(__name__)

class ForwardProbabilityEngine:
    """Generates probabilistic research signals across multiple horizons (1D, 3D, 5D, 10D, 20D, 60D) with confidence scoring."""

    def __init__(self, horizons: list[int] = None):
        self.horizons = horizons or [1, 3, 5, 10, 20, 60]
        self.models = BaselineForecastingModels()

    def calculate_confidence_score(
        self,
        sample_size: int,
        agreement_str: str,
        vol_regime: str,
        score_composite: float
    ) -> dict:
        """
        Calculates transparency-backed confidence score (0-100) and rationale.
        Confidence depends on sample size, model agreement, composite alignment, and volatility regime.
        """
        conf = 50.0
        reasons = []

        # 1. Sample Size factor
        if sample_size >= 200:
            conf += 15.0
            reasons.append(f"Large historical sample size ({sample_size} records)")
        elif sample_size >= 50:
            conf += 5.0
            reasons.append(f"Moderate historical sample size ({sample_size} records)")
        else:
            conf -= 10.0
            reasons.append(f"Small historical sample size ({sample_size} records)")

        # 2. Model Agreement factor
        if agreement_str in ["3/3", "0/3"]:
            conf += 15.0
            reasons.append("Unanimous model agreement")
        else:
            conf += 5.0
            reasons.append(f"Partial model agreement ({agreement_str})")

        # 3. Score alignment factor
        if abs(score_composite - 50.0) > 15.0:
            conf += 10.0
            reasons.append("Strong multi-factor score alignment")

        # 4. Volatility penalty
        if vol_regime == "High Volatility":
            conf -= 10.0
            reasons.append("High volatility regime penalty")

        final_conf = float(np.clip(conf, 10, 95))

        return {
            "confidence_score": round(final_conf, 1),
            "confidence_reasons": reasons
        }

    def generate_horizon_forecasts(
        self,
        symbol: str,
        feature_df: pd.DataFrame,
        composite_score: float,
        vol_regime: str
    ) -> dict:
        """
        Generates probabilistic forecast signals for 1D, 3D, 5D, 10D, 20D, 60D horizons.
        Explicitly states these are probabilistic research signals, not guaranteed future prices.
        """
        results = {}

        for h in self.horizons:
            mf = self.models.train_predict_walk_forward(feature_df, horizon_days=h)

            dir_prob = mf.get("direction_probability", 0.50)
            exp_ret = mf.get("expected_return_pct", 0.0)
            sample_size = mf.get("sample_size", 0)
            agreement = mf.get("model_agreement", "1/1")

            conf_dict = self.calculate_confidence_score(sample_size, agreement, vol_regime, composite_score)

            results[f"{h}d"] = {
                "horizon_days": h,
                "positive_return_probability": round(dir_prob, 2),
                "negative_return_probability": round(1.0 - dir_prob, 2),
                "expected_return_pct": exp_ret,
                "research_signal": "Bullish Signal" if dir_prob > 0.55 else ("Bearish Signal" if dir_prob < 0.45 else "Neutral Signal"),
                "model_agreement": agreement,
                "confidence_score": conf_dict["confidence_score"],
                "confidence_reasons": conf_dict["confidence_reasons"]
            }

        return {
            "symbol": symbol,
            "disclaimer": "All predictions are probabilistic research signals derived from historical patterns, not guaranteed future prices.",
            "forecasts": results
        }
