import pandas as pd
import numpy as np

class MacroRegimeClassifier:
    """Transparent rule-based composite macro regime classifier with explainable component breakdown."""

    def classify_regime(self, macro_snapshot: dict) -> dict:
        """
        Classifies current macro environment based on snapshot indicators:
        - dxy_return_20d
        - us10y_change_20d
        - real_yield_change_20d
        - vix_level
        - oil_return_20d
        - sp500_return_20d
        - inflation_trend (e.g., 'rising', 'falling', 'stable')
        """
        dxy_ret = macro_snapshot.get("dxy_return_20d", 0.0) or 0.0
        yield_change = macro_snapshot.get("us10y_change_20d", 0.0) or 0.0
        real_yield_change = macro_snapshot.get("real_yield_change_20d", 0.0) or 0.0
        vix = macro_snapshot.get("vix_level", 18.0) or 18.0
        oil_ret = macro_snapshot.get("oil_return_20d", 0.0) or 0.0
        sp500_ret = macro_snapshot.get("sp500_return_20d", 0.0) or 0.0
        inflation_trend = macro_snapshot.get("inflation_trend", "stable")

        reasons = []

        # Evaluate risk stance
        if vix > 25.0 or sp500_ret < -0.05:
            risk_regime = "Risk-Off"
            reasons.append(f"High VIX ({vix:.1f}) or negative equity returns ({sp500_ret*100:.1f}%)")
        elif vix < 16.0 and sp500_ret > 0.02:
            risk_regime = "Risk-On"
            reasons.append(f"Low VIX ({vix:.1f}) and strong equity gains ({sp500_ret*100:.1f}%)")
        else:
            risk_regime = "Neutral Risk"

        # Evaluate rate/monetary stance
        if real_yield_change > 0.20 or yield_change > 0.25:
            monetary_regime = "Tightening"
            reasons.append("Rising Treasury / Real yields")
        elif real_yield_change < -0.20 or yield_change < -0.25:
            monetary_regime = "Easing"
            reasons.append("Falling Treasury / Real yields")
        else:
            monetary_regime = "Neutral Monetary"

        # Composite primary regime
        if oil_ret > 0.08 and sp500_ret < -0.02 and inflation_trend in ["rising", "high"]:
            primary_regime = "Stagflationary"
            reasons.append("Rising oil/inflation combined with equity weakness")
        elif oil_ret > 0.05 or inflation_trend == "rising":
            primary_regime = "Inflationary"
            reasons.append("Strong energy returns / rising inflation expectations")
        elif risk_regime == "Risk-Off":
            primary_regime = "Risk-Off"
        elif risk_regime == "Risk-On" and monetary_regime == "Easing":
            primary_regime = "Risk-On Easing"
        elif monetary_regime == "Tightening" and dxy_ret > 0.02:
            primary_regime = "Tightening USD Bull"
        elif monetary_regime == "Easing" and dxy_ret < -0.02:
            primary_regime = "Disinflationary Easing"
        else:
            primary_regime = "Neutral"

        return {
            "primary_regime": primary_regime,
            "risk_regime": risk_regime,
            "monetary_regime": monetary_regime,
            "explainability_reasons": reasons,
            "components": {
                "dxy_return_20d": round(dxy_ret, 4),
                "us10y_change_20d": round(yield_change, 4),
                "vix_level": round(vix, 2),
                "oil_return_20d": round(oil_ret, 4),
                "sp500_return_20d": round(sp500_ret, 4)
            }
        }
