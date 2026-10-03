import pandas as pd
import numpy as np
from typing import Dict, Any, List


class MacroRegimeModel:
    """
    Transparent, rule-based composite macro regime model incorporating rates,
    DXY, inflation expectations, volatility, commodities, and equities.
    """

    @staticmethod
    def classify_macro_regime(macro_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify global macro regime into:
        - Inflationary
        - Disinflationary
        - Deflationary
        - Risk-on
        - Risk-off
        - Tightening
        - Easing
        - Stagflationary
        - Neutral

        Returns final primary regime along with component factors for full explainability.
        """
        dxy = macro_summary.get("dxy", {})
        vix = macro_summary.get("vix", {})
        sp500 = macro_summary.get("sp500", {})
        oil = macro_summary.get("oil", {})
        us10y = macro_summary.get("us10y", {})

        dxy_change = dxy.get("pct_change", 0.0)
        vix_level = vix.get("latest", 15.0) if vix.get("latest") is not None else 15.0
        vix_change = vix.get("pct_change", 0.0)
        sp500_change = sp500.get("pct_change", 0.0)
        oil_change = oil.get("pct_change", 0.0)
        yield_change = us10y.get("abs_change", 0.0)

        # Component scores
        risk_off_score = 0
        if vix_level > 22 or vix_change > 10.0:
            risk_off_score += 2
        if sp500_change < -1.0:
            risk_off_score += 1
        if dxy_change > 0.5:
            risk_off_score += 1

        risk_on_score = 0
        if vix_level < 16 and sp500_change > 0.5:
            risk_on_score += 2
        if dxy_change < -0.3:
            risk_on_score += 1

        inflation_score = 0
        if oil_change > 1.5:
            inflation_score += 2
        if yield_change > 0.05:
            inflation_score += 1

        stagflation_score = 0
        if oil_change > 1.0 and sp500_change < -0.5 and yield_change > 0.03:
            stagflation_score += 3

        # Primary regime assignment
        if stagflation_score >= 3:
            primary_regime = "Stagflationary"
        elif risk_off_score >= 3:
            primary_regime = "Risk-Off"
        elif risk_on_score >= 2:
            primary_regime = "Risk-On"
        elif inflation_score >= 2:
            primary_regime = "Inflationary"
        elif yield_change > 0.05 and dxy_change > 0.3:
            primary_regime = "Tightening"
        elif yield_change < -0.05 and dxy_change < -0.3:
            primary_regime = "Easing"
        elif oil_change < -1.5 and yield_change < -0.03:
            primary_regime = "Disinflationary"
        else:
            primary_regime = "Neutral"

        return {
            "primary_regime": primary_regime,
            "components": {
                "risk_off_score": risk_off_score,
                "risk_on_score": risk_on_score,
                "inflation_score": inflation_score,
                "stagflation_score": stagflation_score,
                "vix_level": vix_level,
                "sp500_daily_pct": sp500_change,
                "oil_daily_pct": oil_change,
                "dxy_daily_pct": dxy_change,
                "yield_10y_abs_change": yield_change,
            },
        }
