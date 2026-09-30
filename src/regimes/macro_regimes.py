import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MacroRegimeClassifier:
    """
    Transparent, rule-based macro regime classification.
    Regimes:
    - Inflationary
    - Disinflationary
    - Deflationary
    - Risk-On
    - Risk-Off
    - Tightening
    - Easing
    - Stagflationary
    - Neutral
    """

    @staticmethod
    def classify_macro_regime(
        us10y_change_20d: float,
        dxy_return_20d: float,
        sp500_return_20d: float,
        oil_return_20d: float,
        vix_level: float,
        cpi_yoy_trend: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classifies composite macro environment with individual component explanations.
        """
        components = {
            "us10y_change_20d": us10y_change_20d,
            "dxy_return_20d": dxy_return_20d,
            "sp500_return_20d": sp500_return_20d,
            "oil_return_20d": oil_return_20d,
            "vix_level": vix_level,
        }

        # Rule evaluation
        is_risk_off = vix_level > 22.0 or sp500_return_20d < -0.04
        is_risk_on = vix_level < 16.0 and sp500_return_20d > 0.02
        is_inflationary = oil_return_20d > 0.08 or us10y_change_20d > 0.25
        is_stagflationary = is_inflationary and sp500_return_20d < -0.03
        is_tightening = us10y_change_20d > 0.15 and dxy_return_20d > 0.015
        is_easing = us10y_change_20d < -0.15 and dxy_return_20d < -0.015

        if is_stagflationary:
            primary_regime = "Stagflationary"
        elif is_risk_off:
            primary_regime = "Risk-Off"
        elif is_tightening:
            primary_regime = "Tightening"
        elif is_easing:
            primary_regime = "Easing"
        elif is_risk_on:
            primary_regime = "Risk-On"
        elif is_inflationary:
            primary_regime = "Inflationary"
        else:
            primary_regime = "Neutral"

        return {
            "primary_regime": primary_regime,
            "is_risk_off": is_risk_off,
            "is_risk_on": is_risk_on,
            "is_inflationary": is_inflationary,
            "is_tightening": is_tightening,
            "is_easing": is_easing,
            "components": components,
            "explainability": f"Macro regime identified as {primary_regime} based on VIX ({vix_level:.1f}), S&P 500 20d return ({sp500_return_20d*100:.1f}%), Oil 20d return ({oil_return_20d*100:.1f}%), and 10Y Yield 20d change ({us10y_change_20d:.2f}bp).",
        }
