import pandas as pd
import numpy as np
from typing import Dict, List, Any


class MacroRegimeDetector:
    """Transparent rule-based composite macro regime detector."""

    @staticmethod
    def detect_regime(
        macro_data: Dict[str, pd.DataFrame],
        date_str: str = None
    ) -> Dict[str, Any]:
        """Classify macro regime based on rates, inflation, equities, oil, DXY, and VIX."""

        components = {}

        # 1. Yields & Rates Stance
        us10y_df = macro_data.get("US10Y")
        us2y_df = macro_data.get("US2Y")
        rate_trend = "NEUTRAL"
        if us10y_df is not None and not us10y_df.empty:
            last_10y = us10y_df.iloc[-1]
            chg_20d = float(last_10y.get("return_20d", 0.0) or 0.0)
            if chg_20d > 0.05:
                rate_trend = "TIGHTENING"
            elif chg_20d < -0.05:
                rate_trend = "EASING"
            components["us10y_20d_change"] = round(chg_20d * 100, 2)

        # 2. Dollar / DXY Stance
        dxy_df = macro_data.get("DXY")
        dxy_trend = "NEUTRAL"
        if dxy_df is not None and not dxy_df.empty:
            dxy_last = dxy_df.iloc[-1]
            dxy_chg = float(dxy_last.get("return_20d", 0.0) or 0.0)
            if dxy_chg > 0.015:
                dxy_trend = "STRONG_USD"
            elif dxy_chg < -0.015:
                dxy_trend = "WEAK_USD"
            components["dxy_20d_change_pct"] = round(dxy_chg * 100, 2)

        # 3. Equities & Volatility / Risk Sentiment
        sp_df = macro_data.get("SP500")
        vix_df = macro_data.get("VIX")
        risk_sentiment = "NEUTRAL"
        vix_val = 15.0
        if vix_df is not None and not vix_df.empty:
            vix_val = float(vix_df.iloc[-1]["close"])
            components["vix"] = round(vix_val, 2)

        sp_ret_20d = 0.0
        if sp_df is not None and not sp_df.empty:
            sp_ret_20d = float(sp_df.iloc[-1].get("return_20d", 0.0) or 0.0)
            components["sp500_20d_return_pct"] = round(sp_ret_20d * 100, 2)

        if vix_val > 22.0 or sp_ret_20d < -0.03:
            risk_sentiment = "RISK_OFF"
        elif vix_val < 16.0 and sp_ret_20d > 0.01:
            risk_sentiment = "RISK_ON"

        # 4. Commodities / Inflation Stance
        oil_df = macro_data.get("CRUDE_OIL")
        oil_ret_20d = 0.0
        if oil_df is not None and not oil_df.empty:
            oil_ret_20d = float(oil_df.iloc[-1].get("return_20d", 0.0) or 0.0)
            components["crude_oil_20d_return_pct"] = round(oil_ret_20d * 100, 2)

        inflation_stance = "NEUTRAL"
        if oil_ret_20d > 0.05 and dxy_trend != "STRONG_USD":
            inflation_stance = "INFLATIONARY"
        elif oil_ret_20d < -0.05:
            inflation_stance = "DISINFLATIONARY"

        # Composite Regime Logic
        if risk_sentiment == "RISK_OFF":
            primary_regime = "Risk-off"
        elif inflation_stance == "INFLATIONARY" and rate_trend == "TIGHTENING" and sp_ret_20d < 0:
            primary_regime = "Stagflationary"
        elif inflation_stance == "INFLATIONARY":
            primary_regime = "Inflationary"
        elif inflation_stance == "DISINFLATIONARY":
            primary_regime = "Disinflationary"
        elif risk_sentiment == "RISK_ON":
            primary_regime = "Risk-on"
        elif rate_trend == "TIGHTENING":
            primary_regime = "Tightening"
        elif rate_trend == "EASING":
            primary_regime = "Easing"
        else:
            primary_regime = "Neutral"

        return {
            "primary_regime": primary_regime,
            "risk_sentiment": risk_sentiment,
            "rate_stance": rate_trend,
            "dxy_stance": dxy_trend,
            "inflation_stance": inflation_stance,
            "components": components,
            "explanation": f"Macro state: {primary_regime} driven by Risk={risk_sentiment}, Rates={rate_trend}, Dollar={dxy_trend}, Inflation={inflation_stance}."
        }
