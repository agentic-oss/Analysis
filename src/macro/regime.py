"""Transparent rule-based multi-factor Macro Regime Classifier."""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MacroRegimeClassifier:
    """
    Classifies macro regime into:
    Inflationary, Disinflationary, Deflationary, Risk-On, Risk-Off, Tightening, Easing, Stagflationary, Neutral.
    Uses multi-factor composite scores incorporating Yields, DXY, VIX, Oil, and Equities.
    """

    @staticmethod
    def classify(macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Classifies current macro regime based on asset return trends and indicators."""
        # Helper to get latest 20d return
        def get_20d_return(df: pd.DataFrame) -> float:
            if df.empty or 'close' not in df.columns or len(df) < 5:
                return 0.0
            closes = df['close'].dropna()
            if len(closes) < 2:
                return 0.0
            start_p = closes.iloc[-min(len(closes), 20)]
            end_p = closes.iloc[-1]
            return float((end_p - start_p) / start_p) if start_p else 0.0

        # Helper to get latest level
        def get_latest_level(df: pd.DataFrame) -> float:
            if df.empty or 'close' not in df.columns:
                return 0.0
            return float(df['close'].dropna().iloc[-1]) if not df['close'].dropna().empty else 0.0

        dxy_ret = get_20d_return(macro_data.get("DXY", pd.DataFrame()))
        sp500_ret = get_20d_return(macro_data.get("SP500", pd.DataFrame()))
        oil_ret = get_20d_return(macro_data.get("CRUDE_OIL", pd.DataFrame()))
        vix_level = get_latest_level(macro_data.get("VIX", pd.DataFrame()))
        us10y_ret = get_20d_return(macro_data.get("US10Y", pd.DataFrame()))

        factors = {
            "dxy_20d_return_pct": round(dxy_ret * 100, 2),
            "sp500_20d_return_pct": round(sp500_ret * 100, 2),
            "oil_20d_return_pct": round(oil_ret * 100, 2),
            "vix_level": round(vix_level, 2),
            "us10y_20d_change_pct": round(us10y_ret * 100, 2)
        }

        # Rule-based logic
        primary_regime = "Neutral"

        if vix_level > 25.0 or (sp500_ret < -0.05 and vix_level > 20.0):
            primary_regime = "Risk-off"
        elif oil_ret > 0.08 and us10y_ret > 0.05:
            primary_regime = "Inflationary"
        elif oil_ret > 0.08 and sp500_ret < -0.03:
            primary_regime = "Stagflationary"
        elif oil_ret < -0.08 and us10y_ret < -0.05:
            primary_regime = "Deflationary"
        elif sp500_ret > 0.03 and vix_level < 18.0:
            primary_regime = "Risk-on"
        elif us10y_ret > 0.05 and dxy_ret > 0.02:
            primary_regime = "Tightening"
        elif us10y_ret < -0.05 and dxy_ret < -0.02:
            primary_regime = "Easing"

        return {
            "primary_regime": primary_regime,
            "components": factors,
            "explanation": f"Classified as {primary_regime} based on VIX ({vix_level:.1f}), S&P 500 20d ({sp500_ret*100:.1f}%), Oil 20d ({oil_ret*100:.1f}%), and US10Y 20d ({us10y_ret*100:.1f}%)."
        }
