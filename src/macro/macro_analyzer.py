import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MacroAnalyzer:
    """
    Analyzes macroeconomic drivers: US Dollar (DXY), Treasury yields (10Y, 2Y),
    Inflation, Energy (Oil), Equities (S&P 500, NIFTY 50), Volatility (VIX), and USD/INR.
    Determines Macro Regime (Inflationary, Disinflationary, Risk-On, Risk-Off, Tightening, Easing, Stagflationary, Neutral).
    """

    @staticmethod
    def calculate_macro_metrics(macro_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Extracts key level, daily change, and rolling return metrics for macro series.
        Expects keys like 'DXY', 'US10Y', 'US2Y', 'OIL_WTI', 'SP500', 'NIFTY50', 'VIX', 'USDINR'.
        """
        metrics = {}
        for name, df in macro_dfs.items():
            if df.empty or "Close" not in df.columns:
                continue
            close = df["Close"]
            latest_val = close.iloc[-1]
            prev_val = close.iloc[-2] if len(close) > 1 else latest_val
            ret_1d = (latest_val - prev_val) / prev_val if prev_val != 0 else 0.0
            ret_20d = (latest_val - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0.0

            metrics[name] = {
                "latest": float(latest_val),
                "change_1d": float(latest_val - prev_val),
                "return_1d_pct": float(ret_1d * 100),
                "return_20d_pct": float(ret_20d * 100)
            }
        return metrics

    @staticmethod
    def classify_macro_regime(macro_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based, transparent, multi-factor macro regime classification.
        """
        dxy_ret = macro_metrics.get("DXY", {}).get("return_20d_pct", 0.0)
        us10y_change = macro_metrics.get("US10Y", {}).get("change_1d", 0.0)
        oil_ret = macro_metrics.get("OIL_WTI", {}).get("return_20d_pct", 0.0)
        sp500_ret = macro_metrics.get("SP500", {}).get("return_20d_pct", 0.0)
        vix_val = macro_metrics.get("VIX", {}).get("latest", 18.0)

        regime_components = []

        # Inflationary vs Deflationary / Disinflationary
        if oil_ret > 5.0 and us10y_change > 0:
            regime_components.append("Inflationary Pressure")
            primary_regime = "Inflationary"
        elif oil_ret < -5.0 and sp500_ret < -3.0:
            regime_components.append("Deflationary Drag")
            primary_regime = "Deflationary"
        else:
            primary_regime = "Disinflationary"

        # Risk Stance
        if vix_val > 22.0 or sp500_ret < -4.0:
            risk_regime = "Risk-Off"
        elif vix_val < 16.0 and sp500_ret > 1.0:
            risk_regime = "Risk-On"
        else:
            risk_regime = "Neutral Risk"

        # Monetary Stance
        if dxy_ret > 2.0 and us10y_change > 0.05:
            monetary_regime = "Tightening"
        elif dxy_ret < -2.0 and us10y_change < -0.05:
            monetary_regime = "Easing"
        else:
            monetary_regime = "Neutral Policy"

        # Stagflationary combo check
        if oil_ret > 5.0 and sp500_ret < -3.0:
            primary_regime = "Stagflationary"

        return {
            "primary_macro_regime": primary_regime,
            "risk_regime": risk_regime,
            "monetary_regime": monetary_regime,
            "composite_summary": f"{primary_regime} | {risk_regime} | {monetary_regime}",
            "dxy_trend_20d": "Strong USD" if dxy_ret > 1.5 else ("Weak USD" if dxy_ret < -1.5 else "Stable USD"),
            "vix_level": vix_val,
            "components": regime_components
        }
