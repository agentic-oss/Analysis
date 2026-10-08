"""
Macro Regime Detection Engine.
Rule-based multi-factor composite macro regime classification.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class MacroRegimeDetector:
    """
    Detects macro regimes (Inflationary, Risk-On, Risk-Off, Tightening, Easing, Stagflationary, Neutral) using rates, inflation proxies, FX, energy, equities, and VIX.
    """

    @staticmethod
    def detect_macro_regime(
        macro_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Classifies current macro regime with explainable score components.
        """
        # Get recent 20-day returns/changes for key drivers if available
        metrics = {}
        for sym, df in macro_dfs.items():
            if df.empty or len(df) < 5:
                continue
            latest_val = float(df["close"].iloc[-1])
            prev_20 = float(df["close"].iloc[-20]) if len(df) >= 20 else float(df["close"].iloc[0])
            pct_change_20 = ((latest_val - prev_20) / prev_20) if prev_20 != 0 else 0.0
            diff_20 = latest_val - prev_20

            metrics[sym] = {
                "latest": latest_val,
                "pct_change_20d": pct_change_20,
                "diff_20d": diff_20,
            }

        dxy_chg = metrics.get("DXY", {}).get("pct_change_20d", 0.0)
        sp500_chg = metrics.get("SP500", {}).get("pct_change_20d", 0.0)
        vix_val = metrics.get("VIX", {}).get("latest", 18.0)
        vix_chg = metrics.get("VIX", {}).get("pct_change_20d", 0.0)
        oil_chg = metrics.get("CRUDE_OIL", {}).get("pct_change_20d", 0.0)
        us10y_chg = metrics.get("US10Y", {}).get("diff_20d", 0.0)

        # Logic rules
        if vix_val > 25.0 or (sp500_chg < -0.05 and vix_chg > 0.20):
            primary_regime = "Risk-Off"
            description = "High volatility, equity market drawdowns, and flight to safety."
        elif sp500_chg > 0.03 and vix_val < 18.0:
            primary_regime = "Risk-On"
            description = "Equity market momentum, low volatility, appetite for risk assets."
        elif oil_chg > 0.08 and us10y_chg > 0.25:
            primary_regime = "Inflationary / Rate-Tightening"
            description = "Surging energy commodities and rising benchmark bond yields."
        elif oil_chg < -0.08 and us10y_chg < -0.25:
            primary_regime = "Disinflationary / Easing"
            description = "Softening energy prices and falling bond yields."
        elif sp500_chg < -0.03 and oil_chg > 0.05:
            primary_regime = "Stagflationary"
            description = "Stagnant/falling growth equities accompanied by high energy prices."
        else:
            primary_regime = "Neutral"
            description = "Balanced cross-market environment with no extreme macro pressures."

        return {
            "primary_regime": primary_regime,
            "description": description,
            "components": {
                "dxy_20d_change_pct": dxy_chg * 100,
                "sp500_20d_change_pct": sp500_chg * 100,
                "vix_latest": vix_val,
                "crude_oil_20d_change_pct": oil_chg * 100,
                "us10y_20d_change": us10y_chg,
            },
        }
