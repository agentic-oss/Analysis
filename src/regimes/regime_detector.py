import pandas as pd
import numpy as np
from typing import Dict, Any

class MacroRegimeDetector:
    """Classifies rule-based macro regime based on composite indicators."""

    @staticmethod
    def classify_macro_regime(row: pd.Series) -> Dict[str, Any]:
        """
        Classifies macro regime into:
        Inflationary, Disinflationary, Risk-On, Risk-Off, Tightening, Easing, Stagflationary, Neutral
        """
        vix = row.get("VIX_close", np.nan)
        dxy_ret = row.get("DXY_close_ret_20d", 0.0)
        yield_chg = row.get("US10Y_close_chg_20d", 0.0)
        oil_ret = row.get("CRUDE_OIL_close_ret_20d", 0.0)
        sp500_ret = row.get("SP500_close_ret_20d", 0.0)

        # Volatility check
        is_risk_off = vix > 22.0 if not pd.isna(vix) else False
        is_inflationary = oil_ret > 0.05 and yield_chg > 0.10
        is_stagflation = is_inflationary and sp500_ret < -0.03

        if is_stagflation:
            regime = "Stagflationary"
        elif is_risk_off or sp500_ret < -0.05:
            regime = "Risk-Off"
        elif sp500_ret > 0.03 and vix < 16.0:
            regime = "Risk-On"
        elif yield_chg > 0.15 or dxy_ret > 0.03:
            regime = "Tightening"
        elif yield_chg < -0.15 or dxy_ret < -0.03:
            regime = "Easing"
        elif oil_ret > 0.04:
            regime = "Inflationary"
        elif oil_ret < -0.04:
            regime = "Disinflationary"
        else:
            regime = "Neutral"

        return {
            "macro_regime": regime,
            "components": {
                "vix": vix,
                "dxy_ret_20d": dxy_ret,
                "yield_chg_20d": yield_chg,
                "oil_ret_20d": oil_ret,
                "sp500_ret_20d": sp500_ret
            }
        }

class MarketRegimeDetector:
    """Classifies market regime for gold and silver (Trend and Volatility)."""

    @staticmethod
    def classify_market_regime(df: pd.DataFrame, prefix: str = "gold_") -> pd.DataFrame:
        df = df.copy()

        close_col = f"{prefix.upper()}close" if f"{prefix.upper()}close" in df.columns else ("GOLD_close" if "GOLD_close" in df.columns else "SILVER_close")
        sma_20_col = f"{prefix}sma_20"
        sma_50_col = f"{prefix}sma_50"
        sma_200_col = f"{prefix}sma_200"
        hist_vol_col = f"{prefix}historical_vol_20"

        trend_list = []
        vol_list = []

        for i, row in df.iterrows():
            c = row.get(close_col, np.nan)
            s20 = row.get(sma_20_col, np.nan)
            s50 = row.get(sma_50_col, np.nan)
            s200 = row.get(sma_200_col, np.nan)
            vol = row.get(hist_vol_col, np.nan)

            if pd.isna(c) or pd.isna(s200):
                trend = "Neutral"
            elif c > s20 and s20 > s50 and s50 > s200:
                trend = "Strong Bullish Trend"
            elif c > s200:
                trend = "Bullish Trend"
            elif c < s20 and s20 < s50 and s50 < s200:
                trend = "Strong Bearish Trend"
            elif c < s200:
                trend = "Bearish Trend"
            else:
                trend = "Range"

            if pd.isna(vol):
                vol_regime = "Normal Volatility"
            elif vol > 0.25:
                vol_regime = "High Volatility"
            elif vol < 0.12:
                vol_regime = "Low Volatility"
            else:
                vol_regime = "Normal Volatility"

            trend_list.append(trend)
            vol_list.append(vol_regime)

        df[f"{prefix}trend_regime"] = trend_list
        df[f"{prefix}volatility_regime"] = vol_list
        return df
