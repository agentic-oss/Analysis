import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MacroRegimeDetector:
    """Rule-based composite framework to classify macro regimes (Inflationary, Disinflationary, Deflationary, Risk-on, Risk-off, Tightening, Easing, Stagflationary, Neutral)."""

    @staticmethod
    def classify_regime(macro_data: Dict[str, pd.DataFrame], target_date: str) -> Dict[str, Any]:
        components = {
            "interest_rates_trend": "neutral",
            "real_yields_trend": "neutral",
            "dxy_trend": "neutral",
            "oil_trend": "neutral",
            "vix_level": "normal",
            "equity_trend": "neutral"
        }

        vix_df = macro_data.get("vix")
        if vix_df is not None and not vix_df.empty:
            v_sub = vix_df[vix_df["date"] <= target_date]
            if not v_sub.empty:
                v_val = v_sub.iloc[-1]["close"]
                if v_val > 25.0:
                    components["vix_level"] = "high"
                elif v_val < 15.0:
                    components["vix_level"] = "low"

        oil_df = macro_data.get("oil_wti")
        if oil_df is not None and not oil_df.empty:
            o_sub = oil_df[oil_df["date"] <= target_date]
            if len(o_sub) >= 20:
                ret_20 = (o_sub.iloc[-1]["close"] - o_sub.iloc[-20]["close"]) / o_sub.iloc[-20]["close"]
                if ret_20 > 0.05:
                    components["oil_trend"] = "up"
                elif ret_20 < -0.05:
                    components["oil_trend"] = "down"

        rates_df = macro_data.get("us10y")
        if rates_df is not None and not rates_df.empty:
            r_sub = rates_df[rates_df["date"] <= target_date]
            if len(r_sub) >= 20:
                ret_20 = r_sub.iloc[-1]["close"] - r_sub.iloc[-20]["close"]
                if ret_20 > 0.20:
                    components["interest_rates_trend"] = "rising"
                elif ret_20 < -0.20:
                    components["interest_rates_trend"] = "falling"

        sp_df = macro_data.get("sp500")
        if sp_df is not None and not sp_df.empty:
            s_sub = sp_df[sp_df["date"] <= target_date]
            if len(s_sub) >= 20:
                ret_20 = (s_sub.iloc[-1]["close"] - s_sub.iloc[-20]["close"]) / s_sub.iloc[-20]["close"]
                if ret_20 > 0.02:
                    components["equity_trend"] = "up"
                elif ret_20 < -0.02:
                    components["equity_trend"] = "down"

        # Classification logic
        regimes = []
        if components["vix_level"] == "high" or components["equity_trend"] == "down":
            regimes.append("Risk-Off")
        elif components["equity_trend"] == "up" and components["vix_level"] == "low":
            regimes.append("Risk-On")

        if components["oil_trend"] == "up" and components["interest_rates_trend"] == "rising":
            regimes.append("Inflationary")
        elif components["oil_trend"] == "down" and components["interest_rates_trend"] == "falling":
            regimes.append("Disinflationary")

        if components["interest_rates_trend"] == "rising":
            regimes.append("Tightening")
        elif components["interest_rates_trend"] == "falling":
            regimes.append("Easing")

        if components["oil_trend"] == "up" and components["equity_trend"] == "down":
            regimes.append("Stagflationary")

        primary_regime = regimes[0] if regimes else "Neutral"

        return {
            "date": target_date,
            "primary_regime": primary_regime,
            "all_active_regimes": regimes if regimes else ["Neutral"],
            "explainable_components": components
        }


class MarketRegimeDetector:
    """Classifies Gold and Silver trend regime (Strong Bullish, Bullish, Range, Bearish, Strong Bearish) and volatility regime (High, Low)."""

    @staticmethod
    def classify_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"trend_regime": "Unknown", "volatility_regime": "Unknown"}

        last = df.iloc[-1]
        price = last["close"]
        sma_20 = last.get("sma_20", price)
        sma_50 = last.get("sma_50", price)
        sma_200 = last.get("sma_200", price)
        rsi = last.get("rsi_14", 50.0)
        vol_20d = last.get("volatility_20d", 0.15)

        # Trend determination
        if price > sma_20 > sma_50 > sma_200 and rsi > 55:
            trend = "Strong Bullish Trend"
        elif price > sma_50 and price > sma_200:
            trend = "Bullish Trend"
        elif price < sma_20 < sma_50 < sma_200 and rsi < 45:
            trend = "Strong Bearish Trend"
        elif price < sma_50 and price < sma_200:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility determination
        hist_vols = df["volatility_20d"].dropna() if "volatility_20d" in df.columns else pd.Series([vol_20d])
        vol_median = hist_vols.median() if not hist_vols.empty else 0.15
        volatility = "High Volatility" if vol_20d > vol_median * 1.25 else ("Low Volatility" if vol_20d < vol_median * 0.8 else "Normal Volatility")

        return {
            "date": str(last["date"]),
            "trend_regime": trend,
            "volatility_regime": volatility,
            "rsi_level": float(rsi),
            "volatility_20d": float(vol_20d)
        }
