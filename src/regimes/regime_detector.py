import pandas as pd
import numpy as np
from typing import Dict, Any

class RegimeDetector:
    """Classifies Macro regimes and Gold/Silver Market regimes."""

    @staticmethod
    def detect_macro_regime(macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based macro regime detection:
        Regimes: Inflationary, Disinflationary, Deflationary, Risk-on, Risk-off, Tightening, Easing, Stagflationary, Neutral
        """
        dxy_return = macro_data.get("dxy_return_20d", 0.0)
        yield_10y = macro_data.get("us10y_yield", 3.0)
        yield_change = macro_data.get("us10y_change_20d", 0.0)
        vix = macro_data.get("vix", 20.0)
        sp500_return = macro_data.get("sp500_return_20d", 0.0)
        oil_return = macro_data.get("oil_return_20d", 0.0)

        scores = {
            "Inflationary": 0,
            "Disinflationary": 0,
            "Risk-on": 0,
            "Risk-off": 0,
            "Tightening": 0,
            "Easing": 0,
            "Stagflationary": 0,
            "Neutral": 1
        }

        if vix > 25.0 or sp500_return < -3.0:
            scores["Risk-off"] += 2
        elif vix < 16.0 and sp500_return > 2.0:
            scores["Risk-on"] += 2

        if oil_return > 5.0 and sp500_return < -2.0:
            scores["Stagflationary"] += 2
            scores["Inflationary"] += 1
        elif oil_return > 5.0:
            scores["Inflationary"] += 2

        if yield_change > 0.3 or dxy_return > 2.0:
            scores["Tightening"] += 2
        elif yield_change < -0.3 or dxy_return < -2.0:
            scores["Easing"] += 2

        primary_regime = max(scores, key=scores.get)

        return {
            "primary_regime": primary_regime,
            "regime_scores": scores,
            "components": {
                "dxy_return_20d": dxy_return,
                "us10y_yield": yield_10y,
                "us10y_change_20d": yield_change,
                "vix": vix,
                "sp500_return_20d": sp500_return,
                "oil_return_20d": oil_return
            }
        }

    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects trend and volatility market regimes for an individual instrument.
        """
        if df.empty or len(df) < 20:
            return {
                "trend_regime": "Range",
                "volatility_regime": "Normal Volatility",
                "composite_label": "Range / Normal Volatility"
            }

        latest = df.iloc[-1]
        close = float(latest.get("close", 0.0))
        sma_20 = float(latest.get("sma_20", close))
        sma_50 = float(latest.get("sma_50", close))
        sma_200 = float(latest.get("sma_200", close))
        volatility = float(latest.get("volatility_20d", 15.0))
        rsi = float(latest.get("rsi_14", 50.0))

        # Trend logic
        if close > sma_20 and sma_20 > sma_50 and sma_50 > sma_200:
            trend = "Strong Bullish Trend"
        elif close > sma_50 and sma_50 > sma_200:
            trend = "Bullish Trend"
        elif close < sma_20 and sma_20 < sma_50 and sma_50 < sma_200:
            trend = "Strong Bearish Trend"
        elif close < sma_50 and sma_50 < sma_200:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility logic
        if volatility > 25.0:
            vol_regime = "High Volatility"
        elif volatility < 12.0:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"

        return {
            "trend_regime": trend,
            "volatility_regime": vol_regime,
            "composite_label": f"{trend} / {vol_regime}",
            "rsi_14": rsi,
            "volatility_20d": volatility
        }
