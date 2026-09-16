import numpy as np
import pandas as pd
from typing import Dict, Any


class TechnicalIndicators:
    """Calculates full suite of technical indicators and price structure metrics."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 5:
            return df

        res = df.copy().sort_values("date").reset_index(drop=True)

        # Moving Averages
        for w in [5, 10, 20, 50, 100, 200]:
            res[f"sma_{w}"] = res["close"].rolling(window=w, min_periods=1).mean()
        for w in [9, 20, 50, 200]:
            res[f"ema_{w}"] = res["close"].ewm(span=w, adjust=False).mean()

        # Returns & Volatility
        res["return_1d"] = res["close"].pct_change(1)
        res["return_5d"] = res["close"].pct_change(5)
        res["return_20d"] = res["close"].pct_change(20)
        res["return_60d"] = res["close"].pct_change(60)
        res["return_120d"] = res["close"].pct_change(120)
        res["return_252d"] = res["close"].pct_change(252)

        res["volatility_20d"] = res["return_1d"].rolling(20, min_periods=5).std() * np.sqrt(252)
        res["volatility_60d"] = res["return_1d"].rolling(60, min_periods=10).std() * np.sqrt(252)

        # Maximum Drawdown
        rolling_max_252 = res["close"].rolling(252, min_periods=1).max()
        res["drawdown_252d"] = (res["close"] - rolling_max_252) / rolling_max_252

        # Momentum: RSI 14
        delta = res["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        res["rsi_14"] = 100 - (100 / (1 + rs))
        res["rsi_14"] = res["rsi_14"].fillna(50)

        # Stochastic RSI
        min_rsi = res["rsi_14"].rolling(14, min_periods=1).min()
        max_rsi = res["rsi_14"].rolling(14, min_periods=1).max()
        res["stoch_rsi"] = (res["rsi_14"] - min_rsi) / (max_rsi - min_rsi + 1e-8)

        # MACD (12, 26, 9)
        ema_12 = res["close"].ewm(span=12, adjust=False).mean()
        ema_26 = res["close"].ewm(span=26, adjust=False).mean()
        res["macd"] = ema_12 - ema_26
        res["macd_signal"] = res["macd"].ewm(span=9, adjust=False).mean()
        res["macd_hist"] = res["macd"] - res["macd_signal"]

        # ROC & Momentum
        res["roc_10"] = res["close"].pct_change(10) * 100
        res["momentum_10"] = res["close"] - res["close"].shift(10)

        # Volatility: ATR 14
        high_low = res["high"] - res["low"]
        high_close = (res["high"] - res["close"].shift(1)).abs()
        low_close = (res["low"] - res["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        res["atr_14"] = tr.rolling(14, min_periods=1).mean()

        # Bollinger Bands (20, 2)
        sma_20 = res["sma_20"]
        std_20 = res["close"].rolling(20, min_periods=1).std()
        res["bollinger_upper"] = sma_20 + (2.0 * std_20)
        res["bollinger_lower"] = sma_20 - (2.0 * std_20)
        res["bollinger_width"] = (res["bollinger_upper"] - res["bollinger_lower"]) / (sma_20 + 1e-8)

        # Volume Indicators
        if "volume" in res.columns:
            res["volume_sma_20"] = res["volume"].rolling(20, min_periods=1).mean()
            res["relative_volume"] = res["volume"] / (res["volume_sma_20"] + 1e-8)
            res["volume_change"] = res["volume"].pct_change()

            # OBV
            obv = [0]
            for i in range(1, len(res)):
                if res.loc[i, "close"] > res.loc[i - 1, "close"]:
                    obv.append(obv[-1] + res.loc[i, "volume"])
                elif res.loc[i, "close"] < res.loc[i - 1, "close"]:
                    obv.append(obv[-1] - res.loc[i, "volume"])
                else:
                    obv.append(obv[-1])
            res["obv"] = obv

        # Price Structure
        res["dist_sma_20"] = (res["close"] - res["sma_20"]) / res["sma_20"]
        res["dist_sma_50"] = (res["close"] - res["sma_50"]) / res["sma_50"]
        res["dist_sma_200"] = (res["close"] - res["sma_200"]) / res["sma_200"]

        high_252 = res["high"].rolling(252, min_periods=1).max()
        low_252 = res["low"].rolling(252, min_periods=1).min()
        res["dist_52w_high"] = (res["close"] - high_252) / high_252
        res["dist_52w_low"] = (res["close"] - low_252) / low_252

        res["gap"] = (res["open"] - res["close"].shift(1)) / (res["close"].shift(1) + 1e-8)
        res["intraday_range"] = (res["high"] - res["low"]) / (res["open"] + 1e-8)

        # Signals & Patterns
        res["golden_cross"] = (res["sma_50"] > res["sma_200"]) & (res["sma_50"].shift(1) <= res["sma_200"].shift(1))
        res["death_cross"] = (res["sma_50"] < res["sma_200"]) & (res["sma_50"].shift(1) >= res["sma_200"].shift(1))

        high_20 = res["high"].rolling(20, min_periods=1).max().shift(1)
        low_20 = res["low"].rolling(20, min_periods=1).min().shift(1)
        res["breakout_20d"] = res["close"] > high_20
        res["breakdown_20d"] = res["close"] < low_20

        res["volatility_expansion"] = res["atr_14"] > (res["atr_14"].rolling(20, min_periods=1).mean() * 1.25)
        res["volatility_contraction"] = res["atr_14"] < (res["atr_14"].rolling(20, min_periods=1).mean() * 0.75)

        return res
