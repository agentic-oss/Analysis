"""
Technical Indicators Engine.
Calculates Moving Averages, Momentum, Volatility, Volume, and Price Structure.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


class TechnicalIndicators:
    """
    Computes technical indicators on daily OHLCV DataFrames.
    """

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 2:
            return df

        df = df.copy()
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_momentum_indicators(df)
        df = TechnicalIndicators.add_volatility_indicators(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        df = TechnicalIndicators.add_price_structure(df)
        return df

    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        for p in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{p}"] = df["close"].rolling(window=p, min_periods=1).mean()

        for p in [9, 20, 50, 200]:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False, min_periods=1).mean()

        return df

    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        # RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()

        rsi_vals = []
        g_vals = gain.values
        l_vals = loss.values
        for g, l in zip(g_vals, l_vals):
            if pd.isna(g) or pd.isna(l):
                rsi_vals.append(50.0)
            elif l == 0:
                rsi_vals.append(100.0 if g > 0 else 50.0)
            else:
                rs = g / l
                rsi_vals.append(100.0 - (100.0 / (1.0 + rs)))

        df["rsi_14"] = rsi_vals

        # Stochastic RSI
        rsi = df["rsi_14"]
        rsi_min = rsi.rolling(window=14, min_periods=1).min()
        rsi_max = rsi.rolling(window=14, min_periods=1).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        df["stoch_rsi"] = stoch_rsi.fillna(0.5)

        # MACD (12, 26, 9)
        ema12 = df["close"].ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = df["close"].ewm(span=26, adjust=False, min_periods=1).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ROC 10 & Momentum
        df["roc_10"] = df["close"].pct_change(10) * 100.0
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        return df

    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        # ATR 14
        high_low = df["high"] - df["low"]
        high_close_prev = (df["high"] - df["close"].shift(1)).abs()
        low_close_prev = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

        # Historical Volatility (20-day annualized)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20d"] = log_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # Bollinger Bands (20, 2)
        sma20 = df["close"].rolling(window=20, min_periods=1).mean()
        std20 = df["close"].rolling(window=20, min_periods=1).std().fillna(0)
        df["bollinger_upper"] = sma20 + (std20 * 2.0)
        df["bollinger_lower"] = sma20 - (std20 * 2.0)
        df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / sma20.replace(0, np.nan)

        return df

    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df["volume_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["relative_volume"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
        df["relative_volume"] = df["relative_volume"].fillna(1.0)

        # On-Balance Volume (OBV)
        obv = [0]
        close_vals = df["close"].values
        vol_vals = df["volume"].values
        for i in range(1, len(df)):
            if close_vals[i] > close_vals[i - 1]:
                obv.append(obv[-1] + vol_vals[i])
            elif close_vals[i] < close_vals[i - 1]:
                obv.append(obv[-1] - vol_vals[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        return df

    @staticmethod
    def add_price_structure(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        df["dist_sma_20"] = (close - df["sma_20"]) / df["sma_20"]
        df["dist_sma_50"] = (close - df["sma_50"]) / df["sma_50"]
        df["dist_sma_200"] = (close - df["sma_200"]) / df["sma_200"]

        # 52-week (252 days) High / Low distance
        high_252 = df["high"].rolling(window=252, min_periods=1).max()
        low_252 = df["low"].rolling(window=252, min_periods=1).min()
        df["dist_52w_high"] = (close - high_252) / high_252
        df["dist_52w_low"] = (close - low_252) / low_252

        # Signals / Patterns
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))

        # Breakouts (20-day high/low breakout)
        df["breakout_high_20"] = close > df["high"].shift(1).rolling(20, min_periods=1).max()
        df["breakdown_low_20"] = close < df["low"].shift(1).rolling(20, min_periods=1).min()

        return df
