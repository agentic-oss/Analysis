import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class TechnicalIndicators:
    """Calculates technical analysis indicators and structure metrics for OHLCV data."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators on given DataFrame."""
        if df is None or df.empty or len(df) < 5:
            return df

        df = df.copy()
        df = TechnicalIndicators.calculate_returns(df)
        df = TechnicalIndicators.calculate_moving_averages(df)
        df = TechnicalIndicators.calculate_momentum(df)
        df = TechnicalIndicators.calculate_volatility(df)
        df = TechnicalIndicators.calculate_volume(df)
        df = TechnicalIndicators.calculate_price_structure(df)
        df = TechnicalIndicators.detect_signals(df)
        return df

    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        df["return_1d"] = df["Close"].pct_change(1)
        df["return_5d"] = df["Close"].pct_change(5)
        df["return_20d"] = df["Close"].pct_change(20)
        df["return_60d"] = df["Close"].pct_change(60)
        df["return_120d"] = df["Close"].pct_change(120)
        df["return_252d"] = df["Close"].pct_change(252)
        return df

    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        for w in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{w}"] = df["Close"].rolling(window=w).mean()

        for w in [9, 20, 50, 200]:
            df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

        return df

    @staticmethod
    def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
        # RSI 14
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, 1e-9))
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        rsi = df["rsi_14"]
        min_rsi = rsi.rolling(window=14).min()
        max_rsi = rsi.rolling(window=14).max()
        df["stoch_rsi"] = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)

        # MACD (12, 26, 9)
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ROC & Momentum
        df["roc_10"] = df["Close"].pct_change(10) * 100
        df["momentum_10"] = df["Close"] - df["Close"].shift(10)

        return df

    @staticmethod
    def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
        # ATR 14
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14).mean()

        # Rolling Volatility
        daily_ret = df["Close"].pct_change()
        for w in [20, 60, 120, 252]:
            df[f"volatility_{w}d"] = daily_ret.rolling(window=w).std() * np.sqrt(252)

        # Bollinger Bands
        sma20 = df["sma_20"]
        std20 = df["Close"].rolling(window=20).std()
        df["bb_upper"] = sma20 + (std20 * 2.0)
        df["bb_lower"] = sma20 - (std20 * 2.0)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-9)

        return df

    @staticmethod
    def calculate_volume(df: pd.DataFrame) -> pd.DataFrame:
        if "Volume" in df.columns:
            df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
            df["relative_volume"] = df["Volume"] / (df["volume_sma_20"] + 1e-9)
            df["volume_change"] = df["Volume"].pct_change()

            # On Balance Volume (OBV)
            obv = [0]
            close = df["Close"].values
            vol = df["Volume"].values
            for i in range(1, len(df)):
                if close[i] > close[i - 1]:
                    obv.append(obv[-1] + vol[i])
                elif close[i] < close[i - 1]:
                    obv.append(obv[-1] - vol[i])
                else:
                    obv.append(obv[-1])
            df["obv"] = obv
        return df

    @staticmethod
    def calculate_price_structure(df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        df["distance_sma_20"] = (c - df["sma_20"]) / (df["sma_20"] + 1e-9)
        df["distance_sma_50"] = (c - df["sma_50"]) / (df["sma_50"] + 1e-9)
        df["distance_sma_200"] = (c - df["sma_200"]) / (df["sma_200"] + 1e-9)

        high_252 = df["High"].rolling(window=252, min_periods=20).max()
        low_252 = df["Low"].rolling(window=252, min_periods=20).min()
        df["distance_52w_high"] = (c - high_252) / (high_252 + 1e-9)
        df["distance_52w_low"] = (c - low_252) / (low_252 + 1e-9)

        df["gap"] = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-9)
        df["intraday_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)

        # ADX / Trend Strength approximation
        df["trend_strength"] = (df["sma_20"] - df["sma_50"]).abs() / (df["sma_50"] + 1e-9)
        return df

    @staticmethod
    def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
        df["signal_golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["signal_death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))

        high_20 = df["High"].rolling(window=20).max().shift(1)
        low_20 = df["Low"].rolling(window=20).min().shift(1)
        df["signal_breakout_20d"] = df["Close"] > high_20
        df["signal_breakdown_20d"] = df["Close"] < low_20

        df["signal_vol_expansion"] = df["bb_width"] > df["bb_width"].rolling(window=20).mean() * 1.5
        df["signal_vol_contraction"] = df["bb_width"] < df["bb_width"].rolling(window=20).mean() * 0.5

        return df
