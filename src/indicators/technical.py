"""
Technical Analysis Engine.
Computes MAs, Momentum, Volatility, Volume, and Price Structure indicators.
Fully reproducible and deterministic.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class TechnicalAnalysisEngine:
    """Calculates full technical indicator suite for OHLCV data."""

    def __init__(
        self,
        sma_periods: List[int] = [5, 10, 20, 50, 100, 200],
        ema_periods: List[int] = [9, 20, 50, 200],
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
    ):
        self.sma_periods = sma_periods
        self.ema_periods = ema_periods
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all technical indicators on a copy of input DataFrame."""
        if df.empty or "close" not in df.columns:
            return df

        df = df.sort_values("date").copy()

        # Returns
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["return_60d"] = df["close"].pct_change(60)
        df["return_120d"] = df["close"].pct_change(120)
        df["return_252d"] = df["close"].pct_change(252)

        # Simple Moving Averages
        for p in self.sma_periods:
            df[f"sma_{p}"] = df["close"].rolling(window=p).mean()

        # Exponential Moving Averages
        for p in self.ema_periods:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        # RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50.0)

        # Stochastic RSI
        rsi_min = df["rsi_14"].rolling(window=self.rsi_period).min()
        rsi_max = df["rsi_14"].rolling(window=self.rsi_period).max()
        df["stoch_rsi"] = (df["rsi_14"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

        # MACD
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of Change (ROC)
        df["roc_12"] = df["close"].pct_change(12) * 100.0

        # ATR
        if "high" in df.columns and "low" in df.columns:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift(1)).abs()
            low_close = (df["low"] - df["close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr"] = tr.rolling(window=self.atr_period).mean()
        else:
            df["atr"] = df["close"].rolling(window=self.atr_period).std()

        # Historical Volatility (20-day annualized)
        df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)

        # Bollinger Bands
        sma_bb = df["close"].rolling(window=self.bollinger_period).mean()
        std_bb = df["close"].rolling(window=self.bollinger_period).std()
        df["bollinger_upper"] = sma_bb + (self.bollinger_std * std_bb)
        df["bollinger_lower"] = sma_bb - (self.bollinger_std * std_bb)
        df["bollinger_middle"] = sma_bb
        df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / (sma_bb + 1e-9)

        # Volume Indicators
        if "volume" in df.columns:
            df["volume_sma_20"] = df["volume"].rolling(20).mean()
            df["relative_volume"] = df["volume"] / (df["volume_sma_20"] + 1e-9)
            # OBV
            obv_direction = np.sign(df["close"].diff()).fillna(0)
            df["obv"] = (obv_direction * df["volume"]).cumsum()

        # Price Structure Metrics
        df["dist_sma_20_pct"] = (df["close"] - df["sma_20"]) / (df["sma_20"] + 1e-9) * 100.0
        df["dist_sma_50_pct"] = (df["close"] - df["sma_50"]) / (df["sma_50"] + 1e-9) * 100.0
        df["dist_sma_200_pct"] = (df["close"] - df["sma_200"]) / (df["sma_200"] + 1e-9) * 100.0

        df["high_52w"] = df["high"].rolling(252, min_periods=20).max() if "high" in df.columns else df["close"].rolling(252, min_periods=20).max()
        df["low_52w"] = df["low"].rolling(252, min_periods=20).min() if "low" in df.columns else df["close"].rolling(252, min_periods=20).min()

        df["dist_52w_high_pct"] = (df["close"] - df["high_52w"]) / (df["high_52w"] + 1e-9) * 100.0
        df["dist_52w_low_pct"] = (df["close"] - df["low_52w"]) / (df["low_52w"] + 1e-9) * 100.0

        # Pattern Flags
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))

        # Breakouts / Breakdowns (20-day high/low)
        high_20d = df["high"].shift(1).rolling(20).max() if "high" in df.columns else df["close"].shift(1).rolling(20).max()
        low_20d = df["low"].shift(1).rolling(20).min() if "low" in df.columns else df["close"].shift(1).rolling(20).min()
        df["breakout_20d"] = df["close"] > high_20d
        df["breakdown_20d"] = df["close"] < low_20d

        return df
