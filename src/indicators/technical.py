import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class TechnicalIndicators:
    """Computes technical indicators and market structure metrics for price series."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "close" not in df.columns:
            return df

        df = df.copy()
        c = df["close"]
        h = df["high"] if "high" in df.columns else c
        l = df["low"] if "low" in df.columns else c
        v = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

        # Returns
        df["return_1d"] = c.pct_change(1)
        df["return_3d"] = c.pct_change(3)
        df["return_5d"] = c.pct_change(5)
        df["return_10d"] = c.pct_change(10)
        df["return_20d"] = c.pct_change(20)
        df["return_60d"] = c.pct_change(60)
        df["return_120d"] = c.pct_change(120)
        df["return_252d"] = c.pct_change(252)

        # Simple Moving Averages
        for p in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{p}"] = c.rolling(window=p, min_periods=1).mean()

        # Exponential Moving Averages
        for p in [9, 20, 50, 200]:
            df[f"ema_{p}"] = c.ewm(span=p, adjust=False, min_periods=1).mean()

        # Momentum: RSI 14
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss.replace(0, 1e-9))
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Stochastic RSI
        rsi = df["rsi_14"]
        min_rsi = rsi.rolling(window=14, min_periods=1).min()
        max_rsi = rsi.rolling(window=14, min_periods=1).max()
        stoch_rsi = (rsi - min_rsi) / ((max_rsi - min_rsi).replace(0, 1e-9))
        df["stoch_rsi"] = stoch_rsi.clip(0, 1)

        # MACD (12, 26, 9)
        ema_12 = c.ewm(span=12, adjust=False, min_periods=1).mean()
        ema_26 = c.ewm(span=26, adjust=False, min_periods=1).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Rate of Change & Momentum
        df["roc_10"] = ((c - c.shift(10)) / c.shift(10).replace(0, 1e-9)) * 100.0
        df["momentum_10"] = c - c.shift(10)

        # Volatility: ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

        # Historical Volatility (20-day annualized)
        df["volatility_20d"] = df["return_1d"].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        df["volatility_60d"] = df["return_1d"].rolling(window=60, min_periods=1).std() * np.sqrt(252)

        # Bollinger Bands (20, 2)
        bb_middle = df["sma_20"]
        bb_std = c.rolling(window=20, min_periods=1).std()
        df["bollinger_upper"] = bb_middle + (2.0 * bb_std)
        df["bollinger_lower"] = bb_middle - (2.0 * bb_std)
        df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / bb_middle.replace(0, 1e-9)

        # Volume Indicators
        if "volume" in df.columns:
            df["volume_sma_20"] = v.rolling(window=20, min_periods=1).mean()
            df["relative_volume"] = v / df["volume_sma_20"].replace(0, 1e-9)
            df["volume_change"] = v.pct_change(1)

            # OBV (On-Balance Volume)
            obv_direction = np.sign(c.diff()).fillna(0)
            df["obv"] = (obv_direction * v).cumsum()

        # Price Structure & Distances
        df["dist_sma_20"] = (c - df["sma_20"]) / df["sma_20"].replace(0, 1e-9)
        df["dist_sma_50"] = (c - df["sma_50"]) / df["sma_50"].replace(0, 1e-9)
        df["dist_sma_200"] = (c - df["sma_200"]) / df["sma_200"].replace(0, 1e-9)

        rolling_52w_high = h.rolling(window=252, min_periods=1).max()
        rolling_52w_low = l.rolling(window=252, min_periods=1).min()
        df["dist_52w_high"] = (c - rolling_52w_high) / rolling_52w_high.replace(0, 1e-9)
        df["dist_52w_low"] = (c - rolling_52w_low) / rolling_52w_low.replace(0, 1e-9)

        df["gap"] = (df["open"] - c.shift(1)) / c.shift(1).replace(0, 1e-9)
        df["intraday_range"] = (h - l) / l.replace(0, 1e-9)

        # Signals & Patterns
        golden_cross = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        death_cross = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
        df["golden_cross"] = golden_cross
        df["death_cross"] = death_cross

        breakout_20d = c > h.shift(1).rolling(window=20, min_periods=1).max()
        breakdown_20d = c < l.shift(1).rolling(window=20, min_periods=1).min()
        df["breakout_20d"] = breakout_20d
        df["breakdown_20d"] = breakdown_20d

        # Rolling Drawdown & Max Drawdown
        rolling_max = c.cummax()
        drawdown = (c - rolling_max) / rolling_max
        df["drawdown"] = drawdown
        df["max_drawdown_252d"] = drawdown.rolling(window=252, min_periods=1).min()

        return df
