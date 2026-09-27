import pandas as pd
import numpy as np
from typing import Dict, Any, List

class TechnicalIndicators:
    """Calculates technical indicators for price time series."""

    @staticmethod
    def calculate_all(df: pd.DataFrame, price_col: str = "close", volume_col: str = "volume") -> pd.DataFrame:
        """Calculates moving averages, RSI, Stoch RSI, MACD, ATR, Bollinger Bands, OBV, and Price Structure."""
        df = df.copy()
        p = df[price_col].astype(float)

        # 1. Moving Averages
        for s in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{s}"] = p.rolling(window=s, min_periods=1).mean()
        for e in [9, 20, 50, 200]:
            df[f"ema_{e}"] = p.ewm(span=e, adjust=False).mean()

        # 2. RSI (14)
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50)

        # Stoch RSI
        rsi = df["rsi_14"]
        min_rsi = rsi.rolling(window=14, min_periods=1).min()
        max_rsi = rsi.rolling(window=14, min_periods=1).max()
        df["stoch_rsi"] = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-8)

        # 3. MACD
        ema_12 = p.ewm(span=12, adjust=False).mean()
        ema_26 = p.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of Change & Momentum
        df["roc_10"] = p.pct_change(periods=10) * 100
        df["momentum_10"] = p.diff(periods=10)

        # 4. Volatility & ATR
        if "high" in df.columns and "low" in df.columns:
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            prev_c = p.shift(1)
            tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()
        else:
            df["atr_14"] = p.pct_change().abs().rolling(window=14, min_periods=1).mean() * p

        df["historical_vol_20"] = p.pct_change().rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # Bollinger Bands
        sma_20 = df["sma_20"]
        std_20 = p.rolling(window=20, min_periods=1).std()
        df["bb_upper"] = sma_20 + 2 * std_20
        df["bb_lower"] = sma_20 - 2 * std_20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20.replace(0, np.nan)

        # 5. Volume
        if volume_col in df.columns:
            v = df[volume_col].astype(float)
            df["volume_sma_20"] = v.rolling(window=20, min_periods=1).mean()
            df["relative_volume"] = v / df["volume_sma_20"].replace(0, np.nan)
            obv = (np.sign(p.diff()).fillna(0) * v).cumsum()
            df["obv"] = obv
        else:
            df["volume_sma_20"] = np.nan
            df["relative_volume"] = np.nan
            df["obv"] = np.nan

        # 6. Price Structure
        df["dist_sma_20"] = (p - df["sma_20"]) / df["sma_20"].replace(0, np.nan)
        df["dist_sma_50"] = (p - df["sma_50"]) / df["sma_50"].replace(0, np.nan)
        df["dist_sma_200"] = (p - df["sma_200"]) / df["sma_200"].replace(0, np.nan)

        high_52w = p.rolling(window=252, min_periods=1).max()
        low_52w = p.rolling(window=252, min_periods=1).min()
        df["dist_52w_high"] = (p - high_52w) / high_52w.replace(0, np.nan)
        df["dist_52w_low"] = (p - low_52w) / low_52w.replace(0, np.nan)

        # Pattern indicators
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
        df["breakout_52w"] = p >= high_52w
        df["breakdown_52w"] = p <= low_52w

        return df
