import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculates full suite of technical indicators and price structure metrics for market time series."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates SMAs, EMAs, RSI, Stochastic RSI, MACD, ATR, Historical Volatility,
        Bollinger Bands, OBV, distances, price gaps, trend strength, and pattern signals.
        Expects a DataFrame sorted chronologically with columns: 'date', 'open', 'high', 'low', 'close', 'volume'
        """
        if df.empty or len(df) < 5:
            return df

        df = df.copy()

        # 1. Moving Averages
        for p in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{p}"] = df["close"].rolling(window=p).mean()

        for p in [9, 20, 50, 200]:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        # 2. RSI (14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        rsi_min = df["rsi_14"].rolling(window=14).min()
        rsi_max = df["rsi_14"].rolling(window=14).max()
        df["stoch_rsi"] = (df["rsi_14"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)

        # 3. MACD (12, 26, 9)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 4. Momentum / ROC
        df["rate_of_change"] = df["close"].pct_change(periods=10) * 100.0
        df["momentum_10"] = df["close"].diff(periods=10)

        # 5. Volatility Indicators
        # ATR (14)
        high_low = df["high"] - df["low"]
        high_cp = (df["high"] - df["close"].shift(1)).abs()
        low_cp = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14).mean()

        # Historical Volatility (20-day annualized)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20d"] = log_ret.rolling(window=20).std() * np.sqrt(252)

        # Bollinger Bands (20, 2)
        sma20 = df["sma_20"]
        rolling_std = df["close"].rolling(window=20).std()
        df["bollinger_upper"] = sma20 + (rolling_std * 2.0)
        df["bollinger_lower"] = sma20 - (rolling_std * 2.0)
        df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / sma20.replace(0, np.nan)

        # 6. Volume Indicators
        if "volume" in df.columns and not df["volume"].isnull().all():
            df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
            df["relative_volume"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
            df["volume_change"] = df["volume"].pct_change()

            # OBV
            obv_direction = np.sign(df["close"].diff()).fillna(0)
            df["obv"] = (obv_direction * df["volume"]).cumsum()

        # 7. Price Structure
        df["dist_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"].replace(0, np.nan) * 100.0
        df["dist_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"].replace(0, np.nan) * 100.0
        df["dist_sma_200"] = (df["close"] - df["sma_200"]) / df["sma_200"].replace(0, np.nan) * 100.0

        # 52-week High / Low distance (approx 252 trading days)
        high_52w = df["high"].rolling(window=252, min_periods=20).max()
        low_52w = df["low"].rolling(window=252, min_periods=20).min()
        df["dist_52w_high"] = (df["close"] - high_52w) / high_52w.replace(0, np.nan) * 100.0
        df["dist_52w_low"] = (df["close"] - low_52w) / low_52w.replace(0, np.nan) * 100.0

        # Gap and Intraday Range
        df["price_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan) * 100.0
        df["intraday_range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan) * 100.0

        # Trend Strength (using slope of SMA 50)
        df["trend_strength"] = (df["sma_50"] - df["sma_50"].shift(10)) / df["sma_50"].shift(10).replace(0, np.nan) * 100.0

        # 8. Pattern Detection Signals
        # Golden Cross (50 SMA crosses above 200 SMA) & Death Cross
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))

        # Breakouts / Breakdowns (closing above 20-day high or below 20-day low)
        high_20d = df["high"].shift(1).rolling(window=20).max()
        low_20d = df["low"].shift(1).rolling(window=20).min()
        df["breakout_20d"] = df["close"] > high_20d
        df["breakdown_20d"] = df["close"] < low_20d

        return df
