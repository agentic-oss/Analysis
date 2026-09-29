import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{period}"] = close.rolling(window=period).mean()

    for period in [9, 20, 50, 200]:
        df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    # Momentum - RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, 1e-9))
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi_min = df["rsi_14"].rolling(window=14).min()
    rsi_max = df["rsi_14"].rolling(window=14).max()
    df["stoch_rsi"] = (df["rsi_14"] - rsi_min) / ((rsi_max - rsi_min).replace(0, 1e-9))

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Rate of Change (ROC 12) & Momentum
    df["roc_12"] = close.pct_change(periods=12) * 100
    df["momentum_10"] = close - close.shift(10)

    # Volatility - ATR 14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14).mean()

    # Historical Volatility (20-day annualized)
    df["log_return"] = np.log(close / close.shift(1))
    df["volatility_20d"] = df["log_return"].rolling(window=20).std() * np.sqrt(252) * 100

    # Bollinger Bands (20, 2)
    sma_20 = df["sma_20"]
    std_20 = close.rolling(window=20).std()
    df["bb_upper"] = sma_20 + (2 * std_20)
    df["bb_lower"] = sma_20 - (2 * std_20)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20

    # Volume Indicators
    df["volume_sma_20"] = volume.rolling(window=20).mean()
    df["rel_volume"] = volume / (df["volume_sma_20"].replace(0, 1.0))
    df["volume_change"] = volume.pct_change()

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv"] = obv

    # Price Structure & Distances
    df["dist_sma_20"] = (close - df["sma_20"]) / df["sma_20"] * 100
    df["dist_sma_50"] = (close - df["sma_50"]) / df["sma_50"] * 100
    df["dist_sma_200"] = (close - df["sma_200"]) / df["sma_200"] * 100

    high_52w = high.rolling(window=252, min_periods=20).max()
    low_52w = low.rolling(window=252, min_periods=20).min()
    df["dist_52w_high"] = (close - high_52w) / high_52w * 100
    df["dist_52w_low"] = (close - low_52w) / low_52w * 100

    df["gap"] = (df["open"] - close.shift(1)) / close.shift(1) * 100
    df["intraday_range"] = (high - low) / low * 100
    df["trend_strength"] = (close - df["sma_50"]).abs() / (df["atr_14"].replace(0, 1.0))

    # Pattern & Cross Signals
    df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
    df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
    df["breakout_20d"] = close > high.shift(1).rolling(window=20).max()
    df["breakdown_20d"] = close < low.shift(1).rolling(window=20).min()

    # Returns
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)
    df["return_252d"] = close.pct_change(252)

    # Drawdown metrics
    cummax = close.cummax()
    df["drawdown"] = (close - cummax) / cummax
    df["max_drawdown_252d"] = df["drawdown"].rolling(window=252, min_periods=20).min()

    return df
