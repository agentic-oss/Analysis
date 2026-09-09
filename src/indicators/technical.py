"""
Technical Analysis & Price Structure Module.
Calculates moving averages, momentum (RSI, Stochastic RSI, MACD, ROC),
volatility (ATR, Historical Volatility, Bollinger Bands), volume (OBV, Rel Vol),
price structure metrics, and technical signals (Golden/Death Cross, Breakouts, Divergences).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and appends them as new columns in reproducible, deterministic manner.
    Input df must have ['date', 'open', 'high', 'low', 'close', 'volume'].
    """
    if df.empty or len(df) < 5:
        return df

    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 1. Moving Averages
    for p in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{p}"] = close.rolling(window=p, min_periods=1).mean()

    for p in [9, 20, 50, 200]:
        df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    # 2. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50.0)

    # Stochastic RSI
    rsi = df["rsi_14"]
    stoch_min = rsi.rolling(window=14, min_periods=1).min()
    stoch_max = rsi.rolling(window=14, min_periods=1).max()
    stoch_denom = (stoch_max - stoch_min).replace(0, np.nan)
    df["stoch_rsi"] = ((rsi - stoch_min) / stoch_denom).fillna(0.5)

    # 3. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 4. Momentum & Rate of Change (ROC 12)
    df["roc_12"] = close.pct_change(periods=12) * 100.0
    df["momentum_10"] = close.diff(periods=10)

    # 5. Volatility (ATR 14, Historical Volatility, Bollinger Bands)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    # Historical Volatility (20d annualized)
    log_ret = np.log(close / prev_close)
    df["volatility_20d"] = log_ret.rolling(window=20, min_periods=2).std() * np.sqrt(252) * 100.0
    df["volatility_60d"] = log_ret.rolling(window=60, min_periods=2).std() * np.sqrt(252) * 100.0

    # Bollinger Bands (20, 2)
    bb_middle = df["sma_20"]
    bb_std = close.rolling(window=20, min_periods=1).std()
    df["bollinger_upper"] = bb_middle + (bb_std * 2.0)
    df["bollinger_lower"] = bb_middle - (bb_std * 2.0)
    bb_denom = bb_middle.replace(0, np.nan)
    df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / bb_denom

    # 6. Volume Indicators
    df["volume_sma_20"] = volume.rolling(window=20, min_periods=1).mean()
    vol_sma_denom = df["volume_sma_20"].replace(0, np.nan)
    df["relative_volume"] = volume / vol_sma_denom
    df["volume_change_pct"] = volume.pct_change() * 100.0

    # On-Balance Volume (OBV)
    direction = np.where(close > prev_close, 1, np.where(close < prev_close, -1, 0))
    df["obv"] = (volume * direction).cumsum()

    # 7. Price Structure Metrics
    df["dist_sma_20_pct"] = ((close - df["sma_20"]) / df["sma_20"].replace(0, np.nan)) * 100.0
    df["dist_sma_50_pct"] = ((close - df["sma_50"]) / df["sma_50"].replace(0, np.nan)) * 100.0
    df["dist_sma_200_pct"] = ((close - df["sma_200"]) / df["sma_200"].replace(0, np.nan)) * 100.0

    high_252 = high.rolling(window=252, min_periods=1).max()
    low_252 = low.rolling(window=252, min_periods=1).min()
    df["dist_52w_high_pct"] = ((close - high_252) / high_252.replace(0, np.nan)) * 100.0
    df["dist_52w_low_pct"] = ((close - low_252) / low_252.replace(0, np.nan)) * 100.0

    df["gap_pct"] = ((df["open"] - prev_close) / prev_close.replace(0, np.nan)) * 100.0
    df["intraday_range_pct"] = ((high - low) / low.replace(0, np.nan)) * 100.0

    # Trend strength: ADX proxy or SMA alignment strength
    df["trend_strength"] = np.where(
        (close > df["sma_20"]) & (df["sma_20"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"]), 1.0,
        np.where((close < df["sma_20"]) & (df["sma_20"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"]), -1.0, 0.0)
    )

    return df


def detect_technical_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Detects active signals on the latest observation of indicator-enhanced DataFrame."""
    if df.empty or "sma_50" not in df.columns or len(df) < 2:
        return {}

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {
        "golden_cross": bool(prev["sma_50"] <= prev["sma_200"] and curr["sma_50"] > curr["sma_200"]),
        "death_cross": bool(prev["sma_50"] >= prev["sma_200"] and curr["sma_50"] < curr["sma_200"]),
        "breakout_20d": bool(curr["close"] >= df["high"].iloc[-21:-1].max() if len(df) >= 21 else False),
        "breakdown_20d": bool(curr["close"] <= df["low"].iloc[-21:-1].min() if len(df) >= 21 else False),
        "new_52w_high": bool(curr["close"] >= df["high"].iloc[-252:-1].max() if len(df) >= 252 else False),
        "new_52w_low": bool(curr["close"] <= df["low"].iloc[-252:-1].min() if len(df) >= 252 else False),
        "rsi_overbought": bool(curr["rsi_14"] > 70.0),
        "rsi_oversold": bool(curr["rsi_14"] < 30.0),
        "volatility_expansion": bool(curr["volatility_20d"] > curr["volatility_60d"] * 1.25),
        "volatility_contraction": bool(curr["volatility_20d"] < curr["volatility_60d"] * 0.75)
    }
    return signals
