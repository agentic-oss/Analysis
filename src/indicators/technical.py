import pandas as pd
import numpy as np
from typing import Dict, Any, List

def calculate_technical_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Calculate standard technical indicators, momentum, volatility, volume, and structural signals."""
    if df.empty or len(df) < 5:
        return df

    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)

    # 1. Moving Averages
    sma_periods = config.get("indicators", {}).get("sma_periods", [5, 10, 20, 50, 100, 200])
    for p in sma_periods:
        df[f"sma_{p}"] = close.rolling(window=p, min_periods=1).mean()

    # Ensure key SMAs exist even if not in config
    for p in [20, 50, 200]:
        if f"sma_{p}" not in df.columns:
            df[f"sma_{p}"] = close.rolling(window=p, min_periods=1).mean()

    ema_periods = config.get("indicators", {}).get("ema_periods", [9, 20, 50, 200])
    for p in ema_periods:
        df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    # 2. RSI 14 & Stochastic RSI
    rsi_p = config.get("indicators", {}).get("rsi_period", 14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_p, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_p, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50.0)

    # Stoch RSI
    rsi_min = df["rsi_14"].rolling(window=rsi_p, min_periods=1).min()
    rsi_max = df["rsi_14"].rolling(window=rsi_p, min_periods=1).max()
    df["stoch_rsi"] = (df["rsi_14"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df["stoch_rsi"] = df["stoch_rsi"].fillna(0.5)

    # 3. MACD
    fast = config.get("indicators", {}).get("macd_fast", 12)
    slow = config.get("indicators", {}).get("macd_slow", 26)
    sig = config.get("indicators", {}).get("macd_signal", 9)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=sig, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # Momentum & ROC
    df["momentum_10"] = close.diff(10)
    df["roc_10"] = close.pct_change(10) * 100.0

    # 4. Volatility: ATR, Historical Volatility, Bollinger Bands
    atr_p = config.get("indicators", {}).get("atr_period", 14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=atr_p, min_periods=1).mean()

    # Historical Volatility (20-day annualized)
    returns_1d = close.pct_change()
    df["volatility_20d"] = returns_1d.rolling(window=20, min_periods=1).std() * np.sqrt(252)

    # Bollinger Bands
    bb_p = config.get("indicators", {}).get("bb_period", 20)
    bb_std_mult = config.get("indicators", {}).get("bb_std", 2.0)
    bb_mid = close.rolling(window=bb_p, min_periods=1).mean()
    bb_std = close.rolling(window=bb_p, min_periods=1).std()
    df["bb_middle"] = bb_mid
    df["bb_upper"] = bb_mid + (bb_std_mult * bb_std)
    df["bb_lower"] = bb_mid - (bb_std_mult * bb_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    # 5. Volume Indicators (if volume exists)
    if volume.notnull().any() and (volume > 0).any():
        df["volume_sma_20"] = volume.rolling(window=20, min_periods=1).mean()
        df["relative_volume"] = volume / df["volume_sma_20"].replace(0, np.nan)
        df["volume_change"] = volume.pct_change()

        # OBV (On-Balance Volume)
        direction = np.sign(close.diff()).fillna(0)
        df["obv"] = (direction * volume).cumsum()
    else:
        df["volume_sma_20"] = np.nan
        df["relative_volume"] = np.nan
        df["volume_change"] = np.nan
        df["obv"] = np.nan

    # 6. Price Structure & Distances
    df["dist_sma_20"] = (close - df["sma_20"]) / df["sma_20"]
    df["dist_sma_50"] = (close - df["sma_50"]) / df["sma_50"]
    df["dist_sma_200"] = (close - df["sma_200"]) / df["sma_200"]

    # 52-week High / Low Distances (approx 252 trading days)
    high_52w = high.rolling(window=252, min_periods=1).max()
    low_52w = low.rolling(window=252, min_periods=1).min()
    df["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
    df["dist_52w_low"] = (close - low_52w) / low_52w.replace(0, np.nan)

    # Gap and Intraday Range
    df["gap"] = (df["open"] - close.shift(1)) / close.shift(1)
    df["intraday_range"] = (high - low) / low.replace(0, np.nan)

    # 7. Pattern Signals
    df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
    df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))

    # Breakouts/Breakdowns (20-day high/low)
    high_20 = high.shift(1).rolling(window=20, min_periods=1).max()
    low_20 = low.shift(1).rolling(window=20, min_periods=1).min()
    df["breakout_20d"] = close > high_20
    df["breakdown_20d"] = close < low_20

    # Trend Strength
    ma_aligned_bull = (df["sma_20"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"])
    ma_aligned_bear = (df["sma_20"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"])
    df["trend_strength"] = np.where(ma_aligned_bull, 1.0, np.where(ma_aligned_bear, -1.0, 0.0))

    return df
