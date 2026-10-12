import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_stoch_rsi(series: pd.Series, period: int = 14, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    rsi = calculate_rsi(series, period)
    min_rsi = rsi.rolling(window=period, min_periods=1).min()
    max_rsi = rsi.rolling(window=period, min_periods=1).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    stoch_rsi = stoch_rsi.fillna(0.5)
    k = stoch_rsi.rolling(window=k_period, min_periods=1).mean()
    d = k.rolling(window=d_period, min_periods=1).mean()
    return pd.DataFrame({"stoch_rsi_k": k * 100, "stoch_rsi_d": d * 100})

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram
    })

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    sma = calculate_sma(series, period)
    std = series.rolling(window=period, min_periods=1).std().fillna(0)
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    return pd.DataFrame({
        "bb_middle": sma,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": bandwidth.fillna(0)
    })

def compute_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with all standard technical indicators."""
    res = df.copy()
    close = res["close"]

    # SMAs & EMAs
    for p in [5, 10, 20, 50, 100, 200]:
        res[f"sma_{p}"] = calculate_sma(close, p)
    for p in [9, 20, 50, 200]:
        res[f"ema_{p}"] = calculate_ema(close, p)

    # Momentum
    res["rsi_14"] = calculate_rsi(close, 14)
    stoch = calculate_stoch_rsi(close, 14)
    res["stoch_rsi_k"] = stoch["stoch_rsi_k"]
    res["stoch_rsi_d"] = stoch["stoch_rsi_d"]

    macd_df = calculate_macd(close, 12, 26, 9)
    res["macd"] = macd_df["macd"]
    res["macd_signal"] = macd_df["macd_signal"]
    res["macd_histogram"] = macd_df["macd_histogram"]
    res["roc_10"] = ((close - close.shift(10)) / close.shift(10).replace(0, np.nan) * 100).fillna(0)

    # Volatility
    res["atr_14"] = calculate_atr(res, 14)
    res["hist_vol_20"] = (close.pct_change().rolling(20, min_periods=1).std() * np.sqrt(252)).fillna(0)
    bb = calculate_bollinger_bands(close, 20, 2.0)
    res["bb_middle"] = bb["bb_middle"]
    res["bb_upper"] = bb["bb_upper"]
    res["bb_lower"] = bb["bb_lower"]
    res["bb_width"] = bb["bb_width"]

    # Price Structure & Distances
    res["dist_sma_20"] = ((close - res["sma_20"]) / res["sma_20"].replace(0, np.nan) * 100).fillna(0)
    res["dist_sma_50"] = ((close - res["sma_50"]) / res["sma_50"].replace(0, np.nan) * 100).fillna(0)
    res["dist_sma_200"] = ((close - res["sma_200"]) / res["sma_200"].replace(0, np.nan) * 100).fillna(0)

    rolling_high_252 = close.rolling(252, min_periods=1).max()
    rolling_low_252 = close.rolling(252, min_periods=1).min()
    res["dist_52w_high"] = ((close - rolling_high_252) / rolling_high_252.replace(0, np.nan) * 100).fillna(0)
    res["dist_52w_low"] = ((close - rolling_low_252) / rolling_low_252.replace(0, np.nan) * 100).fillna(0)

    # Pattern / Signal flags
    res["golden_cross"] = (res["sma_50"] > res["sma_200"]) & (res["sma_50"].shift(1) <= res["sma_200"].shift(1))
    res["death_cross"] = (res["sma_50"] < res["sma_200"]) & (res["sma_50"].shift(1) >= res["sma_200"].shift(1))

    return res
