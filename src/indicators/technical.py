import pandas as pd
import numpy as np

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for a price dataframe.
    Requires columns: close (and optionally open, high, low, volume).
    """
    res = df.copy()

    if "close" not in res.columns:
        return res

    close = res["close"]

    # Moving Averages
    smas = [5, 10, 20, 50, 100, 200]
    for s in smas:
        res[f"sma_{s}"] = close.rolling(window=s, min_periods=1).mean()

    emas = [9, 20, 50, 200]
    for e in emas:
        res[f"ema_{e}"] = close.ewm(span=e, adjust=False, min_periods=1).mean()

    # Momentum: RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    res["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Stochastic RSI (14)
    rsi = res["rsi_14"]
    rsi_min = rsi.rolling(window=14, min_periods=1).min()
    rsi_max = rsi.rolling(window=14, min_periods=1).max()
    res["stoch_rsi"] = (rsi - rsi_min) / ((rsi_max - rsi_min) + 1e-10)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    res["macd"] = ema12 - ema26
    res["macd_signal"] = res["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
    res["macd_histogram"] = res["macd"] - res["macd_signal"]

    # Rate of Change (ROC 12) & Momentum
    res["roc_12"] = close.pct_change(periods=12) * 100.0
    res["momentum_10"] = close.diff(periods=10)

    # Volatility
    if "high" in res.columns and "low" in res.columns:
        high = res["high"]
        low = res["low"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        res["atr_14"] = tr.rolling(window=14, min_periods=1).mean()
    else:
        res["atr_14"] = np.nan

    # Historical Volatility (20-day annualized)
    log_ret = np.log(close / close.shift(1))
    res["volatility_20d"] = log_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100.0

    # Bollinger Bands (20, 2)
    sma20 = res["sma_20"]
    std20 = close.rolling(window=20, min_periods=1).std()
    res["bb_upper"] = sma20 + (std20 * 2.0)
    res["bb_lower"] = sma20 - (std20 * 2.0)
    res["bb_width"] = (res["bb_upper"] - res["bb_lower"]) / (sma20 + 1e-10)

    # Volume indicators
    if "volume" in res.columns:
        vol = res["volume"]
        res["volume_sma_20"] = vol.rolling(window=20, min_periods=1).mean()
        res["relative_volume"] = vol / (res["volume_sma_20"] + 1e-10)
        res["volume_change"] = vol.pct_change() * 100.0

        # OBV
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        res["obv"] = (direction * vol).fillna(0).cumsum()

    # Price structure & distances
    res["dist_sma_20"] = (close - res["sma_20"]) / res["sma_20"] * 100.0
    res["dist_sma_50"] = (close - res["sma_50"]) / res["sma_50"] * 100.0
    res["dist_sma_200"] = (close - res["sma_200"]) / res["sma_200"] * 100.0

    high_52w = close.rolling(window=252, min_periods=1).max()
    low_52w = close.rolling(window=252, min_periods=1).min()
    res["dist_52w_high"] = (close - high_52w) / high_52w * 100.0
    res["dist_52w_low"] = (close - low_52w) / low_52w * 100.0

    if "open" in res.columns:
        res["gap"] = (res["open"] - close.shift(1)) / close.shift(1) * 100.0
    if "high" in res.columns and "low" in res.columns:
        res["intraday_range"] = (res["high"] - res["low"]) / close * 100.0

    # Trend Strength (ADX-like proxy using SMA alignment)
    sma50 = res["sma_50"]
    sma200 = res["sma_200"]
    res["golden_cross"] = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
    res["death_cross"] = (sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))

    # Returns statistics
    res["return_1d"] = close.pct_change(1) * 100.0
    res["return_5d"] = close.pct_change(5) * 100.0
    res["return_20d"] = close.pct_change(20) * 100.0
    res["return_60d"] = close.pct_change(60) * 100.0
    res["return_120d"] = close.pct_change(120) * 100.0
    res["return_252d"] = close.pct_change(252) * 100.0

    # Drawdown
    cummax = close.cummax()
    res["drawdown"] = (close - cummax) / cummax * 100.0

    return res
