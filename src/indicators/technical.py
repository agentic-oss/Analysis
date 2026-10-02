import numpy as np
import pandas as pd

class TechnicalIndicators:
    """Calculates full suite of technical indicators for market time-series data."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates moving averages, momentum, volatility, volume, price structure, and signals.
        Expects a DataFrame sorted by date ascending with 'open', 'high', 'low', 'close', 'volume'.
        """
        if df.empty or "close" not in df.columns:
            return df

        df = df.copy()

        # Ensure numeric type
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        open_p = df["open"] if "open" in df.columns else close
        vol = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

        # 1. Moving Averages
        smas = [5, 10, 20, 50, 100, 200]
        for w in smas:
            df[f"sma_{w}"] = close.rolling(window=w, min_periods=1).mean()

        emas = [9, 20, 50, 200]
        for w in emas:
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

        # 2. Momentum Indicators
        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50.0)

        # Stochastic RSI
        rsi = df["rsi_14"]
        rsi_min = rsi.rolling(window=14, min_periods=1).min()
        rsi_max = rsi.rolling(window=14, min_periods=1).max()
        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        df["stoch_rsi"] = (rsi - rsi_min) / rsi_range
        df["stoch_rsi"] = df["stoch_rsi"].fillna(0.5)

        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of Change & Momentum
        df["roc_10"] = close.pct_change(periods=10) * 100.0
        df["momentum_10"] = close.diff(periods=10)

        # 3. Volatility Indicators
        # True Range & ATR 14
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

        # Historical Volatility (20-day annualized)
        log_ret = np.log(close / close.shift(1))
        df["volatility_20d"] = log_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # Bollinger Bands (20-day, 2 std)
        bb_middle = df["sma_20"]
        bb_std = close.rolling(window=20, min_periods=1).std()
        df["bb_upper"] = bb_middle + (2.0 * bb_std)
        df["bb_lower"] = bb_middle - (2.0 * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_middle.replace(0, np.nan)

        # 4. Volume Indicators
        if "volume" in df.columns:
            df["volume_sma_20"] = vol.rolling(window=20, min_periods=1).mean()
            df["relative_volume"] = vol / df["volume_sma_20"].replace(0, np.nan)
            df["volume_change"] = vol.pct_change()
            # On-Balance Volume (OBV)
            obv_direction = np.sign(close.diff()).fillna(0)
            df["obv"] = (obv_direction * vol).cumsum()

        # 5. Price Structure
        df["dist_sma_20"] = (close - df["sma_20"]) / df["sma_20"].replace(0, np.nan) * 100.0
        df["dist_sma_50"] = (close - df["sma_50"]) / df["sma_50"].replace(0, np.nan) * 100.0
        df["dist_sma_200"] = (close - df["sma_200"]) / df["sma_200"].replace(0, np.nan) * 100.0

        high_52w = high.rolling(window=252, min_periods=1).max()
        low_52w = low.rolling(window=252, min_periods=1).min()
        df["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan) * 100.0
        df["dist_52w_low"] = (close - low_52w) / low_52w.replace(0, np.nan) * 100.0

        df["gap"] = (open_p - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100.0
        df["intraday_range"] = (high - low) / low.replace(0, np.nan) * 100.0
        df["trend_strength"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"].replace(0, np.nan) * 100.0

        # 6. Signals
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
        df["breakout_20d"] = close > high.shift(1).rolling(window=20, min_periods=1).max()
        df["breakdown_20d"] = close < low.shift(1).rolling(window=20, min_periods=1).min()
        df["new_52w_high"] = close >= high_52w
        df["new_52w_low"] = close <= low_52w

        return df
