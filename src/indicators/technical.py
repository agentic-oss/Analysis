"""
Technical indicators and price structure calculations.
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculates standard technical indicators and price structure metrics."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all technical indicators on a DataFrame containing OHLCV.
        Expects columns: ['date', 'close'] and optionally ['open', 'high', 'low', 'volume'].
        Returns DataFrame with additional technical columns.
        """
        if df.empty or "close" not in df.columns:
            return df

        df = df.copy()

        # Simple Moving Averages (SMA)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{period}"] = df["close"].rolling(window=period, min_periods=1).mean()

        # Exponential Moving Averages (EMA)
        for period in [9, 20, 50, 200]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # Returns and Volatility
        df["return_1d"] = df["close"].pct_change()
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["return_60d"] = df["close"].pct_change(60)
        df["return_120d"] = df["close"].pct_change(120)
        df["return_252d"] = df["close"].pct_change(252)

        df["volatility_20d"] = df["return_1d"].rolling(20, min_periods=1).std() * np.sqrt(252)
        df["volatility_60d"] = df["return_1d"].rolling(60, min_periods=1).std() * np.sqrt(252)

        # RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss.replace(0, 1e-9))
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        min_rsi = df["rsi_14"].rolling(14, min_periods=1).min()
        max_rsi = df["rsi_14"].rolling(14, min_periods=1).max()
        rsi_diff = max_rsi - min_rsi
        df["stoch_rsi"] = np.where(rsi_diff > 0, (df["rsi_14"] - min_rsi) / rsi_diff, 0.5)

        # MACD (12, 26, 9)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Rate of Change (ROC 12) & Momentum
        df["roc_12"] = df["close"].pct_change(12) * 100
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        # ATR (Average True Range)
        if "high" in df.columns and "low" in df.columns:
            prev_close = df["close"].shift(1)
            tr1 = df["high"] - df["low"]
            tr2 = (df["high"] - prev_close).abs()
            tr3 = (df["low"] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14, min_periods=1).mean()
        else:
            df["atr_14"] = df["close"] * 0.015  # Fallback approximation

        # Bollinger Bands (20, 2)
        sma_20 = df["sma_20"]
        std_20 = df["close"].rolling(20, min_periods=1).std()
        df["bollinger_upper"] = sma_20 + (2.0 * std_20)
        df["bollinger_lower"] = sma_20 - (2.0 * std_20)
        df["bollinger_bandwidth"] = (df["bollinger_upper"] - df["bollinger_lower"]) / (sma_20.replace(0, 1e-9))

        # Volume metrics if available
        if "volume" in df.columns and not df["volume"].isna().all():
            df["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean()
            df["relative_volume"] = df["volume"] / df["volume_sma_20"].replace(0, 1e-9)
            df["volume_change"] = df["volume"].pct_change()

            # On-Balance Volume (OBV)
            obv = [0]
            close_vals = df["close"].values
            vol_vals = df["volume"].fillna(0).values
            for i in range(1, len(df)):
                if close_vals[i] > close_vals[i - 1]:
                    obv.append(obv[-1] + vol_vals[i])
                elif close_vals[i] < close_vals[i - 1]:
                    obv.append(obv[-1] - vol_vals[i])
                else:
                    obv.append(obv[-1])
            df["obv"] = obv

        # Price structure & distances
        df["dist_sma_20_pct"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        df["dist_sma_50_pct"] = (df["close"] - df["sma_50"]) / df["sma_50"]
        df["dist_sma_200_pct"] = (df["close"] - df["sma_200"]) / df["sma_200"]

        rolling_52w_high = df["close"].rolling(252, min_periods=1).max()
        rolling_52w_low = df["close"].rolling(252, min_periods=1).min()
        df["dist_52w_high_pct"] = (df["close"] - rolling_52w_high) / rolling_52w_high
        df["dist_52w_low_pct"] = (df["close"] - rolling_52w_low) / rolling_52w_low

        if "open" in df.columns:
            df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
            df["intraday_range_pct"] = (df["high"] - df["low"]) / df["open"]

        # Signals / Technical events
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
        df["new_52w_high"] = df["close"] >= rolling_52w_high
        df["new_52w_low"] = df["close"] <= rolling_52w_low

        return df
