import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class TechnicalIndicators:
    """Computes full technical analysis suite for time-series market datasets."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 5:
            return df

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # 1. Historical Returns
        df["return_1d"] = df["close"].pct_change(1)
        df["return_3d"] = df["close"].pct_change(3)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["return_20d"] = df["close"].pct_change(20)
        df["return_60d"] = df["close"].pct_change(60)
        df["return_120d"] = df["close"].pct_change(120)
        df["return_252d"] = df["close"].pct_change(252)

        # 2. Moving Averages
        for p in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{p}"] = df["close"].rolling(window=p, min_periods=1).mean()

        for p in [9, 20, 50, 200]:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False, min_periods=1).mean()

        # 3. Momentum Indicators
        # RSI 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        min_rsi = df["rsi_14"].rolling(window=14, min_periods=1).min()
        max_rsi = df["rsi_14"].rolling(window=14, min_periods=1).max()
        df["stoch_rsi"] = (df["rsi_14"] - min_rsi) / (max_rsi - min_rsi + 1e-10)

        # MACD (12, 26, 9)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of Change (ROC) & Momentum
        df["roc_12"] = df["close"].pct_change(12) * 100
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        # 4. Volatility Indicators
        # ATR 14
        high_low = df["high"] - df["low"]
        high_cp = (df["high"] - df["close"].shift(1)).abs()
        low_cp = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

        # Historical Volatility (20, 60, 252d)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20d"] = log_ret.rolling(20, min_periods=1).std() * np.sqrt(252) * 100
        df["volatility_60d"] = log_ret.rolling(60, min_periods=1).std() * np.sqrt(252) * 100
        df["volatility_252d"] = log_ret.rolling(252, min_periods=1).std() * np.sqrt(252) * 100

        # Bollinger Bands (20, 2.0)
        df["bb_middle"] = df["sma_20"]
        bb_std = df["close"].rolling(20, min_periods=1).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2.0)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2.0)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)

        # 5. Volume Indicators
        if "volume" in df.columns:
            df["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean()
            df["relative_volume"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
            df["volume_change"] = df["volume"].pct_change()

            # OBV
            obv_step = np.where(df["close"] > df["close"].shift(1), df["volume"],
                       np.where(df["close"] < df["close"].shift(1), -df["volume"], 0))
            df["obv"] = pd.Series(obv_step, index=df.index).cumsum()

        # 6. Price Structure & Distances
        df["dist_sma_20"] = ((df["close"] - df["sma_20"]) / df["sma_20"]) * 100
        df["dist_sma_50"] = ((df["close"] - df["sma_50"]) / df["sma_50"]) * 100
        df["dist_sma_200"] = ((df["close"] - df["sma_200"]) / df["sma_200"]) * 100

        high_252 = df["high"].rolling(252, min_periods=1).max()
        low_252 = df["low"].rolling(252, min_periods=1).min()
        df["dist_52w_high"] = ((df["close"] - high_252) / high_252) * 100
        df["dist_52w_low"] = ((df["close"] - low_252) / low_252) * 100

        df["gap_pct"] = ((df["open"] - df["close"].shift(1)) / df["close"].shift(1)) * 100
        df["intraday_range_pct"] = ((df["high"] - df["low"]) / df["low"]) * 100
        df["trend_strength"] = (df["close"] - df["sma_50"]).abs() / (df["atr_14"] + 1e-10)

        # 7. Drawdown Metrics
        cum_max = df["close"].cummax()
        df["drawdown"] = (df["close"] - cum_max) / cum_max
        df["max_drawdown"] = df["drawdown"].cummin()

        # 8. Pattern & Cross Signals
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))

        df["breakout"] = (df["close"] > df["bb_upper"]) | (df["close"] > df["high"].shift(1).rolling(20).max())
        df["breakdown"] = (df["close"] < df["bb_lower"]) | (df["close"] < df["low"].shift(1).rolling(20).min())

        df["new_52w_high"] = df["close"] >= high_252
        df["new_52w_low"] = df["close"] <= low_252

        vol_mean = df["volatility_20d"].rolling(60, min_periods=1).mean()
        df["volatility_expansion"] = df["volatility_20d"] > (vol_mean * 1.5)
        df["volatility_contraction"] = df["volatility_20d"] < (vol_mean * 0.7)

        return df
