import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """
    Calculates moving averages, momentum indicators, volatility metrics, volume metrics,
    and price structure signals for precious metals time series.
    """

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all technical indicators on an OHLCV DataFrame.
        """
        if df.empty or "Close" not in df.columns:
            return df

        res = df.copy()
        close = res["Close"]
        high = res["High"] if "High" in res.columns else close
        low = res["Low"] if "Low" in res.columns else close
        volume = res["Volume"] if "Volume" in res.columns else pd.Series(0, index=res.index)

        # 1. Moving Averages
        for p in [5, 10, 20, 50, 100, 200]:
            res[f"sma_{p}"] = close.rolling(window=p).mean()
        for p in [9, 20, 50, 200]:
            res[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

        # 2. RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        res["rsi_14"] = 100 - (100 / (1 + rs))
        res["rsi_14"] = res["rsi_14"].fillna(50)

        # Stochastic RSI
        rsi = res["rsi_14"]
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        res["stoch_rsi"] = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        res["stoch_rsi"] = res["stoch_rsi"].fillna(0.5)

        # 3. MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        res["macd"] = ema12 - ema26
        res["macd_signal"] = res["macd"].ewm(span=9, adjust=False).mean()
        res["macd_hist"] = res["macd"] - res["macd_signal"]

        # ROC & Momentum
        res["roc_10"] = close.pct_change(10) * 100
        res["momentum_10"] = close - close.shift(10)

        # 4. Volatility (ATR, Bollinger, Historical Vol)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        res["atr_14"] = tr.rolling(14).mean()

        res["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        res["bb_upper"] = bb_mid + (2.0 * bb_std)
        res["bb_lower"] = bb_mid - (2.0 * bb_std)
        res["bb_width"] = (res["bb_upper"] - res["bb_lower"]) / bb_mid.replace(0, np.nan)

        # 5. Volume metrics
        res["volume_sma_20"] = volume.rolling(20).mean()
        res["relative_volume"] = volume / res["volume_sma_20"].replace(0, np.nan)
        res["volume_change"] = volume.pct_change()

        # OBV
        obv_direction = np.sign(close.diff()).fillna(0)
        res["obv"] = (obv_direction * volume).cumsum()

        # 6. Price Structure
        res["distance_sma_20"] = (close - res["sma_20"]) / res["sma_20"]
        res["distance_sma_50"] = (close - res["sma_50"]) / res["sma_50"]
        res["distance_sma_200"] = (close - res["sma_200"]) / res["sma_200"]

        high_52w = high.rolling(252, min_periods=20).max()
        low_52w = low.rolling(252, min_periods=20).min()
        res["distance_52w_high"] = (close - high_52w) / high_52w
        res["distance_52w_low"] = (close - low_52w) / low_52w

        res["intraday_range"] = (high - low) / close

        # 7. Pattern / Crossover Signals
        res["golden_cross"] = (res["sma_50"] > res["sma_200"]) & (res["sma_50"].shift(1) <= res["sma_200"].shift(1))
        res["death_cross"] = (res["sma_50"] < res["sma_200"]) & (res["sma_50"].shift(1) >= res["sma_200"].shift(1))

        res["breakout_20d"] = close >= high.shift(1).rolling(20).max()
        res["breakdown_20d"] = close <= low.shift(1).rolling(20).min()

        return res
