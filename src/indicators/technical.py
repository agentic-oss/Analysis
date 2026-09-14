"""
Technical Indicators & Price Structure Calculation Engine.
Calculates technical indicators, price structures, and pattern triggers reproducibly.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any


class TechnicalAnalysisEngine:
    """
    Computes standard technical indicators: Moving Averages, RSI, MACD, ATR, Bollinger Bands, Volume indicators,
    and price structure distance/pattern triggers.
    """

    @staticmethod
    def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all technical indicators on an OHLCV dataframe sorted by date ascending.
        Returns copy of DataFrame with added indicator columns.
        """
        if df.empty or "close" not in df.columns:
            return df

        df_out = df.copy().sort_values("date").reset_index(drop=True)
        close = df_out["close"].astype(float)
        high = df_out["high"].astype(float) if "high" in df_out.columns else close
        low = df_out["low"].astype(float) if "low" in df_out.columns else close
        volume = df_out["volume"].astype(float) if "volume" in df_out.columns else pd.Series(0.0, index=df_out.index)

        # 1. Standard Returns
        df_out["return_1d"] = close.pct_change(1)
        df_out["return_5d"] = close.pct_change(5)
        df_out["return_20d"] = close.pct_change(20)
        df_out["return_60d"] = close.pct_change(60)
        df_out["return_120d"] = close.pct_change(120)
        df_out["return_252d"] = close.pct_change(252)

        # 2. Moving Averages (SMA & EMA)
        for period in [5, 10, 20, 50, 100, 200]:
            df_out[f"sma_{period}"] = close.rolling(window=period, min_periods=1).mean()

        for period in [9, 20, 50, 200]:
            df_out[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # 3. Momentum: RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df_out["rsi_14"] = 100 - (100 / (1 + rs))
        df_out["rsi_14"] = df_out["rsi_14"].fillna(50.0)

        # Stochastic RSI
        min_rsi = df_out["rsi_14"].rolling(window=14, min_periods=1).min()
        max_rsi = df_out["rsi_14"].rolling(window=14, min_periods=1).max()
        df_out["stoch_rsi"] = (df_out["rsi_14"] - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
        df_out["stoch_rsi"] = df_out["stoch_rsi"].fillna(0.5)

        # 4. MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df_out["macd"] = ema_12 - ema_26
        df_out["macd_signal"] = df_out["macd"].ewm(span=9, adjust=False).mean()
        df_out["macd_hist"] = df_out["macd"] - df_out["macd_signal"]

        # Rate of Change & Momentum
        df_out["roc_10"] = ((close - close.shift(10)) / close.shift(10)) * 100
        df_out["momentum_10"] = close - close.shift(10)

        # 5. Volatility: ATR 14
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_out["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

        # Rolling Volatility (20-day annualized)
        df_out["volatility_20d"] = df_out["return_1d"].rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # Bollinger Bands (20, 2.0)
        ma_20 = df_out["sma_20"]
        std_20 = close.rolling(window=20, min_periods=1).std()
        df_out["bollinger_upper"] = ma_20 + (2.0 * std_20)
        df_out["bollinger_lower"] = ma_20 - (2.0 * std_20)
        df_out["bollinger_width"] = (df_out["bollinger_upper"] - df_out["bollinger_lower"]) / ma_20.replace(0, np.nan)

        # 6. Volume Indicators
        df_out["volume_sma_20"] = volume.rolling(window=20, min_periods=1).mean()
        df_out["relative_volume"] = volume / df_out["volume_sma_20"].replace(0, np.nan)
        df_out["relative_volume"] = df_out["relative_volume"].fillna(1.0)
        df_out["volume_change"] = volume.pct_change(1)

        # On-Balance Volume (OBV)
        obv = np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0))
        df_out["obv"] = pd.Series(obv, index=df_out.index).cumsum()

        # 7. Price Structure & Distances
        df_out["dist_sma_20_pct"] = (close - df_out["sma_20"]) / df_out["sma_20"].replace(0, np.nan) * 100
        df_out["dist_sma_50_pct"] = (close - df_out["sma_50"]) / df_out["sma_50"].replace(0, np.nan) * 100
        df_out["dist_sma_200_pct"] = (close - df_out["sma_200"]) / df_out["sma_200"].replace(0, np.nan) * 100

        rolling_52w_high = close.rolling(window=252, min_periods=1).max()
        rolling_52w_low = close.rolling(window=252, min_periods=1).min()
        df_out["dist_52w_high_pct"] = (close - rolling_52w_high) / rolling_52w_high.replace(0, np.nan) * 100
        df_out["dist_52w_low_pct"] = (close - rolling_52w_low) / rolling_52w_low.replace(0, np.nan) * 100

        df_out["gap_pct"] = (df_out["open"] - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100
        df_out["intraday_range_pct"] = (high - low) / low.replace(0, np.nan) * 100

        # Maximum Drawdown and Drawdown Duration
        rolling_max = close.cummax()
        drawdown = (close - rolling_max) / rolling_max
        df_out["drawdown"] = drawdown

        # 8. Pattern Signals
        df_out["golden_cross"] = (df_out["sma_50"] > df_out["sma_200"]) & (df_out["sma_50"].shift(1) <= df_out["sma_200"].shift(1))
        df_out["death_cross"] = (df_out["sma_50"] < df_out["sma_200"]) & (df_out["sma_50"].shift(1) >= df_out["sma_200"].shift(1))

        # Breakout / Breakdown (20-day high/low)
        high_20d = high.shift(1).rolling(window=20, min_periods=1).max()
        low_20d = low.shift(1).rolling(window=20, min_periods=1).min()
        df_out["breakout_20d"] = close > high_20d
        df_out["breakdown_20d"] = close < low_20d

        # Volatility Expansion / Contraction
        vol_sma_20 = df_out["volatility_20d"].rolling(window=60, min_periods=1).mean()
        df_out["volatility_expansion"] = df_out["volatility_20d"] > 1.25 * vol_sma_20
        df_out["volatility_contraction"] = df_out["volatility_20d"] < 0.75 * vol_sma_20

        return df_out
