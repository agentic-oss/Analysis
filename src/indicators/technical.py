"""Technical indicators calculator for price series."""

import numpy as np
import pandas as pd
from typing import Dict, Any


class TechnicalIndicators:
    """Calculates SMAs, EMAs, RSI, Stochastic RSI, MACD, ATR, HV, Bollinger Bands, OBV, and price structure."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all technical indicators on a DataFrame sorted by date.
        Assumes columns: ['date', 'close', 'high', 'low', 'open', 'volume']
        """
        if df.empty or len(df) < 5:
            return df

        df = df.sort_values('date').copy()
        close = df['close']
        high = df['high']
        low = df['low']
        open_p = df['open']
        volume = df.get('volume', pd.Series(1000, index=df.index))

        # Returns & Volatility
        df['return_1d'] = close.pct_change(1)
        df['return_3d'] = close.pct_change(3)
        df['return_5d'] = close.pct_change(5)
        df['return_20d'] = close.pct_change(20)
        df['return_60d'] = close.pct_change(60)
        df['return_120d'] = close.pct_change(120)
        df['return_252d'] = close.pct_change(252)

        df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
        df['volatility_60d'] = df['return_1d'].rolling(60).std() * np.sqrt(252)

        # Simple Moving Averages
        for p in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{p}'] = close.rolling(window=p, min_periods=1).mean()

        # Exponential Moving Averages
        for p in [9, 20, 50, 200]:
            df[f'ema_{p}'] = close.ewm(span=p, adjust=False).mean()

        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50.0)

        # Stochastic RSI
        rsi_min = df['rsi_14'].rolling(window=14, min_periods=1).min()
        rsi_max = df['rsi_14'].rolling(window=14, min_periods=1).max()
        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        df['stoch_rsi'] = ((df['rsi_14'] - rsi_min) / rsi_range).fillna(0.5) * 100

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Momentum & ROC
        df['roc_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
        df['momentum_10'] = close - close.shift(10)

        # ATR 14
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14, min_periods=1).mean()

        # Bollinger Bands (20, 2)
        bb_middle = df['sma_20']
        bb_std = close.rolling(window=20, min_periods=1).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_middle.replace(0, np.nan)

        # OBV (On-Balance Volume)
        obv_direction = np.where(close > prev_close, 1, np.where(close < prev_close, -1, 0))
        df['obv'] = (obv_direction * volume).cumsum()

        # Distance structure
        df['distance_sma_20'] = (close - df['sma_20']) / df['sma_20']
        df['distance_sma_50'] = (close - df['sma_50']) / df['sma_50']
        df['distance_sma_200'] = (close - df['sma_200']) / df['sma_200']

        high_52w = high.rolling(window=252, min_periods=1).max()
        low_52w = low.rolling(window=252, min_periods=1).min()
        df['distance_52w_high'] = (close - high_52w) / high_52w
        df['distance_52w_low'] = (close - low_52w) / low_52w

        # Max drawdown & duration
        cum_max = close.cummax()
        drawdown = (close - cum_max) / cum_max
        df['max_drawdown'] = drawdown.rolling(window=252, min_periods=1).min()

        # Pattern detections
        df['golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        df['death_cross'] = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
        df['breakout_20d'] = close > high.shift(1).rolling(20, min_periods=1).max()
        df['breakdown_20d'] = close < low.shift(1).rolling(20, min_periods=1).min()

        return df
