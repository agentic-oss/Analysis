"""Technical indicators calculation engine."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class TechnicalIndicators:
    """Calculates SMAs, EMAs, RSI, MACD, ATR, Bollinger Bands, OBV, and price structure metrics."""

    @staticmethod
    def calculate_all(df: pd.DataFrame,
                      sma_periods: List[int] = [5, 10, 20, 50, 100, 200],
                      ema_periods: List[int] = [9, 20, 50, 200],
                      rsi_period: int = 14,
                      atr_period: int = 14) -> pd.DataFrame:
        """Calculates comprehensive suite of technical indicators on an OHLCV dataframe."""
        if df.empty or 'close' not in df.columns:
            return df

        df = df.copy()
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 1. Moving Averages
        for p in sma_periods:
            df[f'sma_{p}'] = df['close'].rolling(window=p, min_periods=1).mean()

        for p in ema_periods:
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False, min_periods=1).mean()

        # 2. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        rs = gain / (loss.replace(0, np.nan))
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50.0)

        # Stochastic RSI
        rsi_min = df['rsi_14'].rolling(window=rsi_period, min_periods=1).min()
        rsi_max = df['rsi_14'].rolling(window=rsi_period, min_periods=1).max()
        stoch_rsi = (df['rsi_14'] - rsi_min) / (rsi_max - rsi_min + 1e-8)
        df['stoch_rsi'] = stoch_rsi.fillna(0.5)

        # 3. MACD
        ema_12 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 4. Rate of Change (ROC) & Momentum
        df['roc_10'] = df['close'].pct_change(periods=10) * 100
        df['momentum_10'] = df['close'].diff(periods=10)

        # 5. ATR (Average True Range)
        if {'high', 'low'}.issubset(df.columns):
            prev_close = df['close'].shift(1)
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - prev_close).abs()
            tr3 = (df['low'] - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=atr_period, min_periods=1).mean()
        else:
            df['atr_14'] = 0.0

        # 6. Historical Volatility (20-day annualized)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20d'] = log_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # 7. Bollinger Bands (20-day, 2 std dev)
        sma_20 = df['sma_20'] if 'sma_20' in df.columns else df['close'].rolling(20, min_periods=1).mean()
        std_20 = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma_20 + 1e-8)

        # 8. On-Balance Volume (OBV)
        if 'volume' in df.columns:
            obv_change = np.where(df['close'] > df['close'].shift(1), df['volume'],
                        np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
            df['obv'] = pd.Series(obv_change, index=df.index).cumsum()
            df['volume_sma_20'] = df['volume'].rolling(20, min_periods=1).mean()
            df['relative_volume'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
        else:
            df['obv'] = 0.0
            df['volume_sma_20'] = 0.0
            df['relative_volume'] = 1.0

        # 9. Distance metrics & Structural flags
        df['dist_sma_20_pct'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-8) * 100
        df['dist_sma_50_pct'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-8) * 100
        df['dist_sma_200_pct'] = (df['close'] - df['sma_200']) / (df['sma_200'] + 1e-8) * 100

        rolling_high_252 = df['high'].rolling(252, min_periods=1).max() if 'high' in df.columns else df['close'].rolling(252, min_periods=1).max()
        rolling_low_252 = df['low'].rolling(252, min_periods=1).min() if 'low' in df.columns else df['close'].rolling(252, min_periods=1).min()

        df['dist_52w_high_pct'] = (df['close'] - rolling_high_252) / (rolling_high_252 + 1e-8) * 100
        df['dist_52w_low_pct'] = (df['close'] - rolling_low_252) / (rolling_low_252 + 1e-8) * 100

        # Crosses & Signals
        df['golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        df['death_cross'] = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))

        return df
