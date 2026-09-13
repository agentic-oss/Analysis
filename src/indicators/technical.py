import numpy as np
import pandas as pd
from typing import Dict, Any

class TechnicalIndicators:
    """Calculates MAs, momentum, volatility, volume, price structure, and signals."""

    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        smas = [5, 10, 20, 50, 100, 200]
        emas = [9, 20, 50, 200]
        for p in smas:
            res[f'sma_{p}'] = res['Close'].rolling(window=p).mean()
        for p in emas:
            res[f'ema_{p}'] = res['Close'].ewm(span=p, adjust=False).mean()
        return res

    @staticmethod
    def calculate_momentum(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
        res = df.copy()

        # RSI 14
        delta = res['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        res[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        rsi = res[f'rsi_{rsi_period}']
        min_rsi = rsi.rolling(window=rsi_period).min()
        max_rsi = rsi.rolling(window=rsi_period).max()
        res['stoch_rsi'] = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)

        # MACD (12, 26, 9)
        ema12 = res['Close'].ewm(span=12, adjust=False).mean()
        ema26 = res['Close'].ewm(span=26, adjust=False).mean()
        res['macd'] = ema12 - ema26
        res['macd_signal'] = res['macd'].ewm(span=9, adjust=False).mean()
        res['macd_histogram'] = res['macd'] - res['macd_signal']

        # ROC (Rate of Change) 10
        res['roc_10'] = res['Close'].pct_change(periods=10) * 100.0
        # Momentum 10
        res['momentum_10'] = res['Close'] - res['Close'].shift(10)

        return res

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, atr_period: int = 14, bb_period: int = 20, bb_std: float = 2.0) -> pd.DataFrame:
        res = df.copy()

        # ATR
        high_low = res['High'] - res['Low']
        high_pc = (res['High'] - res['Close'].shift(1)).abs()
        low_pc = (res['Low'] - res['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        res['atr'] = tr.rolling(window=atr_period).mean()

        # Historical Volatility (20-day annualized)
        log_ret = np.log(res['Close'] / res['Close'].shift(1))
        res['hist_vol_20'] = log_ret.rolling(window=20).std() * np.sqrt(252) * 100.0

        # Bollinger Bands
        sma = res['Close'].rolling(window=bb_period).mean()
        std = res['Close'].rolling(window=bb_period).std()
        res['bb_upper'] = sma + (bb_std * std)
        res['bb_lower'] = sma - (bb_std * std)
        res['bb_middle'] = sma
        res['bb_width'] = (res['bb_upper'] - res['bb_lower']) / (sma + 1e-10)

        return res

    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        if 'Volume' in res.columns:
            res['volume_sma_20'] = res['Volume'].rolling(window=20).mean()
            res['rvol'] = res['Volume'] / (res['volume_sma_20'] + 1e-10)
            res['volume_change'] = res['Volume'].pct_change()

            # OBV
            obv_direction = np.sign(res['Close'].diff()).fillna(0)
            res['obv'] = (obv_direction * res['Volume']).cumsum()
        else:
            res['volume_sma_20'] = np.nan
            res['rvol'] = np.nan
            res['volume_change'] = np.nan
            res['obv'] = np.nan
        return res

    @staticmethod
    def calculate_price_structure(df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()

        # Ensure MAs exist
        if 'sma_20' not in res.columns:
            res['sma_20'] = res['Close'].rolling(20).mean()
        if 'sma_50' not in res.columns:
            res['sma_50'] = res['Close'].rolling(50).mean()
        if 'sma_200' not in res.columns:
            res['sma_200'] = res['Close'].rolling(200).mean()

        res['dist_sma_20'] = (res['Close'] - res['sma_20']) / res['sma_20'] * 100.0
        res['dist_sma_50'] = (res['Close'] - res['sma_50']) / res['sma_50'] * 100.0
        res['dist_sma_200'] = (res['Close'] - res['sma_200']) / res['sma_200'] * 100.0

        res['high_52w'] = res['High'].rolling(252, min_periods=20).max()
        res['low_52w'] = res['Low'].rolling(252, min_periods=20).min()
        res['dist_high_52w'] = (res['Close'] - res['high_52w']) / res['high_52w'] * 100.0
        res['dist_low_52w'] = (res['Close'] - res['low_52w']) / res['low_52w'] * 100.0

        res['gap'] = (res['Open'] - res['Close'].shift(1)) / res['Close'].shift(1) * 100.0
        res['intraday_range'] = (res['High'] - res['Low']) / res['Low'] * 100.0

        # Trend strength (Linear slope over 20 days)
        res['trend_strength'] = (res['Close'] - res['Close'].shift(20)) / res['Close'].shift(20) * 100.0

        # Structural signals
        res['golden_cross'] = (res['sma_50'] > res['sma_200']) & (res['sma_50'].shift(1) <= res['sma_200'].shift(1))
        res['death_cross'] = (res['sma_50'] < res['sma_200']) & (res['sma_50'].shift(1) >= res['sma_200'].shift(1))
        res['breakout_20d'] = res['Close'] > res['High'].shift(1).rolling(20).max()
        res['breakdown_20d'] = res['Close'] < res['Low'].shift(1).rolling(20).min()

        return res

    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df_out = cls.calculate_moving_averages(df)
        df_out = cls.calculate_momentum(df_out)
        df_out = cls.calculate_volatility(df_out)
        df_out = cls.calculate_volume_indicators(df_out)
        df_out = cls.calculate_price_structure(df_out)
        return df_out
