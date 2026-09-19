import numpy as np
import pandas as pd
from typing import Dict, Any, List

class TechnicalIndicators:
    """Calculates quantitative technical indicators for price time-series."""

    @staticmethod
    def calculate_all(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates SMAs, EMAs, RSI, Stoch RSI, MACD, ATR, HV, Bollinger Bands,
        OBV, MA distances, 52-week high/low distances, trend strength, and signals.
        Expects df with columns ['date', 'open', 'high', 'low', 'close', 'volume'].
        """
        if df.empty or len(df) < 5:
            return df

        res = df.copy().sort_values('date').reset_index(drop=True)
        close = res['close'].astype(float)
        high = res['high'].astype(float) if 'high' in res.columns else close
        low = res['low'].astype(float) if 'low' in res.columns else close
        volume = res['volume'].astype(float) if 'volume' in res.columns else pd.Series(0, index=res.index)

        ind_cfg = config.get('indicators', {})

        # 1. Moving Averages
        sma_windows = set(ind_cfg.get('sma_windows', [5, 10, 20, 50, 100, 200])).union({20, 50, 200})
        for w in sorted(list(sma_windows)):
            res[f'sma_{w}'] = close.rolling(window=w, min_periods=1).mean()

        ema_windows = set(ind_cfg.get('ema_windows', [9, 20, 50, 200])).union({9, 20, 50, 200})
        for w in sorted(list(ema_windows)):
            res[f'ema_{w}'] = close.ewm(span=w, adjust=False, min_periods=1).mean()

        # 2. RSI & Stoch RSI
        rsi_p = ind_cfg.get('rsi_period', 14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.ewm(com=rsi_p - 1, min_periods=1).mean()
        avg_loss = loss.ewm(com=rsi_p - 1, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        res['rsi_14'] = 100 - (100 / (1 + rs))

        min_rsi = res['rsi_14'].rolling(window=rsi_p, min_periods=1).min()
        max_rsi = res['rsi_14'].rolling(window=rsi_p, min_periods=1).max()
        res['stoch_rsi'] = (res['rsi_14'] - min_rsi) / (max_rsi - min_rsi + 1e-10)

        # 3. MACD
        fast_w = ind_cfg.get('macd_fast', 12)
        slow_w = ind_cfg.get('macd_slow', 26)
        sig_w = ind_cfg.get('macd_signal', 9)
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        res['macd'] = ema_fast - ema_slow
        res['macd_signal'] = res['macd'].ewm(span=sig_w, adjust=False).mean()
        res['macd_hist'] = res['macd'] - res['macd_signal']

        # 4. Momentum & Rate of Change
        res['momentum_10d'] = close - close.shift(10)
        res['roc_10d'] = (close - close.shift(10)) / (close.shift(10) + 1e-10) * 100

        # 5. Volatility (ATR, HV, Bollinger Bands)
        atr_p = ind_cfg.get('atr_period', 14)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        res['atr_14'] = tr.rolling(window=atr_p, min_periods=1).mean()

        daily_returns = close.pct_change()
        res['volatility_20d'] = daily_returns.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100

        bb_p = ind_cfg.get('bollinger_period', 20)
        bb_std_mult = ind_cfg.get('bollinger_std', 2.0)
        bb_mid = close.rolling(window=bb_p, min_periods=1).mean()
        bb_std = close.rolling(window=bb_p, min_periods=1).std().fillna(0)
        res['bollinger_upper'] = bb_mid + (bb_std_mult * bb_std)
        res['bollinger_lower'] = bb_mid - (bb_std_mult * bb_std)
        res['bollinger_width'] = (res['bollinger_upper'] - res['bollinger_lower']) / (bb_mid + 1e-10)

        # 6. Volume Indicators
        if 'volume' in res.columns and (volume > 0).any():
            res['volume_sma_20'] = volume.rolling(window=20, min_periods=1).mean()
            res['relative_volume'] = volume / (res['volume_sma_20'] + 1e-10)
            obv = (np.sign(delta.fillna(0)) * volume).cumsum()
            res['obv'] = obv
        else:
            res['volume_sma_20'] = 0.0
            res['relative_volume'] = 1.0
            res['obv'] = 0.0

        # 7. Price Structure & Distances
        res['dist_sma_20_pct'] = (close - res['sma_20']) / (res['sma_20'] + 1e-10) * 100
        res['dist_sma_50_pct'] = (close - res['sma_50']) / (res['sma_50'] + 1e-10) * 100
        res['dist_sma_200_pct'] = (close - res['sma_200']) / (res['sma_200'] + 1e-10) * 100

        high_52w = high.rolling(window=252, min_periods=1).max()
        low_52w = low.rolling(window=252, min_periods=1).min()
        res['dist_52w_high_pct'] = (close - high_52w) / (high_52w + 1e-10) * 100
        res['dist_52w_low_pct'] = (close - low_52w) / (low_52w + 1e-10) * 100

        res['intraday_range_pct'] = (high - low) / (low + 1e-10) * 100
        res['gap_pct'] = (res['open'] - close.shift(1)) / (close.shift(1) + 1e-10) * 100

        # Crosses & Breakouts
        res['golden_cross'] = (res['sma_50'] > res['sma_200']) & (res['sma_50'].shift(1) <= res['sma_200'].shift(1))
        res['death_cross'] = (res['sma_50'] < res['sma_200']) & (res['sma_50'].shift(1) >= res['sma_200'].shift(1))
        res['breakout_20d'] = close > high.shift(1).rolling(window=20, min_periods=1).max()
        res['breakdown_20d'] = close < low.shift(1).rolling(window=20, min_periods=1).min()

        # Historical Returns
        res['return_1d'] = close.pct_change(1) * 100
        res['return_5d'] = close.pct_change(5) * 100
        res['return_20d'] = close.pct_change(20) * 100
        res['return_60d'] = close.pct_change(60) * 100
        res['return_252d'] = close.pct_change(252) * 100

        # Max Drawdown
        cum_max = close.cummax()
        drawdown = (close - cum_max) / cum_max * 100
        res['drawdown_pct'] = drawdown
        res['max_drawdown_252d_pct'] = drawdown.rolling(window=252, min_periods=1).min()

        return res
