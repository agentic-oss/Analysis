import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculates SMAs, EMAs, RSI, Stochastic RSI, MACD, ATR, Historical Volatility, Bollinger Bands, Volume metrics, and Price Structure."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 5:
            return df

        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df.get("volume", pd.Series(0, index=df.index))

        # 1. Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{window}"] = c.rolling(window=window, min_periods=1).mean()
        for span in [9, 20, 50, 200]:
            df[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()

        # 2. Daily & Multi-period returns
        df["return_1d"] = c.pct_change(1)
        df["return_5d"] = c.pct_change(5)
        df["return_20d"] = c.pct_change(20)
        df["return_60d"] = c.pct_change(60)
        df["return_120d"] = c.pct_change(120)
        df["return_252d"] = c.pct_change(252)

        # 3. Momentum: RSI 14
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50)

        # Stochastic RSI
        min_rsi = df["rsi_14"].rolling(window=14, min_periods=1).min()
        max_rsi = df["rsi_14"].rolling(window=14, min_periods=1).max()
        stoch_rsi_denom = max_rsi - min_rsi
        df["stoch_rsi"] = np.where(stoch_rsi_denom == 0, 0.5, (df["rsi_14"] - min_rsi) / stoch_rsi_denom)

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ROC & Momentum
        df["roc_10"] = c.pct_change(10) * 100.0
        df["momentum_10"] = c.diff(10)

        # 4. Volatility: ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

        # Historical Volatility (20d annualized)
        log_ret = np.log(c / c.shift(1))
        df["volatility_20d"] = log_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # Bollinger Bands (20d, 2 std)
        bb_mid = df["sma_20"]
        bb_std = c.rolling(window=20, min_periods=1).std()
        df["bb_upper"] = bb_mid + (bb_std * 2)
        df["bb_lower"] = bb_mid - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

        # Drawdowns
        rolling_max_252 = c.rolling(window=252, min_periods=1).max()
        df["drawdown"] = (c - rolling_max_252) / rolling_max_252
        df["max_drawdown_252d"] = df["drawdown"].rolling(window=252, min_periods=1).min()

        # 5. Volume Analysis
        df["volume_sma_20"] = v.rolling(window=20, min_periods=1).mean()
        df["relative_volume"] = v / df["volume_sma_20"].replace(0, np.nan)
        df["obv"] = (np.sign(c.diff().fillna(0)) * v).cumsum()

        # 6. Price Structure & Distances
        df["dist_sma_20"] = (c - df["sma_20"]) / df["sma_20"].replace(0, np.nan)
        df["dist_sma_50"] = (c - df["sma_50"]) / df["sma_50"].replace(0, np.nan)
        df["dist_sma_200"] = (c - df["sma_200"]) / df["sma_200"].replace(0, np.nan)

        high_52w = c.rolling(window=252, min_periods=1).max()
        low_52w = c.rolling(window=252, min_periods=1).min()
        df["dist_52w_high"] = (c - high_52w) / high_52w.replace(0, np.nan)
        df["dist_52w_low"] = (c - low_52w) / low_52w.replace(0, np.nan)

        df["intraday_range"] = (h - l) / l.replace(0, np.nan)
        df["gap"] = (df["open"] - c.shift(1)) / c.shift(1).replace(0, np.nan)

        # Technical Triggers
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))

        return df
