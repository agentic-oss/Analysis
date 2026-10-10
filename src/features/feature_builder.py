import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalAnalysis

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    Builds ML-ready daily feature datasets and target variables for precious metals.
    Ensures zero look-ahead bias: features available on date T use strictly information available by date T.
    Future targets are appended as separate target columns.
    """

    @staticmethod
    def build_feature_dataset(symbol: str, price_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame], gsr_series: Optional[pd.Series] = None) -> pd.DataFrame:
        if price_df.empty or "Close" not in price_df.columns:
            return pd.DataFrame()

        # 1. Technical Indicators
        df = TechnicalAnalysis.calculate_all(price_df)

        # Core Price & Returns
        close = df["Close"]
        df["price"] = close
        df["instrument"] = symbol
        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_20d"] = close.pct_change(20)
        df["return_60d"] = close.pct_change(60)

        # 2. Add Macro Features
        for m_name, m_df in macro_dfs.items():
            if not m_df.empty and "Close" in m_df.columns:
                m_close = m_df["Close"].reindex(df.index, method="ffill")
                df[f"{m_name.lower()}_close"] = m_close
                df[f"{m_name.lower()}_return_1d"] = m_close.pct_change(1)
                df[f"{m_name.lower()}_return_20d"] = m_close.pct_change(20)

        # 3. Add Gold/Silver Ratio Features
        if gsr_series is not None and not gsr_series.empty:
            reindexed_gsr = gsr_series.reindex(df.index, method="ffill")
            df["gold_silver_ratio"] = reindexed_gsr
            mean_gsr = reindexed_gsr.rolling(252, min_periods=30).mean()
            std_gsr = reindexed_gsr.rolling(252, min_periods=30).std().replace(0, np.nan)
            df["gold_silver_ratio_zscore"] = (reindexed_gsr - mean_gsr) / std_gsr

        # 4. Target Variables (Strictly Forward Looking)
        for h in [1, 3, 5, 10, 20, 60]:
            df[f"future_return_{h}d"] = close.pct_change(h).shift(-h)
            df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

        return df
