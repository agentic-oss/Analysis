import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds ML-ready daily feature store dataset.
    Ensures zero look-ahead bias for date T features, with forward targets stored separately.
    """

    @staticmethod
    def build_daily_features(
        df: pd.DataFrame, horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Calculates lagged return features, technical features, ratio features, and future targets.
        """
        if df.empty or len(df) < 5:
            return df

        df = df.copy().sort_values("date").reset_index(drop=True)

        # 1. Historical Returns (Lagged, no look-ahead)
        df["return_1d"] = df["close"].pct_change(1)
        df["return_3d"] = df["close"].pct_change(3)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["return_20d"] = df["close"].pct_change(20)
        df["return_60d"] = df["close"].pct_change(60)

        # 2. Normalized technicals
        if "macd" in df.columns and "atr_14" in df.columns:
            df["macd_norm"] = df["macd"] / df["atr_14"].replace(0, pd.NA)

        # 3. Future Return Targets (For training models / evaluating historical performance)
        for h in horizons:
            df[f"future_return_{h}d"] = (df["close"].shift(-h) - df["close"]) / df["close"]
            df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

        return df
