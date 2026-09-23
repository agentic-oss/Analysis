import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.indicators.technical import calculate_technical_indicators

class FeatureStoreBuilder:
    """
    Builds ML-ready daily feature datasets and explicitly separated future return targets.

    Guarantees look-ahead bias protection:
    - Features at date T only use information up to date T.
    - Future return targets are placed in target variables (e.g., target_future_return_1d, ..., 60d).
    """
    def __init__(self, forecast_horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = forecast_horizons

    def build_feature_dataset(
        self,
        metal_df: pd.DataFrame,
        macro_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        ratio_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if metal_df.empty:
            return pd.DataFrame()

        # 1. Base technical indicators
        features = calculate_technical_indicators(metal_df)

        # 2. Merge Ratio indicators if provided
        if ratio_df is not None and not ratio_df.empty:
            sub_ratio = ratio_df[["timestamp", "gold_silver_ratio", "ratio_zscore"]].copy()
            features = pd.merge(features, sub_ratio, on="timestamp", how="left")

        # 3. Merge Macro indicators if provided
        if macro_dfs:
            for macro_name, m_df in macro_dfs.items():
                if not m_df.empty and "timestamp" in m_df.columns and "close" in m_df.columns:
                    sub_m = m_df[["timestamp", "close"]].copy().rename(columns={"close": f"{macro_name.lower()}_close"})
                    sub_m[f"{macro_name.lower()}_ret_1d"] = sub_m[f"{macro_name.lower()}_close"].pct_change() * 100.0
                    sub_m[f"{macro_name.lower()}_ret_20d"] = sub_m[f"{macro_name.lower()}_close"].pct_change(20) * 100.0
                    features = pd.merge(features, sub_m, on="timestamp", how="left")

        # 4. Merge Events if provided
        if events_df is not None and not events_df.empty and "timestamp" in events_df.columns:
            features = pd.merge(features, events_df, on="timestamp", how="left")

        # 5. Build Future Targets (Look-ahead targets stored separately)
        close = features["close"]
        for h in self.horizons:
            # Shift backwards so at row T we have return from T to T+h
            fwd_price = close.shift(-h)
            features[f"target_future_return_{h}d"] = (fwd_price - close) / close * 100.0
            features[f"target_future_direction_{h}d"] = (features[f"target_future_return_{h}d"] > 0).astype(int)

        return features
