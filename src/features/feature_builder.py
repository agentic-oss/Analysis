import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.indicators.technical import TechnicalIndicators
from src.features.events import EventFeatureBuilder


class FeatureBuilder:
    """Builds ML-ready daily feature store with explicit separation of state features and forward targets."""

    @staticmethod
    def build_feature_dataset(
        metal_df: pd.DataFrame,
        instrument_symbol: str,
        macro_summary: Dict[str, Any] = None,
        ratio_df: pd.DataFrame = None,
        events_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        if metal_df is None or metal_df.empty:
            return pd.DataFrame()

        # Calculate technical indicators
        df = TechnicalIndicators.calculate_all(metal_df)
        df["instrument"] = instrument_symbol

        # Attach ratio metrics if available
        if ratio_df is not None and not ratio_df.empty:
            r_sub = ratio_df[["Date", "ratio", "ratio_zscore", "ratio_percentile"]].copy()
            r_sub["Date"] = pd.to_datetime(r_sub["Date"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = pd.merge(df, r_sub, on="Date", how="left")
        else:
            df["ratio"] = np.nan
            df["ratio_zscore"] = np.nan
            df["ratio_percentile"] = np.nan

        # Attach macro features if available
        if macro_summary:
            for k, v in macro_summary.items():
                if isinstance(v, dict) and "latest" in v:
                    df[f"macro_{k}_latest"] = v.get("latest")
                    df[f"macro_{k}_change"] = v.get("pct_change")

        # Attach event features
        df = EventFeatureBuilder.attach_event_features(df, events_df)

        # Build forward target returns (1D, 3D, 5D, 10D, 20D, 60D)
        for h in [1, 3, 5, 10, 20, 60]:
            df[f"future_return_{h}d"] = df["Close"].pct_change(h).shift(-h)
            df[f"future_dir_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

        return df
