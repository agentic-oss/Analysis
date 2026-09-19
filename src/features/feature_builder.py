import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class FeatureBuilder:
    """Builds ML-ready daily feature store datasets with future target horizons and event features."""

    @staticmethod
    def build_feature_dataset(
        price_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        ratio_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> pd.DataFrame:
        if price_df.empty:
            return pd.DataFrame()

        df = price_df.copy().sort_values('date').reset_index(drop=True)

        # Merge macro data if available
        if macro_df is not None and not macro_df.empty:
            df = pd.merge(df, macro_df, on='date', how='left', suffixes=('', '_macro'))

        # Merge ratio data if available
        if ratio_df is not None and not ratio_df.empty:
            if 'ratio' in ratio_df.columns:
                df = pd.merge(df, ratio_df[['date', 'ratio']], on='date', how='left')

        # Event-based features
        if events_df is not None and not events_df.empty:
            df = pd.merge(df, events_df, on='date', how='left')
            if 'days_to_event' not in df.columns:
                df['days_to_event'] = -1
            if 'days_since_event' not in df.columns:
                df['days_since_event'] = -1
        else:
            df['days_to_event'] = -1
            df['days_since_event'] = -1

        # Target Future Forward Returns (strictly for training/eval - targets stored alongside or shift -h)
        for h in horizons:
            future_close = df['close'].shift(-h)
            df[f'future_return_{h}d'] = (future_close - df['close']) / (df['close'] + 1e-10) * 100
            df[f'future_direction_{h}d'] = (df[f'future_return_{h}d'] > 0).astype(int)

        return df
