import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class HistoricalAnalogueEngine:
    """
    Finds historical market conditions similar to current market state and computes
    forward 1D, 3D, 5D, 10D, 20D, 60D return statistics (without look-ahead bias).
    """

    FEATURE_COLS = [
        'rsi_14', 'macd', 'dist_sma_20', 'dist_sma_50', 'dist_sma_200',
        'hist_vol_20', 'ratio_zscore', 'dxy_return_20d', 'us10y_change_20d',
        'vix_level', 'oil_return_20d', 'sp500_return_20d'
    ]

    def __init__(self, forward_horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.forward_horizons = forward_horizons

    def find_analogues(
        self,
        feature_df: pd.DataFrame,
        current_date: str,
        top_n: int = 15,
        min_historical_buffer_days: int = 65
    ) -> Dict[str, Any]:
        """
        Given daily feature dataframe and current date T, compares T's feature vector
        against all past dates t <= T - min_historical_buffer_days (preventing look-ahead bias).
        """
        if feature_df.empty or current_date not in feature_df.index:
            return {"error": f"Current date {current_date} not in feature dataset."}

        target_dt = pd.to_datetime(current_date)
        # Ensure available features
        available_cols = [c for c in self.FEATURE_COLS if c in feature_df.columns]
        if len(available_cols) == 0:
            return {"error": "No matching feature columns found in feature_df."}

        # Filter historical candidate set (strictly past observations with enough forward window)
        max_past_dt = target_dt - pd.Timedelta(days=min_historical_buffer_days)
        past_df = feature_df[feature_df.index <= max_past_dt].copy()

        if len(past_df) < 30:
            return {
                "sample_count": 0,
                "warning": "Insufficient historical data for analogue matching.",
                "forward_statistics": {}
            }

        # Standardize features across historical + current vector
        combined_features = pd.concat([past_df[available_cols], feature_df.loc[[target_dt], available_cols]])
        scaler = StandardScaler()
        scaled_mat = scaler.fit_transform(combined_features.fillna(0))

        current_vector = scaled_mat[-1]
        past_matrix = scaled_mat[:-1]

        # Calculate Euclidean distances
        distances = np.linalg.norm(past_matrix - current_vector, axis=1)
        similarity_scores = 1.0 / (1.0 + distances)

        past_df['similarity_score'] = similarity_scores
        top_analogues = past_df.sort_values(by='similarity_score', ascending=False).head(top_n)

        # Compute forward returns for top analogues
        forward_stats = {}
        for h in self.forward_horizons:
            # Look up actual price forward return in original dataset
            # Forward return = (price[t+h] - price[t]) / price[t]
            fwd_col = f'future_return_{h}d'
            if fwd_col in top_analogues.columns:
                returns = top_analogues[fwd_col].dropna()
            else:
                # Calculate directly from Close price
                close_series = feature_df['price'] if 'price' in feature_df.columns else feature_df['Close']
                returns_list = []
                for dt in top_analogues.index:
                    try:
                        idx_loc = close_series.index.get_loc(dt)
                        if idx_loc + h < len(close_series):
                            p0 = close_series.iloc[idx_loc]
                            ph = close_series.iloc[idx_loc + h]
                            returns_list.append((ph - p0) / p0)
                    except KeyError:
                        pass
                returns = pd.Series(returns_list)

            if len(returns) > 0:
                pos_prob = float((returns > 0).mean())
                neg_prob = float((returns < 0).mean())
                mean_ret = float(returns.mean())
                med_ret = float(returns.median())
                std_ret = float(returns.std()) if len(returns) > 1 else 0.0
                min_ret = float(returns.min())
                max_ret = float(returns.max())

                forward_stats[f'{h}d'] = {
                    "sample_count": len(returns),
                    "mean_return": round(mean_ret, 4),
                    "median_return": round(med_ret, 4),
                    "positive_probability": round(pos_prob, 4),
                    "negative_probability": round(neg_prob, 4),
                    "expected_volatility": round(std_ret, 4),
                    "max_gain": round(max_ret, 4),
                    "max_loss": round(min_ret, 4),
                    "percentiles": {
                        "p10": round(float(np.percentile(returns, 10)), 4),
                        "p25": round(float(np.percentile(returns, 25)), 4),
                        "p50": round(float(np.percentile(returns, 50)), 4),
                        "p75": round(float(np.percentile(returns, 75)), 4),
                        "p90": round(float(np.percentile(returns, 90)), 4)
                    }
                }

        top_analogues_summary = []
        for idx, row in top_analogues.iterrows():
            top_analogues_summary.append({
                "date": idx.strftime("%Y-%m-%d"),
                "similarity_score": round(float(row['similarity_score']), 4)
            })

        return {
            "current_date": current_date,
            "sample_count": len(top_analogues),
            "top_analogues": top_analogues_summary,
            "forward_statistics": forward_stats
        }
