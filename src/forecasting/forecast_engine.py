import pandas as pd
import numpy as np
from typing import Dict, Any, List


class ProbabilisticForecastEngine:
    """
    Generates probabilistic forward research signals across multiple horizons (1D, 3D, 5D, 10D, 20D, 60D).
    All forecasts are strictly framed as research signals derived from historical empirical distributions.
    """

    @staticmethod
    def generate_forecasts(
        historical_df: pd.DataFrame,
        analogue_output: Dict[str, Any],
        horizons: List[int] = [1, 3, 5, 10, 20, 60],
    ) -> Dict[str, Any]:
        disclaimer = (
            "These statistics represent historical empirical distributions under comparable "
            "market conditions. They are probabilistic research signals and NOT guaranteed future price forecasts."
        )

        forecasts = {}
        analogue_stats = analogue_output.get("forward_statistics", {})

        for h in horizons:
            key = f"{h}d"
            a_stat = analogue_stats.get(key, {})

            if a_stat and a_stat.get("sample_count", 0) > 0:
                pos_prob = a_stat["pos_prob_pct"]
                neg_prob = a_stat["neg_prob_pct"]
                expected_return = a_stat["mean_return_pct"]
                median_return = a_stat["median_return_pct"]
                max_gain = a_stat["max_gain_pct"]
                max_loss = a_stat["max_loss_pct"]
            else:
                # Fallback to general historical unconditional statistics
                if historical_df is not None and not historical_df.empty:
                    ret_h = historical_df["Close"].pct_change(h).dropna() * 100.0
                    pos_prob = round(float((ret_h > 0).mean() * 100), 1)
                    neg_prob = round(100.0 - pos_prob, 1)
                    expected_return = round(float(ret_h.mean()), 2)
                    median_return = round(float(ret_h.median()), 2)
                    max_gain = round(float(ret_h.max()), 2)
                    max_loss = round(float(ret_h.min()), 2)
                else:
                    pos_prob = 50.0
                    neg_prob = 50.0
                    expected_return = 0.0
                    median_return = 0.0
                    max_gain = 0.0
                    max_loss = 0.0

            # Calculate expected volatility for horizon
            if historical_df is not None and not historical_df.empty:
                daily_std = float(historical_df["Close"].pct_change().std() * 100.0)
                horizon_vol = round(daily_std * np.sqrt(h), 2)
            else:
                horizon_vol = 1.0

            forecasts[key] = {
                "horizon_trading_days": h,
                "probability_positive_return_pct": pos_prob,
                "probability_negative_return_pct": neg_prob,
                "expected_mean_return_pct": expected_return,
                "median_return_pct": median_return,
                "expected_horizon_volatility_pct": horizon_vol,
                "historical_min_pct": max_loss,
                "historical_max_pct": max_gain,
            }

        return {
            "disclaimer": disclaimer,
            "forecasts_by_horizon": forecasts,
        }
