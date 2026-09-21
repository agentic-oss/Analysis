import pandas as pd
import numpy as np
from typing import Dict, List, Any


class ProbabilisticForwardEngine:
    """Generates multi-horizon probabilistic forward research signals based on historical distribution and analogue statistics."""

    DISCLAIMER = (
        "RESEARCH NOTICE: All price predictions and forecasting outputs must strictly be framed as "
        "probabilistic research signals based on historical market statistics and model projections, "
        "not guaranteed future prices or financial advice."
    )

    @staticmethod
    def generate_forward_signals(
        df: pd.DataFrame,
        analogue_stats: Dict[str, Any] = None,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        if df is None or df.empty or len(df) < 60:
            return {
                "disclaimer": ProbabilisticForwardEngine.DISCLAIMER,
                "horizons": {}
            }

        df_sorted = df.sort_values("date").reset_index(drop=True)
        latest_price = float(df_sorted.iloc[-1]["close"])

        horizon_signals = {}

        for h in horizons:
            # Historical unconditional distribution
            past_rets = df_sorted["close"].pct_change(h).dropna().values

            # Combine unconditional past returns with conditional analogue returns if available
            if analogue_stats and f"{h}d" in analogue_stats:
                stat_h = analogue_stats[f"{h}d"]
                pos_prob = stat_h["positive_prob_pct"]
                neg_prob = stat_h["negative_prob_pct"]
                mean_ret = stat_h["mean_return_pct"]
                median_ret = stat_h["median_return_pct"]
                min_ret = stat_h["max_loss_pct"]
                max_ret = stat_h["max_gain_pct"]
            else:
                pos_prob = round(float(np.mean(past_rets > 0)) * 100.0, 1) if len(past_rets) > 0 else 50.0
                neg_prob = round(100.0 - pos_prob, 1)
                mean_ret = round(float(np.mean(past_rets)) * 100.0, 2) if len(past_rets) > 0 else 0.0
                median_ret = round(float(np.median(past_rets)) * 100.0, 2) if len(past_rets) > 0 else 0.0
                min_ret = round(float(np.min(past_rets)) * 100.0, 2) if len(past_rets) > 0 else 0.0
                max_ret = round(float(np.max(past_rets)) * 100.0, 2) if len(past_rets) > 0 else 0.0

            vol_est = round(float(np.std(past_rets)) * 100.0, 2) if len(past_rets) > 0 else 1.0

            if pos_prob >= 60.0:
                bias = "Bullish Signal"
            elif pos_prob <= 40.0:
                bias = "Bearish Signal"
            else:
                bias = "Neutral / Mixed Signal"

            horizon_signals[f"{h}d"] = {
                "horizon_trading_days": h,
                "probability_positive_pct": pos_prob,
                "probability_negative_pct": neg_prob,
                "expected_return_pct": mean_ret,
                "median_return_pct": median_ret,
                "expected_volatility_pct": vol_est,
                "historical_min_return_pct": min_ret,
                "historical_max_return_pct": max_ret,
                "signal_bias": bias,
                "implied_median_target_price": round(latest_price * (1.0 + median_ret / 100.0), 2)
            }

        return {
            "disclaimer": ProbabilisticForwardEngine.DISCLAIMER,
            "latest_price": round(latest_price, 2),
            "horizons": horizon_signals
        }
