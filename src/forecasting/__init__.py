import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_probabilistic_signals(analogue_stats: Dict[str, Any], instrument: str = "Gold") -> Dict[str, Any]:
    """
    Transforms historical analogue outcomes into forward probabilistic research signals.
    """
    signals = {}
    horizon_stats = analogue_stats.get("horizon_statistics", {})

    for h in [1, 3, 5, 10, 20, 60]:
        key = f"fwd_{h}d"
        stat = horizon_stats.get(key)
        if stat and stat.get("sample_count", 0) > 0:
            pos_prob = stat["positive_prob"]
            neg_prob = stat["negative_prob"]
            exp_ret = stat["mean_return"]
            med_ret = stat["median_return"]

            signals[f"{h}d"] = {
                "horizon_days": h,
                "positive_return_probability": round(pos_prob, 4),
                "negative_return_probability": round(neg_prob, 4),
                "expected_return": round(exp_ret, 4),
                "median_return": round(med_ret, 4),
                "sample_count": stat["sample_count"],
                "research_signal_type": "PROBABILISTIC_HISTORICAL_SIGNAL",
                "disclaimer": "All price predictions and forecasting outputs must strictly be framed as probabilistic research signals rather than guaranteed future prices."
            }

    return {
        "instrument": instrument,
        "horizons": signals
    }
