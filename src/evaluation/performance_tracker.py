import pandas as pd
import numpy as np
from typing import Dict, Any, List

class PredictionConfidenceScorer:
    """Calculates overall signal confidence score (0-100) and rationale components."""

    @staticmethod
    def calculate_confidence(
        analogue_pos_prob: float,
        model_agreement_ratio: float,
        trend_aligned: bool,
        macro_supportive: bool
    ) -> Dict[str, Any]:
        """
        Combines sample statistics, model agreement, technical trend, and macro support
        into an explainable 0-100 score.
        """
        score = 0.0
        reasons = []

        # 1. Analogue historical probability contribution (max 40 pts)
        prob_score = abs(analogue_pos_prob - 0.5) * 2.0 * 40.0
        score += prob_score
        reasons.append(f"Historical analogue return probability: {analogue_pos_prob*100:.1f}% (+{prob_score:.1f} pts)")

        # 2. Model agreement contribution (max 30 pts)
        model_score = model_agreement_ratio * 30.0
        score += model_score
        reasons.append(f"Model consensus agreement ratio: {model_agreement_ratio*100:.1f}% (+{model_score:.1f} pts)")

        # 3. Technical trend alignment (max 15 pts)
        if trend_aligned:
            score += 15.0
            reasons.append("Technical moving average trend aligned (+15.0 pts)")

        # 4. Macro regime support (max 15 pts)
        if macro_supportive:
            score += 15.0
            reasons.append("Macro environment supportive (+15.0 pts)")

        final_score = int(np.clip(score, 0, 100))

        return {
            "confidence_score": final_score,
            "reasons": reasons
        }

class ForecastPerformanceTracker:
    """Evaluates past predictions generated on date T against actual realized outcomes at T + Horizon."""

    @staticmethod
    def evaluate_predictions(predictions_df: pd.DataFrame, master_price_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compares predicted return/direction with realized future price returns.
        Maintains a continuously updated forecast evaluation database.
        """
        if predictions_df.empty or master_price_df.empty:
            return {"total_evaluated": 0, "accuracy_by_horizon": {}}

        merged = pd.merge(predictions_df, master_price_df, on="date", how="inner")
        if merged.empty:
            return {"total_evaluated": 0, "accuracy_by_horizon": {}}

        eval_records = []
        for i, row in merged.iterrows():
            date = row["date"]
            inst = row.get("instrument", "GOLD")
            horizon = row.get("horizon", "5d")
            pred_dir = row.get("predicted_direction", 1)

            fwd_ret_col = f"target_{inst.lower()}_future_ret_{horizon}"
            if fwd_ret_col in row and not pd.isna(row[fwd_ret_col]):
                actual_ret = row[fwd_ret_col]
                actual_dir = 1 if actual_ret > 0 else 0
                is_correct = (pred_dir == actual_dir)

                eval_records.append({
                    "date": date,
                    "instrument": inst,
                    "horizon": horizon,
                    "predicted_dir": pred_dir,
                    "actual_ret": actual_ret,
                    "is_correct": is_correct
                })

        eval_df = pd.DataFrame(eval_records)
        if eval_df.empty:
            return {"total_evaluated": 0, "accuracy_by_horizon": {}}

        accuracy_by_horizon = {}
        for h, grp in eval_df.groupby("horizon"):
            acc = grp["is_correct"].mean()
            accuracy_by_horizon[h] = {
                "sample_count": len(grp),
                "directional_accuracy": float(acc),
                "mean_actual_return": float(grp["actual_ret"].mean())
            }

        return {
            "total_evaluated": len(eval_df),
            "overall_accuracy": float(eval_df["is_correct"].mean()),
            "accuracy_by_horizon": accuracy_by_horizon
        }
