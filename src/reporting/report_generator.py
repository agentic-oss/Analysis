import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates daily Markdown research reports and machine-readable JSON outputs."""

    def __init__(self, reports_dir: str = "reports/daily", json_dir: str = "data/analysis/daily"):
        self.reports_dir = reports_dir
        self.json_dir = json_dir
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

    def generate_daily_report(
        self,
        date_str: str,
        gold_summary: Dict[str, Any],
        silver_summary: Dict[str, Any],
        ratio_summary: Dict[str, Any],
        macro_summary: Dict[str, Any],
        scores: Dict[str, Any],
        forecast_stats: Dict[str, Any],
        quality_metrics: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Generates:
        1. Markdown research report: reports/daily/YYYY-MM-DD.md
        2. Machine-readable JSON output: data/analysis/daily/YYYY-MM-DD.json
        """
        # Save JSON output
        json_payload = {
            "date": date_str,
            "gold": gold_summary,
            "silver": silver_summary,
            "gold_silver_ratio": ratio_summary,
            "macro": macro_summary,
            "scores": scores,
            "forecast_statistics": forecast_stats,
            "data_quality": quality_metrics
        }
        json_path = os.path.join(self.json_dir, f"{date_str}.json")
        with open(json_path, "w") as f:
            json.dump(json_payload, f, indent=2)

        # Build Markdown Report
        md_content = f"""# Precious Metals Daily Research Report ({date_str})

## Executive Summary

**Gold:**
- Price: ${gold_summary.get('price', 0.0):.2f} USD/oz | ₹{gold_summary.get('inr_price_10g', 0.0):,.0f} INR/10g
- Daily Move: {gold_summary.get('daily_move_pct', 0.0)*100:+.2f}%
- Weekly Move: {gold_summary.get('weekly_move_pct', 0.0)*100:+.2f}%
- Trend: {gold_summary.get('trend_regime', 'N/A')}
- Volatility: {gold_summary.get('volatility_regime', 'N/A')}
- Composite Score: {scores.get('gold_scores', {}).get('composite_score', 50)}/100

**Silver:**
- Price: ${silver_summary.get('price', 0.0):.2f} USD/oz | ₹{silver_summary.get('inr_price_kg', 0.0):,.0f} INR/kg
- Daily Move: {silver_summary.get('daily_move_pct', 0.0)*100:+.2f}%
- Weekly Move: {silver_summary.get('weekly_move_pct', 0.0)*100:+.2f}%
- Trend: {silver_summary.get('trend_regime', 'N/A')}
- Volatility: {silver_summary.get('volatility_regime', 'N/A')}
- Composite Score: {scores.get('silver_scores', {}).get('composite_score', 50)}/100

---

## Gold Analysis
- Technical RSI (14): {gold_summary.get('rsi', 50.0):.1f}
- Key Support (200 SMA): ${gold_summary.get('sma_200', 0.0):.2f}
- Key Resistance (52W High): ${gold_summary.get('high_52w', 0.0):.2f}

---

## Silver & Relative-Value Analysis
- Gold/Silver Ratio: {ratio_summary.get('current_ratio', 0.0):.2f}
- 252-day Z-Score: {ratio_summary.get('zscore_252d', 0.0):.2f}
- 252-day Percentile Rank: {ratio_summary.get('percentile_252d', 50.0):.1f}%

---

## Macro Environment
- DXY (US Dollar Index): {macro_summary.get('dxy', 0.0):.2f}
- US 10-Year Yield: {macro_summary.get('us10y', 0.0):.2f}%
- Crude Oil: ${macro_summary.get('oil', 0.0):.2f}
- Macro Regime: **{macro_summary.get('macro_regime', 'Neutral')}**

---

## Forecast Research (Probabilistic Signals)
*Note: Expressed strictly as historical conditional probabilities, not guaranteed prices.*

- **10-Day Gold Direction Positive Return Frequency:** {forecast_stats.get('gold_10d', {}).get('pos_prob', 0.5)*100:.1f}%
- **10-Day Gold Expected Return (Median):** {forecast_stats.get('gold_10d', {}).get('median_return', 0.0)*100:+.2f}%

---

## Risk Disclaimer & Data Quality
- Data Status: Valid ({quality_metrics.get('valid_records', 0)} records)
- Warnings: {len(quality_metrics.get('warnings', []))}
"""
        md_path = os.path.join(self.reports_dir, f"{date_str}.md")
        with open(md_path, "w") as f:
            f.write(md_content)

        return md_path, json_path
