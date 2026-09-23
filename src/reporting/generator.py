import os
import json
import pandas as pd
from typing import Dict, Any, List

class ReportGenerator:
    """
    Generates daily research reports (Markdown in reports/daily/YYYY-MM-DD.md),
    machine-readable JSON in data/analysis/daily/YYYY-MM-DD.json,
    and alert files in data/analysis/alerts/YYYY-MM-DD.json.
    """
    def __init__(self, reports_dir: str = "reports/daily", analysis_dir: str = "data/analysis/daily", alerts_dir: str = "data/analysis/alerts"):
        self.reports_dir = reports_dir
        self.analysis_dir = analysis_dir
        self.alerts_dir = alerts_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.alerts_dir, exist_ok=True)

    def generate_daily_markdown_report(self, date_str: str, analysis_data: Dict[str, Any]) -> str:
        gold = analysis_data.get("gold", {})
        silver = analysis_data.get("silver", {})
        macro = analysis_data.get("macro", {})
        ratio = analysis_data.get("gold_silver_ratio", {})
        mkt_regime = analysis_data.get("market_regime", {})
        dq = analysis_data.get("data_quality", {})

        inputs = macro.get("inputs", {})
        vix_val = inputs.get("vix_level", "N/A")
        rates_val = inputs.get("rates_change_60d", "N/A")
        dxy_val = inputs.get("dxy_return_60d", "N/A")

        md = f"""# Precious Metals Daily Research Report

**Date:** {date_str}

> **Notice:** All forecasting outputs and return probabilities are probabilistic research signals based on historical market conditions, not guaranteed future outcomes.

---

## Executive Summary

### Gold
* **Current Price:** ${gold.get('price', 0.0):,.2f} USD
* **Daily Move:** {gold.get('daily_return_pct', 0.0):+.2f}%
* **Weekly Move:** {gold.get('weekly_return_pct', 0.0):+.2f}%
* **Monthly Move:** {gold.get('monthly_return_pct', 0.0):+.2f}%
* **Trend Regime:** {mkt_regime.get('gold', {}).get('trend_regime', 'N/A')}
* **Volatility Regime:** {mkt_regime.get('gold', {}).get('volatility_regime', 'N/A')}
* **Composite Technical/Research Score:** {gold.get('composite_score', 0.0)}/100

### Silver
* **Current Price:** ${silver.get('price', 0.0):,.2f} USD
* **Daily Move:** {silver.get('daily_return_pct', 0.0):+.2f}%
* **Weekly Move:** {silver.get('weekly_return_pct', 0.0):+.2f}%
* **Monthly Move:** {silver.get('monthly_return_pct', 0.0):+.2f}%
* **Trend Regime:** {mkt_regime.get('silver', {}).get('trend_regime', 'N/A')}
* **Volatility Regime:** {mkt_regime.get('silver', {}).get('volatility_regime', 'N/A')}
* **Composite Technical/Research Score:** {silver.get('composite_score', 0.0)}/100

---

## Macro Environment & Cross-Market Relationships
* **Primary Macro Regime:** {macro.get('primary_macro_regime', 'N/A')}
* **DXY 60-Day Change:** {f'{dxy_val:+.2f}%' if isinstance(dxy_val, (int, float)) else dxy_val}
* **60-Day Rate Trend Change:** {rates_val}
* **VIX Volatility Index Level:** {vix_val}

---

## Relative Value: Gold/Silver Ratio
* **Current Ratio:** {ratio.get('current_ratio', 'N/A')}
* **Ratio Z-Score (252d):** {ratio.get('current_zscore', 'N/A')}
* **Historical Percentile:** {ratio.get('historical_percentile', 'N/A')}%
* **Long-Term Mean:** {ratio.get('long_term_mean', 'N/A')}

---

## Forecast Research & Probabilities

### Gold Horizon Forecasts
"""
        gold_fwd = gold.get("forward_statistics", {})
        for h, stats in gold_fwd.items():
            md += f"- **{h}:** Prob Positive: {stats.get('probability_positive', 'N/A')}%, Expected Return: {stats.get('expected_return', 'N/A')}%, Agreement: {stats.get('model_agreement', 'N/A')}\n"

        md += "\n### Silver Horizon Forecasts\n"
        silver_fwd = silver.get("forward_statistics", {})
        for h, stats in silver_fwd.items():
            md += f"- **{h}:** Prob Positive: {stats.get('probability_positive', 'N/A')}%, Expected Return: {stats.get('expected_return', 'N/A')}%, Agreement: {stats.get('model_agreement', 'N/A')}\n"

        md += f"""
---

## Key Risks & Thesis Invalidation
* Central bank interest rate surprises or unexpected macroeconomic data releases.
* Geopolitical escalation or abrupt shifts in global USD currency strength.
* Extreme volatility expansion violating recent range assumptions.

---

## Data Quality Summary
* **Total Valid Observations:** {dq.get('valid_records', 0)}
* **Suspicious Records:** {dq.get('suspicious_records', 0)}
* **Errors / Invalid Records:** {dq.get('invalid_records', 0)}
"""

        filepath = os.path.join(self.reports_dir, f"{date_str}.md")
        with open(filepath, "w") as f:
            f.write(md)

        return filepath

    def generate_daily_json_output(self, date_str: str, analysis_data: Dict[str, Any]) -> str:
        filepath = os.path.join(self.analysis_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
        return filepath

    def detect_and_save_alerts(self, date_str: str, analysis_data: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
        alerts = []
        gold = analysis_data.get("gold", {})
        silver = analysis_data.get("silver", {})
        ratio = analysis_data.get("gold_silver_ratio", {})

        # Price jump alerts
        gold_1d = abs(gold.get("daily_return_pct", 0.0))
        if gold_1d >= thresholds.get("price_change_pct_1d", 2.0):
            alerts.append({"type": "GOLD_PRICE_MOVE", "severity": "HIGH", "message": f"Gold moved {gold.get('daily_return_pct'):+.2f}% in 1 day"})

        silver_1d = abs(silver.get("daily_return_pct", 0.0))
        if silver_1d >= thresholds.get("price_change_pct_1d", 2.0):
            alerts.append({"type": "SILVER_PRICE_MOVE", "severity": "HIGH", "message": f"Silver moved {silver.get('daily_return_pct'):+.2f}% in 1 day"})

        # Gold/Silver ratio extreme alert
        gsr_z = abs(ratio.get("current_zscore", 0.0) or 0.0)
        if gsr_z >= thresholds.get("gsr_zscore_extreme", 2.0):
            alerts.append({"type": "GSR_EXTREME", "severity": "MEDIUM", "message": f"Gold/Silver ratio Z-score reached extreme level {ratio.get('current_zscore')}"})

        alert_payload = {
            "date": date_str,
            "alerts_count": len(alerts),
            "alerts": alerts
        }

        filepath = os.path.join(self.alerts_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(alert_payload, f, indent=2)

        return filepath
