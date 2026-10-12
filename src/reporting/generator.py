import os
import json
from typing import Dict, Any

class ReportGenerator:
    """Generates daily Markdown research reports and machine-readable JSON reports."""

    def __init__(
        self,
        reports_dir: str = "reports/daily",
        analysis_json_dir: str = "data/analysis/daily"
    ):
        self.reports_dir = reports_dir
        self.analysis_json_dir = analysis_json_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.analysis_json_dir, exist_ok=True)

    def generate_json_report(self, date_str: str, report_payload: Dict[str, Any]) -> str:
        p = os.path.join(self.analysis_json_dir, f"{date_str}.json")
        with open(p, "w") as f:
            json.dump(report_payload, f, indent=2)
        return p

    def generate_markdown_report(self, date_str: str, payload: Dict[str, Any]) -> str:
        gold = payload.get("gold", {})
        silver = payload.get("silver", {})
        macro = payload.get("macro", {})
        gsr = payload.get("gold_silver_ratio", {})
        analogues = payload.get("historical_analogues", {})
        forecasts = payload.get("forecasts", {})
        quality = payload.get("data_quality", {})

        md = f"""# Precious Metals Daily Research Report ({date_str})

> **Disclaimer**: All price predictions and forecasting outputs must strictly be framed as probabilistic research signals rather than guaranteed future prices.

---

## Executive Summary

### Gold Overview
* **Current Price**: ${gold.get('price', 0.0):,.2f} USD/oz
* **1-Day Move**: {gold.get('return_1d_pct', 0.0):+.2f}%
* **5-Day Move**: {gold.get('return_5d_pct', 0.0):+.2f}%
* **20-Day Move**: {gold.get('return_20d_pct', 0.0):+.2f}%
* **Trend & Volatility**: {gold.get('market_regime', 'N/A')}
* **Research Bias**: {gold.get('research_bias', 'Neutral')}
* **Composite Score**: {gold.get('scores', {}).get('composite_score', 50.0)}/100

### Silver Overview
* **Current Price**: ${silver.get('price', 0.0):,.2f} USD/oz
* **1-Day Move**: {silver.get('return_1d_pct', 0.0):+.2f}%
* **5-Day Move**: {silver.get('return_5d_pct', 0.0):+.2f}%
* **20-Day Move**: {silver.get('return_20d_pct', 0.0):+.2f}%
* **Trend & Volatility**: {silver.get('market_regime', 'N/A')}
* **Research Bias**: {silver.get('research_bias', 'Neutral')}
* **Composite Score**: {silver.get('scores', {}).get('composite_score', 50.0)}/100

---

## Gold Analysis
* **Technical Structure**: RSI(14) = {gold.get('rsi_14', 50.0):.1f}, MACD = {gold.get('macd', 0.0):.2f}, Dist SMA200 = {gold.get('dist_sma_200', 0.0):+.1f}%
* **Key Support**: ${gold.get('sma_200', 0.0):,.2f} (200 DMA)
* **Key Resistance**: ${gold.get('sma_20', 0.0):,.2f} (20 DMA)
* **Forward Probabilities (10D)**: Positive Return Probability = {forecasts.get('gold_10d', {}).get('prob_positive_pct', 50.0):.1f}%

---

## Silver Analysis
* **Technical Structure**: RSI(14) = {silver.get('rsi_14', 50.0):.1f}, MACD = {silver.get('macd', 0.0):.2f}
* **Gold/Silver Ratio**: {gsr.get('current_ratio', 0.0):.2f} (Z-Score: {gsr.get('z_score', 0.0):+.2f}, Percentile: {gsr.get('percentile', 50.0):.1f}%)
* **Relative Value Thesis**: {gsr.get('relative_value_bias', 'Neutral')}

---

## Macro Environment
* **Macro Regime**: **{macro.get('regime', 'Neutral')}**
* **DXY 20D Return**: {macro.get('inputs', {}).get('dxy_return_20d', 0.0)*100:+.2f}%
* **US 10Y Yield Change (20D)**: {macro.get('inputs', {}).get('us10y_change_20d', 0.0):+.2f}%
* **VIX Level**: {macro.get('inputs', {}).get('vix_level', 15.0):.1f}

---

## Historical Analogue Analysis
Top {analogues.get('top_k', 0)} most similar historical market states detected:
"""
        for a in analogues.get("analogues", []):
            md += f"- **Date**: {a.get('date')} | Similarity: {a.get('similarity_score')}% | RSI: {a.get('rsi_14')} | GSR: {a.get('gold_silver_ratio')}\n"

        md += f"""
---

## Forecast Research Signals

| Horizon | Prob Positive (%) | Expected Mean Return (%) | Median Return (%) | Confidence Score |
| :--- | :--- | :--- | :--- | :--- |
"""
        for h in ["1d", "3d", "5d", "10d", "20d", "60d"]:
            f_stat = forecasts.get(f"gold_{h}", {})
            md += f"| **{h.upper()}** | {f_stat.get('prob_positive_pct', 50.0):.1f}% | {f_stat.get('expected_mean_return_pct', 0.0):+.2f}% | {f_stat.get('median_return_pct', 0.0):+.2f}% | {f_stat.get('confidence_score', 50.0)}/100 |\n"

        md += f"""
---

## Risks & Data Quality
* **Invalid Records**: {quality.get('invalid_records', 0)}
* **Suspicious Records**: {quality.get('suspicious_records', 0)}
* **Warnings**: {', '.join(quality.get('warnings', [])) if quality.get('warnings') else 'None'}
"""
        p = os.path.join(self.reports_dir, f"{date_str}.md")
        with open(p, "w") as f:
            f.write(md)

        return p
