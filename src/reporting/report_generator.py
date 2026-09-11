"""
Daily Research Report and Machine-Readable JSON Generator.
Generates comprehensive Markdown daily reports (reports/daily/YYYY-MM-DD.md)
and stable machine-readable JSON files (data/analysis/daily/YYYY-MM-DD.json).
"""

from datetime import datetime, timezone
import json
import os
from typing import Dict, Any, List


class ReportGenerator:
    """Generates daily research reports in Markdown and machine-readable JSON."""

    def generate_markdown_report(self, date_str: str, report_data: Dict[str, Any]) -> str:
        """Constructs detailed Markdown research report matching specification."""
        gold = report_data.get("gold", {})
        silver = report_data.get("silver", {})
        ratio = report_data.get("ratio", {})
        macro = report_data.get("macro", {})
        scores = report_data.get("scores", {})
        analogues = report_data.get("analogues", {})
        signals = report_data.get("signals", {})
        alerts = report_data.get("alerts", [])
        quality = report_data.get("data_quality", {})

        md = f"""# Precious Metals Daily Research Report — {date_str}

*Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*

---

## Executive Summary

### Gold Summary
* **Current Price (USD/oz):** ${gold.get('price_usd', 0.0):,.2f}
* **Current Price (INR/10g):** ₹{gold.get('price_inr_10g', 0.0):,.2f}
* **1-Day Move:** {gold.get('return_1d_pct', 0.0):+.2f}% | **5-Day Move:** {gold.get('return_5d_pct', 0.0):+.2f}% | **20-Day Move:** {gold.get('return_20d_pct', 0.0):+.2f}%
* **Trend Regime:** {gold.get('trend_regime', 'N/A')}
* **Volatility Regime:** {gold.get('volatility_regime', 'N/A')}
* **Research Composite Score:** {scores.get('composite_score', 50.0)}/100

### Silver Summary
* **Current Price (USD/oz):** ${silver.get('price_usd', 0.0):,.2f}
* **Current Price (INR/kg):** ₹{silver.get('price_inr_kg', 0.0):,.2f}
* **1-Day Move:** {silver.get('return_1d_pct', 0.0):+.2f}% | **5-Day Move:** {silver.get('return_5d_pct', 0.0):+.2f}% | **20-Day Move:** {silver.get('return_20d_pct', 0.0):+.2f}%
* **Trend Regime:** {silver.get('trend_regime', 'N/A')}
* **Volatility Regime:** {silver.get('volatility_regime', 'N/A')}

---

## Gold & Silver Technical Analysis

### Technical Indicators
| Asset | Price (USD) | RSI (14) | MACD Hist | Dist 200 SMA | ATR (14) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gold** | ${gold.get('price_usd', 0.0):,.2f} | {gold.get('rsi_14', 50.0):.1f} | {gold.get('macd_hist', 0.0):.2f} | {gold.get('dist_sma_200_pct', 0.0):+.1f}% | ${gold.get('atr', 0.0):.2f} |
| **Silver** | ${silver.get('price_usd', 0.0):,.2f} | {silver.get('rsi_14', 50.0):.1f} | {silver.get('macd_hist', 0.0):.2f} | {silver.get('dist_sma_200_pct', 0.0):+.1f}% | ${silver.get('atr', 0.0):.2f} |

---

## Gold/Silver Relative Value Analysis

* **Gold/Silver Ratio:** {ratio.get('gold_silver_ratio', 0.0):.2f}
* **252-Day Percentile:** {ratio.get('ratio_percentile_252', 50.0):.1f}%
* **Z-Score:** {ratio.get('ratio_zscore', 0.0):+.2f}
* **1D Spread Return (Gold - Silver):** {ratio.get('return_spread_1d_pct', 0.0):+.2f}%

---

## Macro Environment & Cross-Market Drivers

* **Primary Macro Regime:** {macro.get('primary_regime', 'Neutral')}
* **Risk Regime:** {macro.get('risk_regime', 'Neutral')}
* **Monetary Policy Stance:** {macro.get('monetary_regime', 'Neutral')}
* **Key Drivers:**
"""
        for r in macro.get("reasons", ["Normal macro environment"]):
            md += f"  - {r}\n"

        md += """
---

## Historical Analogue Analysis

Top historical analogues matching today's multi-factor market state:
"""
        gold_analogues = analogues.get("gold", {}).get("analogues", [])
        if gold_analogues:
            md += "| Match Date | Historical Gold Price | Similarity Distance |\n| :--- | :--- | :--- |\n"
            for a in gold_analogues[:5]:
                md += f"| {a.get('date')} | ${a.get('close', 0.0):,.2f} | {a.get('distance', 0.0):.3f} |\n"
        else:
            md += "*Insufficient historical analogue data.*\n"

        md += """
---

## Forward Probabilistic Research Signals

*Disclaimer: All forecasting signals represent historical conditional statistics and probabilistic models, NOT guaranteed future outcomes.*

"""
        gold_signals = signals.get("gold", {})
        if gold_signals:
            md += "| Horizon | Signal Direction | Positive Return Prob | Expected Return | Confidence Score |\n"
            md += "| :--- | :--- | :--- | :--- | :--- |\n"
            for h, sig in gold_signals.items():
                md += f"| **{h}** | {sig.get('direction')} | {sig.get('prob_positive')*100:.1f}% | {sig.get('expected_return_pct'):+.2f}% | {sig.get('confidence_score')}/100 |\n"

        md += """
---

## Risk Factors & Alerts

"""
        if alerts:
            for al in alerts:
                md += f"* **[{al.get('severity')}] {al.get('type')}:** {al.get('message')}\n"
        else:
            md += "*No extreme risk alerts triggered for today's session.*\n"

        md += f"""
---

## Data Quality Summary

* **Valid Records:** {quality.get('valid_records', 0)}
* **Suspicious Records:** {quality.get('suspicious_records', 0)}
* **Invalid Records:** {quality.get('invalid_records', 0)}

---
"""
        return md

    def generate_json_report(self, date_str: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs machine-readable JSON structure adhering to stable schema."""
        return {
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scores": report_data.get("scores", {}),
            "gold": report_data.get("gold", {}),
            "silver": report_data.get("silver", {}),
            "gold_silver_ratio": report_data.get("ratio", {}),
            "macro": report_data.get("macro", {}),
            "historical_analogues": report_data.get("analogues", {}),
            "probabilistic_signals": report_data.get("signals", {}),
            "alerts": report_data.get("alerts", []),
            "data_quality": report_data.get("data_quality", {}),
        }
