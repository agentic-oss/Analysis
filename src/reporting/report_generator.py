import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """Generates daily Markdown research reports and machine-readable JSON outputs."""

    @staticmethod
    def generate_json_output(
        date_str: str,
        gold_summary: Dict[str, Any],
        silver_summary: Dict[str, Any],
        ratio_summary: Dict[str, Any],
        macro_summary: Dict[str, Any],
        market_regimes: Dict[str, Any],
        quality_summary: Dict[str, Any],
        alerts_summary: Dict[str, Any],
        filepath: str = None,
    ) -> Dict[str, Any]:
        """Generate machine-readable JSON report."""
        json_data = {
            "date": date_str,
            "gold": gold_summary,
            "silver": silver_summary,
            "gold_silver_ratio": ratio_summary,
            "macro": macro_summary,
            "market_regime": market_regimes,
            "data_quality": quality_summary,
            "alerts": alerts_summary,
            "generated_at": datetime.now().isoformat(),
        }

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, default=str)
            logger.info(f"Saved daily JSON analysis to {filepath}")

        return json_data

    @staticmethod
    def generate_markdown_report(
        date_str: str,
        json_data: Dict[str, Any],
        filepath: str = None,
    ) -> str:
        """Generates detailed Markdown report adhering strictly to Section 23 specifications."""
        gold = json_data.get("gold", {})
        silver = json_data.get("silver", {})
        gsr = json_data.get("gold_silver_ratio", {})
        macro = json_data.get("macro", {})
        regimes = json_data.get("market_regime", {})
        quality = json_data.get("data_quality", {})
        alerts = json_data.get("alerts", {})

        g_price = gold.get("price", 0.0)
        g_inr = gold.get("price_inr_10g", 0.0)
        g_move = gold.get("daily_move_pct", 0.0)
        g_bias = gold.get("confidence", {}).get("research_bias", "Neutral")
        g_conf = gold.get("confidence", {}).get("confidence_score", 50)
        g_regime = gold.get("regime", {}).get("trend_regime", "Range")
        g_vol_regime = gold.get("regime", {}).get("volatility_regime", "Normal Volatility")

        s_price = silver.get("price", 0.0)
        s_inr = silver.get("price_inr_kg", 0.0)
        s_move = silver.get("daily_move_pct", 0.0)
        s_bias = silver.get("confidence", {}).get("research_bias", "Neutral")
        s_conf = silver.get("confidence", {}).get("confidence_score", 50)
        s_regime = silver.get("regime", {}).get("trend_regime", "Range")
        s_vol_regime = silver.get("regime", {}).get("volatility_regime", "Normal Volatility")

        md = f"""# Precious Metals Daily Research Report — {date_str}

> **Disclaimer**: All price predictions and forecasting outputs in this document are strictly framed as probabilistic research signals rather than guaranteed future prices.

---

## Executive Summary

### Gold
* **Current Price (USD/oz)**: ${g_price:,.2f}
* **Indian Spot Price (INR/10g)**: ₹{g_inr:,.2f}
* **Daily Move**: {g_move:+.2f}%
* **Trend Regime**: {g_regime}
* **Volatility Regime**: {g_vol_regime}
* **Research Bias**: **{g_bias}** (Confidence: {g_conf}/100)

### Silver
* **Current Price (USD/oz)**: ${s_price:,.2f}
* **Indian Spot Price (INR/kg)**: ₹{s_inr:,.2f}
* **Daily Move**: {s_move:+.2f}%
* **Trend Regime**: {s_regime}
* **Volatility Regime**: {s_vol_regime}
* **Research Bias**: **{s_bias}** (Confidence: {s_conf}/100)

---

## Gold Analysis

* **Technical Structure**: RSI 14: {gold.get('rsi', 50.0):.1f} | 200 DMA Distance: {gold.get('dist_200dma', 0.0):+.2f}%
* **Key Support**: {', '.join([str(x) for x in gold.get('support', [])])}
* **Key Resistance**: {', '.join([str(x) for x in gold.get('resistance', [])])}
* **Research Reasons**:
"""
        for r in gold.get("confidence", {}).get("reasons", []):
            md += f"  - {r}\n"

        md += f"""
---

## Silver Analysis

* **Technical Structure**: RSI 14: {silver.get('rsi', 50.0):.1f} | 200 DMA Distance: {silver.get('dist_200dma', 0.0):+.2f}%
* **Gold/Silver Ratio**: {gsr.get('current_ratio', 0.0):.2f} (Z-Score: {gsr.get('zscore', 0.0):+.2f}, Percentile: {gsr.get('percentile', 50.0):.1f}%)
* **GSR Regime**: {gsr.get('regime', 'Normal')}
* **Key Support**: {', '.join([str(x) for x in silver.get('support', [])])}
* **Key Resistance**: {', '.join([str(x) for x in silver.get('resistance', [])])}

---

## Macro Environment

* **US Dollar Index (DXY)**: {macro.get('dxy', {}).get('latest', 0.0)} ({macro.get('dxy', {}).get('pct_change', 0.0):+.2f}%)
* **US 10Y Yield**: {macro.get('us10y', {}).get('latest', 0.0)}% ({macro.get('us10y', {}).get('abs_change', 0.0):+.2f} bps)
* **Crude Oil (WTI)**: ${macro.get('oil', {}).get('latest', 0.0)} ({macro.get('oil', {}).get('pct_change', 0.0):+.2f}%)
* **S&P 500**: {macro.get('sp500', {}).get('latest', 0.0)} ({macro.get('sp500', {}).get('pct_change', 0.0):+.2f}%)
* **NIFTY 50**: {macro.get('nifty50', {}).get('latest', 0.0)} ({macro.get('nifty50', {}).get('pct_change', 0.0):+.2f}%)
* **VIX**: {macro.get('vix', {}).get('latest', 0.0)}
* **Macro Regime**: **{regimes.get('primary_macro_regime', 'Neutral')}**

---

## Historical Analogue & Forecast Research

### Gold Forward Probabilities
"""
        g_fwd = gold.get("forward_statistics", {})
        for h, fstat in g_fwd.items():
            md += f"- **{h} Trading Horizon**: Positive Return Prob: **{fstat.get('probability_positive_return_pct', 50)}%** | Expected Return: **{fstat.get('expected_mean_return_pct', 0.0):+.2f}%** | Expected Volatility: **{fstat.get('expected_horizon_volatility_pct', 0.0):.2f}%**\n"

        md += """
### Silver Forward Probabilities
"""
        s_fwd = silver.get("forward_statistics", {})
        for h, fstat in s_fwd.items():
            md += f"- **{h} Trading Horizon**: Positive Return Prob: **{fstat.get('probability_positive_return_pct', 50)}%** | Expected Return: **{fstat.get('expected_mean_return_pct', 0.0):+.2f}%** | Expected Volatility: **{fstat.get('expected_horizon_volatility_pct', 0.0):.2f}%**\n"

        md += f"""
---

## Active Market Alerts

* Total Triggered Alerts: **{alerts.get('alert_count', 0)}**
"""
        for a in alerts.get("alerts", []):
            md += f"- [{a.get('level', 'info').upper()}] {a.get('type')}: {a.get('message')}\n"

        md += f"""
---

## Data Quality Summary

* Valid Records Evaluated: {quality.get('total_valid_records', 0)}
* Suspicious Records: {quality.get('total_suspicious_records', 0)}
* Invalid Records: {quality.get('total_invalid_records', 0)}
* Data Warnings: {len(quality.get('warnings', []))}
"""

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md)
            logger.info(f"Saved daily Markdown report to {filepath}")

        return md
