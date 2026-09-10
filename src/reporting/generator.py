import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DailyReportGenerator:
    """Generates daily research report in Markdown (reports/daily/YYYY-MM-DD.md) and JSON (data/analysis/daily/YYYY-MM-DD.json)."""

    def generate_markdown_report(
        self,
        date_str: str,
        gold_summary: Dict[str, Any],
        silver_summary: Dict[str, Any],
        macro_summary: Dict[str, Any],
        rv_summary: Dict[str, Any],
        analogue_summary: Dict[str, Any],
        forecast_signals: Dict[str, Any],
        quality_summary: Dict[str, Any],
        output_dir: str = "reports/daily"
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, f"{date_str}.md")

        g_price = gold_summary.get("price", 0.0)
        g_1d = gold_summary.get("daily_move_pct", 0.0)
        g_bias = gold_summary.get("technical_bias", "Neutral")

        s_price = silver_summary.get("price", 0.0)
        s_1d = silver_summary.get("daily_move_pct", 0.0)
        s_bias = silver_summary.get("technical_bias", "Neutral")

        macro_regime = macro_summary.get("primary_macro_regime", "Neutral")
        gs_ratio = rv_summary.get("gold_silver_ratio", 0.0)

        content = f"""# Precious Metals Daily Research Report ({date_str})

## Executive Summary

### Gold
* **Current Price:** ${g_price:.2f} / oz
* **Daily Move:** {g_1d:+.2f}%
* **Weekly Move:** {gold_summary.get('weekly_move_pct', 0.0):+.2f}%
* **Monthly Move:** {gold_summary.get('monthly_move_pct', 0.0):+.2f}%
* **Technical Bias:** {g_bias}
* **RSI (14):** {gold_summary.get('rsi_14', 50.0):.1f}
* **20D Volatility:** {gold_summary.get('volatility_20d', 0.0)*100:.1f}%

### Silver
* **Current Price:** ${s_price:.2f} / oz
* **Daily Move:** {s_1d:+.2f}%
* **Weekly Move:** {silver_summary.get('weekly_move_pct', 0.0):+.2f}%
* **Monthly Move:** {silver_summary.get('monthly_move_pct', 0.0):+.2f}%
* **Technical Bias:** {s_bias}
* **RSI (14):** {silver_summary.get('rsi_14', 50.0):.1f}
* **20D Volatility:** {silver_summary.get('volatility_20d', 0.0)*100:.1f}%

---

## Macro Environment & Regime

* **Primary Macro Regime:** `{macro_regime}`
* **Gold / Silver Ratio:** `{gs_ratio:.2f}` (Z-Score: `{rv_summary.get('ratio_zscore_60', 0.0):+.2f}`)
* **Key Macro Drivers:**
  * DXY: `{macro_summary.get('inputs', {}).get('dxy_return_20d', 0.0)*100:+.2f}%` (20-day return)
  * Crude Oil: `{macro_summary.get('inputs', {}).get('oil_return_20d', 0.0)*100:+.2f}%` (20-day return)
  * S&P 500: `{macro_summary.get('inputs', {}).get('sp500_return_20d', 0.0)*100:+.2f}%` (20-day return)
  * VIX: `{macro_summary.get('inputs', {}).get('vix_level', 0.0):.1f}`

---

## Cross-Market & Relative Value Analysis

* **Spread Return (Gold - Silver):** `{rv_summary.get('spread_return_1d', 0.0)*100:+.2f}%`
* **Spread Volatility (20D):** `{rv_summary.get('spread_volatility_20d', 0.0)*100:.1f}%`
* **Ratio Extreme Status:** Gold Overvalued: `{rv_summary.get('extreme_high_gold_overvalued', False)}` | Silver Overvalued: `{rv_summary.get('extreme_low_silver_overvalued', False)}`

---

## Historical Analogue Analysis

Found `{analogue_summary.get('sample_count', 0)}` historical analogues.

### Forward Return Probabilities
"""
        signals = forecast_signals.get("signals", {})
        for h, sig in signals.items():
            content += f"- **{h} Horizon:** {sig.get('bias')} | Positive Return Prob: `{sig.get('positive_return_probability', 0.5)*100:.1f}%` | Expected Return: `{sig.get('expected_return', 0.0)*100:+.2f}%`\n"

        content += f"""
> **Disclaimer:** {forecast_signals.get('disclaimer', '')}

---

## Data Quality & System Status

* **Status:** `{quality_summary.get('overall_status', 'valid')}`
* **Evaluated Instruments:** `{quality_summary.get('instruments_evaluated', 0)}`
* **Warnings Count:** `{len(quality_summary.get('all_warnings', []))}`
* **Errors Count:** `{len(quality_summary.get('all_errors', []))}`
"""
        with open(md_path, "w") as f:
            f.write(content)

        logger.info(f"Generated Markdown report at {md_path}")
        return md_path

    def generate_json_report(
        self,
        date_str: str,
        gold_summary: Dict[str, Any],
        silver_summary: Dict[str, Any],
        macro_summary: Dict[str, Any],
        rv_summary: Dict[str, Any],
        scores: Dict[str, Any],
        forecast_signals: Dict[str, Any],
        quality_summary: Dict[str, Any],
        output_dir: str = "data/analysis/daily"
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{date_str}.json")

        json_data = {
            "date": date_str,
            "gold": gold_summary,
            "silver": silver_summary,
            "gold_silver_ratio": rv_summary,
            "macro": macro_summary,
            "scores": scores,
            "forecast_signals": forecast_signals,
            "data_quality": quality_summary
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Generated JSON report at {json_path}")
        return json_path
