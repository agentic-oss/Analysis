"""Daily Markdown Report, JSON Analysis, and Alert Generator."""

import os
from typing import Dict, Any, List


class ReportGenerator:
    """Generates Markdown daily research reports, structured JSON reports, and alerts."""

    def generate_markdown_report(
        self,
        date_str: str,
        gold_data: Dict[str, Any],
        silver_data: Dict[str, Any],
        ratio_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        gold_scores: Dict[str, Any],
        silver_scores: Dict[str, Any],
        gold_signals: Dict[str, Any],
        silver_signals: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        quality_data: Dict[str, Any]
    ) -> str:
        """Generates comprehensive daily research report at reports/daily/YYYY-MM-DD.md."""

        md = f"""# Precious Metals Daily Research Report — {date_str}

## Executive Summary

**Gold:**
* **Current Price:** ${gold_data.get('price', 0.0):,.2f} USD ({gold_data.get('price_inr', 0.0):,.2f} INR/10g)
* **1D Return:** {gold_data.get('return_1d', 0.0):+.2%}
* **5D Return:** {gold_data.get('return_5d', 0.0):+.2%}
* **20D Return:** {gold_data.get('return_20d', 0.0):+.2%}
* **Market Regime:** {gold_data.get('regime', {}).get('trend_regime', 'N/A')} ({gold_data.get('regime', {}).get('volatility_regime', 'N/A')})
* **Composite Research Score:** {gold_scores.get('composite_score', 0)} / 100

**Silver:**
* **Current Price:** ${silver_data.get('price', 0.0):,.2f} USD ({silver_data.get('price_inr', 0.0):,.2f} INR/kg)
* **1D Return:** {silver_data.get('return_1d', 0.0):+.2%}
* **5D Return:** {silver_data.get('return_5d', 0.0):+.2%}
* **20D Return:** {silver_data.get('return_20d', 0.0):+.2%}
* **Market Regime:** {silver_data.get('regime', {}).get('trend_regime', 'N/A')} ({silver_data.get('regime', {}).get('volatility_regime', 'N/A')})
* **Composite Research Score:** {silver_scores.get('composite_score', 0)} / 100

---

## Gold Analysis

* **Technical Structure:** RSI(14) = {gold_data.get('indicators', {}).get('rsi_14', 50):.1f}, MACD Hist = {gold_data.get('indicators', {}).get('macd_hist', 0):.2f}
* **Macro Drivers Correlation (60D):** DXY = {gold_data.get('correlations', {}).get('DXY', {}).get('corr_60d', 0):+.2f}, US10Y = {gold_data.get('correlations', {}).get('US10Y', {}).get('corr_60d', 0):+.2f}
* **Forecast Signals (10-Day Horizon):**
  * Probability Positive: {gold_signals.get('forecast_signals', {}).get('10d', {}).get('probability_positive', 0.5):.0%}
  * Expected Return: {gold_signals.get('forecast_signals', {}).get('10d', {}).get('expected_return', 0):+.2%}
* **Prediction Confidence:** {gold_signals.get('confidence', {}).get('score', 0)} / 100

---

## Silver Analysis & Gold/Silver Ratio

* **Gold/Silver Ratio:** {ratio_data.get('current_ratio', 0):.2f} (Z-Score: {ratio_data.get('z_score_252d', 0):+.2f}, Percentile: {ratio_data.get('percentile_rank', 0):.1f}%)
* **Technical Structure:** RSI(14) = {silver_data.get('indicators', {}).get('rsi_14', 50):.1f}
* **Forecast Signals (10-Day Horizon):**
  * Probability Positive: {silver_signals.get('forecast_signals', {}).get('10d', {}).get('probability_positive', 0.5):.0%}
  * Expected Return: {silver_signals.get('forecast_signals', {}).get('10d', {}).get('expected_return', 0):+.2%}

---

## Macro Environment

* **Macro Regime:** {macro_data.get('primary_regime', 'Neutral')}
* **Key Components:**
  * Rates Trend: {macro_data.get('explainable_components', {}).get('rates_trend', 'flat')}
  * Dollar Index (DXY) Trend: {macro_data.get('explainable_components', {}).get('dxy_trend', 'flat')}
  * Volatility State (VIX): {macro_data.get('explainable_components', {}).get('vix_state', 'calm')}

---

## Daily Alerts

"""
        if alerts:
            for a in alerts:
                md += f"* **[{a.get('type')}]** {a.get('message')}\n"
        else:
            md += "* No alert thresholds triggered today.\n"

        md += f"""
---

## Data Quality Summary

* **Records Validated:** {quality_data.get('total_records', 0)}
* **Valid:** {quality_data.get('classifications', {}).get('valid', 0)} | **Suspicious:** {quality_data.get('classifications', {}).get('suspicious', 0)} | **Invalid:** {quality_data.get('classifications', {}).get('invalid', 0)}
* **Warnings:** {len(quality_data.get('warnings', []))}
* **Errors:** {len(quality_data.get('errors', []))}

---

*Disclaimer: All price predictions and forecasting outputs are strictly framed as probabilistic research signals rather than guaranteed future prices.*
"""
        os.makedirs("reports/daily", exist_ok=True)
        report_path = os.path.join("reports/daily", f"{date_str}.md")
        with open(report_path, "w") as f:
            f.write(md)

        return report_path


class AlertDetector:
    """Detects market anomalies, breakouts, and ratio extremes against configured thresholds."""

    @staticmethod
    def detect_alerts(
        date_str: str,
        gold_df: Dict[str, Any],
        silver_df: Dict[str, Any],
        ratio_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        alerts = []

        g_ret = gold_df.get('return_1d', 0.0)
        s_ret = silver_df.get('return_1d', 0.0)
        z_score = ratio_data.get('z_score_252d', 0.0)

        if abs(g_ret) > 0.02:
            alerts.append({
                "date": date_str,
                "type": "GOLD_LARGE_MOVE",
                "message": f"Gold moved {g_ret:+.2%} today (threshold: 2.0%)."
            })

        if abs(s_ret) > 0.03:
            alerts.append({
                "date": date_str,
                "type": "SILVER_LARGE_MOVE",
                "message": f"Silver moved {s_ret:+.2%} today (threshold: 3.0%)."
            })

        if abs(z_score) > 2.0:
            alerts.append({
                "date": date_str,
                "type": "RATIO_EXTREME",
                "message": f"Gold/Silver ratio Z-score reached extreme level of {z_score:+.2f}."
            })

        return alerts
