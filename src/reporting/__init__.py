import os
import json
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger("ReportGenerator")


class ReportGenerator:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = os.path.join(reports_dir, "daily")
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_daily_report(
        self,
        date_str: str,
        gold_summary: Dict[str, Any],
        silver_summary: Dict[str, Any],
        ratio_summary: Dict[str, Any],
        macro_summary: Dict[str, Any],
        gold_signals: Dict[str, Any],
        silver_signals: Dict[str, Any],
        scores: Dict[str, Any],
        data_quality: Dict[str, Any]
    ) -> str:
        md_content = f"""# Precious Metals Daily Research Report - {date_str}

## Executive Summary

### Gold (Spot USD / Indian INR)
- **Current Price (USD/oz)**: ${gold_summary.get('close', 0.0):,.2f}
- **Current Price (INR/10g)**: ₹{ratio_summary.get('gold_inr_10g', 0.0):,.2f}
- **Daily Move**: {gold_summary.get('return_1d', 0.0)*100:+.2f}%
- **5-Day Move**: {gold_summary.get('return_5d', 0.0)*100:+.2f}%
- **20-Day Move**: {gold_summary.get('return_20d', 0.0)*100:+.2f}%
- **Market Regime**: {gold_summary.get('market_regime', 'N/A')}
- **Volatility (20D Ann.)**: {gold_summary.get('volatility_20d', 0.0):.2f}%
- **Composite Research Score**: {scores.get('composite_score', 0.0)}/100

### Silver (Spot USD / Indian INR)
- **Current Price (USD/oz)**: ${silver_summary.get('close', 0.0):,.2f}
- **Current Price (INR/kg)**: ₹{ratio_summary.get('silver_inr_kg', 0.0):,.2f}
- **Daily Move**: {silver_summary.get('return_1d', 0.0)*100:+.2f}%
- **5-Day Move**: {silver_summary.get('return_5d', 0.0)*100:+.2f}%
- **20-Day Move**: {silver_summary.get('return_20d', 0.0)*100:+.2f}%
- **Market Regime**: {silver_summary.get('market_regime', 'N/A')}
- **Volatility (20D Ann.)**: {silver_summary.get('volatility_20d', 0.0):.2f}%

---

## Macro Environment & Cross-Market Drivers

- **Macro Regime**: {macro_summary.get('macro_regime', 'Neutral')}
- **USD / INR Rate**: {ratio_summary.get('usd_inr_rate_used', 83.0):.2f}
- **DXY 20D Return**: {macro_summary.get('components', {}).get('dxy_return_20d', 0.0)*100:+.2f}%
- **US 10Y Real Yield 20D Change**: {macro_summary.get('components', {}).get('real_yield_change_20d', 0.0):+.2f} bp
- **VIX Level**: {macro_summary.get('components', {}).get('vix_level', 0.0):.2f}

---

## Relative Value & Gold/Silver Ratio

- **Current Gold/Silver Ratio**: {ratio_summary.get('current_ratio', 0.0):.2f}
- **Long-term Average**: {ratio_summary.get('long_term_average', 0.0):.2f}
- **Ratio Z-Score**: {ratio_summary.get('z_score', 0.0):+.2f}
- **Historical Percentile**: {ratio_summary.get('historical_percentile', 0.0):.1f}%

---

## Technical Scores & Breakdown

- **Technical Structure Score**: {scores.get('technical_score', 0.0)} / 100
- **Macro Drivers Score**: {scores.get('macro_score', 0.0)} / 100
- **Momentum Score**: {scores.get('momentum_score', 0.0)} / 100
- **Volatility Score**: {scores.get('volatility_score', 0.0)} / 100
- **Relative Value Score**: {scores.get('relative_value_score', 0.0)} / 100
- **Historical Pattern Score**: {scores.get('historical_pattern_score', 0.0)} / 100

---

## Probabilistic Forward Research Signals (Gold)

*Disclaimer: All forecasting outputs are strictly probabilistic historical research signals, not guaranteed future prices.*

| Horizon | Positive Return Prob | Expected Return | Median Return | Historical Sample Count |
| :--- | :--- | :--- | :--- | :--- |
"""
        horizons = gold_signals.get("horizons", {})
        for h_key in ["1d", "3d", "5d", "10d", "20d", "60d"]:
            h_data = horizons.get(h_key)
            if h_data:
                md_content += f"| {h_key.upper()} | {h_data.get('positive_return_probability', 0.0)*100:.1f}% | {h_data.get('expected_return', 0.0)*100:+.2f}% | {h_data.get('median_return', 0.0)*100:+.2f}% | {h_data.get('sample_count', 0)} |\n"

        md_content += f"""
---

## Data Quality & System Status

- **Valid Records**: {data_quality.get('valid_records', 0)}
- **Suspicious Records**: {data_quality.get('suspicious_records', 0)}
- **Warnings**: {len(data_quality.get('warnings', []))}
- **Errors**: {len(data_quality.get('errors', []))}

*Report generated at {pd.Timestamp.now().isoformat()}*
"""

        filepath = os.path.join(self.reports_dir, f"{date_str}.md")
        with open(filepath, "w") as f:
            f.write(md_content)

        return filepath


def detect_daily_alerts(
    date_str: str,
    gold_summary: Dict[str, Any],
    silver_summary: Dict[str, Any],
    ratio_summary: Dict[str, Any],
    macro_summary: Dict[str, Any]
) -> Dict[str, Any]:
    alerts = []

    # RSI extremes
    rsi_g = gold_summary.get("rsi_14", 50.0)
    if rsi_g >= 70:
        alerts.append({"type": "GOLD_RSI_OVERBOUGHT", "severity": "MEDIUM", "message": f"Gold RSI is {rsi_g:.1f} (>=70)"})
    elif rsi_g <= 30:
        alerts.append({"type": "GOLD_RSI_OVERSOLD", "severity": "MEDIUM", "message": f"Gold RSI is {rsi_g:.1f} (<=30)"})

    # Gold/Silver Ratio extremes
    ratio = ratio_summary.get("current_ratio", 80.0)
    if ratio >= 90.0:
        alerts.append({"type": "GOLD_SILVER_RATIO_EXTREME_HIGH", "severity": "HIGH", "message": f"Gold/Silver ratio is {ratio:.2f} (>=90)"})
    elif ratio <= 65.0:
        alerts.append({"type": "GOLD_SILVER_RATIO_EXTREME_LOW", "severity": "HIGH", "message": f"Gold/Silver ratio is {ratio:.2f} (<=65)"})

    # Price moves
    ret_g = gold_summary.get("return_1d", 0.0)
    if abs(ret_g) >= 0.02:
        alerts.append({"type": "GOLD_LARGE_DAILY_MOVE", "severity": "HIGH", "message": f"Gold moved {ret_g*100:+.2f}% in 1 day"})

    return {
        "date": date_str,
        "alerts_count": len(alerts),
        "alerts": alerts
    }
