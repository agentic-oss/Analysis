import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """Generates Markdown research reports, machine-readable JSON outputs, and alert summaries."""

    def __init__(
        self,
        reports_dir: str = "reports/daily",
        analysis_dir: str = "data/analysis/daily",
        alerts_dir: str = "data/analysis/alerts",
    ):
        self.reports_dir = reports_dir
        self.analysis_dir = analysis_dir
        self.alerts_dir = alerts_dir

        for d in [self.reports_dir, self.analysis_dir, self.alerts_dir]:
            os.makedirs(d, exist_ok=True)

    def generate_json_output(self, date_str: str, analysis_payload: Dict[str, Any]) -> str:
        """Saves machine-readable JSON output to data/analysis/daily/YYYY-MM-DD.json."""
        filepath = os.path.join(self.analysis_dir, f"{date_str}.json")
        with open(filepath, "w") as f:
            json.dump(analysis_payload, f, indent=2)
        return filepath

    def generate_alerts(self, date_str: str, alerts_list: List[Dict[str, Any]]) -> str:
        """Saves daily alert triggers to data/analysis/alerts/YYYY-MM-DD.json."""
        filepath = os.path.join(self.alerts_dir, f"{date_str}.json")
        payload = {
            "date": date_str,
            "timestamp": datetime.utcnow().isoformat(),
            "alert_count": len(alerts_list),
            "alerts": alerts_list,
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        return filepath

    def generate_markdown_report(self, date_str: str, data: Dict[str, Any]) -> str:
        """Generates structured Precious Metals Daily Research Report in Markdown format."""
        filepath = os.path.join(self.reports_dir, f"{date_str}.md")

        gold_info = data.get("gold", {})
        silver_info = data.get("silver", {})
        macro_info = data.get("macro", {})
        ratio_info = data.get("gold_silver_ratio", {})
        alerts_info = data.get("alerts", [])
        quality_info = data.get("data_quality", {})

        md = []
        md.append(f"# Precious Metals Daily Research Report — {date_str}\n")
        md.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

        # Executive Summary
        md.append("## Executive Summary\n")
        md.append("### Gold")
        md.append(f"- **Current Price:** ${gold_info.get('price', 0.0):,.2f}/oz")
        md.append(f"- **Daily Move:** {gold_info.get('daily_move_pct', 0.0):+.2f}%")
        md.append(f"- **Weekly Move:** {gold_info.get('weekly_move_pct', 0.0):+.2f}%")
        md.append(f"- **Monthly Move:** {gold_info.get('monthly_move_pct', 0.0):+.2f}%")
        md.append(f"- **Market Regime:** {gold_info.get('regime', 'N/A')}")
        md.append(f"- **Research Score:** {gold_info.get('scores', {}).get('composite_score', 50)}/100 (Confidence: {gold_info.get('scores', {}).get('confidence_score', 50)}/100)")
        md.append(f"- **Research Bias:** {gold_info.get('bias', 'Neutral')}\n")

        md.append("### Silver")
        md.append(f"- **Current Price:** ${silver_info.get('price', 0.0):,.2f}/oz")
        md.append(f"- **Daily Move:** {silver_info.get('daily_move_pct', 0.0):+.2f}%")
        md.append(f"- **Weekly Move:** {silver_info.get('weekly_move_pct', 0.0):+.2f}%")
        md.append(f"- **Monthly Move:** {silver_info.get('monthly_move_pct', 0.0):+.2f}%")
        md.append(f"- **Market Regime:** {silver_info.get('regime', 'N/A')}")
        md.append(f"- **Research Score:** {silver_info.get('scores', {}).get('composite_score', 50)}/100")
        md.append(f"- **Research Bias:** {silver_info.get('bias', 'Neutral')}\n")

        # Gold Analysis
        md.append("## Gold Analysis")
        md.append(f"- **Technical Structure:** SMA20: ${gold_info.get('sma_20', 0):,.2f} | SMA50: ${gold_info.get('sma_50', 0):,.2f} | SMA200: ${gold_info.get('sma_200', 0):,.2f}")
        md.append(f"- **RSI (14):** {gold_info.get('rsi_14', 50):.1f}")
        md.append(f"- **ATR (14):** ${gold_info.get('atr_14', 0):.2f}")
        md.append(f"- **Key Support (20d Low):** ${gold_info.get('support_20d', 0):,.2f}")
        md.append(f"- **Key Resistance (20d High):** ${gold_info.get('resistance_20d', 0):,.2f}\n")

        # Silver & Relative Value Analysis
        md.append("## Silver & Relative Value Analysis")
        md.append(f"- **Gold/Silver Ratio:** {ratio_info.get('current_ratio', 0.0):.2f}")
        md.append(f"- **200-day Ratio SMA:** {ratio_info.get('sma_200', 0.0):.2f}")
        md.append(f"- **Ratio Z-Score:** {ratio_info.get('zscore', 0.0):+.2f}")
        md.append(f"- **Ratio Percentile (252d):** {ratio_info.get('percentile', 50.0):.1f}%\n")

        # Macro Environment
        md.append("## Macro Environment")
        md.append(f"- **Composite Macro Regime:** {macro_info.get('macro_regime', 'Neutral')}")
        md.append(f"- **VIX:** {macro_info.get('vix', 0.0):.2f}")
        md.append(f"- **Crude Oil 1D Return:** {macro_info.get('oil_1d_change', 0.0)*100:+.2f}%")
        md.append(f"- **DXY 1D Return:** {macro_info.get('dxy_1d_change', 0.0)*100:+.2f}%")
        md.append(f"- **S&P 500 1D Return:** {macro_info.get('sp500_1d_change', 0.0)*100:+.2f}%\n")

        # Forecast Research Signals
        md.append("## Forecast Research Signals & Probabilities")
        md.append("_Note: Signals represent historical conditional statistics and probabilistic models, not guaranteed outcomes._\n")
        md.append("| Horizon | Pos Return Prob | Expected Return | Research Bias |")
        md.append("|---|---|---|---|")
        gold_fwd = gold_info.get("forward_statistics", {})
        for h, stats in gold_fwd.items():
            prob = stats.get("probability_positive", 0.5) * 100.0
            exp_ret = stats.get("expected_return", 0.0) * 100.0
            bias = stats.get("research_bias", "Neutral")
            md.append(f"| {h} | {prob:.1f}% | {exp_ret:+.2f}% | {bias} |")
        md.append("\n")

        # Risks & Alerts
        md.append("## Risks & Active Alerts")
        if alerts_info:
            for alert in alerts_info:
                md.append(f"- ⚠️ **{alert.get('type')}**: {alert.get('message')}")
        else:
            md.append("- No abnormal risk alerts triggered for today.\n")

        # Data Quality
        md.append("## Data Quality & Integrity")
        md.append(f"- **Analyzed Instruments:** {quality_info.get('instruments_analyzed', 0)}")
        md.append(f"- **Warnings Count:** {quality_info.get('overall_warnings_count', 0)}")
        md.append(f"- **Errors Count:** {quality_info.get('overall_errors_count', 0)}\n")

        content = "\n".join(md)
        with open(filepath, "w") as f:
            f.write(content)

        return filepath
