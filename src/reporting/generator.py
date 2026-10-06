import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates daily research Markdown reports and structured JSON files."""

    @staticmethod
    def generate_json_report(date_str: str, gold_summary: Dict[str, Any], silver_summary: Dict[str, Any], ratio_summary: Dict[str, Any], macro_summary: Dict[str, Any], quality_summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "date": date_str,
            "gold": gold_summary,
            "silver": silver_summary,
            "gold_silver_ratio": ratio_summary,
            "macro": macro_summary,
            "data_quality": quality_summary
        }

    @staticmethod
    def generate_markdown_report(date_str: str, gold_info: Dict[str, Any], silver_info: Dict[str, Any], ratio_info: Dict[str, Any], macro_info: Dict[str, Any], analogues_info: Dict[str, Any], fwd_prob_info: Dict[str, Any], quality_info: Dict[str, Any]) -> str:
        md = []
        md.append(f"# Precious Metals Daily Research Report — {date_str}\n")

        md.append("## Executive Summary\n")

        g_price = gold_info.get("price", 0.0)
        g_1d = gold_info.get("return_1d", 0.0) * 100.0
        g_5d = gold_info.get("return_5d", 0.0) * 100.0
        g_20d = gold_info.get("return_20d", 0.0) * 100.0

        md.append(f"### Gold Spot\n")
        md.append(f"- **Current Price**: ${g_price:,.2f} USD/oz")
        md.append(f"- **1D Move**: {g_1d:+.2f}%")
        md.append(f"- **Weekly Move (5D)**: {g_5d:+.2f}%")
        md.append(f"- **Monthly Move (20D)**: {g_20d:+.2f}%")
        md.append(f"- **Research Bias**: {gold_info.get('research_bias', 'Neutral')}\n")

        s_price = silver_info.get("price", 0.0)
        s_1d = silver_info.get("return_1d", 0.0) * 100.0
        s_5d = silver_info.get("return_5d", 0.0) * 100.0
        s_20d = silver_info.get("return_20d", 0.0) * 100.0

        md.append(f"### Silver Spot\n")
        md.append(f"- **Current Price**: ${s_price:,.2f} USD/oz")
        md.append(f"- **1D Move**: {s_1d:+.2f}%")
        md.append(f"- **Weekly Move (5D)**: {s_5d:+.2f}%")
        md.append(f"- **Monthly Move (20D)**: {s_20d:+.2f}%")
        md.append(f"- **Research Bias**: {silver_info.get('research_bias', 'Neutral')}\n")

        md.append("## Gold Analysis\n")
        md.append(f"- **RSI (14)**: {gold_info.get('rsi_14', 50.0):.1f}")
        md.append(f"- **MACD Histogram**: {gold_info.get('macd_hist', 0.0):.4f}")
        md.append(f"- **20D Support**: ${gold_info.get('support_20d', 0.0):,.2f}")
        md.append(f"- **20D Resistance**: ${gold_info.get('resistance_20d', 0.0):,.2f}\n")

        md.append("## Silver Analysis & Gold/Silver Ratio\n")
        md.append(f"- **Gold/Silver Ratio**: {ratio_info.get('current_ratio', 0.0):.2f}")
        md.append(f"- **Ratio 252D Z-Score**: {ratio_info.get('ratio_zscore', 0.0):+.2f}")
        md.append(f"- **Ratio Percentile**: {ratio_info.get('ratio_percentile', 50.0):.1f}%\n")

        md.append("## Macro Environment\n")
        md.append(f"- **Primary Macro Regime**: {macro_info.get('primary_regime', 'Neutral')}")
        md.append(f"- **Active Sub-Regimes**: {', '.join(macro_info.get('all_active_regimes', ['Neutral']))}\n")

        md.append("## Historical Analogue Analysis\n")
        md.append(f"Found {analogues_info.get('sample_count', 0)} top historical analogues.")
        f_stats = analogues_info.get("forward_stats", {}).get("5d", {})
        if f_stats:
            md.append(f"- **5D Positive Return Frequency**: {f_stats.get('prob_positive', 0.5)*100:.1f}%")
            md.append(f"- **5D Mean Historical Return**: {f_stats.get('mean_return', 0.0)*100:+.2f}%\n")

        md.append("## Forecast Research & Probabilities\n")
        md.append("> *Notice: All statistics represent historical conditional research signals and do not guarantee future prices.*\n")

        horizons = fwd_prob_info.get("horizons", {})
        for h_key, h_data in horizons.items():
            md.append(f"- **{h_key} Horizon**: Positive Prob = {h_data.get('prob_positive', 0.5)*100:.1f}%, Expected Return = {h_data.get('expected_return', 0.0)*100:+.2f}%")

        md.append("\n## Data Quality Summary\n")
        md.append(f"- **Gold Records**: {quality_info.get('gold_records', 0)}")
        md.append(f"- **Silver Records**: {quality_info.get('silver_records', 0)}")
        md.append(f"- **Warnings**: {len(quality_info.get('warnings', []))}")
        md.append(f"- **Errors**: {len(quality_info.get('errors', []))}\n")

        return "\n".join(md)
