import datetime
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.indicators.technical import TechnicalIndicators
from src.regimes.regime_classifier import MacroRegimeClassifier, MarketRegimeClassifier

logger = logging.getLogger(__name__)

class FeatureStoreBuilder:
    """
    Builds ML-ready daily feature datasets for gold & silver.
    Ensures strict temporal correctness and prevents look-ahead bias.
    """

    FORWARD_HORIZONS = [1, 3, 5, 10, 20, 60]

    def build_daily_features(
        self,
        target_symbol: str,
        prices_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        ratio_df: Optional[pd.DataFrame] = None,
        event_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Builds feature DataFrame for target symbol given price and macro dataframes.
        """
        if prices_df.empty:
            return pd.DataFrame()

        # 1. Technical features
        df = TechnicalIndicators.calculate_all(prices_df).copy()
        df['price'] = df['Close']
        df['instrument'] = target_symbol

        # Returns
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)

        # 2. Ratio features if available
        if ratio_df is not None and not ratio_df.empty:
            df['gold_silver_ratio'] = ratio_df['ratio']
            df['gold_silver_ratio_zscore'] = ratio_df['ratio_zscore']
        else:
            df['gold_silver_ratio'] = np.nan
            df['gold_silver_ratio_zscore'] = np.nan

        # 3. Macro features
        for key, m_df in macro_dfs.items():
            if not m_df.empty and 'Close' in m_df.columns:
                m_close = m_df['Close'].rename(f'{key.lower()}_close')
                df = df.join(m_close, how='left')
                df[f'{key.lower()}_close'] = df[f'{key.lower()}_close'].ffill()

                if key in ['DXY', 'CRUDE_OIL', 'SP500', 'NIFTY50']:
                    df[f'{key.lower()}_return_20d'] = df[f'{key.lower()}_close'].pct_change(20)
                elif key in ['US10Y', 'US2Y', 'REAL_YIELD', 'VIX', 'INDIAVIX']:
                    df[f'{key.lower()}_change_20d'] = df[f'{key.lower()}_close'].diff(20)

        # Fill default column names if missing
        for col in ['dxy_return_20d', 'us10y_change_20d', 'real_yield_change_20d', 'vix_change_20d', 'crude_oil_return_20d', 'sp500_return_20d', 'nifty50_return_20d']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # 4. Regimes
        macro_regimes = []
        market_regimes = []
        for idx, row in df.iterrows():
            m_reg = MacroRegimeClassifier.classify_macro_regime(
                us10y_change_20d=row.get('us10y_change_20d', 0.0),
                real_yield_change_20d=row.get('real_yield_change_20d', 0.0),
                dxy_return_20d=row.get('dxy_return_20d', 0.0),
                vix_level=row.get('vix_close', 18.0),
                oil_return_20d=row.get('crude_oil_return_20d', 0.0),
                equity_return_20d=row.get('sp500_return_20d', 0.0)
            )
            mk_reg = MarketRegimeClassifier.classify_market_regime(
                price=row['Close'],
                sma_50=row.get('sma_50', row['Close']),
                sma_200=row.get('sma_200', row['Close']),
                rsi_14=row.get('rsi_14', 50.0),
                hist_vol_20=row.get('hist_vol_20', 15.0)
            )
            macro_regimes.append(m_reg['macro_regime'])
            market_regimes.append(mk_reg['composite_market_regime'])

        df['macro_regime'] = macro_regimes
        df['market_regime'] = market_regimes

        # 5. Event features
        if event_df is not None and not event_df.empty:
            df['days_to_event'] = 0
            df['days_since_event'] = 0
            df['event_type'] = "none"
            df['event_importance'] = "low"
        else:
            df['days_to_event'] = 99
            df['days_since_event'] = 99
            df['event_type'] = "none"
            df['event_importance'] = "none"

        # 6. Forward Targets (Shifted into the future so T only sees future at target time)
        for h in self.FORWARD_HORIZONS:
            df[f'future_return_{h}d'] = df['Close'].pct_change(h).shift(-h)
            df[f'future_direction_{h}d'] = (df[f'future_return_{h}d'] > 0).astype(int)

        return df
