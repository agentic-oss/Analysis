# Architecture Overview

The Precious Metals Market Data, Deep Analysis, Research, and Forecasting Pipeline consists of modular layers:

```
                  [ Instrument & System Configs ]
                               |
                   [ Ingestion Layer (Provider) ]
                               |
                  [ Data Quality & Validation ]
                               |
                  [ Storage & Parquet Repository ]
                               |
      +------------------------+------------------------+
      |                        |                        |
[ Technical Engine ]   [ Macro & Correlation ]   [ Relative Value Ratio ]
      |                        |                        |
      +------------------------+------------------------+
                               |
                  [ Regime Detection Engine ]
                               |
                  [ Historical Analogue Engine ]
                               |
                 [ Feature Store & Builder ]
                               |
               [ Transparent Scoring Framework ]
                               |
             [ Probabilistic Forecasting Signals ]
                               |
           [ Reports, Alerts & Pipeline Runner ]
```
