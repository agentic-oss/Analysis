# Data Schemas

This document outlines the data schemas across raw, processed, feature store, predictions, and report outputs.

## Raw Ingestion Schema

| Field | Type | Description |
| --- | --- | --- |
| `date` | String (YYYY-MM-DD) | Observation date |
| `symbol` | String | Instrument symbol |
| `ticker` | String | Data source ticker |
| `open` | Float | Open price |
| `high` | Float | High price |
| `low` | Float | Low price |
| `close` | Float | Close price |
| `adj_close` | Float | Adjusted close price |
| `volume` | Float | Volume |
| `provider` | String | Provider identifier (`yahoo`, `mock`) |
| `source` | String | Data source |
| `retrieval_timestamp` | String (ISO 8601) | Timestamp when data was retrieved |

For converted precious metals:
| Field | Type | Description |
| --- | --- | --- |
| `usdinr_rate` | Float | FX rate used for conversion |
| `price_inr` | Float | Price in INR (per 10g for Gold, per kg for Silver) |
| `unit_inr` | String | Units (`10g`, `kg`) |
| `conversion_timestamp` | String (ISO 8601) | Timestamp of FX conversion |

## Daily Quality Report Schema (`data/quality/YYYY-MM-DD.json`)

```json
{
  "date": "YYYY-MM-DD",
  "total_records": 167,
  "gold_records": 10,
  "silver_records": 10,
  "symbols_validated": ["GOLD", "SILVER", "USDINR", "..."],
  "warnings": [],
  "errors": [],
  "symbol_details": {}
}
```

## Prediction Tracking Schema (`data/predictions/prediction_history.parquet`)

| Field | Type | Description |
| --- | --- | --- |
| `prediction_date` | String (YYYY-MM-DD) | Date prediction was generated (Date T) |
| `instrument` | String | Instrument (`GOLD`, `SILVER`) |
| `horizon` | Int | Horizon trading days (1, 3, 5, 10, 20, 60) |
| `predicted_direction` | String | `UP` or `DOWN` |
| `predicted_return` | Float | Expected % return |
| `confidence` | Float | Confidence score / Positive return probability % |
| `market_regime` | String | Trend regime on prediction date |
| `actual_return` | Float | Realized % return at T + horizon (updated when realized) |
| `prediction_correct` | Int (0 or 1) | Directional accuracy indicator |
