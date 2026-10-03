# Model Evaluation & Tracking

## Validation Strategy
- **Time-Series Split**: Never randomly shuffle financial observations.
- **Expanding Window Walk-Forward**: Models train on historical window [0..T] and evaluate on out-of-sample block [T..T+step].

## Forecast Tracker (`ForecastTracker`)
- Logs daily predictions to `data/predictions/daily/YYYY-MM-DD_INSTRUMENT.json`.
- As realized future prices materialize, compares predicted direction/magnitude against realized outcome.
- Updates global performance stats in `data/models/model-performance.json`.
