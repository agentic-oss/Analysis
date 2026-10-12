# Model Evaluation & Forecast Performance Tracking

The forecast evaluation tracker (`src/evaluation/tracker.py`) saves daily predictions generated on date $T$ and evaluates actual outcomes over time.
Metrics tracked:
* Overall directional accuracy (%)
* Accuracy by horizon (1D, 3D, 5D, 10D, 20D, 60D)
* Accuracy by instrument and market regime
Outputs are recorded in `data/models/forecast_evaluation.json`.
