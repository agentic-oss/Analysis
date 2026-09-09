# Methodology & Forecasting

## Non-Sensational Probabilistic Principles
All model outputs are expressed as historical conditional return probabilities and expected values. The system strictly avoids deterministic claims ("Gold will rise X%").

## Look-Ahead Protection
Features on date T are constructed strictly using observations available on or before date T. Forward targets (`future_return_5d`) are stored separately and omitted from model inputs during live signal generation.
