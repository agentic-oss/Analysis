# Research Methodology

## Data Normalization & Unit Conversion
- Gold: Spot / Futures (USD/oz). Indian conversion: INR / 10g = (Price per oz / 31.1034768) * 10 * USDINR.
- Silver: Spot / Futures (USD/oz). Indian conversion: INR / kg = (Price per oz / 31.1034768) * 1000 * USDINR.

## Relative Value & Gold/Silver Ratio
- Gold/Silver Ratio = Price(Gold) / Price(Silver).
- Z-score = (Current Ratio - 252d Rolling Mean) / 252d Rolling Standard Deviation.

## Look-Ahead-Bias Protection
- Features calculated at date T rely strictly on observations up to date T.
- Target variables (`future_return_hd`) are computed via backward shifting (`shift(-h)`) for model training and historical evaluation.
