# Probabilistic Forecasting & Analogue Engine

The historical analogue engine calculates normalized weighted Euclidean distance between today's feature vector and historical daily observations:

$$d(x_T, x_t) = \sqrt{\frac{\sum_{i} w_i \cdot \left(\frac{x_{T,i} - x_{t,i}}{\sigma_i}\right)^2}{\sum_i w_i}}$$

Top $K$ closest historical observations are selected to compute forward return probability distributions across 1D, 3D, 5D, 10D, 20D, and 60D horizons.
