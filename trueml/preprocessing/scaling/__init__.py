"""
What is Feature Scaling?

Feature scaling is a data preprocessing technique used in machine learning to normalize or standardize the range of independent variables (features).

Since features in a dataset may have very different units and scales (e.g., age in years vs. income in dollars), models that rely on distance or gradient calculations can be biased toward features with larger numeric ranges. Feature scaling ensures that all features contribute proportionally to the model.

Why Feature Scaling Matters

- Improves model performance: Algorithms like gradient descent converge faster when features are normalized, since they don’t have to “zig-zag” across uneven scales

- Interpretability: Standardized features (mean 0, variance 1) make it easier to compare the relative importance of coefficients in linear models

- Better Accuracy: Distance-based models such as k-nearest neighbors (KNN), k-means, and support vector machines (SVMs) perform more reliably with scaled features

- Faster Convergence: Neural networks and gradient descent optimizers reach optimal solutions more quickly when features are scaled
"""

from .min_max_scaler import MinMaxScaler

__all__ = [
    "MinMaxScaler",
]
