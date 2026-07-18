from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MinMaxScaler:
    """
    Scale features to a specified range.

    The transformation is defined as::

        X_scaled = ((X - X_min) / (X_max - X_min))
                   * (range_max - range_min)
                   + range_min

    Each feature is scaled independently.

    Parameters
    ----------
    feature_range : tuple[float, float], default=(0.0, 1.0)
        Desired range of transformed data.

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Minimum value observed for each feature.

    max_ : ndarray of shape (n_features,)
        Maximum value observed for each feature.
    """

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        range_min, range_max = feature_range

        if range_min >= range_max:
            raise ValueError(
                "feature_range must be of the form (min, max) with min < max."
            )

        self.feature_range = feature_range
        self.min_: NDArray | None = None
        self.max_: NDArray | None = None

    def fit(self, x: ArrayLike) -> Self:
        """
        Compute the minimum and maximum value of each feature.

        Parameters
        ----------
        x : ArrayLike of shape (n_samples,) or (n_samples, n_features)
            Input data.

        Returns
        -------
        Self
            Fitted scaler.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)

        return self

    def transform(self, x: ArrayLike) -> NDArray:
        """
        Scale input data using the fitted statistics.

        Parameters
        ----------
        x : ArrayLike of shape (n_samples,) or (n_samples, n_features)
            Input data.

        Returns
        -------
        NDArray
            Scaled data.
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("MinMaxScaler has not been fitted.")

        x = np.asarray(x, dtype=float)

        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(-1, 1)

        scale = self.max_ - self.min_
        scale = np.where(scale == 0, 1, scale)

        range_min, range_max = self.feature_range

        x_scaled = ((x - self.min_) / scale) * (range_max - range_min) + range_min

        if is_1d:
            return x_scaled.ravel()

        return x_scaled

    def fit_transform(self, x: ArrayLike) -> NDArray:
        """
        Fit the scaler to the data and transform it.

        Parameters
        ----------
        x : ArrayLike of shape (n_samples,) or (n_samples, n_features)
            Input data.

        Returns
        -------
        NDArray
            Scaled data.
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x: ArrayLike) -> NDArray:
        """
        Undo the scaling transformation.

        Parameters
        ----------
        x : ArrayLike of shape (n_samples,) or (n_samples, n_features)
            Scaled data.

        Returns
        -------
        NDArray
            Data transformed back to the original scale.
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("MinMaxScaler has not been fitted.")

        x = np.asarray(x, dtype=float)

        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(-1, 1)

        scale = self.max_ - self.min_
        scale = np.where(scale == 0, 1, scale)

        range_min, range_max = self.feature_range

        x_original = ((x - range_min) / (range_max - range_min)) * scale + self.min_

        if is_1d:
            return x_original.ravel()

        return x_original
