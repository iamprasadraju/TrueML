import numpy as np
from numpy.typing import ArrayLike


class MinMaxScaler:
    """
    Scale features to a specified range.

    MinMaxScaler is a preprocessing technique that linearly rescales each
    feature independently by subtracting its minimum value and dividing by
    its range (maximum - minimum), then mapping the result to the desired
    feature range.

    The transformation is defined as::

        X_scaled = ((X - X_min) / (X_max - X_min))
                   * (range_max - range_min)
                   + range_min

    If a feature has zero range (all values are identical), its scale is
    treated as 1 to avoid division by zero. Such features are mapped to
    the minimum value of the specified feature range.

    Attributes:
        min_ (numpy.ndarray):
            Minimum value observed for each feature during fitting.

        max_ (numpy.ndarray):
            Maximum value observed for each feature during fitting.

        feature_range (tuple[float, float]):
            Desired range of transformed data.

    Examples:
        >>> import numpy as np
        >>> from trueml.preprocessing import MinMaxScaler

        >>> X = np.array([
        ...     [1, 10],
        ...     [2, 20],
        ...     [3, 30],
        ... ])

        >>> scaler = MinMaxScaler()
        >>> X_scaled = scaler.fit_transform(X)

        >>> X_scaled
        array([[0. , 0. ],
               [0.5, 0.5],
               [1. , 1. ]])

        >>> scaler.min_
        array([ 1., 10.])

        >>> scaler.max_
        array([ 3., 30.])

        >>> scaler = MinMaxScaler(feature_range=(-1, 1))
        >>> scaler.fit_transform(np.array([1, 2, 3]))
        array([-1.,  0.,  1.])
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        """
        Initialize a MinMaxScaler.

        Args:
            feature_range:
                Desired range of transformed data as ``(min, max)``.

        Raises:
            ValueError:
                If ``feature_range`` is not of the form ``(min, max)``
                with ``min < max``.
        """
        range_min, range_max = feature_range

        if range_min >= range_max:
            raise ValueError(
                "feature_range must be of the form (min, max) with min < max."
            )

        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, x: ArrayLike):
        """
        Compute the minimum and maximum value of each feature.

        Args:
            x (ArrayLike):
                Input data of shape ``(n_samples, n_features)`` or
                ``(n_samples,)``.

        Returns:
            MinMaxScaler:
                The fitted scaler instance.
        """
        x = np.asarray(x, dtype=float)

        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)

        return self

    def transform(self, x: ArrayLike):
        """
        Scale input data using the fitted feature-wise statistics.

        Args:
            x (ArrayLike):
                Input data of shape ``(n_samples, n_features)`` or
                ``(n_samples,)``.

        Returns:
            numpy.ndarray:
                The scaled data with values in the specified
                ``feature_range``.

        Raises:
            ValueError:
                If the scaler has not been fitted before calling
                ``transform``.
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("MinMaxScaler has not been fitted.")

        x = np.asarray(x, dtype=float)

        scale = self.max_ - self.min_
        scale[scale == 0] = 1  # Prevent division by zero

        range_min, range_max = self.feature_range

        return (x - self.min_) / scale * (range_max - range_min) + range_min

    def fit_transform(self, x: ArrayLike):
        """
        Fit the scaler to the data and immediately transform it.

        This is equivalent to calling::

            scaler.fit(x)
            scaler.transform(x)

        Args:
            x (ArrayLike):
                Input data of shape ``(n_samples, n_features)`` or
                ``(n_samples,)``.

        Returns:
            numpy.ndarray:
                The scaled data with values in the specified
                ``feature_range``.
        """
        return self.fit(x).transform(x)
