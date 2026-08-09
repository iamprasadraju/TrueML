import numpy as np


class MinMaxScaler:
    def __init__(
        self,
        feature_range=(0.0, 1.0),
    ):
        range_min, range_max = feature_range

        if range_min >= range_max:
            raise ValueError(
                "feature_range must be of the form (min, max) with min < max."
            )

        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)

        return self

    def transform(self, x):
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

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
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
