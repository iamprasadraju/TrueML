import numpy as np


class SimpleImputer:
    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
    ):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, x):
        x = np.asarray(x)

        if np.isnan(self.missing_values):
            mask = np.isnan(x.astype(float))
        else:
            mask = x == self.missing_values

        valid = x[~mask]

        if self.strategy == "mean":
            self.statistics_ = np.mean(valid.astype(float))

        elif self.strategy == "median":
            self.statistics_ = np.median(valid.astype(float))

        elif self.strategy == "most_frequent":
            values, counts = np.unique(valid, return_counts=True)
            self.statistics_ = values[np.argmax(counts)]

        elif self.strategy == "constant":
            self.statistics_ = self.fill_value

        else:
            raise ValueError(
                "strategy must be one of "
                "{'mean', 'median', 'most_frequent', 'constant'}"
            )

        return self

    def transform(self, x):
        if self.statistics_ is None:
            raise ValueError("Call 'fit' before 'transform'.")

        x = np.asarray(x).copy()

        if np.isnan(self.missing_values):
            mask = np.isnan(x.astype(float))
        else:
            mask = x == self.missing_values

        x[mask] = self.statistics_

        return x

    def fit_transform(self, x):
        return self.fit(x).transform(x)
