from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SimpleImputer:
    """
    Simple imputer for replacing missing values.

    Parameters
    ----------
    missing_values : Any, default=np.nan
        The placeholder for missing values.

    strategy : {"mean", "median", "most_frequent", "constant"}, default="mean"
        Strategy used to replace missing values.

        - "mean": Replace missing values with the mean of each feature.
        - "median": Replace missing values with the median of each feature.
        - "most_frequent": Replace missing values with the most frequent value.
        - "constant": Replace missing values with `fill_value`.

    fill_value : Any, default=None
        Value used when `strategy="constant"`.

    Attributes
    ----------
    statistics_ : Any
        The computed replacement value learned during :meth:`fit`.
    """

    def __init__(
        self,
        missing_values: Any = np.nan,
        strategy: str = "mean",
        fill_value: Any = None,
    ) -> None:
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Any = None

    def fit(self, x: ArrayLike) -> Self:
        """
        Compute the replacement value from the input data.

        Parameters
        ----------
        x : ArrayLike
            Input data containing missing values.

        Returns
        -------
        Self
            Fitted imputer.
        """
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

    def transform(self, x: ArrayLike) -> NDArray:
        """
        Replace missing values using the statistic computed during :meth:`fit`.

        Parameters
        ----------
        x : ArrayLike
            Input data to transform.

        Returns
        -------
        NDArray
            Transformed array with missing values replaced.

        Raises
        ------
        ValueError
            If the imputer has not been fitted.
        """
        if self.statistics_ is None:
            raise ValueError("Call 'fit' before 'transform'.")

        x = np.asarray(x).copy()

        if np.isnan(self.missing_values):
            mask = np.isnan(x.astype(float))
        else:
            mask = x == self.missing_values

        x[mask] = self.statistics_

        return x

    def fit_transform(self, x: ArrayLike) -> NDArray:
        """
        Fit the imputer and transform the input data.

        Parameters
        ----------
        x : ArrayLike
            Input data.

        Returns
        -------
        NDArray
            Transformed array with missing values replaced.
        """
        return self.fit(x).transform(x)
