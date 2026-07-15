"""
Record and store metrics during model training.

The :class:`History` object is a lightweight key-value store that records
arbitrary named metrics at each training epoch. It is created automatically
by :class:`~trueml.linear_model.LinearModel` when ``history=True``.

Example
-------
>>> history = History()
>>> history.append(0, loss=0.5, accuracy=0.8)
>>> history.append(1, loss=0.3, accuracy=0.9)
>>> history["loss"]
[0.5, 0.3]
>>> history.keys()
dict_keys(['loss', 'accuracy'])
"""


class History:
    """Record and store metrics during model training.

    A lightweight key-value store that records arbitrary named metrics
    at each training epoch. Use :meth:`append` to log metrics and
    subscript notation (``history["loss"]``) to retrieve them.

    Examples
    --------
    >>> history = History()
    >>> history.append(0, loss=0.5, accuracy=0.8)
    >>> history.append(1, loss=0.3, accuracy=0.9)
    >>> history["loss"]
    [0.5, 0.3]
    >>> history.keys()
    dict_keys(['loss', 'accuracy'])
    >>> history.items()
    dict_items([('loss', [0.5, 0.3]), ('accuracy', [0.8, 0.9])])
    """

    def __init__(self):
        self.epochs: list[int] = []
        """List of epoch indices recorded so far."""

        self.history: dict[str, list[float]] = {}
        """Dictionary mapping metric names to lists of values."""

    def append(self, epoch: int, **metrics: float) -> None:
        """Record one epoch's worth of metrics.

        Parameters
        ----------
        epoch : int
            The epoch number (zero-indexed).
        **metrics : float
            Arbitrary keyword arguments where each key is a metric name
            and each value is the scalar score for this epoch.

        Examples
        --------
        >>> history = History()
        >>> history.append(0, loss=0.5, accuracy=0.8)
        >>> history.append(1, loss=0.3, accuracy=0.9)
        """
        self.epochs.append(epoch)
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)

    def __getitem__(self, key: str) -> list[float]:
        """Retrieve the recorded values for a given metric.

        Parameters
        ----------
        key : str
            The name of the metric to retrieve.

        Returns
        -------
        list[float]
            The list of recorded values for the metric.

        Raises
        ------
        KeyError
            If ``key`` has not been recorded.
        """
        return self.history[key]

    def keys(self):
        """Return the names of all recorded metrics.

        Returns
        -------
        dict_keys
            A view of all metric names.
        """
        return self.history.keys()

    def items(self):
        """Return all recorded metrics as ``(name, values)`` pairs.

        Returns
        -------
        dict_items
            A view of ``(metric_name, list_of_values)`` pairs.
        """
        return self.history.items()
