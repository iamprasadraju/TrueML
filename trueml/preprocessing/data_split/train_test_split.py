import numpy as np
from numpy.typing import ArrayLike


def train_test_split(
    *arrays: ArrayLike,
    train_split: float = 0.8,
    test_split: float | None = None,
    seed: float | int = 42,
):
    """
    Randomly shuffle and split one or more arrays into training and testing sets.

    All input arrays must have the same number of samples along the first
    dimension. The arrays are shuffled using a single permutation so that
    corresponding samples remain aligned across all inputs.

    Parameters
    ----------
    *arrays : ArrayLike
        One or more input arrays to split. All arrays must have the same
        length along axis 0.

    train_split : float, default=0.8
        Proportion of samples to include in the training set. Must be between
        0 and 1.

    test_split : float, optional
        Proportion of samples to include in the test set. Currently unused.
        If provided, it should satisfy
        ``train_split + test_split == 1``.

    seed : int or float, default=42
        Seed used to initialize NumPy's random number generator for
        reproducible shuffling.

    Returns
    -------
    tuple
        Returns the train and test splits for each input array in the same
        order they were provided.

        For example:

        - One input array:
          ``(X_train, X_test)``

        - Two input arrays:
          ``(X_train, X_test, y_train, y_test)``

        - Three input arrays:
          ``(X_train, X_test, y_train, y_test, z_train, z_test)``

    Raises
    ------
    ValueError
        If the input arrays do not all have the same number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.arange(10)

    Split features and labels:

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, train_split=0.8, seed=42
    ... )

    Split a single array:

    >>> train, test = train_test_split(np.arange(10), train_split=0.7)

    Notes
    -----
    The function uses ``numpy.random.default_rng`` to generate a single
    permutation of the sample indices. The same permutation is applied to
    every input array, ensuring that corresponding samples remain synchronized.
    """
    np_arrays = [np.asarray(arr) for arr in arrays]

    sample_size = np_arrays[0].shape[0]
    if not all(arr.shape[0] == sample_size for arr in np_arrays):
        raise ValueError(
            "All input arrays must have the same number of samples (axis 0)."
        )

    rng = np.random.default_rng(seed)

    shuffled_indices = rng.permutation(sample_size)

    split_idx = int(sample_size * train_split)

    results = []
    for arr in np_arrays:
        shuffled_arr = arr[shuffled_indices]
        train_part = shuffled_arr[:split_idx]
        test_part = shuffled_arr[split_idx:]
        results.extend([train_part, test_part])

    return tuple(results)
