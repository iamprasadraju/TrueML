from __future__ import annotations

import numpy as np


def train_test_split(*arrays, train_split=0.8, test_split=None, seed=42):
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
