import numpy as np
from numpy.typing import ArrayLike


def train_test_split(
    *arrays: ArrayLike,
    train_split: float = 0.8,
    test_split: float = None,
    seed: float | int = 42,
):
    np_arrays = [np.asarray(arr) for arr in arrays]

    sample_size = np_arrays[0].shape[0]
    if not all(arr.shape[0] == sample_size for arr in np_arrays):
        raise ValueError("All input arrays must have the same number of samples (axis 0).")

    #seed the modern NumPy generator
    rng = np.random.default_rng(seed)
    
    # 4. Generate ONE set of indices to keep X and y synchronized
    shuffled_indices = rng.permutation(sample_size)
    
    # 5. Calculate the index integer threshold for the split
    split_idx = int(sample_size * train_split)
    
    # 6. Apply indices, split the arrays, and collect the outputs
    results = []
    for arr in np_arrays:
        shuffled_arr = arr[shuffled_indices]
        train_part = shuffled_arr[:split_idx]
        test_part = shuffled_arr[split_idx:]
        results.extend([train_part, test_part])
        
    return tuple(results)