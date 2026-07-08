import numpy as np
from numpy.typing import ArrayLike

def std(x: ArrayLike) -> float:
    """
    Compute the standard deviation of the input array.

    std: measures how dispersed the data is relative to the average (mean)
    
    Parameters
    ----------
    x : ArrayLike
        Input array.
        
    Returns
    -------
    float
        Standard deviation of the input array.

    formula
    -------
    σ = √(Σ(x - μ)² / n)
    
    where σ is the standard deviation, μ is the mean, and n is the number of elements.    
    Examples
    --------
    >>> std(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    1.4142135623730951
    """
    return np.std(x)