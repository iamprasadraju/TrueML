import numpy as np
from numpy.typing import ArrayLike

def max(x: ArrayLike) -> float:
    """
    Compute the maximum of the input array.

    max: maximum is the highest numerical value in a dataset. It tells you the absolute upper boundary of your observations, reveals the maximum possible cost or measurement, and helps identify data limits or potential errors.
    
    Parameters
    ----------
    x : np.array
        Input array.
        
    Returns
    -------
    float
        Maximum of the input array.
    
    Examples
    --------
    >>> max(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    5.0
    """
    return np.max(x)
