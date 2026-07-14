import numpy as np
from numpy.typing import ArrayLike


class LabelEncoder:
    """Encode categorical labels as consecutive integers.

    Example:
        >>> encoder = LabelEncoder()
        >>> y = ["cat", "dog", "cat", "bird"]
        >>> encoded = encoder.fit_transform(y)
        >>> encoded
        array([1, 2, 1, 0])

        >>> encoder.inverse_transform(encoded)
        array(['cat', 'dog', 'cat', 'bird'], dtype='<U4')
    """

    def __init__(self):
        # Sorted unique class labels.
        self.classes_ = None

        # Mapping from original label -> integer index.
        self.mapping_ = None

    def fit(self, y: ArrayLike):
        """Learn the unique class labels.

        Args:
            y: Array-like sequence of categorical labels.

        Returns:
            The fitted LabelEncoder instance.
        """
        y = np.asarray(y)

        # Store the sorted unique labels.
        self.classes_ = np.unique(y)

        # Create a mapping from label to integer.
        self.mapping_ = {label: idx for idx, label in enumerate(self.classes_)}

        return self

    def transform(self, y: ArrayLike):
        """Encode labels as integers.

        Args:
            y: Array-like sequence of labels.

        Returns:
            A NumPy array containing integer-encoded labels.
        """
        y = np.asarray(y)

        return np.array(
            [self.mapping_[label] for label in y],
            dtype=int,
        )

    def inverse_transform(self, y: ArrayLike):
        """Convert encoded integers back to the original labels.

        Args:
            y: Array-like sequence of integer-encoded labels.

        Returns:
            A NumPy array containing the original labels.

        Raises:
            ValueError: If any encoded label is out of range.
        """
        y = np.asarray(y)

        # Validate encoded labels.
        if np.any((y < 0) | (y >= len(self.classes_))):
            raise ValueError("Encoded labels contain invalid values.")

        return self.classes_[y]

    def fit_transform(self, y: ArrayLike):
        """Fit the encoder and transform the labels.

        Args:
            y: Array-like sequence of categorical labels.

        Returns:
            A NumPy array containing integer-encoded labels.
        """
        return self.fit(y).transform(y)
