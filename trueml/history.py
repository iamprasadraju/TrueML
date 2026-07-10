class History:
    """
    Record and store metrics during model training.

    Example
    -------
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
        self.epochs = []
        self.history = {}

    def append(self, epoch: int, **metrics):
        self.epochs.append(epoch)
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)

    def __getitem__(self, key: str) -> list:
        return self.history[key]

    def keys(self) -> list:
        return self.history.keys()

    def items(self) -> list:
        return self.history.items()
