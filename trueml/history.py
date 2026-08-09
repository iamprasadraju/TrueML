class History:
    def __init__(self):
        self.epochs: list[int] = []
        self.history: dict[str, list[float]] = {}

    def append(self, epoch: int, **metrics: float) -> None:
        self.epochs.append(epoch)
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)

    def __getitem__(self, key: str) -> list[float]:
        return self.history[key]

    def keys(self):
        return self.history.keys()

    def items(self):
        return self.history.items()

    def plot(self, *metrices):
        import matplotlib.pyplot as plt

        if not metrices:
            metrices = self.history.keys()

        for metric in metrices:
            plt.plot(self.epochs, self.history[metric], label=metric)

        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.show()
