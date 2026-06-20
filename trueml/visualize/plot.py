import plotly.graph_objects as go


def _in_jupyter():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False

class LivePlot:
    def __init__(self, title="Training Loss", labels=None, xlabel="Epoch", ylabel="Loss"):
        if isinstance(title, (int, float)):
            raise TypeError(
                "LivePlot is now a class, not a function. "
                "Use `plot = LivePlot(\"Training Loss\")` then `plot(value)` "
                "instead of `LivePlot(value)`."
            )

        self.title = title
        self.labels = labels if labels else ["loss"]
        self.n_traces = len(self.labels)
        self.values = [[] for _ in range(self.n_traces)]

        if _in_jupyter():
            self._init_plotly(xlabel, ylabel)
        else:
            self.fig = None

    def _init_plotly(self, xlabel, ylabel):
        try:
            from IPython.display import display
        except ImportError:
            self.fig = None
            return

        data = [
            go.Scatter(x=[], y=[], mode="lines", name=label)
            for label in self.labels
        ]
        self.fig = go.FigureWidget(data=data)
        self.fig.update_layout(
            title=self.title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
        )
        display(self.fig)

    def update(self, *values):
        if not values:
            return

        if len(values) == 1 and self.n_traces > 1:
            values = values * self.n_traces

        if len(values) != self.n_traces:
            raise ValueError(
                f"Expected {self.n_traces} value(s) for {self.n_traces} trace(s) "
                f"({self.labels}), got {len(values)}."
            )

        for i, v in enumerate(values):
            self.values[i].append(float(v))

        if self.fig is not None:
            self._update_plotly()

    def _update_plotly(self):
        epochs = list(range(len(self.values[0])))
        with self.fig.batch_update():
            for i, trace in enumerate(self.fig.data):
                trace.x = epochs
                trace.y = self.values[i]

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)
