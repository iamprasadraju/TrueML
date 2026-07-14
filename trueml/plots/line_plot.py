import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def line(
    y: ArrayLike,
    x: ArrayLike | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    marker: str | None = None,
    color: str = "C0",
    ls: str | None = None,
    alpha: float = 1.0,
    figsize: tuple[int, int] | None = None,
):
    """
    Create a line plot.

    A line plot visualizes the relationship between two variables by connecting
    data points with straight line segments. It is commonly used to show trends,
    changes over time, or continuous relationships.

    If ``x`` is not provided, the indices of ``y`` are used as the x-axis values.

    Args:
        y: Values for the y-axis.
        x: Values for the x-axis. If ``None``, the indices of ``y`` are used.
            Defaults to ``None``.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``None``.
        marker: Marker style for each data point (for example, ``"o"``,
            ``"s"``, or ``"^"``). Defaults to ``None``.
        color: Color of the line and markers. Defaults to ``"C0"``.
        ls: Line style. Common values include ``"-"``, ``"--"``, ``"-."``,
            and ``":"``. If ``None``, a solid line is used unless
            ``marker`` is specified, in which case only markers are drawn.
            Defaults to ``None``.
        alpha: Transparency of the line, ranging from ``0.0`` (fully
            transparent) to ``1.0`` (fully opaque). Defaults to ``1.0``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.

    Returns:
        None. Displays the line plot using Matplotlib.

    Example:
        >>> import numpy as np
        >>> from trueml.plots import line

        >>> y = np.array([0.1, 0.4, 0.2, 0.8])
        >>> line(y)

        >>> epochs = np.arange(1, 5)
        >>> line(
        ...     y,
        ...     x=epochs,
        ...     title="Training Loss",
        ...     xlabel="Epoch",
        ...     ylabel="Loss",
        ...     marker="o",
        ...     color="crimson",
        ... )
    """
    y = np.asarray(y)

    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    if ls is None:
        ls = "-" if marker is None else ""

    fig, ax = plt.subplots(figsize=figsize or (6, 4))

    ax.plot(
        x,
        y,
        color=color,
        linestyle=ls,
        marker=marker,
        alpha=alpha,
    )

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    plt.show()
