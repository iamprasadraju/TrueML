import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Literal


def line(
    X: ArrayLike,
    Y: ArrayLike,
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
    """Create a line plot.

    Args:
        X: The x-axis data.
        Y: The y-axis data.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        marker: The marker style. {'o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '', ...}
        color: The color of the line.
        ls: The linestyle of the line. {'-', '--', '-.', ':', '', ...}
        alpha: The transparency of the line.
        figsize: The size of the figure.

    Returns:
        None (displays the plot using matplotlib)

    Example:
        line(X, Y, title="Line Plot", xlabel="X", ylabel="Y")
        line(X, Y, color="red", alpha=0.5)
        line(X, Y, marker="o", color="blue")
    """
    if ls is None:
        ls = "-" if marker is None else ""

    fig, ax = plt.subplots(figsize=figsize or (6, 4))

    ax.plot(
        X,
        Y,
        color=color,
        ls=ls,
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
