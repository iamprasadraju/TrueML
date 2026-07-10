import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Literal


def hist(
    X: ArrayLike,
    bins: int = 50,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Frequency",
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
    color: str = "C0",
    alpha: float = 0.7,
    figsize: tuple[int, int] | None = None,
):
    """Create a histogram.

    Args:
        X: The data to plot.
        bins:  "bin" (or "bucket") the range of values— divide the entire range of values into a series of intervals
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        histtype: The type of histogram to plot.
        color: The color of the bars.
        alpha: The transparency of the bars.
        figsize: The size of the figure.
    
    Returns:
        None (displays the plot using matplotlib)
    
    Example:
        >>> hist(X, bins=50, title="Histogram", xlabel="X", ylabel="Frequency")
        >>> hist(X, color="red", alpha=0.5)
    """

    fig, ax = plt.subplots(figsize=figsize or (6, 4))

    ax.hist(
        X,
        bins=bins,
        histtype=histtype,
        color=color,
        alpha=alpha,
    )

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    plt.show()