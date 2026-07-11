import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Literal


def hist(
    x: ArrayLike,
    bins: int = 50,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Frequency",
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
    color: str = "C0",
    alpha: float = 0.7,
    figsize: tuple[int, int] | None = None,
    density: bool = False,
    range: tuple[float, float] | None = None,
):
    """
    Create a histogram.

    A histogram visualizes the distribution of a numerical variable by grouping
    observations into bins and displaying the frequency or probability density
    within each bin.

    Args:
        x: One-dimensional numerical data to plot.
        bins: Number of equal-width bins used to group the data.
            Defaults to ``50``.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``"Frequency"``.
        histtype: Type of histogram to draw. One of ``"bar"``,
            ``"barstacked"``, ``"step"``, or ``"stepfilled"``.
            Defaults to ``"bar"``.
        color: Color of the histogram. Defaults to ``"C0"``.
        alpha: Transparency of the histogram, ranging from ``0.0`` (fully
            transparent) to ``1.0`` (fully opaque). Defaults to ``0.7``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.
        density: If ``True``, normalize the histogram so that the total area
            under the bars equals ``1``. If ``False``, display frequencies.
            Defaults to ``False``.
        range: Lower and upper range of the bins as ``(min, max)``.
            Values outside this range are ignored. Defaults to ``None``.

    Returns:
        None. Displays the histogram using Matplotlib.

    Example:
        >>> hist(
        ...     x=ages,
        ...     bins=20,
        ...     title="Age Distribution",
        ...     xlabel="Age (years)",
        ...     ylabel="Density",
        ...     histtype="stepfilled",
        ...     color="steelblue",
        ...     alpha=0.6,
        ...     figsize=(8, 5),
        ...     density=True,
        ...     range=(18, 80),
        ... )
    """

    fig, ax = plt.subplots(figsize=figsize or (6, 4))

    ax.hist(
        x,
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
