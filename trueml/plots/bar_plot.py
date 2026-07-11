import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Any, Literal


def bar(
    x: ArrayLike,
    height: ArrayLike,
    width: float | ArrayLike = 0.8,
    bottom: float | ArrayLike | None = None,
    *,
    align: Literal["center", "edge"] = "center",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] | None = None,
    data: Any | None = None,
    color: str | None = None,
    alpha: float = 1.0,
):
    """
    Create a bar plot.

    A bar plot compares numerical values across discrete categories. Each bar
    represents a category, and its height corresponds to the associated value.

    Args:
        x: Category labels or x-axis positions for the bars.
        height: Heights of the bars.
        width: Width of the bars. Defaults to ``0.8``.
        bottom: Starting y-coordinate of the bars. Defaults to ``None``.
        align: Alignment of the bars. Must be ``"center"`` or ``"edge"``.
            Defaults to ``"center"``.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``None``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.
        data: Object supplying variables referenced by ``x`` and ``height``.
            Defaults to ``None``.
        color: Color of the bars. Defaults to Matplotlib's default color.
        alpha: Transparency of the bars, ranging from ``0.0`` (fully transparent)
            to ``1.0`` (fully opaque). Defaults to ``1.0``.

    Returns:
        None. Displays the bar plot using Matplotlib.

    Example:
        >>> bar(
        ...     x=["A", "B", "C"],
        ...     height=[10, 15, 8],
        ...     title="Sales by Category",
        ...     xlabel="Category",
        ...     ylabel="Sales",
        ...     color="steelblue",
        ...     alpha=0.8,
        ... )
    """

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()

    plt.bar(
        x,
        height,
        width=width,
        bottom=bottom,
        align=align,
        data=data,
        color=color,
        alpha=alpha,
    )

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    plt.show()
