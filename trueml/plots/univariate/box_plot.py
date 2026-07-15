from collections.abc import Sequence

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


def box(
    x: ArrayLike | Sequence[ArrayLike],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] | None = None,
    labels: Sequence[str] | None = None,
    showfliers: bool = True,
    vert: bool = True,
):
    """Create a box plot.

    A box plot (also known as a box-and-whisker plot) summarizes the
    distribution of one or more numerical datasets using the median,
    quartiles, whiskers, and potential outliers. It is useful for
    comparing the spread, central tendency, skewness, and outliers
    across datasets.

    Args:
        x: One-dimensional numerical data or a sequence of numerical
            datasets. If multiple datasets are provided, a separate
            box plot is drawn for each dataset.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``None``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.
        labels: Labels for each box when plotting multiple datasets.
            Defaults to ``None``.
        showfliers: If ``True``, display outliers. Defaults to ``True``.
        vert: If ``True``, draw vertical box plots. If ``False``,
            draw horizontal box plots. Defaults to ``True``.

    Returns:
        None. Displays the box plot using Matplotlib.

    Example:
        >>> box(
        ...     x=[1, 2, 3, 4, 5, 6, 7],
        ...     title="Sample Data",
        ...     ylabel="Value",
        ... )

        >>> box(
        ...     x=[
        ...         [1, 2, 3, 4, 5],
        ...         [2, 3, 4, 5, 6],
        ...     ],
        ...     labels=["Dataset A", "Dataset B"],
        ...     title="Dataset Comparison",
        ...     ylabel="Value",
        ...     showfliers=False,
        ... )
    """

    fig, ax = plt.subplots(figsize=figsize or (6, 4))

    ax.boxplot(
        x,
        tick_labels=labels,
        showfliers=showfliers,
        orientation="vertical" if vert else "horizontal",
    )

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()
