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
    """Create a bar plot."""

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
