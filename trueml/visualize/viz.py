import numpy as np
import plotly.graph_objects as go


def viz2d(func, x_range=(-5, 5), resolution=50, figsize=(500, 500)):
    """
    Visualize a single-variable function as a 2D line plot.

    Parameters
    ----------
    func : callable
        Function accepting a NumPy array and returning a NumPy array.

    x_range : tuple, default=(-5, 5)
        Range of values for the x-axis.

    resolution : int, default=50
        Number of points sampled along the x-axis.

    figsize : tuple, default=(500, 500)
        Figure size as (width, height) in pixels.

    Examples
    --------
    >>> from trueml.activations import sigmoid
    >>> viz2d(sigmoid, x_range=(-10, 10))
    """
    x = np.linspace(*x_range, resolution)
    y = func(x)

    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines")])
    plot_name = func.__qualname__.split(".")[0]
    fig.update_layout(
        template="plotly_white",
        title=plot_name,
        xaxis_title="x",
        yaxis_title="value",
        width=figsize[0],
        height=figsize[1],
    )

    fig.show()


def viz3d(func, x_range=(-5, 5), y_range=(-5, 5), resolution=50, figsize=(500, 500)):
    """
    Visualize a two-variable function as a 3D surface.

    Parameters
    ----------
    func : callable
        Function accepting two NumPy arrays (X, Y) and returning Z.

    x_range : tuple, default=(-5, 5)
        Range of values for the x-axis.

    y_range : tuple, default=(-5, 5)
        Range of values for the y-axis.

    resolution : int, default=50
        Number of points sampled along each axis.

    figsize : tuple, default=(500, 500)
        Figure size as (width, height) in pixels.

    Examples
    --------
    >>> viz3d(lambda x, y: np.abs(x - y))
    """
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)

    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="viridis")])
    plot_name = func.__qualname__.split(".")[0]
    fig.update_layout(
        template="plotly_white",
        title=plot_name,
        width=figsize[0],
        height=figsize[1],
        scene=dict(xaxis_title="y_true", yaxis_title="y_pred", zaxis_title="value"),
    )

    fig.show()
