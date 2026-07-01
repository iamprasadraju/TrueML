import numpy as np
import plotly.graph_objects as go


def viz(func, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
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

    Examples
    --------
    >>> visualize(lambda x, y: np.abs(x - y))
    """

    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)

    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    # Use the class name for static/class methods; otherwise use the function name.
    plot_name = func.__qualname__.split(".")[0]
    fig.update_layout(
        title=plot_name,
        scene=dict(xaxis_title="y_true", yaxis_title="y_pred", zaxis_title="value"),
    )

    fig.show()
