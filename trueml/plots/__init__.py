from . import bivariate, multivariate, univariate
from .bivariate import line, scatter
from .contour_plot import contour
from .function_plot import function2d, function3d
from .imshow import imshow
from .multivariate import heatmap
from .plot_history import history
from .univariate import bar, box, hist

__all__ = [
    # Subpackages
    "univariate",
    "bivariate",
    "multivariate",
    # Plot functions
    "bar",
    "box",
    "hist",
    "line",
    "scatter",
    "heatmap",
    "contour",
    "function2d",
    "function3d",
    "imshow",
    "history",
]
