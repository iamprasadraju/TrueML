"""
tinymlx — minimal, high-performance machine learning library.

A no-abstraction mathematical sandbox where every operation in the
learning pipeline (forward pass, loss, gradient, update) is an
explicit first-class function you invoke.

Subpackages
-----------
linearmodel     — supervised linear models with explicit forward/backward
lossfunc        — loss functions for error measurement and gradient computation
activations     — element-wise nonlinearities (sigmoid, relu, tanh, ...)
linalg          — low-level linear algebra primitives (matmul, ...)
preprocessing   — data transformations (scalers, encoders)
metrics         — post-training evaluation scores
datasets        — synthetic data generators for experiments
"""

from . import linearmodel
from . import lossfunc
from . import activations
from . import linalg
from . import preprocessing
from . import metrics
from . import datasets

__version__ = "0.1.0"