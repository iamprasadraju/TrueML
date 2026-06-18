"""
Preprocessing — data transformations applied before training.

Standalone transformers that apply deterministic or fitted transforms
to input data. Not part of the model pipeline — use these to prepare
data before passing it to model.forward().

The separation clarifies responsibility: models make predictions,
preprocessing prepares inputs. No preprocessing state leaks into
model parameters.

Future transformers:
    StandardScaler    — zero mean, unit variance
    MinMaxScaler      — scale to [0, 1] range
    OneHotEncoder     — categorical encoding
"""
