"""
Datasets — synthetic data generators for experiments and testing.

Generate reproducible synthetic datasets without requiring external
files. Useful for tutorials, unit tests, and quick prototyping of
new models or optimizers.

Separating data generation from model code keeps models focused
on prediction logic and makes it easy to create controlled
experimental conditions (known true parameters, specific noise
levels, etc.).

Available generators:
    make_regression(n, d, noise, seed)  — linear regression data
"""
