"""
Public package API for the custom PCA implementation.

This module re-exports the PCA class and utility functions so callers can use:
    from preprocessing.pca.src import PCA, scree_plot, frobenius_norm, relative_frobenius_norm, mse
"""

from .pca import PCA
from .pca_utils import frobenius_norm, relative_frobenius_norm, mse, scree_plot

__all__ = [
    "PCA",
    "frobenius_norm",
    "relative_frobenius_norm",
    "mse",
    "scree_plot",
]
