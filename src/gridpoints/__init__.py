"""Monotonic Lagrangian sorting for NumPy, CuPy, and PyTorch arrays."""

from .mlg_sort import argsort, sort, invert_permutation

__all__ = (
    "argsort",
    "sort",
    "invert_permutation",
)

__version__ = "1.0.4"