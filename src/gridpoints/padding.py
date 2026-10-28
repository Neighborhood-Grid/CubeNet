"""
Helper to complete point clouds so that they fit a grid.
"""

from .array_libs import arange, astype, empty


def pad(X, n_total, value="nan"):
    """Complete point clouds up to the size n_total, using fictious nan or inf points.

    Parameters
    ----------
    X : array (N, D), array (B, N, D), or list of arrays (N_b, D)
        One point cloud, a batch, or point clouds of different sizes.
    n_total : int
        Pad up to n_total points. Must be >= the size of the raw point clouds
    value : str, "nan" or "inf"
        Determines how fictitious points are filled.
        The kind of pading used will significantly affects the behavior of `gridpoints.sort`.
        - "nan": Fictitious points will placed at random positions
        in the grid.
        - "inf": Fictitious points will pushed toward the grid's boundaries.
        Note that value = "inf" will pad the point cloud with both +inf and -inf values.
    Returns
    -------
    X_padded : array (n_total, D) or (B, n_total, D)
    """
    assert value in ("nan", "inf"), f"value must be 'nan' or 'inf', got {value!r}"
    single = hasattr(X, 'ndim') and X.ndim == 2
    clouds = [X] if single else list(X)
    assert max(len(c) for c in clouds) <= n_total, "more points than n_total"

    ref = clouds[0]
    dtype = str(ref.dtype).replace('torch.', '')
    if 'float' not in dtype:   # integers cannot hold nan / inf
        dtype = 'float64'
    D = ref.shape[-1]
    out = empty((len(clouds), n_total, D), dtype, ref)
    out[...] = float(value)

    for b, cloud in enumerate(clouds):
        out[b, :len(cloud)] = astype(cloud, dtype)
        if value == "inf":   # point k gets the signs given by the bits of k, one per coordinate
            k = arange(n_total - len(cloud), 'int64', ref)[:, None]
            bits = (k >> (D - 1 - arange(D, 'int64', ref))[None, :]) & 1
            out[b, len(cloud):] = astype(1 - 2 * bits, dtype) * float("inf")

    return out[0] if single else out