"""
Helper to complete point clouds so that they fit a grid.
"""

from .array_libs import arange, astype, empty, get_backend


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
    is_array = hasattr(X, 'ndim')
    ref = X if is_array else X[0]
    assert (X.shape[-2] if is_array else max(len(c) for c in X)) <= n_total, "more points than n_total"

    dtype = str(ref.dtype).replace('torch.', '')
    if 'float' not in dtype:   # integers cannot hold nan / inf
        dtype = 'float64'
    D = ref.shape[-1]
    shape = (n_total, D) if is_array and X.ndim == 2 else (len(X), n_total, D)

    # 1. Prefill with nan, then copy the true points in.
    out = empty(shape, dtype, ref)
    out[...] = float('nan')
    if is_array:
        out[..., :X.shape[-2], :] = astype(X, dtype)
    else:
        for b, cloud in enumerate(X):
            out[b, :len(cloud)] = astype(cloud, dtype)

    # 2. Turn the nans into +/-inf: point k, coordinate d gets the sign of bit d of k.
    if value == "inf":
        xp, _ = get_backend(out)
        k = arange(n_total, 'int64', ref)[:, None]
        d = arange(D, 'int64', ref)[None, :]
        signs = astype(1 - 2 * ((k // 2 ** (D - 1 - d)) % 2), dtype)   # (n_total, D)
        out = xp.where(xp.isnan(out), signs * float('inf'), out)

    return out