"""
Helpers:
Unified array syntax to support numpy, cupy and pytorch arrays

Here we just define basic primitives that we will need later
like arange, copy, shuffle etc.
"""


def precision(N):
    """Names of the (float, int) dtypes suited to point clouds of N points.

    - float32 resolves the gaps (~1/N) between neighbouring points only up to
      a few 10**4 points, float64 is used beyond.
    - the diagonal sorts store values up to 6N: int32 is safe up to N ~ 3.5e8.
    """
    return ('float32' if N <= 2**16 else 'float64'), ('int32' if 6 * N < 2**31 else 'int64')


def get_dtype(dtp, xp):
    if dtp == 'bool':
        return getattr(xp, 'bool_', None) or xp.bool
    return getattr(xp, dtp) if isinstance(dtp, str) else dtp


def get_backend(X):
    module = type(X).__module__.split('.')[0]
    if module == 'torch':
        import torch
        return torch, 'torch'
    elif module == 'cupy':
        import cupy
        return cupy, 'cupy'
    else:
        import numpy
        return numpy, 'numpy'

def astype(X, dtype_str, reference_array=None):
    xp, backend_name = get_backend(X)
    dtype = get_dtype(dtype_str, xp)
    if backend_name == 'torch':
        device = reference_array.device if reference_array is not None else X.device
        return X.to(dtype=dtype, device=device)
    return X.astype(dtype, copy=False)


def inplace_sort(X, axis):
    xp, backend_name = get_backend(X)
    if backend_name == 'torch':
        X.copy_(X.sort(dim=axis)[0])
    elif backend_name == "cupy":
        X[...] = xp.sort(xp.ascontiguousarray(X), axis=axis)
    else:
        X.sort(axis=axis)


def sort(X, axis):
    xp, backend_name = get_backend(X)
    return X.sort(dim=axis)[0] if backend_name == 'torch' else xp.sort(X, axis=axis)


def _alloc(fn, shape, dtype_str, reference_array):
    xp, backend_name = get_backend(reference_array)
    dtype = get_dtype(dtype_str, xp)
    if backend_name == 'torch':
        return getattr(xp, fn)(shape, dtype=dtype, device=reference_array.device)
    return getattr(xp, fn)(shape, dtype=dtype)

def empty(shape, dtype_str, reference_array):
    return _alloc('empty', shape, dtype_str, reference_array)

def zeros(shape, dtype_str, reference_array):
    return _alloc('zeros', shape, dtype_str, reference_array)

def ones(shape, dtype_str, reference_array):
    return _alloc('ones', shape, dtype_str, reference_array)


def arange(N, dtype_str, reference_array):
    xp, backend_name = get_backend(reference_array)
    dtype = get_dtype(dtype_str, xp)
    if backend_name == 'torch':
        return xp.arange(N, dtype=dtype, device=reference_array.device)
    return xp.arange(N, dtype=dtype)

def random(shape, dtype_str, reference_array):
    """Uniform random numbers in [0, 1)."""
    xp, backend_name = get_backend(reference_array)
    dtype = get_dtype(dtype_str, xp)
    if backend_name == 'torch':
        return xp.rand(shape, dtype=dtype, device=reference_array.device)
    elif backend_name == 'cupy':
        return xp.random.rand(*shape).astype(dtype)
    return xp.random.default_rng(42).random(shape, dtype=dtype)

def copy(arr):
    _, backend_name = get_backend(arr)
    if backend_name == 'torch':
        return arr.clone()
    return arr.copy()

def amax(X, axis):
    return X.amax(axis) if get_backend(X)[1] == 'torch' else X.max(axis)

def amin(X, axis):
    return X.amin(axis) if get_backend(X)[1] == 'torch' else X.min(axis)

def all_true(tensor):
    return bool(tensor.all())

def any_true(tensor):
    return bool(tensor.any())


def nan_infs_to_num_and_tiebreaker(arr):
    """Make a batch of clouds (B, N, D) finite, and break ties with a tiny noise.

    +inf / -inf are put just beyond the finite points of their cloud and axis
    (they end up on the border of the grid), NaN are replaced by the value of a
    random finite point of their cloud and axis (they end up anywhere).
    """
    xp, _ = get_backend(arr)
    out = arr
    finite = xp.isfinite(arr)
    if not all_true(finite):
        hi = amax(xp.where(finite, arr, -xp.inf), 1)[:, None, :]
        lo = amin(xp.where(finite, arr, xp.inf), 1)[:, None, :]
        out = xp.where(xp.isposinf(arr), hi + 1, arr)
        out = xp.where(xp.isneginf(arr), lo - 1, out)
        nan = xp.isnan(arr)
        if any_true(nan):
            finite_first = sort(xp.where(finite, arr, xp.inf), 1)
            n_finite = finite.sum(1)[:, None, :]
            pick = astype(random(arr.shape, arr.dtype, arr) * n_finite, 'int64')
            out = xp.where(nan, take_along_axis(finite_first, pick, 1), out)
    eps = 1e-5 if '32' in str(arr.dtype) else 1e-10
    return out + eps * random(out.shape, arr.dtype, out)

def flip(X, axis=None):
    xp, _ = get_backend(X)
    if axis is None or axis == () or axis == []:
        return X

    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    if hasattr(X, "flip") and callable(X.flip):
        return X.flip(axis)

    return xp.flip(X, axis=axis)

def argsort(X):
    xp, backend_name = get_backend(X)
    return xp.argsort(X, dim=-1) if backend_name == 'torch' else xp.argsort(X, axis=-1)

def argpartition(X, mid):
    """Indices that put the `mid` smallest values of the last axis first."""
    xp, backend_name = get_backend(X)
    if backend_name == 'torch':
        return xp.argsort(X, dim=-1)
    else:
        return xp.argpartition(X, mid - 1, axis=-1)

def take_along_axis(arr, idx, axis):
    xp, backend_name = get_backend(arr)
    if backend_name == 'torch':
        return xp.take_along_dim(arr, idx.long(), dim=axis)
    else:
        return xp.take_along_axis(arr, idx, axis=axis)

def concatenate(arrays, axis = 0):
    xp, name = get_backend(arrays[0])
    if name == "torch":
        return xp.cat(list(arrays), dim=axis)
    return xp.concatenate(arrays, axis=axis)

def transpose(X, axes):
    _, name = get_backend(X)
    if name == "torch":
        return X.permute(*axes)
    return X.transpose(axes)
