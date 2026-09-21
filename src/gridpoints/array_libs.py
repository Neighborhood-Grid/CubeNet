"""
Helpers:
Unified array syntax to support numpy, cupy and pytorch arrays

Here we just define basic primitives that we will need later 
like arange, copy, shuffle etc.
"""

#This have to be promoted to float64 and int64 for very big point clouds (more than 1 billion points)
_FLOAT = 'float32'
_INT = 'int32'

EPS = 10**(-5) if (_FLOAT == 'float32') else 10**(-10)


def get_dtype(dtp, xp):
    if dtp == 'float':
        dtp = _FLOAT
    elif dtp in ('int', 'integer'):
        dtp = _INT
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
    return X.astype(dtype)


def inplace_sort(X, axis):
    xp, backend_name = get_backend(X)
    if backend_name == 'torch':
        X.copy_(X.sort(dim=axis)[0])
    if backend_name == "cupy":
        X[...] = xp.sort(xp.ascontiguousarray(X), axis=axis)
    else:
        X.sort(axis=axis)


def empty(shape, dtype_str, reference_array):
    xp, backend_name = get_backend(reference_array)
    dtype = get_dtype(dtype_str, xp)
    if backend_name == 'torch':
        return xp.empty(shape, dtype=dtype, device=reference_array.device)
    return xp.empty(shape, dtype=dtype)


def arange(N, dtype_str, reference_array):
    xp, backend_name = get_backend(reference_array)
    dtype = get_dtype(dtype_str, xp)
    if backend_name == 'torch':
        return xp.arange(N, dtype=dtype, device=reference_array.device)
    return xp.arange(N, dtype=dtype)

def argmax(arr, axis):
    xp, backend_name = get_backend(arr)
    if backend_name == 'torch':
        if arr.dtype == xp.bool:
            arr = arr.to(xp.int32)
        return arr.argmax(dim=axis)
    return arr.argmax(axis=axis)

def random_noise(shape, reference_array):
    xp, backend_name = get_backend(reference_array)
    if backend_name == 'torch':
        return EPS * xp.rand(*shape, device=reference_array.device)
    elif backend_name == 'cupy':
        return EPS * xp.random.rand(*shape)
    else:
        rng = xp.random.default_rng(42)
        return EPS * rng.random(shape)

def copy(arr):
    _, backend_name = get_backend(arr)
    if backend_name == 'torch':
        return arr.clone()
    return arr.copy()

def nan_to_num(arr):
    xp, backend = get_backend(arr)
    out = copy(arr)

    for d in range(arr.shape[1]):
        x = arr[:, d]
        finite = xp.isfinite(x)
        vals = x[finite]

        out[xp.isposinf(x), d] = vals.max() + 1
        out[xp.isneginf(x), d] = vals.min() - 1

        nan = xp.isnan(x)
        n = nan.sum().item() if backend == 'torch' else int(nan.sum())
        if n:
            idx = (xp.randint(len(vals), (n,), device=arr.device)
                   if backend == 'torch' else xp.random.randint(len(vals), size=n))
            out[nan, d] = vals[idx]

    return out + random_noise(out.shape, out)

def all_true(tensor):
    _, backend_name = get_backend(tensor)
    if backend_name == 'torch':
        return tensor.all().item()
    return bool(tensor.all())

def random_permutation(arr):
    xp, backend_name = get_backend(arr)
    if backend_name == 'torch':
        perm = xp.randperm(len(arr), device=arr.device)
        return arr[perm]
    return xp.random.permutation(arr)

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

def argpartition(X, mid, axis):
    xp, backend_name = get_backend(X)
    if backend_name == 'torch':
        return xp.argsort(X, dim=axis)
    else:
        return xp.argpartition(X, mid - 1, axis=-1)

def take_along_axis(idx, val, axis):
    xp, backend_name = get_backend(val)
    if backend_name == 'torch':
        return xp.take_along_dim(idx, val, dim=axis)
    else:
        return xp.take_along_axis(idx, val, axis=axis)

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
