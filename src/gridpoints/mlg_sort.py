"""
Main fonctions of the library are defined here: 
argsort and sort
"""


import math
import warnings
from .array_libs import (
    arange, get_backend, all_true, 
    inplace_sort, empty, astype, nan_to_num, 
    random_permutation, argmax, copy, flip
)
from .diagonal_tricks import sort_diag_trick, sort_tridiag_trick
from .kdtree import kdtree_order

# ============================================
#                   Sorting API
# ============================================

def argsort(X, gridshape, level=2, init="kdtree", n_iter=50, verbose = 2):
    """Compute a multi-directional monotonic sorted ordering of a point cloud.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input point cloud.
    gridshape : tuple of int
        Target shape of the sorting grid. The product of the dimensions must 
        equal N and its length must equal D. E.g., for X of shape (42, 2), 
        `shape` can be (6, 7); for (60, 3), `shape` can be (3, 4, 5).

        Note: Point clouds can be padded with +/- infinity placeholders to reach 
        a valid grid size. NaNs can also be used, but will be placed randomly 
        rather than pushed to grid boundaries.
    level : {1, 2, 3}, default=2
        Maximum order of lattice directions considered. Higher values improve 
        grid quality at the expense of computational speed. Note that `level=3` 
        is only supported if D >= 3.
    init : {'kdtree', 'shuffle'} or None, default="kdtree"
        Initialization method for the permutation order. Try init = None
        if the point cloud is already nearly sorted.
    n_iter : int, default=50
        Maximum number of optimization iterations.
    verbose : int, default 2
        0 = silent, 1 = log progress, 2 = 1 + warn if convergence failed

    Returns
    -------
    order : ndarray of shape (N,)
        Permutation indices such that `X[order].reshape(*shape, D)` forms a 
        monotonically sorted grid.
    converged : bool
        `True` if the algorithm converged within `n_iter` iterations, `False` otherwise.
        If not converged, the fonction will also raise a convergence warning
    """
    N, D = X.shape
    assert math.prod(gridshape) == N
    assert len(gridshape) == D
    
    level = min(level, X.shape[-1])
    if init == "kdtree":
            if verbose >=1:
                print("performing kdtree initialisation")
            order = kdtree_order(X, gridshape)
            if verbose >=1:
                print("done")
            converged = True
            if level == 1:
                return order, converged
    else:
        order = arange(len(X), 'int', reference_array=X)
    if init == "shuffle":
        order = random_permutation(order)

    ordernew, converged = _mlg_step(X[order], gridshape, level=level, n_iter=n_iter, verbose = verbose)
    order = order[ordernew]
    
    if not converged:
        while level >= 2:
            level = level - 1
            ordernew, converged = _mlg_step(
                X[order], gridshape, level=level, n_iter=n_iter, verbose = verbose
            )
            order = order[ordernew]
        if not converged and verbose >=2:
            warnings.warn(
                f"Algorithm failed to converge after {n_iter} iterations. " \
                "Consider checking if the grid structure is already suitable" \
                "for the given task, which is higly probable, or increase the" \
                "`n_iter` parameter if strict convergence is required.",
                category=ConvergenceWarning,
                stacklevel=2,
            )
    if verbose:
        print("done")
    return order, converged


def sort(X, gridshape, level=2, init="kdtree", inplace=False, n_iter=50, verbose = 2):
    """Sort a point cloud `X` using the monotonic Lagrangian algorithm.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input point cloud.
    gridshape : tuple of int
        Target shape of the sorting grid. The product of the dimensions must 
        equal N and its length must equal D. E.g., for X of shape (42, 2), 
        `shape` can be (6, 7); for (60, 3), `shape` can be (3, 4, 5).

        Note: Point clouds can be padded with +/- infinity placeholders to reach 
        a valid grid size. NaNs can also be used, but will be placed randomly 
        rather than pushed to grid boundaries.
    level : {1, 2, 3}, default=1
        Maximum order of lattice directions considered. Higher values improve 
        grid quality at the expense of computational speed. Note that `level=3` 
        is only supported if D >= 3.
    init : {'kdtree', 'shuffle'} or None, default="kdtree"
        Initialization method for the permutation order. Try init = None
        if the point cloud is already nearly sorted.
    inplace : bool, default=False
        If True, modify the point cloud `X` in place.
    n_iter : int, default=50
        Maximum number of optimization iterations.
    verbose : int, default 2
         0 = silent, 1 = log progress, 2 = 1 + warn if convergence failed

    Returns
    -------
    X_sorted : ndarray of shape (N, D) or None
        The sorted point cloud such that `X_sorted.reshape(*shape, D)` forms a 
        monotonically sorted grid. Returns `None` if `inplace=True`.
    """
    order, _ = argsort(X, gridshape, level=level, init = init, n_iter=n_iter,verbose = verbose)

    if inplace:
        X[:] = X[order]
        return None
        
    return X[order]

def invert_permutation(sigma):
    """
    invert permutation helper, sigma_inv[sigma] = identity
    """
    xp, _ = get_backend(sigma)
    return xp.argsort(sigma)

# ============================================
#                   Helpers
# ============================================

class ConvergenceWarning(UserWarning):
    """Warning for convergence issues."""
    pass

def _canonical_directions(directions):
    first_nz = argmax(directions != 0, axis=1)
    idx = arange(len(directions), 'int', reference_array=directions)
    first_nz_sign = directions[idx, first_nz]
    return directions[first_nz_sign > 0]


def _direction_blocks(level, xp, dim=3):
    offsets = xp.arange(-1, 2)
    lattice = xp.stack(xp.meshgrid(*(offsets,) * dim, indexing="ij"), axis=-1).reshape(-1, dim)
    sq_norms = xp.sum(lattice ** 2, axis=-1)
    dirs = _canonical_directions(lattice[(sq_norms > 0) & (sq_norms <= level ** 2)])
    return [dirs[xp.sum(dirs ** 2, axis=-1) == k] for k in range(1, level + 1)]


def _sort_along_direction(X, direction):
    xp, _ = get_backend(X)
    level = int(xp.abs(direction).sum())

    if level == 1:
        axis = int(xp.argmax(xp.abs(direction)))
        if all_true(xp.diff(X, axis=axis) >= 0):
            return True
        inplace_sort(X, axis=axis)
        return False

    elif level == 2:
        ax0, ax1 = xp.where(direction != 0)[0]
        ax0, ax1 = int(ax0), int(ax1)
        Y = xp.moveaxis(X, [ax0, ax1], [-2, -1])
        flipaxes = []
        for ax, _ax in zip([ax0, ax1], [-2, -1]):
            if direction[ax] == -1:
                flipaxes.append(_ax)
        Y[...] = flip(Y, flipaxes)
        was_sorted = sort_diag_trick(Y)
        Y[...] = flip(Y, flipaxes)
        return was_sorted

    elif level == 3:
        ax0, ax1, ax2 = xp.where(direction != 0)[0]
        ax0, ax1, ax2 = int(ax0), int(ax1), int(ax2)
        Y = xp.moveaxis(X, [ax0, ax1, ax2], [-3, -2, -1])
        flipaxes = []
        for ax, _ax in zip([ax0, ax1, ax2], [-3, -2, -1]):
            if direction[ax] == -1:
                flipaxes.append(_ax)
        Y[...] = flip(Y, flipaxes)
        was_sorted = sort_tridiag_trick(Y)
        Y[...] = flip(Y, flipaxes)
        return was_sorted

    raise ValueError(f"Unsupported direction with L1-norm {level}")


def _build_monotonic_lagrangian_loop(X, directions):
    xp, _ = get_backend(X)
    N, K = len(X), len(directions)
    projections = X @ directions.T

    rank = empty((K, N), 'int', reference_array=X)
    for k in range(K):
        order = xp.argsort(projections[:, k])
        rank[k, order] = arange(N, 'int', reference_array=X)

    identity = arange(N, 'int', reference_array=X)
    h = [identity] + [rank[k] for k in range(K)]
    h_next = [rank[k] for k in range(K)] + [identity]

    perms = []
    for hk, hk_next in zip(h, h_next):
        sigma = empty((N,), 'int', reference_array=X)
        sigma[hk] = hk_next
        perms.append(sigma)

    init_perm, end_perm = copy(perms[0]), copy(perms[-1])
    cycle_perms = perms[:-1]
    cycle_perms[0] = init_perm[end_perm]

    return init_perm, cycle_perms, end_perm


def _sort_loop(loop, directions, shape, n_iter, verbose):
    idx_to_cycle, directions_cycle, cycle_to_idx = loop
    grid = idx_to_cycle.reshape(shape)
    first_step = True

    for it in range(n_iter):
        if verbose:
            print("-", end="", flush=True)
        fully_sorted = True
        for direction, along_direction in zip(directions, directions_cycle):
            if not first_step:
                grid = along_direction[grid]
            first_step = False
            fully_sorted = fully_sorted & _sort_along_direction(grid, direction)
        if fully_sorted:
            break

    print("it:", it)
    return (cycle_to_idx[grid]).ravel(), fully_sorted

def _mlg_step(X, shape, level, n_iter, verbose):
    xp, _ = get_backend(X)     
    directions_raw = xp.vstack(_direction_blocks(level=level, xp=xp, dim=X.shape[-1]))
    directions = astype(directions_raw, 'float', reference_array=X)
    X_float = nan_to_num(astype(X, 'float'))
    
    loop = _build_monotonic_lagrangian_loop(X_float, directions)
    if verbose >=1:
        print(f"level [{level}]. sorting")
    return _sort_loop(loop, directions, shape, n_iter, verbose)