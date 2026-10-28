"""
Main fonctions of the library are defined here:
argsort and sort

They work on one point cloud (N, D) or on a batch of point clouds (B, N, D).
"""


import math
import itertools
import warnings
from .array_libs import (
    arange, get_backend, any_true, inplace_sort, empty, zeros, ones,
    astype, nan_infs_to_num_and_tiebreaker, random, argsort as _argsort,
    take_along_axis, concatenate, flip, precision,
)
from .diagonal_tricks import sort_diag_trick, sort_tridiag_trick, sorted_flags
from .kdtree import kdtree_order

# ============================================
#                   Sorting API
# ============================================

def argsort(X, gridshape, verbose = 2, level=2, init="kdtree", n_iter=40, chunk_size=None):
    """Compute a multi-directional monotonic sorted ordering of point clouds.

    Parameters
    ----------
    X : ndarray of shape (N, D) or (B, N, D)
        Input point cloud, or batch of B point clouds sharing the same gridshape.
    gridshape : tuple of int
        Target shape of the sorting grid. The product of the dimensions must
        equal N and its length must equal D. E.g., for X of shape (42, 2),
        `shape` can be (6, 7); for (60, 3), `shape` can be (3, 4, 5).

        Note: Point clouds can be padded with +/- infinity placeholders to reach
        a valid grid size (see `gridpoints.pad`). NaNs can also be used, but will
        be placed randomly rather than pushed to grid boundaries.
    verbose : int, default 2
        0 = silent, 1 = log progress, 2 = 1 + warn if convergence failed
    level : {1, 2, 3}, default=2
        Maximum order of lattice directions considered. Higher values improve
        grid quality at the expense of computational speed. Note that `level=3`
        is only supported if D >= 3.
    init : {'kdtree', 'shuffle'} or None, default="kdtree"
        Initialization method for the permutation order. Try init = None
        if the point cloud is already nearly sorted.
    n_iter : int, default=40
        Maximum number of optimization iterations (for each point cloud).
    chunk_size : int, optional
        Maximum number of point clouds sorted simultaneously, to limit the memory
        usage. Default: the whole batch. About (K + 16) * 4 * chunk_size * N bytes
        are needed, where K is the number of lattice directions (9 for 3D points
        with level=2).

    Returns
    -------
    order : ndarray of shape (N,) or (B, N)
        Permutation indices such that `X[order].reshape(*shape, D)` forms a
        monotonically sorted grid (`X[b][order[b]]` for the batch).
    """
    batched = X.ndim == 3
    if not batched:
        X = X[None]
    B, N, D = X.shape
    assert len(gridshape) == D, (
        f"gridshape={gridshape} has {len(gridshape)} entries but the points are {D}D: "
        "gridshape needs exactly one specified shape per axis. "
        "For example, with 3D points, (32, 32) is invalid (z shape is missing), "
        "while (32, 32, 1) is valid (z size is 1)."
    )
    assert math.prod(gridshape) == N, (
        f"a grid with given gridshape ={gridshape} contains {math.prod(gridshape)} cells "
        f"but there are N={N} points. "
        "If there are fewer points than cells, pad the points with "
        "dummy points (e.g. NaN/inf, see gridpoints.pad). "
        "If there are more points than cells, use a bigger gridshape."
    )
    level = min(level, D)
    chunk_size = chunk_size or max(B, 1)

    chunks = [_argsort_chunk(X[i:i + chunk_size], gridshape, verbose, level, init, n_iter)
              for i in range(0, B, chunk_size)]
    order = concatenate([o for o, _ in chunks])
    n_failed = sum(int((~ok).sum()) for _, ok in chunks)

    if n_failed and verbose >= 2:
        warnings.warn(
            f"Algorithm failed to converge after {n_iter} iterations "
            f"({n_failed} point cloud{'s' if n_failed > 1 else ''}).\n "
            "Consider checking if the grid structure is already suitable \n"
            "for the given task, which is higly probable, or increase the \n"
            "`n_iter` parameter if strict convergence is required.\n"
            "(to shutup this warning, run the sort with verbose <= 1)",
            category=ConvergenceWarning,
            stacklevel=2
        )
    return order if batched else order[0]


def sort(X, gridshape, verbose = 2, level=2, init="kdtree", inplace=False, n_iter=40, chunk_size=None):
    """Sort point clouds `X` using the monotonic Lagrangian algorithm.

    Parameters
    ----------
    X : ndarray of shape (N, D) or (B, N, D)
        Input point cloud, or batch of point clouds.
    inplace : bool, default=False
        If True, modify the point cloud `X` in place.

    See `argsort` for the other parameters.

    Returns
    -------
    X_sorted : ndarray of the shape of X, or None
        The sorted point clouds such that `X_sorted.reshape(*shape, D)` forms a
        monotonically sorted grid. Returns `None` if `inplace=True`.
    """
    order = argsort(X, gridshape, verbose=verbose, level=level, init=init,
                    n_iter=n_iter, chunk_size=chunk_size)
    X_sorted = take_along_axis(X, order[..., None], X.ndim - 2)

    if inplace:
        X[...] = X_sorted
        return None

    return X_sorted

def invert_permutation(sigma):
    """
    invert permutation helper, sigma_inv[sigma] = identity
    (along the last axis, for a batch of permutations)
    """
    xp, _ = get_backend(sigma)
    return xp.argsort(sigma)

# ============================================
#                   Helpers
# ============================================

class ConvergenceWarning(UserWarning):
    """Warning for convergence issues."""
    pass


def _argsort_chunk(X, gridshape, verbose, level, init, n_iter):
    """argsort of a batch X (B, N, D). Returns the orders and which clouds converged."""
    B, N, D = X.shape
    float_dtype, int_dtype = precision(N)
    X = nan_infs_to_num_and_tiebreaker(astype(X, float_dtype))

    if verbose >= 1:
        print(f"[gridsort] initialisation: {init}")
    if init == "kdtree":
        order = kdtree_order(X, gridshape, int_dtype)
        if level == 1:
            return order, ones((B,), 'bool', X)
    elif init == "shuffle":
        order = astype(_argsort(random((B, N), float_dtype, X)), int_dtype)
    else:
        order = empty((B, N), int_dtype, X)
        order[:] = arange(N, int_dtype, X)

    # Each cloud is sorted until it converges. The ones that do not are
    # sorted again with simpler directions, on their own.
    converged = zeros((B,), 'bool', X)
    todo = arange(B, 'int64', X)
    while True:
        Xo = take_along_axis(X[todo], order[todo][..., None], 1)
        ordernew, ok = _mlg_step(Xo, gridshape, level, n_iter, verbose)
        order[todo] = take_along_axis(order[todo], ordernew, 1)
        converged[todo] = ok
        todo = todo[~ok]
        if level == 1 or len(todo) == 0:
            return order, converged
        level -= 1


def _directions(level, dim):
    """Lattice directions with at most `level` non-zero entries, each in {-1, 0, 1}.
    Only one of v and -v is kept: the first non-zero entry is positive."""
    lattice = itertools.product((-1, 0, 1), repeat=dim)
    dirs = [v for v in lattice
            if 0 < sum(map(abs, v)) <= level and next(x for x in v if x) > 0]
    return sorted(dirs, key=lambda v: sum(map(abs, v)))


def _sort_along_direction(grid, direction):
    """Sort in place all the fibers of the grids (B, *shape) along `direction`.
    Returns for each grid whether it was already sorted: bool array (B,)."""
    xp, _ = get_backend(grid)
    axes = [a + 1 for a, v in enumerate(direction) if v]    # + 1: batch axis
    level = len(axes)

    if level == 1:
        was_sorted = sorted_flags(grid, axes[0])
        inplace_sort(grid, axes[0])
        return was_sorted

    trick = {2: sort_diag_trick, 3: sort_tridiag_trick}[level]
    Y = xp.moveaxis(grid, axes, tuple(range(-level, 0)))
    flipaxes = [i - level for i, a in enumerate(axes) if direction[a - 1] < 0]
    Y[...] = flip(Y, flipaxes)
    was_sorted = trick(Y)
    Y[...] = flip(Y, flipaxes)
    return was_sorted


def _build_monotonic_lagrangian_loop(X, directions):
    """
    A cell of the grid contains the rank of its point along the direction
    being sorted. Going from a direction to the next one is a permutation
    of these ranks: perms[k] maps the ranks along direction k-1 to those along
    direction k (perms[0]: point index -> rank along the first direction,
    perms[-1]: rank along the last direction -> point index).
    """
    B, N, _ = X.shape
    int_dtype = precision(N)[1]
    rows = arange(B, 'int64', X)[:, None]
    identity = arange(N, int_dtype, X)

    perms, previous = [], identity
    for direction in directions + [None]:
        if direction is None:
            rank = identity
        else:
            rank = empty((B, N), int_dtype, X)
            rank[rows, _argsort(sum(v * X[:, :, a] for a, v in enumerate(direction) if v))] = identity
        sigma = empty((B, N), int_dtype, X)
        sigma[rows, previous] = rank
        perms.append(sigma)
        previous = rank

    init_perm, end_perm, cycle_perms = perms[0], perms[-1], perms[:-1]
    cycle_perms[0] = take_along_axis(init_perm, end_perm, 1)

    return init_perm, cycle_perms, end_perm


def _sort_loop(loop, directions, shape, n_iter, verbose):
    """Sort the grids along all the directions, cyclically, until no grid changes.

    A grid that went through the whole cycle without any change will never
    change again: it is set aside, and the loop goes on with the others only.
    """
    idx_to_cycle, cycle, cycle_to_idx = loop
    B, N = idx_to_cycle.shape
    grid = idx_to_cycle.reshape(B, *shape)
    order = empty((B, N), cycle_to_idx.dtype, grid)
    converged = zeros((B,), 'bool', grid)
    active = arange(B, 'int64', grid)    # position in the batch of the grids still in the loop
    first_step = True

    for it in range(n_iter):
        if verbose:
            print("-", end="", flush=True)
        fully_sorted = ones((len(active),), 'bool', grid)
        for direction, along_direction in zip(directions, cycle):
            if not first_step:
                grid = take_along_axis(along_direction, grid.reshape(len(active), N), 1).reshape(grid.shape)
            first_step = False
            fully_sorted &= _sort_along_direction(grid, direction)

        if any_true(fully_sorted):
            order[active[fully_sorted]] = take_along_axis(
                cycle_to_idx[fully_sorted], grid[fully_sorted].reshape(-1, N), 1)
            converged[active[fully_sorted]] = True
            keep = ~fully_sorted
            grid, cycle_to_idx, active = grid[keep], cycle_to_idx[keep], active[keep]
            cycle = [p[keep] for p in cycle]
            if len(active) == 0:
                break

    if len(active):
        order[active] = take_along_axis(cycle_to_idx, grid.reshape(len(active), N), 1)

    if verbose:
        print("\n it:", it, "converged:", int(converged.sum()), "/", B)
    return order, converged


def _mlg_step(X, shape, level, n_iter, verbose):
    """One run of the monotonic Lagrangian loop on the points X (B, N, D), in grid order.
    Returns the new order of each cloud, and which clouds converged."""
    directions = _directions(level, X.shape[-1])
    loop = _build_monotonic_lagrangian_loop(X, directions)
    if verbose >=1:
        print(f"level [{level}]. sorting")
    return _sort_loop(loop, directions, shape, n_iter, verbose)
