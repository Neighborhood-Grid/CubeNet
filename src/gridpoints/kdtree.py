"""
Helpers:
anything related to kdtree

The kdtree is built for a batch of point clouds at once. Its structure only
depends on the grid shape, so the destination cells (`pos`) are shared by all
the clouds, and only the point indices (`idx`) have a batch axis:
    idx : (B, nodes, n)   point indices of each node, for each cloud
    pos : (nodes, n)      destination grid cells of each node
"""

import math
from .array_libs import (
    transpose,
    arange,
    empty,
    argsort,
    argpartition,
    take_along_axis,
    concatenate,
)


def kdtree_order_dyadic(X, int_dtype='int32'):
    """
    Sorting version for the (easy) dyadic case, were X has a power of 2**D shape.
    Mainly for pedagogical reason. It will help to understand what the general
    version do, when the number of points is not necesarily a power of two and
    we have to deal with potential median hyperplan.

    Parameters
    ----------
    X    : array-like, shape (B, N, D). Batch of point clouds

    Returns
    -------
    order : array, shape (B, N), based on the kdtree cuts, such that
        Xgrid = X[b][order[b]].reshape(*gridshape, D) is monotonic along each axis:
        Xgrid[i+1, j, k][0] >= Xgrid[i, j, k][0]
        Xgrid[i, j+1, k][1] >= Xgrid[i, j, k][1]

    """
    B, N, D = X.shape
    M = int(N**(1/D) + 0.1)
    K = int(math.log2(M))

    idx = empty((B, N), int_dtype, X)
    idx[:] = arange(N, int_dtype, X)

    for step in range(K * D):
        axis = step % D
        size = idx.shape[-1]
        mid = size // 2
        X_curr = take_along_axis(X[:, :, axis], idx.reshape(B, -1), 1).reshape(idx.shape)
        p = argpartition(X_curr, mid)
        idx = take_along_axis(idx, p, -1)
        idx = idx.reshape(*idx.shape[:-1], 2, mid)

    idx = idx.reshape(B, *[2] * (K * D))
    transpose_axes = [0] + [1 + d + k * D for d in range(D) for k in range(K)]
    idx = transpose(idx, transpose_axes)

    return idx.reshape(B, N)



def kdtree_order(X, gridshape, int_dtype='int32'):
    """
    Parameters
    ----------
    X         : array-like, shape (B, N, D). Batch of point clouds
    gridshape : sequence of int, length D, product == N

    Returns
    -------
    order : array, shape (B, N), based on the kdtree cuts, such that
        Xgrid = X[b][order[b]].reshape(*gridshape, D) is monotonic along each axis:
        Xgrid[i+1, j, k][0] >= Xgrid[i, j, k][0]
        Xgrid[i, j+1, k][1] >= Xgrid[i, j, k][1]

    Note:
    -----
    It is the natural generalisation of the kdtree_order_dyadic version.
    What make the general case more complex is simply to deal with
    the median hyperplan when cuting a non even shape of the grid hyperrectangle.
    """
    B, N, D = X.shape
    shape = tuple(int(s) for s in gridshape)
    assert math.prod(shape) == N
    assert len(shape) == D

    def next_axis(sh, start):
        for off in range(D):
            a = (start + off) % D
            if sh[a] > 1:
                return a
        return None

    def multi_index_on_axis(pos, axis):
        """Multi-index component on `axis` for C-order linear indices `pos`."""
        stride = int(math.prod(shape[axis + 1 :])) if axis + 1 < D else 1
        return (pos // stride) % shape[axis]

    def cut(idx, axis, k):
        """Reorder idx along its last axis: its first k points are the k lowest along `axis`."""
        vals = take_along_axis(X[:, :, axis], idx.reshape(B, -1), 1).reshape(idx.shape)
        return take_along_axis(idx, argpartition(vals, k), 2)

    output = empty((B, N), int_dtype, X)

    if N == 1:
        output[:] = 0
        return output

    # idx = point indices, pos = fixed destination slots (C-order of the grid)
    idx_init = empty((B, 1, N), int_dtype, X)
    idx_init[:] = arange(N, int_dtype, X)
    pos_init = arange(N, int_dtype, X)[None, :]

    ax0 = next_axis(shape, 0)
    worklist = {(shape, ax0): ([idx_init], [pos_init])}

    def push(idx_part, pos_part, child_shape, child_n_axis):
        if idx_part.shape[-1] == 1 or child_n_axis is None:
            output[:, pos_part[:, 0]] = idx_part[:, :, 0]
        else:
            idx_list, pos_list = worklist.setdefault((child_shape, child_n_axis), ([], []))
            idx_list.append(idx_part)
            pos_list.append(pos_part)

    while worklist:
        key = next(iter(worklist))
        idx_list, pos_list = worklist.pop(key)
        cur_shape, axis = key

        idx_b = concatenate(idx_list, 1) if len(idx_list) > 1 else idx_list[0]
        pos_b = concatenate(pos_list, 0) if len(pos_list) > 1 else pos_list[0]

        n = idx_b.shape[-1]
        size  = cur_shape[axis]
        slab  = n // size
        lsize = size // 2
        ln    = lsize * slab
        mid_n = slab if (size % 2) else 0

        # destinations = current pos sorted by their multi-index on `axis`
        mi = multi_index_on_axis(pos_b, axis)
        pos_sorted = take_along_axis(pos_b, argsort(mi), 1)

        child_shape = list(cur_shape)
        child_shape[axis] = lsize
        child_shape = tuple(child_shape)
        nax = next_axis(child_shape, (axis + 1) % D)

        if size % 2 == 0:
            idx_b = cut(idx_b, axis, ln)
            push(concatenate([idx_b[:, :, :ln], idx_b[:, :, ln:]], 1),
                 concatenate([pos_sorted[:, :ln], pos_sorted[:, ln:]], 0),
                 child_shape, nax)

        else:
            # odd: left + median slab + right
            idx_b = cut(idx_b, axis, ln + mid_n)
            idx_r  = idx_b[:, :, ln + mid_n:]
            idx_lm = idx_b[:, :, :ln + mid_n]
            if ln > 0:
                idx_lm = cut(idx_lm, axis, ln)
                push(concatenate([idx_lm[:, :, :ln], idx_r], 1),
                     concatenate([pos_sorted[:, :ln], pos_sorted[:, ln + mid_n:]], 0),
                     child_shape, nax)

            child_m = list(cur_shape); child_m[axis] = 1; child_m = tuple(child_m)
            push(idx_lm[:, :, ln:], pos_sorted[:, ln : ln + mid_n],
                 child_m, next_axis(child_m, (axis + 1) % D))

    return output
