"""
Helpers:
anything related to kdtree
"""

import math
from collections import defaultdict
from .array_libs import (
    get_backend,
    transpose,
    arange,
    empty,
    argpartition,
    take_along_axis,
    concatenate,
)


def kdtree_order_dyadic(X):
    """
    Sorting version for the (easy) dyadic case, were X has a power of 2**D shape.
    Mainly for pedagogical reason. It will help to understand what the general
    version do, when the number of points is not necesarily a power of two and 
    we have to deal with potential median hyperplan. 

    Parameters
    ----------
    X    : array-like, shape (N, D). Point cloud

    Returns
    -------
    order : array, shape (N,), based on the kdtree cuts, such that
        Xgrid = X[order].reshape(*gridshape, D) is monotonic along each axis:
        Xgrid[i+1, j, k][0] >= Xgrid[i, j, k][0]
        Xgrid[i, j+1, k][1] >= Xgrid[i, j, k][1]

    """
    N, D = X.shape
    M = int(N**(1/D) + 0.1)
    K = int(math.log2(M))
    
    idx = arange(N, 'int', reference_array=X)
    
    for step in range(K * D):
        axis = step % D
        size = idx.shape[-1]
        mid = size // 2
        X_curr = X[idx, axis]
        p = argpartition(X_curr, mid, axis = -1)
        idx = take_along_axis(idx, p, axis = -1)    
        idx = idx.reshape(*idx.shape[:-1], 2, mid)
        
    idx = idx.squeeze(-1)
    transpose_axes = [d + k * D for d in range(D) for k in range(K)]
    idx = transpose(idx, transpose_axes)

    return idx.flatten()



def kdtree_order(X, gridshape):
    """
    Parameters
    ----------
    X         : array-like, shape (N, D). Point cloud
    gridshape : sequence of int, length D, product == N

    Returns
    -------
    order : array, shape (N,), based on the kdtree cuts, such that
        Xgrid = X[order].reshape(*gridshape, D) is monotonic along each axis:
        Xgrid[i+1, j, k][0] >= Xgrid[i, j, k][0]
        Xgrid[i, j+1, k][1] >= Xgrid[i, j, k][1]

    Note: 
    -----
    It is the natural generalisation of the kdtree_order_dyadic version.
    What make the general case more complex is simply to deal with 
    the median hyperplan when cuting a non even shape of the grid hyperrectangle.
    """
    xp, _ = get_backend(X)
    N, D = X.shape
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

    output = empty(N, 'int', reference_array=X)

    if N == 1:
        output[0] = 0
        return output

    # idx = point indices, pos = fixed destination slots (C-order of the grid)
    idx_init = arange(N, 'int', reference_array=X)[None, :]
    pos_init = arange(N, 'int', reference_array=X)[None, :]

    ax0 = next_axis(shape, 0)
    worklist = defaultdict(lambda: ([], []))
    worklist[(shape, ax0)][0].append(idx_init)
    worklist[(shape, ax0)][1].append(pos_init)

    while worklist:
        key = next(iter(worklist))
        idx_list, pos_list = worklist.pop(key)
        cur_shape, cur_axis = key

        idx_b = concatenate(idx_list) if len(idx_list) > 1 else idx_list[0]
        pos_b = concatenate(pos_list) if len(pos_list) > 1 else pos_list[0]

        n = idx_b.shape[-1]
        if n == 1:
            output[pos_b[:, 0]] = idx_b[:, 0]
            continue

        axis  = cur_axis
        size  = cur_shape[axis]
        slab  = n // size
        lsize = size // 2
        ln    = lsize * slab
        mid_n = slab if (size % 2) else 0

        def push(idx_part, pos_part, child_shape, child_n_axis):
            m = idx_part.shape[-1]
            if m == 1 or child_n_axis is None:
                output[pos_part[:, 0]] = idx_part[:, 0]
            else:
                ckey = (child_shape, child_n_axis)
                worklist[ckey][0].append(idx_part)
                worklist[ckey][1].append(pos_part)

        # --- partition the POINTS only ---
        if size % 2 == 0:
            vals = X[idx_b, axis]
            p = argpartition(vals, ln, axis=-1)
            idx_b = take_along_axis(idx_b, p, axis=-1)
            idx_l, idx_r = idx_b[:, :ln], idx_b[:, ln:]

            # destinations = current pos sorted by their multi-index on `axis`
            mi = multi_index_on_axis(pos_b, axis)
            order_mi = xp.argsort(mi, axis=-1)
            pos_sorted = take_along_axis(pos_b, order_mi, axis=-1)
            pos_l, pos_r = pos_sorted[:, :ln], pos_sorted[:, ln:]

            child_shape = list(cur_shape)
            child_shape[axis] = lsize
            child_shape = tuple(child_shape)
            nax = next_axis(child_shape, (axis + 1) % D)

            push(concatenate([idx_l, idx_r]),
                 concatenate([pos_l, pos_r]),
                 child_shape, nax)

        else:
            # odd: left + median slab + right
            total_keep = ln + mid_n
            vals = X[idx_b, axis]
            p = argpartition(vals, total_keep, axis=-1)
            idx_b = take_along_axis(idx_b, p, axis=-1)
            idx_r  = idx_b[:, total_keep:]
            idx_lm = idx_b[:, :total_keep]

            if ln > 0:
                vals_lm = X[idx_lm, axis]
                p2 = argpartition(vals_lm, ln, axis=-1)
                idx_lm = take_along_axis(idx_lm, p2, axis=-1)
                idx_l = idx_lm[:, :ln]
                idx_m = idx_lm[:, ln:]
            else:
                idx_l = None
                idx_m = idx_lm

            mi = multi_index_on_axis(pos_b, axis)
            order_mi = xp.argsort(mi, axis=-1)
            pos_sorted = take_along_axis(pos_b, order_mi, axis=-1)
            pos_l = pos_sorted[:, :ln] if ln > 0 else None
            pos_m = pos_sorted[:, ln : ln + mid_n]
            pos_r = pos_sorted[:, ln + mid_n :]

            child_l = list(cur_shape); child_l[axis] = lsize; child_l = tuple(child_l)
            child_m = list(cur_shape); child_m[axis] = 1;     child_m = tuple(child_m)

            nax_lr = next_axis(child_l, (axis + 1) % D)
            nax_m  = next_axis(child_m, (axis + 1) % D)

            if idx_l is not None:
                push(concatenate([idx_l, idx_r]),
                     concatenate([pos_l, pos_r]),
                     child_l, nax_lr)
            push(idx_m, pos_m, child_m, nax_m)

    return output