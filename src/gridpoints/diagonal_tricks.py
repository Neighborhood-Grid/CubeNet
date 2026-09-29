"""
Helpers:
Diagonal trick is based on a tensor rotation that allow 
to sort arrays efficiently along arbitrary diagonal fibers.
"""


from .array_libs import (
    get_backend,
    arange,
    inplace_sort,
    all_true
)

def sort_diag_trick(X):
    """ 
    we rotate the tensor along the main diagonal 
    to place diagonals on the last axis, then 
    perform a standard sort(axis = -1), and
    rotate back the diagonals
    """
    xp, _ = get_backend(X)
    #bigM for safety, protect
    #overlapping diagonals
    bigM = (X.max() - X.min()) + 1

    n, m = X.shape[-2:]

    #move the smallest axis to last position
    if m > n: 
        X = xp.moveaxis(X, [-1, -2], [-2, -1]) 
        n, m = m, n #now n >= m
        
    i = arange(n, 'int', X)[:, None]
    j = arange(m, 'int', X)[None, :]
    
    di = (i - j)
    i_ = di % n
    ires = di // n 
    groupkey = bigM * ires 
    
    X[..., i_, j] = X[..., i, j] - groupkey #Rotation
    # ========================
    # here is the trick: as we rotated the tensor
    # sorting on the diagonal is just working on
    # axis -1 of the rotated view
    # ========================
    was_sorted = all_true(xp.diff(X, axis=-1) >= 0)
    inplace_sort(X, axis=-1)
    X[..., i, j] = X[..., i_, j] + groupkey
    
    return was_sorted

def sort_tridiag_trick(X):
    """ 
    we rotate the tensor along the tridiagonal, 
    then perform a standard sort(axis = -1), 
    and rotate back the tridiagonals
    """
    xp, _ = get_backend(X)
    #bigM for safety, protect
    #overlapping diagonals
    bigM = (X.max() - X.min()) + 1

    #move the smallest axes to last positions
    abc = X.shape
    dst_axes = (-3, -2, -1)
    src_axes = tuple(sorted(dst_axes, key=lambda ax: -abc[ax]))
    X = xp.moveaxis(X, src_axes, dst_axes)   
    a, b, c = X.shape[-3:]  # now a >= b >= c 
    
    i = arange(a, 'int', X)[:, None, None]
    j = arange(b, 'int', X)[None, :, None]
    k = arange(c, 'int', X)[None, None, :]
    
    di = i - k
    dj = j - k
    di_ = di % a
    dj_ = dj % b
    dires = di // a         
    djres = dj // b
    groupkey = bigM * (dires * 4 + djres)
    
    X[..., di_, dj_, k] = X[..., i, j, k] - groupkey #Trirotation
    # ========================
    # Again, thanks to the rotation trick
    # we just need to .sort(X, axis = -1)
    # to sort the tridiagonal
    # ========================
    was_sorted = all_true(xp.diff(X, axis=-1) >= 0)
    inplace_sort(X, axis=-1)  
    X[..., i, j, k] = X[..., di_, dj_, k] + groupkey
    
    return was_sorted