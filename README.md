## ❒ Gridsort — Multi dimensional sort for point clouds

Gridsort maps unstructured **point clouds** to structured grids through a **bijective transformation**: one point, one cell, no overlap, fully invertible. It replaces and enhances the squarenet project with a faster sorting algorithm.

Take raw point cloud `X(N, D)`  
→ find a grid shape and a permutation `order` such that  
Xgrid = X[order].reshape(*gridshape, D) is sorted along every axis of the grid.  
→ On the `Xgrid` view of `X`, neighbor queries become a simple stencil look-up  
neighborhood[i, j, k] = {Xgrid[i±di, j±dj, k±dk] | (di, dj, dk) ≤ R}, where `R` is a radius cutoff to determine,  
allowing local operations in linear time.  
→ Standard operations (grid convolution, clustering, …) can then be applied  
directly on the `Xgrid` view instead of relying on complex graph convolutions  
or other point-cloud techniques.

`X` can be a NumPy, PyTorch or CuPy array of any dimension (N, D).  
To allow natural padding when the grid has more cells than points,  
NaNs and Infs are supported in a consistent manner: 
- nans → random position
- (+-) infs → border of the grid
This allow to deal with prime or variable N, as long as one is ready to 
deal with void/special grid cells.

Expected runtime for sorting 1 million points: CPU → < 10s, GPU → < 100 ms

---

### Installation

```bash
pip install gridsort          # core only
pip install gridsort[demo]    # for the demonstration notebook, see `notebook.ipynb`
```

### Quickstart

```python
import gridsort as grid
import numpy as np

# Raw point cloud (numpy, pytorch or cupy)
X = np.random.rand(1_000_000, 3)

# Sorted view: place the points inside the grid
order = grid.argsort(X, gridshape=(100, 100, 100))
Yflat = X[order]
Ygrid = Yflat.reshape(100, 100, 100, 3)

# Rest of your pipeline, working with grids
Zgrid = apply_something(Ygrid)

# Back to the original points indexing
Zflat = Zgrid.reshape(-1, 2)
orderinv = grid.invert_permutation(order)
Z = Zflat[orderinv]   # matches the initial points order
```

---


<img src="https://raw.githubusercontent.com/Neighborhood-Grid/CubeNet/ballgrid.png">

---

### Note on the cutoff radius R

There is no strict theoretical guarantee about what the cutof radius R should be for a given task. E.g the relative grid position between a point and its nearest neighbors can't be garanted to be in the exact adjacent grid cells. What is guaranteed from the gridsort ordering is only **grid monotonicity**: *x* coordinates increase along rows, *y* coordinates along columns, and so on.

As an example, empirical results in 2-D show that `R = 5` is enough for ~99 % of the nearest neighbors; some outlier neighbors will sit further apart for complex geometries with pronounced peaks, holes or any non-smoothness. When a stricter neighborhood is required, or in high dimensional setting, the best practice is to build an assembly of grid experts, each working on a rotated / projected view of the points, as discussed in [this topic](https://github.com/glotzerlab/freud/discussions/1417).

### Note on efficient stencil operations

The typical use-case of `gridsort` is to allow fast local operations on arbitrary point clouds using stencil kernels:

```text
output(i, j, k) = f( Xgrid[i±di, j±dj, k±dk] | di, dj, dk in local window )
```
To go beyond standard (slow) python loops, this can be accelerated with native grid convolution operations of standard libraries whenever possible, or with `pystencils` or `taichi` compilers for complex/non linear grid kernels.