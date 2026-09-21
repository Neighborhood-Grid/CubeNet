## ❒ Gridpoints — Multi dimensional sort for point clouds

`Gridpoints` maps unstructured **point clouds** to structured grids through a **bijective transformation**: one point, one cell, no overlap, fully invertible. It replaces and enhances the squarenet project with a more powerfull sorting algorithm.

What it does: Take raw point cloud `P(N, D)`  and find a grid shape and an index permutation `order` such that Pgrid = P[order].reshape(*gridshape, D) 
is sorted along every axis of the grid. E.g. in 3D, for Pgrid = (x, y, z):
```text
x[i+1, j, k] >= x[i, j, k]
y[i, j+1, k] >= y[i, j, k]
z[i, j, k+1] >= z[i, j, k]
```
→ On the `Pgrid` view of `P`, neighbor queries become a simple stencil look-up :
```text
neighborhood[i, j, k] = {Pgrid[i±di, j±dj, k±dk] | (di, dj, dk) ≤ R},
````
where `R` is a radius cutoff to determine, allowing local operations in linear time.  
→ Standard operations (convolution, clustering, …) can then be applied  
directly on the `Pgrid` view instead of relying on complex graph convolutions  
or other point-cloud techniques.

`P` can be a NumPy, PyTorch or CuPy array of any dimension (N, D).  
To allow natural padding when the grid has more cells than points,  
NaNs and Infs are supported in a consistent manner: 
- nans → random position
- (+-) infs → border of the grid
This allow to gridsort prime or variable number of points N, as long as one is ready to deal with void/special grid cells.

Expected runtime for sorting 1 million points: CPU → < 10s, GPU → < 500 ms

---

### Installation

```bash
pip install gridpoints          # core only
pip install gridpoints[demo]    # for the demonstration notebook, see `notebook.ipynb`
```

### Quickstart

```python
import gridpoints as grid
import numpy as np

# Raw point cloud (numpy, pytorch or cupy)
A = np.random.rand(1_000_000, 3)

# Sorted view: place the points inside the grid
order = grid.argsort(A, gridshape=(100, 100, 100))
Bflat = A[order]
Bgrid = Bflat.reshape(100, 100, 100, 3)

# Rest of your pipeline, working with grids
Cgrid = apply_something(Bgrid)

# Back to the original points indexing
Cflat = Cgrid.reshape(-1, 3)
orderinv = grid.invert_permutation(order)
C = Cflat[orderinv]   # matches the initial points order
```

<img src="https://raw.githubusercontent.com/Neighborhood-Grid/CubeNet/main/ballexemple.png">

---

### Note on the cutoff radius R

There is no strict theoretical guarantee about what the cutof radius R should be for a given task. E.g the relative grid position between a point and its nearest neighbors can't be garanted to be in the exact adjacent grid cells. What is guaranteed from the sorted ordering is only **grid monotonicity**: *x* coordinates increase along rows, *y* coordinates along columns, and so on.

As an example, empirical results in 2-D show that `R = 5` is enough for ~99 % of the nearest neighbors; some outlier neighbors will sit further apart for complex geometries with pronounced peaks, holes or any non-smoothness. When a stricter neighborhood is required, or in high dimensional setting, the best practice is to build an assembly of grid experts, each working on a rotated / projected view of the points, as discussed in [this topic](https://github.com/glotzerlab/freud/discussions/1417).

### Note on efficient stencil operations

The typical use-case of `Gridpoints` is to allow fast local operations on arbitrary point clouds using stencil kernels:

```text
output(i, j, k) = f( Pgrid[i±di, j±dj, k±dk] | di, dj, dk in local window )
```
To go beyond standard (slow) python loops, this can be accelerated with native grid convolution operations of standard libraries whenever possible, or with `pystencils` or `taichi` compilers for complex/non linear grid kernels.
