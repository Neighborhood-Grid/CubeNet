"""
Helpers: 
sample various datasets,
plot grids as a mesh, 
check neighbormap counts

Required:
pip install squarenet
"""

def sample(method, size, backend="numpy"):
    """Sample a dataset  and convert it to a backend array.

    Parameters
    ----------
    method : str
        Dataset sampling method:
        "ball", "square", "gaussian", "spiky", "holy"
        "france", "germany" (2D only)
    size : (N, D)
        Number of points to sample and dimension.
    backend : {"numpy", "cupy", "torch"}, default="numpy"
        Array backend for the returned points.

    Returns
    -------
    array-like (N, D)
        Sampled point cloud.
    """
    try:
        import squarenet
    except:
        print("Missing dependencies for the demo. pip install gridsort[demo]")
    from squarenet import samplepoints
    points = samplepoints(method = method, size = size, plot_points = False)
    if backend == "cupy":
        import cupy as cp
        points = cp.asarray(points)
    if backend == "torch":
        import torch
        points = torch.tensor(points)
    if method == "everest": #just a bug fix
        points = points[..., [0, 2, 1]]
    return points + 1

def show_result(X, animate=False):
    """Visualize a gridified point cloud as a mesh using ``squarenet``.
    The point cloud must be sorted by the monotonic_lagrangian_sort 
    fonction first, and reshaped as a grid (X.reshape(*gridshape, D))

    Parameters
    ----------
    X : array-like (*G, D)
        Point cloud arranged on a grid.
    animate : bool, default=False
        Whether to request an animated visualization.
    """
    try:
        import squarenet
    except:
        print("Missing dependencies for the demo. pip install gridsort[demo]")
    from squarenet import SquareNet
    import numpy as np
    try:
        X = np.asarray(X)
    except:
        print("Array still on GPU, move it back to gpu first befor showing e.g. show_result(X.cpu())")
    sn = SquareNet(gridshape = X.shape[:-1])
    sn.pointsmaped = X
    sn.plot(style = "mesh", animate = animate, save = animate)
     #neighbormap counts the number of points found at a given relative delta = [i'-i, j'-j, k] such that
     #P[i', j', k] = argmin(i', j') dist(P[i,j,k], P[i',j',k])
    sn.neighbormap(projection = (0, 1)) #can also be projection = (1, 2)/(0, 1)

def print_zoom(Ygrid, center, side=1):
    """
    Print a local cross-section of a 2D or 3D grid around ``center``.

    Parameters
    ----------
    Ygrid : array-like
        Grid of points, with shape ``(*grid_shape, D)`` where D is 2 or 3.
    center : tuple of int
        Index of the central grid point.
    side : int, default=1
        Number of cells shown on each side of the center.
    """
    import numpy as np
    try:
        Ygrid = np.asarray(Ygrid)
    except:
        print("Array still on GPU, move it back to gpu first befor showing e.g. print_zoom(X.cpu())")
    ndim = Ygrid.ndim - 1
    if ndim not in (2, 3):
        raise ValueError("printzoom only for 2D/3D grids")
    def _format_point(point):
        return "(" + ", ".join(f"{float(x):.3f}" for x in point) + ")"
    print(f"\n zooming arround the grid position {center} :\n")
    if ndim == 2:
        i, j = center
        for point in Ygrid[i, j-side:j]:
            print("." * (15*side-1), _format_point(point), "." * (15*side-1))
        for point in Ygrid[i-side:i+side+1, j]:
            print(_format_point(point), end=" ")
        print("")
        for point in Ygrid[i, j+1:j+side+1]:
            print("." * (15*side-1), _format_point(point), "." * (15*side-1))
    else:
        i, j, k = center
        print("axis 0:")
        for ii in range(i-side, i+side+1):
            print(_format_point(Ygrid[ii, j, k]), end=" ")
        print()
        print("axis 1:")
        for jj in range(j-side, j+side+1):
            print(_format_point(Ygrid[i, jj, k]), end=" ")
        print()
        print("axis 2:")
        for kk in range(k-side, k+side+1):
            print(_format_point(Ygrid[i, j, kk]), end=" ")
        print()