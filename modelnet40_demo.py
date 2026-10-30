"""
==============================================================
Requirements:
==============================================================

I. Dependencies: numpy, gridpoints:
    pip install numpy gridpoints

II. Getting ModelNet40 (~2 GB):
    wget http://modelnet.cs.princeton.edu/ModelNet40.zip
    mkdir -p ModelNet40 && unzip ModelNet40.zip -d ModelNet40

ModelNet40 is distributed by Princeton (project page:
https://modelnet.cs.princeton.edu). It is subject to Princeton's terms of use
(non-commercial research and education only, no public redistribution), so it
is not shipped with this code. 

==============================================================
Usage:
==============================================================
    python modelnet40_demo.py --obj table --num 6
    python modelnet40_demo.py --obj 33 --num 6

"""

import argparse
import math
import os

import numpy as np

import griddemo as demo
import gridpoints as grid

DATA_DIR = "ModelNet40/ModelNet40"

CATEGORIES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]


def modelnet_path(category, index, split="train", data_dir=DATA_DIR):
    """Return the path of a ModelNet40 .off file.

    Args:
        category: Category name, one of `CATEGORIES`.
        index: Object number within the category (e.g. 6 -> "table_0006.off").
        split: "train" or "test".
        data_dir: Root folder containing one subfolder per category.

    Returns:
        Path to the .off file, as a string.
    """
    filename = f"{category}_{index:04d}.off"
    return os.path.join(data_dir, category, split, filename)


def read_off(filepath, sample=None, normalize=False, rng=None):
    """Read a mesh in OFF format, optionally sampling points on its surface.

    Args:
        filepath: Path to the .off file.
        sample: If None, return the mesh vertices and faces as stored in the
            file (the number of vertices varies from one object to another).
            If an integer N, return exactly N points sampled uniformly on the
            surface of the mesh (requires a triangle mesh, as in ModelNet40).
        normalize: If True, center the points at the origin and scale them
            to fit in the unit sphere.
        rng: Optional `np.random.Generator`, for reproducible sampling.

    Returns:
        If `sample` is None: a tuple `(verts, faces)` with `verts` of shape
        (V, 3), float32, and `faces` of shape (F, 3), int32.
        Otherwise: an array of shape (sample, 3), float32.
    """
    with open(filepath, "r") as f:
        # Header. In some ModelNet40 files the "OFF" keyword is glued to the
        # counts on the first line ("OFF490 518 0") instead of being on its own.
        line = f.readline().strip()
        if line.upper() == "OFF":
            line = f.readline().strip()
        else:
            line = line[3:].strip()
        n_verts, n_faces, _ = map(int, line.split())  # 3rd value: edges, unused

        verts = np.array(
            [list(map(float, f.readline().split())) for _ in range(n_verts)],
            dtype=np.float32,
        )

        # Each face line is "n i_1 ... i_n", optionally followed by a color.
        faces = []
        for _ in range(n_faces):
            parts = list(map(int, f.readline().split()))
            faces.append(parts[1:1 + parts[0]])
        faces = np.array(faces, dtype=np.int32)

    if sample is None:
        pts = verts
    else:
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{filepath}: surface sampling needs a triangle mesh.")
        rng = np.random.default_rng() if rng is None else rng

        # Pick triangles with probability proportional to their area, so that
        # the points are uniformly spread over the surface.
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        idx = rng.choice(len(faces), size=sample, p=areas / areas.sum())

        # Uniform point in each chosen triangle: draw (u, v) in the unit square
        # and fold the upper half (u + v > 1) back onto the triangle.
        u = rng.random((sample, 1))
        v = rng.random((sample, 1))
        outside = (u + v) > 1
        u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
        pts = v0[idx] + u * (v1[idx] - v0[idx]) + v * (v2[idx] - v0[idx])
        pts = pts.astype(np.float32)

    if normalize:
        pts = pts - pts.mean(axis=0)
        pts = pts / np.max(np.linalg.norm(pts, axis=1))

    if sample is None:
        return pts, faces
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default="table", help="category name or index (0-39)")
    parser.add_argument("--num", type=int, default=6, help="object number")
    args = parser.parse_args()

    # --obj accepts either a name ("table") or an index ("33").
    category = CATEGORIES[int(args.obj)] if args.obj.isdigit() else args.obj
    path = modelnet_path(category, args.num)
    print(f"Loading {path}")

    # 16^3 points, so that they exactly fill a 32 x 32 x 32 grid.
    x = read_off(path, sample=32**3)
    print(f"Point cloud: {x.shape}")

    # Smallest cubic grid that holds all the points; padding fills the
    # remaining cells when N is not a perfect cube.
    n_points = x.shape[0]
    side = math.ceil(np.cbrt(n_points))
    grid_shape = (side, side, side)
    n_cells = side**3

    x_sorted = grid.sort(
        grid.pad(x, n_total=n_cells),
        gridshape=grid_shape,
        n_iter=100,
        level=3,
    )

    demo.show_result(x_sorted.reshape(*grid_shape, 3), animate=True)


if __name__ == "__main__":
    main()
