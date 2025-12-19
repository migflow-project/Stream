import stream.mesh as stm
import numpy as np 
import matplotlib.pyplot as plt
from time import perf_counter
import gmsh
import sys

plot = True

if len(sys.argv) < 2: 
    print("Usage : python mesh_from_gmsh.py <.geo file>")

fname = sys.argv[1]

gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.35)
gmsh.open(fname)
gmsh.model.mesh.generate(2)

nodeTags, coords, _ = gmsh.model.mesh.getNodes(includeBoundary=False, returnParametricCoord=False)
coords = coords.reshape((-1, 3))[:, :2].astype(np.float32)
n = len(coords)

mesh = stm.Mesh2D()
mesh.set_nodes(coords)
t0 = perf_counter()
mesh.insert_BVH_neighbors()
mesh.insert_iterative()
print(f"Time to mesh {n} points : {perf_counter() - t0}")
mesh.remove_super_nodes()

if plot:
    ordered_nodes = mesh.get_ordered_nodes()
    elems = mesh.get_elem()

    fig, ax = plt.subplots()
    ax.triplot(ordered_nodes[:, 0], ordered_nodes[:, 1], elems, linewidth=2)
    ax.scatter(ordered_nodes[:, 0], ordered_nodes[:, 1])
    ax.set_aspect("equal")
    plt.show()

gmsh.finalize()
