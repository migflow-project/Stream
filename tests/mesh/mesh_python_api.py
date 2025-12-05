import stream.mesh as stm
import numpy as np 
import matplotlib.pyplot as plt
from time import perf_counter

plot = True
n = 100
np.random.seed(42)

t = np.linspace(0, 1, n, endpoint=True)

nodes = [
    [x, y] for x in t for y in t
]
nodes = np.array(nodes).astype(np.float32)
nodes += np.random.normal(0, 0.0001, size=(n*n, 2))

# nodes = np.random.uniform(0, 1, size=(n*n, 2)).astype(np.float32)

mesh = stm.Mesh2D()
mesh.set_nodes(nodes)
t0 = perf_counter()
mesh.insert_BVH_neighbors()
# mesh.insert_quadrant_neighbors()
mesh.insert_iterative()
print(f"Time to mesh {n*n} points : {perf_counter() - t0}")
mesh.remove_super_nodes()

if plot:
    ordered_nodes = mesh.get_ordered_nodes()
    elems = mesh.get_elem()

    fig, ax = plt.subplots()
    ax.triplot(ordered_nodes[:, 0], ordered_nodes[:, 1], elems, linewidth=2)
    ax.scatter(ordered_nodes[:, 0], ordered_nodes[:, 1])
    ax.set_aspect("equal")
    plt.show()
