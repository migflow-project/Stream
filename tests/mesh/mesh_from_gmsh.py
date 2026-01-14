# 
# Stream - Copyright (C) <2025-2026>
# <Universite catholique de Louvain (UCL), Belgique>
# 
# List of the contributors to the development of Stream: see AUTHORS file.
# Description and complete License: see LICENSE file.
# 
# This file is part of Stream. Stream is free software:
# you can redistribute it and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
# 
# Stream is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License along with Stream. 
# If not, see <https://www.gnu.org/licenses/>.
# 
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
