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
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.collections as mc
import matplotlib.patches as mp
import sys

def get_circumcircle(nodes, elem):
    a = nodes[elem[:, 0]]
    b = nodes[elem[:, 1]]
    c = nodes[elem[:, 2]]

    dir1 = b - a
    dir2 = c - a

    c1 = -0.5 * np.sum(dir1*(b+a), axis=1)
    c2 = -0.5 * np.sum(dir2*(c+a), axis=1)

    coeff = dir1[:, 0]*dir2[:, 1] - dir1[:, 1]*dir2[:, 0]

    x = (-c1*dir2[:, 1] + dir1[:, 1]*c2) / coeff
    y = (c1*dir2[:, 0] - dir1[:, 0]*c2) / coeff

    r = np.sqrt((x - a[:, 0])**2 + (y - a[:, 1])**2)
    return np.array([x, y]).T, r

def get_circumcircle_old(pts):
    a, b, c = pts[:3]

    dir1 = b - a
    dir2 = c - a

    c1 = -0.5 * np.dot(dir1, b+a)
    c2 = -0.5 * np.dot(dir2, c+a)

    coeff = dir1[0]*dir2[1] - dir1[1]*dir2[0]

    x = (-c1*dir2[1] + dir1[1]*c2) / coeff
    y = (c1*dir2[0] - dir1[0]*c2) / coeff

    r = np.hypot(x - a[0], y - a[1])

    return [x, y], r

fnode = sys.argv[1]
felem = sys.argv[2]
fnode_complete = sys.argv[3]

nodes = np.loadtxt(fnode, np.float32, delimiter=" ")
elems = np.loadtxt(felem, np.uint32, delimiter=" ")

# c, r = get_circumcircle(nodes, elems)
# circles = [mp.Circle(ci, ri, fill=False) for ci, ri in zip(c, r)]
# circle_coll = mc.PatchCollection(circles, match_original=True)

node_complete = np.loadtxt(fnode_complete, np.bool)
node_complete = np.hstack([node_complete, [1, 1, 1]])

fig, ax = plt.subplots()
# ax.add_collection(circle_coll)
ax.triplot(nodes[:, 0], nodes[:, 1], elems, linewidth=2)
ax.scatter(nodes[:, 0], nodes[:, 1], c=node_complete)
ax.set_aspect("equal")
plt.show()

