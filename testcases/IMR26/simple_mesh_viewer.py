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

fnode = sys.argv[1]
felem = sys.argv[2]

nodes = np.loadtxt(fnode, np.float32, delimiter=" ")
elems = np.loadtxt(felem, np.uint32, delimiter=" ")

fig, ax = plt.subplots()
ax.triplot(nodes[:, 0], nodes[:, 1], elems, linewidth=2)
ax.scatter(nodes[:, 0], nodes[:, 1])
ax.set_aspect("equal")
plt.show()
