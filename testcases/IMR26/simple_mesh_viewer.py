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

import argparse
import os

# Initialize the parser
parser = argparse.ArgumentParser(
    description="Graph the mesh described by a node and an element text files."
)

parser.add_argument("node_file", help="Path to the input node file")
parser.add_argument("elem_file", help="Path to the input element file")
args = parser.parse_args()

if not os.path.exists(args.node_file):
    print(f"Error: {args.node_file} does not exist.")
if not os.path.exists(args.elem_file):
    print(f"Error: {args.elem_file} does not exist.")

nodes = np.loadtxt(args.node_file, np.float32, delimiter=" ")
elems = np.loadtxt(args.elem_file, np.uint32, delimiter=" ")

fig, ax = plt.subplots()
ax.triplot(nodes[:, 0], nodes[:, 1], elems, linewidth=2)
ax.scatter(nodes[:, 0], nodes[:, 1])
ax.set_aspect("equal")
plt.show()
