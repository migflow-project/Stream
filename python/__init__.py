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
import pathlib

# List the possible search path in order of preferences
# 1. The output directory
# 2. The install dir
search_paths = [
    "@CMAKE_BINARY_DIR@",
    "/usr/local/lib",
]

libdir_path = ""
for path in search_paths:
    if pathlib.Path(path).exists():
        libdir_path = path
        break

if libdir_path == "":
    raise ValueError("Could not the libraries")

libgeo  = np.ctypeslib.load_library("libgeometry", libdir_path)
libnum  = np.ctypeslib.load_library("libnumerics", libdir_path)
libmesh = np.ctypeslib.load_library("libmesh",     libdir_path)
