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
