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
from .. import libmesh
import ctypes
import numpy as np
import numpy.typing as npt


# ============================ Bindings to C functions ========================
AlphaShape2D_ptr = ctypes.c_void_p

AlphaShape2D_create = libmesh.AlphaShape2D_create
AlphaShape2D_create.restype = AlphaShape2D_ptr
AlphaShape2D_create.argtypes = []

AlphaShape2D_destroy = libmesh.AlphaShape2D_destroy
AlphaShape2D_destroy.restype = None
AlphaShape2D_destroy.argtypes = [AlphaShape2D_ptr]

AlphaShape2D_set_nodes = libmesh.AlphaShape2D_set_nodes
AlphaShape2D_set_nodes.restype = None
AlphaShape2D_set_nodes.argtypes = [
    AlphaShape2D_ptr,                                                          # ashape
    ctypes.c_uint32,                                                           # nnodes
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=("C", "ALIGNED")),  # coords
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags=("C", "ALIGNED"))   # alpha
]

AlphaShape2D_init = libmesh.AlphaShape2D_init
AlphaShape2D_init.restype = None
AlphaShape2D_init.argtypes = [AlphaShape2D_ptr]

AlphaShape2D_compute = libmesh.AlphaShape2D_compute
AlphaShape2D_compute.restype = None
AlphaShape2D_compute.argtypes = [AlphaShape2D_ptr]

AlphaShape2D_get_nelem = libmesh.AlphaShape2D_get_nelem
AlphaShape2D_get_nelem.restype = ctypes.c_uint32
AlphaShape2D_get_nelem.argtypes = [AlphaShape2D_ptr]

AlphaShape2D_get_elem = libmesh.AlphaShape2D_get_elem
AlphaShape2D_get_elem.restype = None
AlphaShape2D_get_elem.argtypes = [
    AlphaShape2D_ptr,
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2,
                           flags=("C", "ALIGNED", "WRITEABLE"))
]

AlphaShape2D_get_ordered_nodes = libmesh.AlphaShape2D_get_ordered_nodes
AlphaShape2D_get_ordered_nodes.restype = None
AlphaShape2D_get_ordered_nodes.argtypes = [
    AlphaShape2D_ptr,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                           flags=("C", "ALIGNED", "WRITEABLE"))
]


class AlphaShape2D:

    def __init__(self):
        self.mesh = AlphaShape2D_create()
        self.nnodes = 0

    def __del__(self):
        AlphaShape2D_destroy(self.mesh)

    def set_nodes(self, nodes: npt.NDArray[np.float32], alpha: npt.NDArray[np.float32]) -> None:
        nodes = nodes.astype(np.float32)
        alpha = alpha.astype(np.float32)
        self.nnodes = nodes.shape[0]
        AlphaShape2D_set_nodes(self.mesh, self.nnodes, nodes, alpha)
        self.init()

    def init(self) -> None:
        AlphaShape2D_init(self.mesh)

    def compute(self) -> None:
        AlphaShape2D_compute(self.mesh)

    def get_elem(self) -> npt.NDArray[np.uint32]:

        nelems = AlphaShape2D_get_nelem(self.mesh)
        elems = np.empty((nelems, 3), dtype=np.uint32)
        AlphaShape2D_get_elem(self.mesh, elems)

        return elems

    def get_ordered_nodes(self) -> npt.NDArray[np.float32]:

        nodes = np.empty((self.nnodes, 2), dtype=np.float32)
        AlphaShape2D_get_ordered_nodes(self.mesh, nodes)

        return nodes

