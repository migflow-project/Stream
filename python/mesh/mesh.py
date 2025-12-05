from .. import libmesh
import ctypes
import numpy as np
import numpy.typing as npt

# ============================ Bindings to C functions ========================
mesh2D_ptr = ctypes.c_void_p

mesh2D_create = libmesh.Mesh2D_create
mesh2D_create.restype = mesh2D_ptr
mesh2D_create.argtypes = []

mesh2D_destroy = libmesh.Mesh2D_destroy
mesh2D_destroy.restype = None
mesh2D_destroy.argtypes = [mesh2D_ptr]

mesh2D_set_nodes = libmesh.Mesh2D_set_nodes
mesh2D_set_nodes.restype = None 
mesh2D_set_nodes.argtypes = [
    mesh2D_ptr, 
    ctypes.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=("C", "ALIGNED")) 
]

mesh2D_init = libmesh.Mesh2D_init
mesh2D_init.restype = None
mesh2D_init.argtypes = [mesh2D_ptr]

mesh2D_insert_morton_neighbors = libmesh.Mesh2D_insert_morton_neighbors
mesh2D_insert_morton_neighbors.restype = None
mesh2D_insert_morton_neighbors.argtypes = [mesh2D_ptr]

mesh2D_insert_quadrant_neighbors = libmesh.Mesh2D_insert_quadrant_neighbors
mesh2D_insert_quadrant_neighbors.restype = None
mesh2D_insert_quadrant_neighbors.argtypes = [mesh2D_ptr]

mesh2D_insert_BVH_neighbors = libmesh.Mesh2D_insert_BVH_neighbors
mesh2D_insert_BVH_neighbors.restype = None
mesh2D_insert_BVH_neighbors.argtypes = [mesh2D_ptr]

mesh2D_insert_iterative = libmesh.Mesh2D_insert_iterative
mesh2D_insert_iterative.restype = None
mesh2D_insert_iterative.argtypes = [mesh2D_ptr]

mesh2D_remove_super_nodes = libmesh.Mesh2D_remove_super_nodes
mesh2D_remove_super_nodes.restype = None
mesh2D_remove_super_nodes.argtypes = [mesh2D_ptr]

mesh2D_get_nelem = libmesh.Mesh2D_get_nelem
mesh2D_get_nelem.restype = ctypes.c_uint32
mesh2D_get_nelem.argtypes = [mesh2D_ptr]

mesh2D_get_elem = libmesh.Mesh2D_get_elem
mesh2D_get_elem.restype = None
mesh2D_get_elem.argtypes = [
    mesh2D_ptr,
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags=("C", "ALIGNED", "WRITEABLE")) 
]

mesh2D_get_ordered_nodes = libmesh.Mesh2D_get_ordered_nodes
mesh2D_get_ordered_nodes.restype = None
mesh2D_get_ordered_nodes.argtypes = [
    mesh2D_ptr,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=("C", "ALIGNED", "WRITEABLE")) 
]

class Mesh2D:

    def __init__(self):
        self.mesh = mesh2D_create()
        self.nnodes = 0

    def __del__(self):
        mesh2D_destroy(self.mesh)

    def init(self) -> None:
        mesh2D_init(self.mesh)

    def set_nodes(self, nodes : npt.NDArray[np.float32]) -> None:
        nodes = nodes.astype(np.float32)
        self.nnodes = nodes.shape[0]
        mesh2D_set_nodes(self.mesh, self.nnodes, nodes)
        self.init()

    def insert_morton_neighbors(self) -> None:
        mesh2D_insert_morton_neighbors(self.mesh)

    def insert_quadrant_neighbors(self) -> None:
        mesh2D_insert_quadrant_neighbors(self.mesh)

    def insert_BVH_neighbors(self) -> None:
        mesh2D_insert_BVH_neighbors(self.mesh)

    def insert_iterative(self) -> None:
        mesh2D_insert_iterative(self.mesh)

    def remove_super_nodes(self) -> None:
        mesh2D_remove_super_nodes(self.mesh)

    def get_elem(self) -> npt.NDArray[np.uint32]:

        nelems = mesh2D_get_nelem(self.mesh)
        elems = np.empty((nelems, 3), dtype=np.uint32)
        mesh2D_get_elem(self.mesh, elems)

        return elems

    def get_ordered_nodes(self) -> npt.NDArray[np.float32]:

        nodes = np.empty((self.nnodes, 2), dtype=np.float32)
        mesh2D_get_ordered_nodes(self.mesh, nodes)

        return nodes
