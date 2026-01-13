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
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2,
                           flags=("C", "ALIGNED", "WRITEABLE"))
]

mesh2D_get_ordered_nodes = libmesh.Mesh2D_get_ordered_nodes
mesh2D_get_ordered_nodes.restype = None
mesh2D_get_ordered_nodes.argtypes = [
    mesh2D_ptr,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                           flags=("C", "ALIGNED", "WRITEABLE"))
]


class Mesh2D:

    def __init__(self):
        self.mesh = mesh2D_create()
        self.nnodes = 0

    def __del__(self):
        mesh2D_destroy(self.mesh)

    def init(self) -> None:
        mesh2D_init(self.mesh)

    def set_nodes(self, nodes: npt.NDArray[np.float32]) -> None:
        nodes = nodes.astype(np.float32)
        self.nnodes = nodes.shape[0]
        mesh2D_set_nodes(self.mesh, self.nnodes, nodes)
        self.init()

    def insert_morton_neighbors(self) -> None:
        mesh2D_insert_morton_neighbors(self.mesh)

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



# ================================= AlphaShape2D ==============================

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



# ================================= AlphaShape3D ==============================

# ============================ Bindings to C functions ========================
AlphaShape3D_ptr = ctypes.c_void_p

AlphaShape3D_create = libmesh.AlphaShape3D_create
AlphaShape3D_create.restype = AlphaShape3D_ptr
AlphaShape3D_create.argtypes = []

AlphaShape3D_destroy = libmesh.AlphaShape3D_destroy
AlphaShape3D_destroy.restype = None
AlphaShape3D_destroy.argtypes = [AlphaShape3D_ptr]

AlphaShape3D_set_nodes = libmesh.AlphaShape3D_set_nodes
AlphaShape3D_set_nodes.restype = None
AlphaShape3D_set_nodes.argtypes = [
    AlphaShape3D_ptr,                                                          # ashape
    ctypes.c_uint32,                                                           # nnodes
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=("C", "ALIGNED")),  # coords
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags=("C", "ALIGNED"))   # alpha
]

AlphaShape3D_init = libmesh.AlphaShape3D_init
AlphaShape3D_init.restype = None
AlphaShape3D_init.argtypes = [AlphaShape3D_ptr]

AlphaShape3D_compute = libmesh.AlphaShape3D_compute
AlphaShape3D_compute.restype = None
AlphaShape3D_compute.argtypes = [AlphaShape3D_ptr]

AlphaShape3D_get_nelem = libmesh.AlphaShape3D_get_nelem
AlphaShape3D_get_nelem.restype = ctypes.c_uint32
AlphaShape3D_get_nelem.argtypes = [AlphaShape3D_ptr]

AlphaShape3D_get_elem = libmesh.AlphaShape3D_get_elem
AlphaShape3D_get_elem.restype = None
AlphaShape3D_get_elem.argtypes = [
    AlphaShape3D_ptr,
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2,
                           flags=("C", "ALIGNED", "WRITEABLE"))
]

AlphaShape3D_get_ordered_nodes = libmesh.AlphaShape3D_get_ordered_nodes
AlphaShape3D_get_ordered_nodes.restype = None
AlphaShape3D_get_ordered_nodes.argtypes = [
    AlphaShape3D_ptr,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                           flags=("C", "ALIGNED", "WRITEABLE"))
]


class AlphaShape3D:

    def __init__(self):
        self.mesh = AlphaShape3D_create()
        self.nnodes = 0

    def __del__(self):
        AlphaShape3D_destroy(self.mesh)

    def set_nodes(self, nodes: npt.NDArray[np.float32], alpha: npt.NDArray[np.float32]) -> None:
        nodes = nodes.astype(np.float32)
        alpha = alpha.astype(np.float32)
        self.nnodes = nodes.shape[0]
        AlphaShape3D_set_nodes(self.mesh, self.nnodes, nodes, alpha)
        self.init()

    def init(self) -> None:
        AlphaShape3D_init(self.mesh)

    def compute(self) -> None:
        AlphaShape3D_compute(self.mesh)

    def get_elem(self) -> npt.NDArray[np.uint32]:

        nelems = AlphaShape3D_get_nelem(self.mesh)
        elems = np.empty((nelems, 4), dtype=np.uint32)
        AlphaShape3D_get_elem(self.mesh, elems)

        return elems

    def get_ordered_nodes(self) -> npt.NDArray[np.float32]:

        nodes = np.empty((self.nnodes, 3), dtype=np.float32)
        AlphaShape3D_get_ordered_nodes(self.mesh, nodes)

        return nodes
