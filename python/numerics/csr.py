from .. import libnum
import ctypes
import numpy as np
import numpy.typing as npt


# ========================== Bindings to C functions =========================
d_csr_ptr = ctypes.c_void_p
h_csr_ptr = ctypes.c_void_p

d_csr_create = libnum.d_csr_create
d_csr_create.restype = d_csr_ptr
d_csr_create.argtypes = []

d_csr_destroy = libnum.d_csr_destroy
d_csr_destroy.restype = None
d_csr_destroy.argtypes = [d_csr_ptr]

h_csr_create = libnum.h_csr_create
h_csr_create.restype = h_csr_ptr
h_csr_create.argtypes = [
    ctypes.c_uint32,                                                           # n
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags=("C", "ALIGNED")),   # row
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags=("C", "ALIGNED")),   # col
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags=("C", "ALIGNED"))   # val
]

h_csr_destroy = libnum.h_csr_destroy
h_csr_destroy.restype = None
h_csr_destroy.argtypes = [h_csr_ptr]

d_csr_h2d = libnum.d_csr_h2d
d_csr_h2d.restype = None
d_csr_h2d.argtypes = [
    d_csr_ptr,
    h_csr_ptr
]


# ========================== Python wrapper classes =========================
class HostCSR:
    def __init__(
            self,
            row: npt.NDArray[np.uint32],
            col: npt.NDArray[np.uint32],
            val: npt.NDArray[np.float32]
            ):
        self.n = len(row) - 1
        self.nnz = row[self.n]

        # Init C++ host CSR from numpy arrays
        self.hcsr = h_csr_create(
            self.n,
            row.astype(np.uint32),
            col.astype(np.uint32),
            val.astype(np.float32)
        )

    def __del__(self):
        h_csr_destroy(self.hcsr)


class DeviceCSR:
    def __init__(self, host_csr: HostCSR):
        # Init empty C++ device csr
        self.n = host_csr.n
        self.dcsr = d_csr_create()

        # Move the csr from host to device
        d_csr_h2d(self.dcsr, host_csr.hcsr)

    def __del__(self):
        d_csr_destroy(self.dcsr)
