from .. import libnum
from .csr import d_csr_ptr, DeviceCSR

import ctypes
import numpy as np
import numpy.typing as npt


# ========================== Bindings to C functions =========================
linsys_ptr = ctypes.c_void_p

LinSys_create = libnum.LinSys_create
LinSys_create.restype = linsys_ptr
LinSys_create.argtypes = []

LinSys_destroy = libnum.LinSys_destroy
LinSys_destroy.restype = None
LinSys_destroy.argtypes = [linsys_ptr]

LinSys_set = libnum.LinSys_set
LinSys_set.restype = None
LinSys_set.argtypes = [
    linsys_ptr,                                                                # System to set
    d_csr_ptr,                                                                 # Matrix
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags=("C", "ALIGNED"))   # independant vector
]


# ========================== Python wrapper classes =========================

class LinSys:
    def __init__(
            self,
            dcsr: DeviceCSR,
            b: npt.NDArray[np.float32]
            ):

        self.sys = LinSys_create()
        self.set(dcsr, b)

    def set(self, dcsr, b):
        LinSys_set(self.sys, dcsr.dcsr, b.astype(np.float32))

    def __del__(self):
        LinSys_destroy(self.sys)
