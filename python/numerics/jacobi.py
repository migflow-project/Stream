from .. import libnum
from .csr import d_csr_ptr, DeviceCSR

import ctypes

# ========================== Bindings to C functions =========================

prec_jacobi_ptr = ctypes.c_void_p

PrecJacobi_create = libnum.PrecJacobi_create
PrecJacobi_create.restype = prec_jacobi_ptr
PrecJacobi_create.argtypes = [d_csr_ptr]

PrecJacobi_destroy = libnum.PrecJacobi_destroy
PrecJacobi_destroy.restype = None
PrecJacobi_destroy.argtypes = [prec_jacobi_ptr]


# ========================== Python wrapper classes =========================

class PrecJacobi:
    def __init__(self, dcsr: DeviceCSR):
        self.prec = PrecJacobi_create(dcsr.dcsr)

    def __del__(self):
        PrecJacobi_destroy(self.prec)
