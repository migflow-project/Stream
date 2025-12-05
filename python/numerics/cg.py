from .. import libnum
from .csr import d_csr_ptr, DeviceCSR
from .jacobi import prec_jacobi_ptr, PrecJacobi
from .linsys import linsys_ptr, LinSys

import ctypes
import numpy as np
import numpy.typing as npt


# ========================== Bindings to C functions =========================
solver_cg_ptr = ctypes.c_void_p

solver_cg_create = libnum.SolverCG_create
solver_cg_create.restype = solver_cg_ptr
solver_cg_create.argtypes = []

solver_cg_destroy = libnum.SolverCG_destroy
solver_cg_destroy.restype = None
solver_cg_destroy.argtypes = [solver_cg_ptr]

solver_cg_jacobi_solve = libnum.SolverCG_jacobi_solve
solver_cg_jacobi_solve.restype = ctypes.c_uint32
solver_cg_jacobi_solve.argtypes = [
    # solver
    solver_cg_ptr,
    # system
    linsys_ptr,
    # preconditioner
    prec_jacobi_ptr,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags=("C", "ALIGNED", "WRITEABLE"))   # val
]


# ========================== Python wrapper classes =========================


class SolverCG:
    def __init__(self):
        self.solver = solver_cg_create()

    def __del__(self):
        solver_cg_destroy(self.solver)

    def solve_jacobi(
            self,
            sys: LinSys,
            prec: PrecJacobi,
            x: npt.NDArray[np.float32]
    ):

        if (type(prec) is not PrecJacobi):
            raise TypeError(
                f"SolverCG.solve_jacobi expects a 'PrecJacobi' preconditioner, not a '{type(prec)}'")

        sol = np.copy(x).astype(np.float32)
        niter = solver_cg_jacobi_solve(self.solver, sys.sys, prec.prec, sol)
        x[:] = sol[:]
        return niter
